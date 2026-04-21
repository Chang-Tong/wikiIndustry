from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.integrations.neo4j.client import GraphEdge, GraphNode, Neo4jClient
from app.services.rag_engine import RAGEngine
from app.store.sqlite import SqliteStore

router = APIRouter()


class DocDetail(BaseModel):
    doc_id: str
    title: str
    text: str


@router.get("/docs/{doc_id}", response_model=DocDetail)
async def get_doc(doc_id: str, request: Request) -> DocDetail:
    """Get document by ID."""
    store: SqliteStore = request.app.state.store
    doc = store.get_doc(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="doc_not_found")
    return DocDetail(doc_id=doc.doc_id, title=doc.title, text=doc.text)


class RetrievedChunk(BaseModel):
    text: str
    score: float | None = None
    meta: dict[str, str] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    doc_id: str | None = None
    top_k: int = Field(default=6, ge=1, le=20)


class AskResponse(BaseModel):
    rag_enabled: bool
    llm_enabled: bool
    answer: str | None
    chunks: list[RetrievedChunk]


class AskManyRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    doc_ids: list[str] = Field(min_length=1, max_length=200)
    top_k: int = Field(default=12, ge=1, le=40)


class AskKnowledgeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=12, ge=1, le=40)
    max_docs: int = Field(default=60, ge=1, le=400)


def _split_text_chunks(text: str) -> list[str]:
    if not text:
        return []

    def split_long(s: str, limit: int) -> list[str]:
        parts = re.split(r"(?<=[。！？.!?])\s+|\n{1,}", s.strip())
        res: list[str] = []
        buf: list[str] = []
        size = 0
        for p in [x.strip() for x in parts if x and x.strip()]:
            if size + len(p) > limit and buf:
                res.append(" ".join(buf).strip())
                buf = []
                size = 0
            buf.append(p)
            size += len(p) + 1
        if buf:
            res.append(" ".join(buf).strip())
        return res

    chunks: list[str] = []

    if re.search(r"^【\d+】标题：", text, flags=re.M):
        raw = re.split(r"(?=^【\d+】标题：)", text.strip(), flags=re.M)
        for part in [p.strip() for p in raw if p and p.strip()]:
            if len(part) <= 1600:
                chunks.append(part)
            else:
                chunks.extend(split_long(part, 1600))
        return chunks

    raw = re.split(r"\n{2,}", text.strip())
    for part in [p.strip() for p in raw if p.strip()]:
        if len(part) <= 1600:
            chunks.append(part)
        else:
            chunks.extend(split_long(part, 1600))
    return chunks


def _simple_retrieve(*, question: str, text: str, top_k: int) -> list[RetrievedChunk]:
    chunks = _split_text_chunks(text)
    q_terms = [t for t in re.split(r"[\s，。；、,.!?/|]+", question) if t]

    scored: list[tuple[int, str]] = []
    for ch in chunks:
        s = 0
        for t in q_terms:
            if t and t in ch:
                s += 1
        if s > 0:
            scored.append((s, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[:top_k]
    if not picked:
        picked = [(0, ch) for ch in chunks[:top_k]]

    return [RetrievedChunk(text=ch, score=float(s)) for s, ch in picked]


async def _expand_query_with_llm(question: str) -> list[str]:
    """使用大模型提取问题中的关键词和实体，替代硬编码的停用词过滤。"""
    if not settings.openai_api_key or not settings.openai_base_url:
        # 降级：如果没有配置 LLM，使用基础分词
        tokens = [t.strip() for t in re.split(r"[\s，。；、,.!?！？:：;；/|（）()【】\[\]\"'“”‘’]+", question) if t.strip()]
        return [t for t in tokens if len(t) > 1] or tokens

    system = "你是一个检索词提取器。请从用户问题中提取最重要的核心实体、专有名词、地名、主题词作为检索关键词。直接输出关键词列表，词与词之间用空格分隔。不要输出任何解释，不要包含'有哪些'、'什么'、'新闻'等停用词。"
    req: dict[str, object] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "temperature": 0.1,
        "max_tokens": 100,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.openai_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{settings.openai_base_url}/chat/completions", headers=headers, json=req)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                # 分割并清理返回的关键词
                terms = [t.strip() for t in re.split(r"[\s，、,]+", content) if t.strip()]
                return terms
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"LLM query expansion failed: {e}")
        
    # 降级：基础分词
    tokens = [t.strip() for t in re.split(r"[\s，。；、,.!?！？:：;；/|（）()【】\[\]\"'“”‘’]+", question) if t.strip()]
    return [t for t in tokens if len(t) > 1] or tokens


async def _graph_retrieve(
    *, question: str, nodes: list[GraphNode], edges: list[GraphEdge], top_k: int
) -> list[RetrievedChunk]:
    """
    从 Neo4j 图谱中检索相关信息，返回结构化的知识片段。
    加入 BM25 风格的词频相关性计算，替代纯字符串匹配。
    """
    q_terms = await _expand_query_with_llm(question)
    # 如果 LLM 没有提取出词，或者问题太短，回退加上原始问题
    if not q_terms or len(q_terms) == 0:
        q_terms = [question]
    
    # 确保原始问题也在词表中，以防 LLM 丢失关键信息
    original_tokens = [t.strip() for t in re.split(r"[\s，。；、,.!?！？:：;；/|（）()【】\[\]\"'“”‘’]+", question) if t.strip()]
    q_terms = list(set(q_terms + [t for t in original_tokens if len(t) > 1]))

    node_by_id = {n.id: n for n in nodes}
    scored: list[tuple[float, RetrievedChunk]] = []

    node_hit: dict[str, float] = {}
    
    # 简单的文本相关性评分函数 (词频 * 权重)
    def calculate_score(text: str, terms: list[str]) -> float:
        score = 0.0
        text_lower = text.lower()
        for term in terms:
            term_lower = term.lower()
            if term_lower in text_lower:
                # 完整匹配给予更高权重，部分匹配按长度比例
                if term_lower == text_lower:
                    score += 3.0
                else:
                    count = text_lower.count(term_lower)
                    score += count * 1.0
        return score

    if q_terms:
        for n in nodes:
            # 节点名称和类型评分
            name_score = calculate_score(n.name, q_terms)
            type_score = calculate_score(n.type, q_terms) * 0.5  # 类型权重略低
            
            hits = name_score + type_score
            if hits > 0:
                node_hit[n.id] = hits
                scored.append(
                    (
                        hits,
                        RetrievedChunk(
                            text=f"【{n.type}】{n.name}",
                            score=hits,
                            meta={
                                "source": "neo4j",
                                "kind": "node",
                                "node_type": n.type,
                            },
                        ),
                    )
                )

    for e in edges:
        source = node_by_id.get(e.source_id)
        target = node_by_id.get(e.target_id)
        if not source or not target:
            continue

        rel_desc = _build_relation_description(source, target, e)
        edge_text = f"{rel_desc}\n{e.evidence or ''}".strip()

        # 边内容评分
        term_hits = calculate_score(edge_text, q_terms)
        
        # 结合节点分数 (上下文扩散)
        neighbor_hits = node_hit.get(source.id, 0.0) + node_hit.get(target.id, 0.0)
        
        # 综合相关性分数
        score = term_hits + neighbor_hits * 0.5
        if score <= 0:
            continue

        score += 1.0  # 基础分
        scored.append(
            (
                score,
                RetrievedChunk(
                    text=rel_desc,
                    score=score,
                    meta={
                        "source": "neo4j",
                        "kind": "edge",
                        "edge_type": e.type,
                        "source_type": source.type,
                        "target_type": target.type,
                    },
                ),
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]


def _build_relation_description(source: GraphNode, target: GraphNode, edge: GraphEdge) -> str:
    """构建关系的人类可读描述"""
    rel_type = edge.type
    evidence = edge.evidence or ""

    # 根据关系类型定制描述模板
    templates: dict[str, str] = {
        "来自": "{source} 的来源是 {target}",
        "一级分类": "{source} 属于一级分类 [{target}]",
        "子栏目": "{source} 属于子栏目 [{target}]",
        "相关部门": "{source} 涉及部门: {target}",
        "行业标签": "{source} 所属行业: {target}",
        "省份标签": "{source} 涉及省份: {target}",
        "地市标签": "{source} 涉及地市: {target}",
        "主题标签": "{source} 的主题是: {target}",
        "内容类型": "{source} 的内容类型是: {target}",
        "提及": "{source} 内容中提及了 {target}",
        "关联(共现)": "{source} 与 {target} 共同提及相同实体: {evidence}",
        "标签相似": "{source} 与 {target} 具有标签相似性 ({evidence})",
        "摘要相似推荐": "{source} 与 {target} 内容相似 ({evidence})",
        "重复": "{source} 与 {target} 内容重复",
        "网址": "{source} 的链接地址: {target}",
        "日期": "{source} 的发布时间是: {target}",
    }

    template = templates.get(rel_type, "{source} -[{rel_type}]-> {target}")
    desc = template.format(
        source=source.name,
        target=target.name,
        rel_type=rel_type,
        evidence=evidence[:100] if evidence else ""
    )

    if evidence and rel_type not in templates:
        desc += f" [证据: {evidence[:100]}]"

    return desc


def _merge_chunks(primary: list[RetrievedChunk], secondary: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for ch in [*primary, *secondary]:
        key = ch.text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(ch)
        if len(out) >= top_k:
            break
    return out


def _rerank_chunks_by_question(*, chunks: list[RetrievedChunk], question: str, top_k: int) -> list[RetrievedChunk]:
    q_terms = [t for t in re.split(r"[\s，。；、,.!?/|]+", question) if t]
    scored: list[tuple[float, RetrievedChunk]] = []
    for ch in chunks:
        text = ch.text or ""
        base = ch.score if isinstance(ch.score, (int, float)) else 0.0
        hit = float(sum(1 for t in q_terms if t in text))
        score = base + hit
        if score <= 0 and q_terms:
            continue
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return chunks[:top_k]
    return [x[1] for x in scored[:top_k]]


async def _llm_answer(*, question: str, chunks: list[RetrievedChunk]) -> str | None:
    """
    使用 LLM 生成答案，提供完整的知识图谱上下文。
    """
    if not settings.openai_api_key or not settings.openai_model:
        return None

    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    # 分离图谱和文本来源
    graph_chunks = [c for c in chunks if c.meta and c.meta.get("source") == "neo4j"]
    text_chunks = [c for c in chunks if c.meta and c.meta.get("source") != "neo4j"]

    # 构建上下文
    context_parts = []

    # 添加图谱结构说明
    if graph_chunks:
        context_parts.append("=" * 40)
        context_parts.append("【知识图谱信息】")
        context_parts.append("-" * 40)
        context_parts.append("""
知识图谱结构说明:
- NewsItem(新闻条目): 每条新闻摘要，是核心节点
- Organization(组织机构): 如教育部、学校、公司等
- Category/SubCategory(分类): 新闻的栏目分类
- ContentType(内容类型): 如公告、政策法规、新闻动态
- IndustryTag(行业标签): 如教育、数字经济、新能源等
- ProvinceTag/CityTag(地区标签): 涉及的省份和地市
- ThemeTag(主题标签): 如项目建设、资金补贴、安全教育等
- Entity(通用实体): 从文本抽取的人名、地名、机构等

关系类型:
- 来自: 新闻的来源网站
- 一级分类/子栏目: 新闻的分类信息
- 相关部门: 涉及的部门或机构
- 行业/省份/地市/主题标签: 各类标签关系
- 提及: 新闻内容中提及的实体
- 关联(共现): 多个新闻共同提及的实体
- 标签相似/摘要相似推荐: 基于相似度的推荐关系
""")
        context_parts.append("\n【检索到的图谱关系】:")
        for i, c in enumerate(graph_chunks[:8], 1):
            context_parts.append(f"[{i}] {c.text}")

    if text_chunks:
        context_parts.append("\n" + "=" * 40)
        context_parts.append("【文本检索信息】")
        context_parts.append("-" * 40)
        for i, c in enumerate(text_chunks[:8], 1):
            context_parts.append(f"[{i}] {c.text}")

    context = "\n\n".join(context_parts).strip()

    system = """你是教育领域知识库问答助手，专门分析基于知识图谱构建的新闻和政策文档。

你的任务是:
1. 基于提供的知识图谱信息和文本片段回答问题
2. 理解图谱中的实体关系（新闻-标签-部门-地区等）
3. 如果问题涉及实体关联（如"某部门发布了哪些政策"），优先使用图谱关系回答
4. 如果问题涉及时序或具体内容，结合文本片段回答
5. 明确说明信息来源是"知识图谱"还是"文本检索"
6. 若无法从片段中得到结论，请明确说明"无法确定"

回答格式:
- 首先给出直接答案
- 然后说明推理依据（引用了哪些图谱关系或文本）
- 如有多个相关信息，按重要性排序"""

    user = f"问题：{question}\n\n{context}"

    req: dict[str, object] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=req)
        resp.raise_for_status()
        data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.strip() if isinstance(content, str) and content.strip() else None


@router.post("/qa/ask-many", response_model=AskResponse)
async def qa_ask_many(payload: AskManyRequest, request: Request) -> AskResponse:
    doc_ids = [x for x in payload.doc_ids if x.strip()][:200]
    if not doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids_required")
    return await _qa_across_docs(question=payload.question, top_k=payload.top_k, doc_ids=doc_ids, request=request)


async def _qa_across_docs(*, question: str, top_k: int, doc_ids: list[str], request: Request) -> AskResponse:
    store: SqliteStore = request.app.state.store
    neo4j: Neo4jClient | None = request.app.state.neo4j
    all_text_chunks: list[RetrievedChunk] = []
    all_graph_chunks: list[RetrievedChunk] = []

    per_doc_k = max(2, min(6, top_k // 2))
    for doc_id in doc_ids:
        doc = store.get_doc(doc_id)
        if doc is None:
            continue
        text_chunks: list[RetrievedChunk] = []
        if not text_chunks:
            text_chunks = _simple_retrieve(question=question, text=doc.text, top_k=per_doc_k)
        for ch in text_chunks:
            meta = dict(ch.meta or {})
            meta["doc_id"] = doc_id
            all_text_chunks.append(RetrievedChunk(text=ch.text, score=ch.score, meta=meta))

        if neo4j is not None:
            try:
                nodes, edges = await neo4j.read_graph_by_doc_id(doc_id=doc_id)
                gchs = await _graph_retrieve(question=question, nodes=nodes, edges=edges, top_k=per_doc_k)
                for ch in gchs:
                    meta = dict(ch.meta or {})
                    meta["doc_id"] = doc_id
                    all_graph_chunks.append(RetrievedChunk(text=ch.text, score=ch.score, meta=meta))
            except Exception:
                pass

    merged = _merge_chunks(all_graph_chunks, all_text_chunks, top_k=max(top_k * 3, top_k))
    chunks = _rerank_chunks_by_question(chunks=merged, question=question, top_k=top_k)
    answer = await _llm_answer(question=question, chunks=chunks)
    llm_enabled = bool(settings.openai_api_key and settings.openai_model)
    return AskResponse(rag_enabled=False, llm_enabled=llm_enabled, answer=answer, chunks=chunks)


@router.post("/qa/ask-kb", response_model=AskResponse)
async def qa_ask_kb(payload: AskKnowledgeRequest, request: Request) -> AskResponse:
    store: SqliteStore = request.app.state.store
    doc_ids = store.list_finished_doc_ids(limit=payload.max_docs)
    if not doc_ids:
        raise HTTPException(status_code=404, detail="no_finished_docs_in_kb")
    return await _qa_across_docs(question=payload.question, top_k=payload.top_k, doc_ids=doc_ids, request=request)


# ==================== Graph-based Q&A (RAGFlow-free) ====================


class GraphAskRequest(BaseModel):
    """Request for graph-based Q&A (no RAGFlow dependency)."""

    question: str = Field(min_length=1, max_length=1000, description="用户问题")
    doc_id: str | None = Field(default=None, description="指定文档ID（可选）")
    top_k: int = Field(default=10, ge=1, le=50, description="返回的最大结果数")
    use_schema: bool = Field(default=True, description="是否向LLM提供图谱Schema说明")


class GraphSource(BaseModel):
    """Source information for graph-based answer."""

    type: str  # "node" or "edge"
    name: str
    node_type: str | None = None
    relation_type: str | None = None
    source: str | None = None
    target: str | None = None
    evidence: str | None = None
    score: float | None = None


class QueryLogInfo(BaseModel):
    """Neo4j query log information."""

    query: str
    parameters: dict[str, Any]
    duration_ms: float
    result_count: int


class GraphAskResponse(BaseModel):
    """Response for graph-based Q&A."""

    answer: str
    cypher_query: str | None = None
    sources: list[GraphSource]
    total_nodes: int
    total_edges: int
    query_logs: list[QueryLogInfo] = []


@router.post("/qa/ask-graph", response_model=GraphAskResponse)
async def qa_ask_graph(payload: GraphAskRequest, request: Request) -> GraphAskResponse:
    """基于知识图谱 + 文档 RAG 的联合问答。

    Args:
        payload: 问答请求
        request: FastAPI请求

    Returns:
        包含答案、Cypher查询和来源的响应
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    store: SqliteStore | None = request.app.state.store
    if neo4j is None:
        raise HTTPException(status_code=503, detail="Neo4j not available")

    try:
        await neo4j.open()

        # 使用 RAGEngine 执行图数据库 + 文档 RAG 联合检索
        engine = RAGEngine(neo4j_client=neo4j, sqlite_store=store)
        result = await engine.answer(
            question=payload.question,
            doc_id=payload.doc_id,
            top_k=payload.top_k,
        )

        # 获取图谱统计
        stats = await neo4j.get_schema_stats()

        # 提取 Cypher 查询（用于透明度展示）
        cypher_queries: list[str] = []
        for log in result.query_logs:
            if log.get("step") == "generate_cypher":
                cypher_queries = log.get("queries", [])
                break
        cypher_query = "\n".join(cypher_queries) if cypher_queries else None

        # 映射 sources 到 GraphSource
        sources: list[GraphSource] = []
        for chunk in result.sources:
            meta = chunk.get("meta", {}) if isinstance(chunk, dict) else {}
            source_type = chunk.get("type", "unknown") if isinstance(chunk, dict) else "unknown"
            if source_type == "graph":
                sources.append(GraphSource(
                    type="edge",
                    name=str(chunk.get("text", ""))[:200],
                    score=float(chunk.get("score", 0.0)) if isinstance(chunk.get("score"), (int, float)) else None,
                    relation_type=meta.get("kind") if isinstance(meta, dict) else None,
                ))
            elif source_type == "document":
                sources.append(GraphSource(
                    type="document",
                    name=str(meta.get("title", "文档片段"))[:200] if isinstance(meta, dict) else "文档片段",
                    doc_id=str(meta.get("doc_id", "")) if isinstance(meta, dict) else None,
                    score=float(chunk.get("score", 0.0)) if isinstance(chunk.get("score"), (int, float)) else None,
                ))
            else:
                sources.append(GraphSource(
                    type="unknown",
                    name=str(chunk.get("text", str(chunk)))[:200],
                ))

        # 构建 query_logs（兼容展示）
        query_logs: list[QueryLogInfo] = []
        for log in result.query_logs:
            step = log.get("step", "unknown")
            if step == "generate_cypher":
                query_logs.append(QueryLogInfo(
                    query="; ".join(log.get("queries", [])),
                    parameters={},
                    duration_ms=0.0,
                    result_count=0,
                ))
            elif step == "graph_retrieval":
                query_logs.append(QueryLogInfo(
                    query="GRAPH RETRIEVAL",
                    parameters={},
                    duration_ms=0.0,
                    result_count=log.get("chunks_count", 0),
                ))
            elif step == "doc_retrieval":
                query_logs.append(QueryLogInfo(
                    query="DOCUMENT RETRIEVAL",
                    parameters={},
                    duration_ms=0.0,
                    result_count=log.get("chunks_count", 0),
                ))

        return GraphAskResponse(
            answer=result.answer or "无法从知识库中找到答案。",
            cypher_query=cypher_query,
            sources=sources,
            total_nodes=stats.get("total_nodes", 0),
            total_edges=stats.get("total_edges", 0),
            query_logs=query_logs,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph Q&A failed: {repr(e)}")
    finally:
        await neo4j.close()


async def _get_recent_graph_data(neo4j: Neo4jClient, limit: int = 1000) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Get recent graph data from Neo4j.

    Args:
        neo4j: Neo4j client
        limit: Maximum number of nodes to retrieve

    Returns:
        Tuple of (nodes, edges)
    """
    try:
        # Use the first available doc_id from the graph
        # Query for any doc_id that has nodes
        from neo4j import AsyncDriver
        driver: AsyncDriver = neo4j._driver  # type: ignore

        async with driver.session() as session:
            # Get a sample doc_id
            doc_result = await session.run(
                "MATCH (n:Entity) RETURN DISTINCT n.doc_id as doc_id LIMIT 1"
            )
            doc_record = await doc_result.single()
            if not doc_record or not doc_record.get("doc_id"):
                return [], []

            doc_id = doc_record.get("doc_id")
            return await neo4j.read_graph_by_doc_id(doc_id=doc_id)
    except Exception:
        return [], []


def _generate_cypher_query(question: str) -> str | None:
    """Generate a sample Cypher query based on the question.

    This is a heuristic-based query generator for transparency.
    """
    # Extract potential entity names from question
    import re

    # Common patterns
    if "哪些" in question or "什么" in question:
        # Likely asking for list of items
        if "政策" in question or "文件" in question:
            return "MATCH (n:Entity {type: 'NewsItem'}) RETURN n.name LIMIT 10"
        if "部门" in question or "机构" in question:
            return "MATCH (n:Entity {type: 'Organization'}) RETURN n.name LIMIT 10"

    # Look for organization names
    org_pattern = r"(教育部|国务院|.+?[部厅委局会])"
    org_match = re.search(org_pattern, question)
    if org_match:
        org = org_match.group(1)
        return f"MATCH (o:Entity {{name: '{org}'}})-[r:REL]-(n) RETURN n.name, r.type LIMIT 20"

    # Default query
    return "MATCH (n:Entity)-[r:REL]-(m:Entity) RETURN n, r, m LIMIT 50"


def _build_graph_sources(chunks: list[RetrievedChunk]) -> list[GraphSource]:
    """Build source information from graph chunks."""
    sources: list[GraphSource] = []
    seen: set[str] = set()

    for chunk in chunks:
        meta = chunk.meta or {}
        kind = meta.get("kind")

        if kind == "node":
            key = f"node:{chunk.text}"
            if key not in seen:
                seen.add(key)
                # Parse "【type】name" format
                text = chunk.text
                if text.startswith("【") and "】" in text:
                    type_end = text.index("】")
                    node_type = text[1:type_end]
                    name = text[type_end + 1:]
                    sources.append(GraphSource(
                        type="node",
                        name=name,
                        node_type=node_type,
                        score=chunk.score,
                    ))

        elif kind == "edge":
            edge_type = meta.get("edge_type", "关联")
            evidence = chunk.text
            # Try to extract source/target from evidence text
            # Format: "source relation target"
            if " -[" in evidence and "]-> " in evidence:
                parts = evidence.split(" -[")
                if len(parts) == 2:
                    source = parts[0]
                    rest = parts[1]
                    if "]-> " in rest:
                        target = rest.split("]-> ")[1]
                        key = f"edge:{source}:{target}:{edge_type}"
                        if key not in seen:
                            seen.add(key)
                            sources.append(GraphSource(
                                type="edge",
                                name=f"{source} → {target}",
                                relation_type=edge_type,
                                source=source,
                                target=target,
                                evidence=evidence[:200],
                                score=chunk.score,
                            ))

    return sources[:20]  # Limit to top 20 sources


async def _llm_graph_answer(
    *,
    question: str,
    chunks: list[RetrievedChunk],
    use_schema: bool = True,
) -> str | None:
    """Generate answer using LLM with graph context.

    Args:
        question: User question
        chunks: Retrieved graph chunks
        use_schema: Whether to include schema description

    Returns:
        Generated answer
    """
    if not settings.openai_api_key or not settings.openai_model:
        # Fallback: return structured graph info
        return _build_fallback_answer(question, chunks)

    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    # Build context from chunks
    context_parts = []

    if use_schema:
        context_parts.append("""
【知识图谱Schema说明】
节点类型：
- NewsItem: 新闻/政策条目，包含标题、摘要、URL
- Organization: 组织机构（教育部、学校、企业等）
- Category: 一级分类（教育要闻、政策解读等）
- IndustryTag: 行业标签（教育、数字经济等）
- ProvinceTag/CityTag: 省份/城市标签
- ThemeTag: 主题标签（项目建设、资金补贴等）
- Time: 时间/日期
- Entity: 通用实体

关系类型：
- 来自: 新闻来源网站
- 一级分类: 新闻所属分类
- 行业/省份/地市/主题标签: 各类标签关系
- 提及: 新闻内容中提及的实体
- 发布/启动/召开: 组织行为
""")

    # Add retrieved graph data
    if chunks:
        context_parts.append("\n【检索到的图谱数据】:")
        for i, chunk in enumerate(chunks[:15], 1):
            context_parts.append(f"{i}. {chunk.text}")

    context = "\n".join(context_parts)

    system = """你是基于知识图谱的问答助手。

任务要求：
1. 基于提供的知识图谱数据回答用户问题
2. 理解图谱中的实体类型（Organization, NewsItem, ThemeTag等）和关系类型
3. 回答时引用具体的实体和关系作为证据
4. 如果涉及统计（如"多少"、"几个"），请基于数据计算
5. 如果信息不足，明确说明"根据现有图谱数据无法确定"

回答格式：
- 首先给出直接答案
- 然后列出推理依据（引用了哪些实体和关系）
- 可以补充"相关信息"部分提供额外上下文"""

    user = f"问题：{question}\n\n{context}"

    req: dict[str, object] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 2000,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=req)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip() if isinstance(content, str) and content.strip() else None
    except Exception as e:
        # Fallback on error
        return _build_fallback_answer(question, chunks) + f"\n\n(LLM调用失败: {repr(e)})"


def _build_fallback_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Build fallback answer when LLM is not available."""
    if not chunks:
        return "知识图谱中未找到相关信息。"

    lines = ["基于知识图谱找到以下信息：", ""]

    for i, chunk in enumerate(chunks[:10], 1):
        meta = chunk.meta or {}
        kind = meta.get("kind", "unknown")

        if kind == "node":
            lines.append(f"{i}. 【实体】{chunk.text}")
        elif kind == "edge":
            lines.append(f"{i}. 【关系】{chunk.text}")
        else:
            lines.append(f"{i}. {chunk.text}")

    lines.append("")
    lines.append("您的问题：" + question)
    lines.append("（注：当前未配置LLM，仅展示原始图谱数据）")

    return "\n".join(lines)


# ==================== Full RAG Q&A (Graph + Documents + Web) ====================


class RAGAskRequest(BaseModel):
    """Request for full RAG Q&A with multi-source retrieval."""

    question: str = Field(min_length=1, max_length=1000, description="用户问题")
    doc_id: str | None = Field(default=None, description="指定文档ID（可选）")
    top_k: int = Field(default=10, ge=1, le=50, description="返回的最大结果数")


class RAGSource(BaseModel):
    """Source information for RAG answer."""

    type: str  # "graph", "document", "web"
    name: str
    node_type: str | None = None
    relation_type: str | None = None
    doc_id: str | None = None
    url: str | None = None
    score: float = 0.0


class RAGConfidence(BaseModel):
    """Confidence information for RAG answer."""

    level: str = Field(..., description="Confidence level: high/medium/low")
    avg_score: float = Field(..., description="Average retrieval score")
    chunks_count: int = Field(..., description="Total chunks used")
    graph_chunks: int = Field(..., description="Number of graph chunks")
    doc_chunks: int = Field(..., description="Number of document chunks")


class RAGAskResponse(BaseModel):
    """Response for full RAG Q&A (with hallucination mitigation)."""

    answer: str
    sources: list[RAGSource]
    graph_nodes: int
    graph_edges: int
    doc_count: int
    web_results: int
    confidence: RAGConfidence  # 置信度信息


@router.post("/qa/rag", response_model=RAGAskResponse)
async def qa_rag(payload: RAGAskRequest, request: Request) -> RAGAskResponse:
    """
    完整 RAG 问答 - 自适应 LLM 驱动检索（V2）

    使用 AdaptiveRAGEngine V2：
    1. 动态 Schema 发现 - 实时获取图谱结构
    2. LLM 自主决策 - 自己决定检索策略
    3. 自我修正机制 - 支持多轮迭代优化
    """
    from app.services.rag_engine import RAGEngine

    neo4j: Neo4jClient | None = request.app.state.neo4j
    store: SqliteStore | None = request.app.state.store

    if neo4j is None:
        raise HTTPException(status_code=503, detail="Neo4j not available")

    try:
        # 创建 RAG 引擎
        engine = RAGEngine(
            neo4j_client=neo4j,
            sqlite_store=store,
        )

        # 执行 RAG
        result = await engine.answer(
            question=payload.question,
            doc_id=payload.doc_id,
            top_k=payload.top_k,
        )

        # 获取图谱统计
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        try:
            await neo4j.open()
            if payload.doc_id:
                nodes, edges = await neo4j.read_graph_by_doc_id(doc_id=payload.doc_id)
            else:
                nodes, edges = await neo4j.read_all_graph(limit=1000)
        except Exception:
            pass

        # 转换 sources 格式（V2 格式兼容）
        sources: list[RAGSource] = []
        for s in result.sources:
            # V2 的 source 可能包含 source/target/relationship 或简单 dict
            if "source" in s and "target" in s:
                # 路径格式
                sources.append(RAGSource(
                    type="graph_path",
                    name=f"{s['source'].get('name', '')} -> {s['target'].get('name', '')}",
                    node_type=s["source"].get("type"),
                    relation_type=s.get("relationship", {}).get("type") if isinstance(s.get("relationship"), dict) else None,
                ))
            else:
                # 简单格式
                sources.append(RAGSource(
                    type=s.get("type", "unknown"),
                    name=s.get("name", str(s)[:100]),
                    node_type=s.get("node_type"),
                    relation_type=s.get("relation_type"),
                    doc_id=s.get("doc_id"),
                    url=s.get("url"),
                    score=s.get("score", 0.0),
                ))

        # 在回答末尾添加推理过程（如果存在）
        final_answer = result.answer
        if result.reasoning_process:
            final_answer += f"\n\n[推理过程]\n{result.reasoning_process[:500]}..."

        return RAGAskResponse(
            answer=final_answer,
            sources=sources,
            graph_nodes=len(nodes),
            graph_edges=len(edges),
            doc_count=len([s for s in result.sources if s.get("type") == "document"]),
            web_results=0,
            confidence=RAGConfidence(
                level=result.confidence if isinstance(result.confidence, str) else "unknown",
                avg_score=0.0,  # V2 不计算平均分
                chunks_count=len(result.sources),
                graph_chunks=len([s for s in result.sources if s.get("type") in ["graph_path", "graph"]]),
                doc_chunks=len([s for s in result.sources if s.get("type") == "document"]),
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG failed: {repr(e)}")
