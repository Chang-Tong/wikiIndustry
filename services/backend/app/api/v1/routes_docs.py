from __future__ import annotations

import json
import logging
import re
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.settings import settings
from app.integrations.neo4j.client import GraphEdge, GraphNode, Neo4jClient, stable_id
from app.integrations.ragflow.client import RAGFlowClient
from app.services.embedding_service import EmbeddingService
from app.store.sqlite import SqliteStore

router = APIRouter()

# EmbeddingService 单例 + 问题向量缓存（避免每次检索都重复请求 Ollama）
_embed_svc_instance: EmbeddingService | None = None
_question_emb_cache: dict[str, list[float]] = {}
_MAX_EMB_CACHE = 50


def _get_embed_svc() -> EmbeddingService:
    global _embed_svc_instance
    if _embed_svc_instance is None:
        _embed_svc_instance = EmbeddingService()
    return _embed_svc_instance


async def _get_question_embedding(question: str) -> list[float] | None:
    """获取问题的 embedding，带简单内存缓存。"""
    if question in _question_emb_cache:
        return _question_emb_cache[question]
    svc = _get_embed_svc()
    emb = await svc.embed_single(question)
    if len(_question_emb_cache) >= _MAX_EMB_CACHE:
        _question_emb_cache.pop(next(iter(_question_emb_cache)))
    _question_emb_cache[question] = emb
    return emb


@router.get("/debug/settings")
async def debug_settings() -> dict[str, Any]:
    """Debug endpoint to check settings."""
    return {
        "openai_base_url": settings.openai_base_url,
        "openai_api_key_set": bool(settings.openai_api_key),
        "openai_api_key_prefix": settings.openai_api_key[:10] + "..." if settings.openai_api_key else None,
        "openai_model": settings.openai_model,
        "require_ollama_embedding": settings.require_ollama_embedding,
        "ollama_base_url": settings.ollama_base_url,
    }


@router.post("/debug/rag-test")
async def debug_rag_test(request: Request) -> dict[str, Any]:
    """Debug endpoint to test RAG engine."""
    from app.services.rag_engine_v2 import AdaptiveRAGEngine
    neo4j: Neo4jClient | None = request.app.state.neo4j

    if neo4j is None:
        return {"error": "Neo4j not available"}

    # Read question from request body
    try:
        body = await request.json()
        question = body.get("question", "最近有哪些教育政策？")
    except Exception:
        question = "最近有哪些教育政策？"

    engine = AdaptiveRAGEngine(neo4j_client=neo4j)

    # Test full RAG pipeline
    result = await engine.answer(question=question, top_k=10)

    return {
        "answer": result.answer,
        "sources_count": len(result.sources),
        "query_plans": [
            {
                "thinking": plan.thinking,
                "queries": plan.queries,
                "needs_direct_analysis": plan.needs_direct_analysis,
                "follow_up_needed": plan.follow_up_needed,
            }
            for plan in result.query_plans
        ],
        "confidence": result.confidence,
        "reasoning_process": result.reasoning_process,
        "settings_api_key_set": bool(settings.openai_api_key),
        "settings_base_url": settings.openai_base_url,
    }


class CreateDocRequest(BaseModel):
    title: str = Field(default="untitled", min_length=1, max_length=200)
    text: str = Field(min_length=1)


class CreateDocResponse(BaseModel):
    doc_id: str


@router.post("/docs", response_model=CreateDocResponse)
async def create_doc(payload: CreateDocRequest, request: Request) -> CreateDocResponse:
    store: SqliteStore = request.app.state.store
    doc_id = uuid4().hex
    store.create_doc(doc_id=doc_id, title=payload.title, text=payload.text)
    return CreateDocResponse(doc_id=doc_id)


class RetrievedChunk(BaseModel):
    text: str
    score: float | None = None
    meta: dict[str, Any] | None = None


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


def _extract_chunks_from_ragflow_response(data: object) -> list[RetrievedChunk]:
    if not isinstance(data, dict):
        return []
    for key in ("chunks", "contexts", "data", "results"):
        v = data.get(key)
        if isinstance(v, list):
            out: list[RetrievedChunk] = []
            for row in v:
                if isinstance(row, str) and row.strip():
                    out.append(RetrievedChunk(text=row.strip()))
                    continue
                if not isinstance(row, dict):
                    continue
                text = row.get("text") or row.get("content") or row.get("chunk") or ""
                if not isinstance(text, str) or not text.strip():
                    continue
                score = row.get("score")
                meta = row.get("meta")
                out.append(
                    RetrievedChunk(
                        text=text.strip(),
                        score=float(score) if isinstance(score, (int, float)) else None,
                        meta=meta if isinstance(meta, dict) else None,
                    )
                )
            return out
    return []


_STOPWORDS: set[str] = {
    "的", "是", "在", "和", "了", "有", "我", "都", "个", "与", "也", "对", "为", "能", "很",
    "可以", "就", "不", "会", "要", "没有", "我们的", "哪些", "什么", "怎么", "为什么",
    "哪里", "谁", "多少", "几", "一个", "一些", "这些", "那些", "这个", "那个", "吗",
    "呢", "吧", "啊", "嗯", "哦", "发布", "涉及", "关于", "相关", "政策", "新闻", "文件",
    "有哪些", "有什么", "是什么", "分别是", "分别是哪些", "分别是什", "分别是",
}

_QUESTION_WORDS: set[str] = {"哪些", "什么", "怎么", "为什么", "哪里", "谁", "多少", "几", "吗", "呢", "吧"}


def _filter_query_terms(terms: list[str]) -> list[str]:
    """过滤停用词和过短/过长的检索词。"""
    out: list[str] = []
    for t in terms:
        t = t.strip("。？！，；：\"'（）()【】《》")  # noqa: RUF001
        if not t or t in _STOPWORDS:
            continue
        if len(t) < 2:
            continue
        # 过滤包含疑问词的完整句子/片段
        if any(qw in t for qw in _QUESTION_WORDS):
            continue
        # 过滤过长的非实体词（如整句话）
        if len(t) > 20 and " " not in t and len(set(t)) < 5:
            continue
        out.append(t)
    return out


def _extract_ngram_terms(question: str, max_len: int = 3, min_len: int = 2) -> list[str]:
    """用滑动窗口提取中文 n-gram 作为关键词候选，过滤明显无意义的片段。"""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", question)
    terms: set[str] = set()
    for length in range(max_len, min_len - 1, -1):
        for i in range(len(text) - length + 1):
            term = text[i:i + length]
            # 过滤以常见无意义字开头/结尾的 n-gram
            if term[0] in "的了呢吗和在是为了就与不" or term[-1] in "的了呢吗和在是为了就与不":
                continue
            # 过滤包含过多停用词的 n-gram
            stop_count = sum(1 for ch in term if ch in "的了呢吗和在是为了就与不")
            if stop_count / len(term) >= 0.5:
                continue
            terms.add(term)
    return list(terms)


async def _expand_query_with_llm(question: str) -> list[str]:
    """使用大模型提取问题中的关键词和实体，替代硬编码的停用词过滤。"""
    if not settings.openai_api_key or not settings.openai_base_url:
        tokens = [t.strip() for t in re.split(r"[\s，。；、,.!?！？:：;；/|（）()【】\[\]\"'']+", question) if t.strip()]
        return _filter_query_terms(list(set(tokens + _extract_ngram_terms(question))))

    system = (
        "你是一个检索词提取器。请从用户问题中提取最重要的核心实体、专有名词、地名、机构名、主题词作为检索关键词。"
        "要求：\n"
        "1. 只输出关键词，词与词之间用空格分隔\n"
        "2. 不要输出完整句子、不要输出解释、不要输出标点\n"
        "3. 不要包含停用词和疑问词（如：的、是、有、哪些、什么、怎么、为什么、发布、政策、新闻）\n"
        "4. 示例：\"教育部有哪些就业政策？\" -> \"教育部 就业\"\n"
        "5. 示例：\"浙江省出台了什么补贴？\" -> \"浙江省 补贴\""
    )
    req: dict[str, object] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"问题：{question}\n关键词："},
        ],
        "temperature": 0.0,
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
                # 如果模型返回了原句或接近原句，视为无效
                clean_content = content.strip().replace(" ", "").replace("，", "").replace(",", "")
                clean_question = question.strip().replace(" ", "").replace("，", "").replace(",", "")
                if clean_content == clean_question or len(clean_content) > len(clean_question) * 0.8:
                    raise ValueError("LLM returned sentence instead of keywords")
                terms = [t.strip() for t in re.split(r"[\s，、,]+", content) if t.strip()]
                filtered = _filter_query_terms(terms)
                if filtered:
                    return filtered
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"LLM query expansion failed: {e}")

    tokens = [t.strip() for t in re.split(r"[\s，。；、,.!?！？:：;；/|（）()【】\[\]\"'']+", question) if t.strip()]
    return _filter_query_terms(list(set(tokens + _extract_ngram_terms(question))))


async def _graph_retrieve(
    *,
    question: str,
    nodes: list[GraphNode] | None = None,
    edges: list[GraphEdge] | None = None,
    top_k: int,
    neo4j: Neo4jClient | None = None,
    doc_id: str | None = None,
    rel_types: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    从 Neo4j 图谱中检索相关信息，返回结构化的知识片段。
    优先在数据库层完成子图提取，避免全图加载到 Python 内存。
    """
    logger = logging.getLogger(__name__)

    q_terms = await _expand_query_with_llm(question)
    # 不再回退到完整问题，整句对 Neo4j CONTAINS 检索无意义
    if not q_terms:
        q_terms = []

    # 优先走 Neo4j 侧检索，避免全图内存加载（仅在显式传入 neo4j 且未传入 nodes 时启用）
    if neo4j is not None and nodes is None:
        kw_nodes, kw_edges = await neo4j.retrieve_subgraph_by_keywords(q_terms, doc_id=doc_id, rel_types=rel_types)
        node_by_id: dict[str, GraphNode] = {n.id: n for n in kw_nodes}
        edge_by_id: dict[str, GraphEdge] = {e.id: e for e in kw_edges}

        q_emb: list[float] | None = None
        try:
            q_emb = await _get_question_embedding(question)
        except Exception as e:
            logger.warning(f"问题 embedding 生成失败，退化为纯关键词检索: {e}")

        if q_emb:
            emb_nodes, emb_edges = await neo4j.retrieve_subgraph_by_embedding(q_emb, doc_id=doc_id)
            for n in emb_nodes:
                node_by_id.setdefault(n.id, n)
            for e in emb_edges:
                edge_by_id.setdefault(e.id, e)

        nodes = list(node_by_id.values())
        edges = list(edge_by_id.values())
    else:
        nodes = nodes or []
        edges = edges or []

    if not nodes:
        return []

    # 获取问题语义向量（用于内存层精排）
    q_emb: list[float] | None = None
    try:
        q_emb = await _get_question_embedding(question)
    except Exception as e:
        logger.warning(f"问题 embedding 生成失败，退化为纯关键词检索: {e}")

    node_by_id = {n.id: n for n in nodes}
    scored: list[tuple[float, RetrievedChunk]] = []

    node_hit: dict[str, float] = {}
    node_semantic: dict[str, float] = {}

    def calculate_score(text: str, terms: list[str]) -> float:
        score = 0.0
        text_lower = text.lower()
        for term in terms:
            term_lower = term.lower()
            if term_lower in text_lower:
                if term_lower == text_lower:
                    score += 3.0
                else:
                    count = text_lower.count(term_lower)
                    score += count * 1.0
        return score

    # 节点检索
    for n in nodes:
        name_score = calculate_score(n.name, q_terms)
        type_score = calculate_score(n.type, q_terms) * 0.5
        hits = name_score + type_score

        semantic = 0.0
        if q_emb:
            node_emb = n.properties.get("embedding")
            if isinstance(node_emb, list) and len(node_emb) > 0:
                semantic = EmbeddingService.cosine_similarity(q_emb, node_emb)

        if hits > 0 or semantic > 0.5:
            node_hit[n.id] = hits
            node_semantic[n.id] = semantic
            hybrid = hits * 0.6 + semantic * 5.0 * 0.4
            scored.append(
                (
                    hybrid,
                    RetrievedChunk(
                        text=f"【{n.type}】{n.name}",
                        score=round(hybrid, 4),
                        meta={
                            "source": "neo4j",
                            "kind": "node",
                            "node_type": n.type,
                            "bm25_score": round(hits, 4),
                            "semantic_score": round(semantic, 4),
                        },
                    ),
                )
            )

    rel_scored: list[tuple[float, RetrievedChunk]] = []
    corr_scored: list[tuple[float, RetrievedChunk]] = []

    for e in edges:
        source = node_by_id.get(e.source_id)
        target = node_by_id.get(e.target_id)
        if not source or not target:
            continue

        rel_desc = _build_relation_description(source, target, e)
        edge_text = f"{rel_desc}\n{e.evidence or ''}".strip()
        term_hits = calculate_score(edge_text, q_terms)
        neighbor_hits = node_hit.get(source.id, 0.0) + node_hit.get(target.id, 0.0)
        keyword_part = term_hits + neighbor_hits * 0.5

        edge_semantic = max(
            node_semantic.get(source.id, 0.0),
            node_semantic.get(target.id, 0.0),
        )

        is_corr = e.type == "CORRELATED_WITH"

        if is_corr:
            if keyword_part <= 0 and edge_semantic <= 0.5:
                continue
            score = keyword_part * 0.6 + edge_semantic * 5.0 * 0.4
        else:
            if keyword_part <= 0 and edge_semantic <= 0.25:
                continue
            score = (keyword_part + 1.0) * 0.8 + edge_semantic * 5.0 * 0.4
            score *= 1.3

        chunk = RetrievedChunk(
            text=rel_desc,
            score=round(score, 4),
            meta={
                "source": "neo4j",
                "kind": "edge",
                "edge_type": e.type,
                "source_type": source.type,
                "target_type": target.type,
                "source_name": source.name,
                "target_name": target.name,
                "bm25_score": round(keyword_part + (0.0 if is_corr else 1.0), 4),
                "semantic_score": round(edge_semantic, 4),
            },
        )

        if is_corr:
            corr_scored.append((score, chunk))
        else:
            rel_scored.append((score, chunk))

    rel_scored.sort(key=lambda x: x[0], reverse=True)
    corr_scored.sort(key=lambda x: x[0], reverse=True)
    scored.sort(key=lambda x: x[0], reverse=True)

    node_quota = max(2, int(top_k * 0.3))
    edge_quota = top_k - node_quota
    rel_quota = max(1, int(edge_quota * 0.4))
    corr_quota = edge_quota - rel_quota

    selected = scored[:node_quota] + rel_scored[:rel_quota] + corr_scored[:corr_quota]
    used_texts = {c.text for _, c in selected}

    for s, c in rel_scored[rel_quota:]:
        if c.text not in used_texts and len(selected) < top_k:
            selected.append((s, c))
            used_texts.add(c.text)

    pool = scored[node_quota:] + corr_scored[corr_quota:]
    pool.sort(key=lambda x: x[0], reverse=True)
    for s, c in pool:
        if c.text not in used_texts and len(selected) < top_k:
            selected.append((s, c))
            used_texts.add(c.text)

    selected.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in selected[:top_k]]

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


def _format_chunks_for_llm(chunks: list[RetrievedChunk]) -> str:
    """将检索到的 chunks 整理为 LLM 友好的 Markdown 表格/结构化文本。"""
    graph_chunks = [c for c in chunks if c.meta and c.meta.get("source") == "neo4j"]
    text_chunks = [c for c in chunks if c.meta and c.meta.get("source") != "neo4j"]

    parts: list[str] = []

    if graph_chunks:
        parts.append("## 知识图谱数据")
        node_rows: list[str] = []
        edge_rows: list[str] = []
        for i, c in enumerate(graph_chunks[:20], 1):
            meta = c.meta or {}
            if meta.get("kind") == "node":
                node_rows.append(f"| {i} | {meta.get('node_type', 'Entity')} | {c.text} | {c.score or '-'} |")
            elif meta.get("kind") == "edge":
                edge_rows.append(
                    f"| {i} | {meta.get('source_name', '')} | {meta.get('edge_type', 'REL')} | "
                    f"{meta.get('target_name', '')} | {c.score or '-'} |"
                )
            else:
                node_rows.append(f"| {i} | - | {c.text} | {c.score or '-'} |")

        if node_rows:
            parts.append("### 相关实体")
            parts.append("| 序号 | 类型 | 名称 | 相关度 |")
            parts.append("|------|------|------|--------|")
            parts.extend(node_rows)

        if edge_rows:
            parts.append("\n### 相关关系")
            parts.append("| 序号 | 源实体 | 关系 | 目标实体 | 相关度 |")
            parts.append("|------|--------|------|----------|--------|")
            parts.extend(edge_rows)

    if text_chunks:
        parts.append("\n## 文本检索数据")
        parts.append("| 序号 | 内容 | 相关度 |")
        parts.append("|------|------|--------|")
        for i, c in enumerate(text_chunks[:10], 1):
            text = c.text.replace('\n', ' ')[:200]
            parts.append(f"| {i} | {text} | {c.score or '-'} |")

    return "\n".join(parts) if parts else "无检索结果"


async def _decompose_question(question: str) -> list[str]:
    """使用 LLM 将复杂问题拆解为 2-4 个子问题，便于分别检索和回答。"""
    if not settings.openai_api_key or not settings.openai_base_url:
        return [question]

    # 简单启发式：过短的查询、纯实体名词、没有疑问词的查询不进行拆解
    _question_marks = {"哪些", "什么", "怎么", "为什么", "如何", "多少", "几", "吗", "呢", "?", "？"}
    is_likely_entity_only = len(question) <= 12 and not any(q in question for q in _question_marks)
    if is_likely_entity_only:
        return [question]

    system = """你是一个问题拆解专家。请判断用户问题是否需要拆解：
- 如果问题复杂（包含多个并列子问题、跨多个主题、需要分步推理），拆解为 2-4 个具体的子问题
- 如果问题简单（只有一个明确意图、纯实体查询、简单的"是什么/有哪些"），直接返回原问题，不要强行拆解
- 子问题必须与原始问题的核心意图一致，禁止编造原问题没有提到的主题（如地理、气候、资源等）
- 只输出 JSON 数组，不要任何解释"""

    req: dict[str, object] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"问题：{question}\n请输出 JSON 数组："},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.openai_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{settings.openai_base_url}/chat/completions", headers=headers, json=req)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = json.loads(content)
            if isinstance(parsed, list) and len(parsed) > 0:
                return [str(x) for x in parsed]
            # fallback: try to extract array from markdown
            import re
            arr_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
            if arr_match:
                return [q.strip('"').strip("'") for q in re.findall(r'"([^"]+)"', arr_match.group(0))] or [question]
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"问题拆解失败: {e}")
    return [question]


async def _llm_answer(*, question: str, chunks: list[RetrievedChunk]) -> str | None:
    """使用 LLM 生成答案，提供完整的知识图谱上下文。"""
    if not settings.openai_api_key or not settings.openai_model:
        return None

    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    context = _format_chunks_for_llm(chunks)

    system = """你是教育领域知识库问答助手，专门分析基于知识图谱构建的新闻和政策文档。

任务要求：
1. 基于提供的知识图谱信息和文本片段回答用户问题
2. 理解图谱中的实体关系（新闻-标签-部门-地区等）
3. 如果问题涉及实体关联（如"某部门发布了哪些政策"），优先使用图谱关系回答
4. 如果问题涉及时序或具体内容，结合文本片段回答
5. 若无法从片段中得到结论，请明确说明"无法确定"

回答时必须包含以下四个部分（使用 Markdown 标题）：

## 直接答案
用 2-4 句话给出核心结论，不要绕弯子。

## 详细分析
分点列出支持该结论的关键信息。每一点都必须引用具体的实体、关系或文本片段作为证据。如果涉及多个地区、部门或主题，请分别阐述。

## 数据依据
以 Markdown 表格形式列出引用的关键实体和关系（最多 10 条）。
| 来源类型 | 名称/关系 | 说明 |
|----------|-----------|------|

## 信息缺口
明确指出哪些子问题在现有数据中找不到答案，禁止编造。

约束：
- 禁止输出 "根据提供的知识图谱..." 这类套话
- 禁止编造不存在的数据
- 如果有定量问题（如"多少条"），必须基于数据依据中的条目计数"""

    user = f"问题：{question}\n\n{context}"

    req: dict[str, object] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=req)
        resp.raise_for_status()
        data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.strip() if isinstance(content, str) and content.strip() else None


@router.post("/rag/ask", response_model=AskResponse)
async def rag_ask(payload: AskRequest, request: Request) -> AskResponse:
    ragflow_url = settings.ragflow_base_url.strip()
    if settings.require_real_ragflow:
        if not ragflow_url:
            raise HTTPException(status_code=400, detail="RAGFLOW_BASE_URL 未配置，请填写真实 RAGFlow 服务地址")
        if not settings.ragflow_api_key.strip():
            raise HTTPException(status_code=400, detail="RAGFLOW_API_KEY 未配置，请填写真实 RAGFlow API Key")
        if ragflow_url.startswith("http://localhost:8000/api/v1/ragflow"):
            raise HTTPException(status_code=400, detail="RAGFLOW_BASE_URL 指向本地 mock，请改为真实 RAGFlow 服务地址")
    store: SqliteStore = request.app.state.store
    ragflow = RAGFlowClient(
        settings.ragflow_base_url,
        api_key=settings.ragflow_api_key,
        dataset_name=settings.ragflow_dataset_name,
    )

    chunks: list[RetrievedChunk] = []
    rag_enabled = bool(settings.ragflow_base_url and settings.ragflow_api_key)

    if rag_enabled:
        try:
            data = await ragflow.query(query=payload.question, top_k=payload.top_k, doc_id=payload.doc_id)
            chunks = _extract_chunks_from_ragflow_response(data)
        except Exception as e:
            if settings.require_real_ragflow:
                raise HTTPException(status_code=502, detail=f"ragflow_query_failed: {repr(e)}")
            chunks = []

    if not chunks:
        if not payload.doc_id:
            raise HTTPException(status_code=400, detail="doc_id_required_when_ragflow_disabled")
        doc = store.get_doc(payload.doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="doc_not_found")
        chunks = _simple_retrieve(question=payload.question, text=doc.text, top_k=payload.top_k)

    answer = await _llm_answer(question=payload.question, chunks=chunks)
    llm_enabled = bool(settings.openai_api_key and settings.openai_model)
    return AskResponse(rag_enabled=rag_enabled, llm_enabled=llm_enabled, answer=answer, chunks=chunks)


class RagflowIngestRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    title: str = Field(default="untitled", min_length=1, max_length=200)
    text: str = Field(min_length=1)


class RagflowQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=6, ge=1, le=20)
    doc_id: str | None = None


@router.post("/ragflow/ingest")
async def ragflow_ingest(payload: RagflowIngestRequest, request: Request) -> dict[str, object]:
    store: SqliteStore = request.app.state.store
    store.upsert_doc(doc_id=payload.doc_id, title=payload.title, text=payload.text)
    return {"ok": True, "doc_id": payload.doc_id}


@router.post("/ragflow/query")
async def ragflow_query(payload: RagflowQueryRequest, request: Request) -> dict[str, object]:
    store: SqliteStore = request.app.state.store
    if not payload.doc_id:
        raise HTTPException(status_code=400, detail="doc_id_required")
    doc = store.get_doc(payload.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="doc_not_found")
    chunks = _simple_retrieve(question=payload.query, text=doc.text, top_k=payload.top_k)
    out = []
    for c in chunks:
        out.append(
            {
                "text": c.text,
                "score": c.score,
                "meta": {"doc_id": doc.doc_id, "title": doc.title},
            }
        )
    return {"chunks": out}


@router.post("/qa/ask", response_model=AskResponse)
async def qa_ask(payload: AskRequest, request: Request) -> AskResponse:
    ragflow_url = settings.ragflow_base_url.strip()
    if settings.require_real_ragflow:
        if not ragflow_url:
            raise HTTPException(status_code=400, detail="RAGFLOW_BASE_URL 未配置，请填写真实 RAGFlow 服务地址")
        if not settings.ragflow_api_key.strip():
            raise HTTPException(status_code=400, detail="RAGFLOW_API_KEY 未配置，请填写真实 RAGFlow API Key")
        if ragflow_url.startswith("http://localhost:8000/api/v1/ragflow"):
            raise HTTPException(status_code=400, detail="RAGFLOW_BASE_URL 指向本地 mock，请改为真实 RAGFlow 服务地址")
    store: SqliteStore = request.app.state.store
    if not payload.doc_id:
        raise HTTPException(status_code=400, detail="doc_id_required")
    doc = store.get_doc(payload.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="doc_not_found")

    ragflow = RAGFlowClient(
        settings.ragflow_base_url,
        api_key=settings.ragflow_api_key,
        dataset_name=settings.ragflow_dataset_name,
    )
    rag_enabled = bool(settings.ragflow_base_url and settings.ragflow_api_key)
    text_chunks: list[RetrievedChunk] = []
    if rag_enabled:
        try:
            data = await ragflow.query(query=payload.question, top_k=payload.top_k, doc_id=payload.doc_id)
            text_chunks = _extract_chunks_from_ragflow_response(data)
        except Exception as e:
            if settings.require_real_ragflow:
                raise HTTPException(status_code=502, detail=f"ragflow_query_failed: {repr(e)}")
            text_chunks = []
    if not text_chunks:
        text_chunks = _simple_retrieve(question=payload.question, text=doc.text, top_k=payload.top_k)

    graph_chunks: list[RetrievedChunk] = []
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is not None:
        try:
            nodes, edges = await neo4j.read_graph_by_doc_id(doc_id=payload.doc_id)
            graph_chunks = await _graph_retrieve(
                question=payload.question,
                nodes=nodes,
                edges=edges,
                top_k=payload.top_k,
            )
        except Exception:
            graph_chunks = []

    chunks = _merge_chunks(graph_chunks, text_chunks, payload.top_k)
    answer = await _llm_answer(question=payload.question, chunks=chunks)
    llm_enabled = bool(settings.openai_api_key and settings.openai_model)
    return AskResponse(rag_enabled=rag_enabled, llm_enabled=llm_enabled, answer=answer, chunks=chunks)


@router.post("/qa/ask-many", response_model=AskResponse)
async def qa_ask_many(payload: AskManyRequest, request: Request) -> AskResponse:
    ragflow_url = settings.ragflow_base_url.strip()
    if settings.require_real_ragflow:
        if not ragflow_url:
            raise HTTPException(status_code=400, detail="RAGFLOW_BASE_URL 未配置，请填写真实 RAGFlow 服务地址")
        if not settings.ragflow_api_key.strip():
            raise HTTPException(status_code=400, detail="RAGFLOW_API_KEY 未配置，请填写真实 RAGFlow API Key")
        if ragflow_url.startswith("http://localhost:8000/api/v1/ragflow"):
            raise HTTPException(status_code=400, detail="RAGFLOW_BASE_URL 指向本地 mock，请改为真实 RAGFlow 服务地址")
    doc_ids = [x for x in payload.doc_ids if x.strip()][:200]
    if not doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids_required")
    return await _qa_across_docs(question=payload.question, top_k=payload.top_k, doc_ids=doc_ids, request=request)


async def _qa_across_docs(*, question: str, top_k: int, doc_ids: list[str], request: Request) -> AskResponse:
    store: SqliteStore = request.app.state.store
    ragflow = RAGFlowClient(
        settings.ragflow_base_url,
        api_key=settings.ragflow_api_key,
        dataset_name=settings.ragflow_dataset_name,
    )
    rag_enabled = bool(settings.ragflow_base_url and settings.ragflow_api_key)
    neo4j: Neo4jClient | None = request.app.state.neo4j
    all_text_chunks: list[RetrievedChunk] = []
    all_graph_chunks: list[RetrievedChunk] = []

    per_doc_k = max(2, min(6, top_k // 2))
    for doc_id in doc_ids:
        doc = store.get_doc(doc_id)
        if doc is None:
            continue
        text_chunks: list[RetrievedChunk] = []
        if rag_enabled:
            try:
                data = await ragflow.query(query=question, top_k=per_doc_k, doc_id=doc_id)
                text_chunks = _extract_chunks_from_ragflow_response(data)
            except Exception as e:
                if settings.require_real_ragflow:
                    raise HTTPException(status_code=502, detail=f"ragflow_query_failed: {repr(e)}")
                text_chunks = []
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
    return AskResponse(rag_enabled=rag_enabled, llm_enabled=llm_enabled, answer=answer, chunks=chunks)


@router.post("/qa/ask-kb", response_model=AskResponse)
async def qa_ask_kb(payload: AskKnowledgeRequest, request: Request) -> AskResponse:
    ragflow_url = settings.ragflow_base_url.strip()
    if settings.require_real_ragflow:
        if not ragflow_url:
            raise HTTPException(status_code=400, detail="RAGFLOW_BASE_URL 未配置，请填写真实 RAGFlow 服务地址")
        if not settings.ragflow_api_key.strip():
            raise HTTPException(status_code=400, detail="RAGFLOW_API_KEY 未配置，请填写真实 RAGFlow API Key")
        if ragflow_url.startswith("http://localhost:8000/api/v1/ragflow"):
            raise HTTPException(status_code=400, detail="RAGFLOW_BASE_URL 指向本地 mock，请改为真实 RAGFlow 服务地址")
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
    """基于知识图谱的问答（不依赖RAGFlow）。

    Args:
        payload: 问答请求
        request: FastAPI请求

    Returns:
        包含答案和图谱来源的响应
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(status_code=503, detail="Neo4j not available")

    try:
        # Clear previous query logs
        neo4j.clear_query_logs()

        # 预取 Schema 摘要（供 LLM 策略决策和 Cypher 生成共用）
        schema_summary: dict[str, Any] = {}
        try:
            schema_summary = await neo4j.get_schema_summary()
        except Exception:
            pass

        # Step 0: LLM 决定关键词检索时应优先扩展哪些关系类型
        preferred_rel_types: list[str] | None = None
        if schema_summary:
            preferred_rel_types = await _llm_decide_expansion_strategy(payload.question, schema_summary)

        # Step 1: 问题拆解，获取子问题列表
        sub_questions = await _decompose_question(payload.question)

        # Step 2: 对每个子问题分别检索子图，合并去重
        seen_texts: set[str] = set()
        merged_chunks: list[RetrievedChunk] = []
        for sq in sub_questions[:4]:
            chunks = await _graph_retrieve(
                question=sq,
                top_k=payload.top_k,
                neo4j=neo4j,
                doc_id=payload.doc_id,
                rel_types=preferred_rel_types,
            )
            for ch in chunks:
                if ch.text not in seen_texts:
                    seen_texts.add(ch.text)
                    merged_chunks.append(ch)

        # Step 3: LLM 生成 Cypher 做语义级检索（发挥 LLM 泛化理解能力）+ 自我修正
        executed_cyphers: list[str] = []
        if schema_summary:
            cypher_queries = await _llm_generate_cypher_queries(payload.question, schema_summary, max_queries=2)
            for q in cypher_queries:
                cypher_nodes: list[GraphNode] = []
                cypher_edges: list[GraphEdge] = []
                success = False
                try:
                    cypher_nodes, cypher_edges = await _run_cypher_and_collect_graph(neo4j, q)
                    if cypher_nodes or cypher_edges:
                        success = True
                        executed_cyphers.append(q)
                    else:
                        raise ValueError("查询返回空结果")
                except Exception as e:
                    # 尝试自我修正
                    fixed_q = await _llm_fix_cypher_query(payload.question, q, str(e), schema_summary)
                    if fixed_q and fixed_q != q:
                        try:
                            cypher_nodes, cypher_edges = await _run_cypher_and_collect_graph(neo4j, fixed_q)
                            if cypher_nodes or cypher_edges:
                                success = True
                                executed_cyphers.append(fixed_q)
                        except Exception:
                            pass

                if success and (cypher_nodes or cypher_edges):
                    cypher_chunks = await _graph_retrieve(
                        question=payload.question,
                        nodes=cypher_nodes,
                        edges=cypher_edges,
                        top_k=payload.top_k,
                    )
                    for ch in cypher_chunks:
                        if ch.text not in seen_texts:
                            seen_texts.add(ch.text)
                            merged_chunks.append(ch)

        # 如果拆解后没有拿到足够结果，再用原问题补一次
        if len(merged_chunks) < payload.top_k // 2:
            extra = await _graph_retrieve(
                question=payload.question,
                top_k=payload.top_k,
                neo4j=neo4j,
                doc_id=payload.doc_id,
                rel_types=preferred_rel_types,
            )
            for ch in extra:
                if ch.text not in seen_texts:
                    seen_texts.add(ch.text)
                    merged_chunks.append(ch)

        graph_chunks = merged_chunks[:payload.top_k * 2]

        # 获取全图统计（轻量 count）
        try:
            stats = await neo4j.get_schema_stats()
            total_nodes = stats.get("total_nodes", 0)
            total_edges = stats.get("total_edges", 0)
        except Exception:
            total_nodes = 0
            total_edges = 0

        if not graph_chunks and total_nodes == 0:
            return GraphAskResponse(
                answer="知识图谱中暂无数据，请先上传JSON文件构建图谱。",
                sources=[],
                total_nodes=0,
                total_edges=0,
            )

        # Generate Cypher query for transparency
        cypher_query = executed_cyphers[0] if executed_cyphers else _generate_cypher_query(payload.question)

        # Generate answer with LLM（传入拆解后的子问题作为提示）
        enriched_question = payload.question
        if len(sub_questions) > 1:
            enriched_question += "\n\n（拆解后的关注点：" + "；".join(sub_questions[1:]) + "）"

        answer = await _llm_graph_answer(
            question=enriched_question,
            chunks=graph_chunks,
            use_schema=payload.use_schema,
        )

        # Build sources
        sources = _build_graph_sources(graph_chunks)

        # Get query logs
        query_logs = [
            QueryLogInfo(
                query=log["query"],
                parameters=log["parameters"],
                duration_ms=log["duration_ms"],
                result_count=log["result_count"],
            )
            for log in neo4j.get_query_logs()
        ]

        return GraphAskResponse(
            answer=answer or "无法从知识图谱中找到答案。",
            cypher_query=cypher_query,
            sources=sources,
            total_nodes=total_nodes,
            total_edges=total_edges,
            query_logs=query_logs,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph Q&A failed: {repr(e)}")


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

    # Look for organization names (general pattern for Chinese gov agencies)
    org_match = re.search(r"(.+?[部厅委局会])", question)
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
            source = meta.get("source_name") or ""
            target = meta.get("target_name") or ""
            # Fallback: try to extract from evidence text
            if not source or not target:
                if " -[" in evidence and "]-> " in evidence:
                    parts = evidence.split(" -[")
                    if len(parts) == 2:
                        source = parts[0]
                        rest = parts[1]
                        if "]-> " in rest:
                            target = rest.split("]-> ")[1]
            if source and target:
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


async def _llm_generate_cypher_queries(
    question: str,
    schema: dict[str, Any],
    max_queries: int = 3,
) -> list[str]:
    """让 LLM 基于 Schema 生成 Cypher 查询候选。"""
    if not settings.openai_api_key or not settings.openai_base_url:
        return []

    node_types = [t["type"] for t in schema.get("node_types", [])]
    rel_types = [t["type"] for t in schema.get("rel_types_sample", [])]
    samples = schema.get("samples", [])

    schema_text = """图谱结构说明：
- 所有节点的标签都是 :Entity，实体类型通过 n.type 属性区分
- 常见节点类型: """ + ", ".join(node_types[:15]) + """
- 常见关系类型（存在 r.type 属性中）: """ + ", ".join(rel_types[:15]) + """
- 数据样本:
"""
    for s in samples[:3]:
        schema_text += f"  ({s['source_type']}){s['source']} -[{s['rel']}]-> ({s['target_type']}){s['target']}\n"

    system = f"""你是一位 Neo4j Cypher 专家。请基于以下图谱 Schema，为用户问题生成最多 {max_queries} 个 Cypher 查询。
{schema_text}

约束：
1. 只返回 Cypher 查询语句，每行一个，不要序号、不要代码块标记、不要解释
2. 节点标签统一使用 :Entity，通过 n.type 过滤具体类型
3. 模糊匹配请直接硬编码具体值，如 n.name CONTAINS '教育部'，不要用 $term 参数占位
4. 返回节点和关系供后续展示，优先 RETURN n, r, m 或 RETURN n LIMIT 20
5. 如果问题涉及统计，可用 count(*) 等聚合函数
6. 关系过滤不要捏造 Schema 中不存在的关系类型；如果不确定，宁可不写关系类型过滤，直接 MATCH (n)-[r]-(m)
7. 如果无法写出有意义的查询，返回空行"""

    req: dict[str, object] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"问题：{question}\nCypher："},
        ],
        "temperature": 0.0,
        "max_tokens": 800,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.openai_api_key}"}
    queries: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{settings.openai_base_url}/chat/completions", headers=headers, json=req)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            for line in content.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r"^```cypher\s*", "", line)
                line = re.sub(r"^```\s*", "", line)
                line = re.sub(r"^```$", "", line)
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                if line.upper().startswith("MATCH") or line.upper().startswith("CALL") or line.upper().startswith("UNWIND"):
                    queries.append(line)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"LLM Cypher 生成失败: {e}")
    return queries[:max_queries]


async def _run_cypher_and_collect_graph(
    neo4j: Neo4jClient,
    query: str,
    parameters: dict[str, Any] | None = None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """执行 Cypher 并把结果转换为 GraphNode/GraphEdge。"""
    try:
        records = await neo4j.execute_cypher(query, parameters or {})
    except Exception:
        return [], []

    nodes_by_id: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    for record in records:
        for key, value in record.items():
            # Neo4j Node
            if hasattr(value, "labels"):
                props: dict[str, Any] = dict(value)
                node_id = str(props.get("id") or stable_id(props.get("name", ""), props.get("type", "Entity")))
                if node_id in nodes_by_id:
                    continue
                nodes_by_id[node_id] = GraphNode(
                    id=node_id,
                    name=props.get("name", ""),
                    type=props.get("type", "Entity"),
                    doc_id=props.get("doc_id", ""),
                    properties={k: v for k, v in props.items() if k not in ("id", "name", "type", "doc_id")},
                )
            # Neo4j Relationship
            elif hasattr(value, "type") and hasattr(value, "start_node"):
                props = dict(value)
                start = value.start_node
                end = value.end_node
                start_props = dict(start)
                end_props = dict(end)
                src_id = str(start_props.get("id") or stable_id(start_props.get("name", ""), start_props.get("type", "Entity")))
                tgt_id = str(end_props.get("id") or stable_id(end_props.get("name", ""), end_props.get("type", "Entity")))
                edge_id = str(props.get("id") or stable_id(src_id, tgt_id, props.get("type", "")))
                edges.append(
                    GraphEdge(
                        id=edge_id,
                        source_id=src_id,
                        target_id=tgt_id,
                        type=props.get("type", "REL"),
                        label=props.get("type", "REL"),
                        doc_id=props.get("doc_id", ""),
                        evidence=props.get("evidence"),
                        properties={k: v for k, v in props.items() if k not in ("id", "source_id", "target_id", "type", "doc_id", "evidence")},
                    )
                )

    return list(nodes_by_id.values()), edges


async def _llm_fix_cypher_query(
    question: str,
    original_query: str,
    error_info: str,
    schema: dict[str, Any],
) -> str | None:
    """LLM 基于错误信息修正 Cypher 查询，生成更保守、更泛化的版本。"""
    if not settings.openai_api_key or not settings.openai_base_url:
        return None

    node_types = [t["type"] for t in schema.get("node_types", [])]
    rel_types = [t["type"] for t in schema.get("rel_types_sample", [])]

    schema_text = """图谱结构：
- 节点标签统一为 :Entity，类型通过 n.type 区分
- 常见节点类型: """ + ", ".join(node_types[:15]) + """
- 常见关系类型（存在 r.type 属性中）: """ + ", ".join(rel_types[:15])

    system = f"""你是一位 Neo4j Cypher 专家。用户的上一条 Cypher 查询执行失败或返回空，请基于错误信息修正它。
{schema_text}

修正原则（按优先级）：
1. 如果原查询过滤了过于具体的 type（如 type='政府机构'），请放宽为模糊匹配（如 n.name CONTAINS '教育部'）或不限制 type
2. 如果关系类型过滤太死（返回空结果常见原因），请删除关系类型过滤，直接用 (n)-[r]-(m) 匹配所有关系
3. 优先使用 n.name CONTAINS '关键词' 做实体定位，不要依赖精确 name 匹配
4. 返回格式：只输出一条修正后的 Cypher 语句，不要任何解释、不要代码块标记"""

    req: dict[str, object] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"问题：{question}\n原查询：{original_query}\n错误/空结果原因：{error_info}\n修正后的 Cypher："},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.openai_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{settings.openai_base_url}/chat/completions", headers=headers, json=req)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            content = re.sub(r"^```cypher\s*", "", content)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"```$", "", content).strip()
            if content.upper().startswith("MATCH") or content.upper().startswith("CALL") or content.upper().startswith("UNWIND"):
                return content
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"LLM Cypher 修正失败: {e}")
    return None


async def _llm_decide_expansion_strategy(
    question: str,
    schema: dict[str, Any],
) -> list[str] | None:
    """让 LLM 根据问题和 Schema，决定关键词检索时应优先扩展哪些关系类型。"""
    if not settings.openai_api_key or not settings.openai_base_url:
        return None

    rel_types = [t["type"] for t in schema.get("rel_types_sample", [])]
    if not rel_types:
        return None

    rel_list = ", ".join(rel_types[:20])
    system = f"""你是一位图谱检索策略专家。请根据用户问题，从以下关系类型中挑选最多 5 个最相关的类型。

可用关系类型: {rel_list}

要求：
1. 只输出关系类型名称，用空格分隔
2. 不要输出解释、不要输出额外标点
3. 如果问题很宽泛（如只给一个地名或机构名），优先选择能连接到新闻/政策/事件的关系（如来自、提及、省份标签、主题标签等）
4. 如果看不出该选哪些，返回空字符串"""

    req: dict[str, object] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"问题：{question}\n优先扩展的关系类型："},
        ],
        "temperature": 0.0,
        "max_tokens": 100,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.openai_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{settings.openai_base_url}/chat/completions", headers=headers, json=req)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not content:
                return None
            selected = [t.strip() for t in re.split(r"[\s，、,]+", content) if t.strip()]
            # 只保留 Schema 中确实存在的关系类型
            valid = [s for s in selected if s in rel_types]
            return valid[:5] if valid else None
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"LLM 扩展策略决策失败: {e}")
    return None


async def _llm_graph_answer(
    *,
    question: str,
    chunks: list[RetrievedChunk],
    use_schema: bool = True,
) -> str | None:
    """Generate answer using LLM with graph context."""
    if not settings.openai_api_key or not settings.openai_model:
        return _build_fallback_answer(question, chunks)

    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    schema_part = ""
    if use_schema:
        schema_part = """
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
"""

    context = schema_part + "\n" + _format_chunks_for_llm(chunks)

    system = """你是基于知识图谱的问答助手，专门分析教育政策和新闻文档。

任务要求：
1. 基于提供的知识图谱数据回答用户问题
2. 理解图谱中的实体类型（Organization, NewsItem, ThemeTag等）和关系类型
3. 回答时引用具体的实体和关系作为证据
4. 如果涉及统计（如"多少"、"几个"），请基于数据计算
5. 如果信息不足，明确说明"根据现有图谱数据无法确定"

回答时必须包含以下四个部分（使用 Markdown 标题）：

## 直接答案
用 2-4 句话给出核心结论，不要绕弯子。

## 详细分析
分点列出支持该结论的关键信息。每一点都必须引用具体的实体、关系或文本片段作为证据。如果涉及多个地区、部门或主题，请分别阐述。

## 数据依据
以 Markdown 表格形式列出引用的关键实体和关系（最多 10 条）。
| 来源类型 | 名称/关系 | 说明 |
|----------|-----------|------|

## 信息缺口
明确指出哪些子问题在现有数据中找不到答案，禁止编造。

约束：
- 禁止输出 "根据提供的知识图谱..." 这类套话
- 禁止编造不存在的数据
- 如果有定量问题（如"多少条"），必须基于数据依据中的条目计数"""

    user = f"问题：{question}\n\n{context}"

    req: dict[str, object] = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=req)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip() if isinstance(content, str) and content.strip() else None
    except Exception as e:
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
    from app.services.rag_engine_v2 import AdaptiveRAGEngine

    neo4j: Neo4jClient | None = request.app.state.neo4j
    store: SqliteStore | None = request.app.state.store

    if neo4j is None:
        raise HTTPException(status_code=503, detail="Neo4j not available")

    try:
        # 创建自适应 RAG 引擎 V2
        engine = AdaptiveRAGEngine(
            neo4j_client=neo4j,
            sqlite_store=store,
        )

        # 执行 RAG V2
        result = await engine.answer(
            question=payload.question,
            doc_id=payload.doc_id,
            top_k=payload.top_k,
        )

        # 获取图谱统计
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        try:
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
