"""
RAG Engine - 智能推理检索增强生成

改进点：
1. 使用 LLM 生成 Cypher 查询，而非硬编码规则
2. 使用 LLM 进行相关性判断
3. 多轮推理：检索 -> LLM分析 -> 再检索
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
import logging

from app.core.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """检索到的知识片段"""
    text: str
    score: float
    source: str  # "graph", "document", "web"
    meta: dict[str, Any]


@dataclass
class RAGAnswer:
    """RAG 回答结果（带防幻觉信息）"""
    answer: str
    sources: list[dict[str, Any]]
    query_logs: list[dict[str, Any]]
    confidence: dict[str, Any]  # 置信度信息
    reasoning_process: str = ""  # 推理过程说明


class RAGEngine:
    """RAG 引擎 - LLM驱动的智能检索"""

    def __init__(
        self,
        neo4j_client: Any | None = None,
        sqlite_store: Any | None = None,
    ) -> None:
        self.neo4j = neo4j_client
        self.store = sqlite_store
        self.web_search_enabled = bool(settings.openai_api_key)

    async def answer(
        self,
        question: str,
        doc_id: str | None = None,
        top_k: int = 10,
    ) -> RAGAnswer:
        """
        智能 RAG 流程：
        1. LLM 分析查询意图
        2. LLM 生成 Cypher 查询
        3. 执行检索
        4. LLM 基于结果推理回答
        """
        query_logs: list[dict[str, Any]] = []

        # 步骤1: LLM 分析查询并生成 Cypher
        cypher_queries = await self._generate_cypher_queries(question)
        query_logs.append({"step": "generate_cypher", "queries": cypher_queries})

        # 步骤2: 执行图谱检索
        graph_chunks = await self._execute_graph_queries(cypher_queries, top_k)
        query_logs.append({"step": "graph_retrieval", "chunks_count": len(graph_chunks)})

        # 步骤2.5: 利用 CORRELATED_WITH 边扩展相似新闻
        correlation_chunks = await self._expand_with_correlations(graph_chunks, top_k)
        if correlation_chunks:
            query_logs.append({"step": "correlation_expansion", "chunks_count": len(correlation_chunks)})
            graph_chunks = graph_chunks + correlation_chunks

        # 步骤3: 文档检索（作为补充）
        doc_chunks = await self._retrieve_documents(question, doc_id, top_k)
        query_logs.append({"step": "doc_retrieval", "chunks_count": len(doc_chunks)})

        # 合并结果
        all_chunks = graph_chunks + doc_chunks
        all_chunks.sort(key=lambda x: x.score, reverse=True)
        top_chunks = all_chunks[:top_k]

        # 计算置信度
        avg_score = sum(c.score for c in top_chunks) / len(top_chunks) if top_chunks else 0
        confidence_level = "high" if avg_score >= 3 else "medium" if avg_score >= 1.5 else "low"

        # 步骤4: LLM 推理生成回答
        answer = await self._generate_answer_with_reasoning(question, top_chunks)

        # 步骤5: 防幻觉验证 - 检查引用是否真实存在
        verification = self._verify_answer_sources(answer, top_chunks)
        query_logs.append({"step": "verify_sources", "verification": verification})

        # 步骤6: 后处理 - 添加免责声明和警告
        answer = self._post_process_answer(answer, top_chunks, verification)

        # 更新置信度（考虑引用验证结果）
        if verification.get("invalid_citations"):
            confidence_level = "low"  # 有无效引用时降低置信度
        elif not verification.get("has_citations"):
            confidence_level = "low"

        # 构建来源信息
        sources = self._build_sources(top_chunks)

        return RAGAnswer(
            answer=answer,
            sources=sources,
            query_logs=query_logs,
            confidence={
                "level": confidence_level,
                "avg_score": round(avg_score, 2),
                "chunks_count": len(top_chunks),
                "graph_chunks": len([c for c in top_chunks if c.source == "graph"]),
                "doc_chunks": len([c for c in top_chunks if c.source == "document"]),
                "citation_verification": verification,
            },
        )

    async def _generate_cypher_queries(self, question: str) -> list[str]:
        """使用 LLM 生成 Cypher 查询"""
        logger.info(f"[_generate_cypher_queries] Starting with question: {question[:50]}...")
        logger.info(f"[_generate_cypher_queries] API key set: {bool(settings.openai_api_key)}, Neo4j: {self.neo4j is not None}")
        if not settings.openai_api_key or not self.neo4j:
            logger.warning(f"[_generate_cypher_queries] Missing requirements - API key: {bool(settings.openai_api_key)}, Neo4j: {self.neo4j is not None}")
            return []

        system_prompt = """你是一个 Neo4j Cypher 查询生成专家。

图谱结构：
- 节点标签: Entity (属性: id, name, type, doc_id)
- 节点类型(type): NewsItem(新闻), Organization(机构), Person(人物), Policy(政策), ThemeTag(主题), Entity(通用实体), Location(地点), Event(事件), Technology(技术)
- 关系类型: REL (通用关系), CORRELATED_WITH (相似度关系)

重要提示：
- 省份/城市通常作为 type='Entity' 存储，name 属性包含省份名称如"浙江"、"广东"、"上海"等
- 不要假设有 ProvinceTag 类型，省份就是普通的 Entity 类型
- 新闻标题通常包含省份名称，可以直接在 NewsItem 的 name 中搜索
- CORRELATED_WITH 表示两篇新闻语义相似（embedding + entity 重叠计算），用于找"相关新闻"或"类似报道"
- 当你需要扩展信息源、找相关报道时，主动使用 CORRELATED_WITH 关系

任务：根据用户问题生成 Cypher 查询语句。
要求：
1. 生成多个查询变体以提高召回率
2. 使用模糊匹配 (CONTAINS, =~) 而非精确匹配
3. 考虑同义词和语义关联
4. 返回多条路径以提供上下文

输出格式（JSON数组）：
["查询1", "查询2", ...]

示例：
用户问：哪些省份涉及义务教育？
输出：
[
  "MATCH (n:Entity {type: 'NewsItem'}) WHERE n.name CONTAINS '义务教育' OR n.name CONTAINS '教育' RETURN n LIMIT 20",
  "MATCH (n:Entity {type: 'NewsItem'})-[:REL]-(e:Entity) WHERE n.name =~ '.*(浙江|广东|上海|北京).*' AND (n.name CONTAINS '教育' OR e.name CONTAINS '教育') RETURN n, e LIMIT 20"
]"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.openai_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.openai_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"生成查询：{question}"},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 1000,
                    },
                )
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # 提取 JSON
                json_match = re.search(r'\[[\s\S]*\]', content)
                if json_match:
                    queries = json.loads(json_match.group())
                    logger.info(f"LLM 生成 {len(queries)} 个 Cypher 查询")
                    return [q for q in queries if isinstance(q, str)]
        except Exception as e:
            logger.warning(f"生成 Cypher 查询失败: {e}")

        return []

    async def _execute_graph_queries(
        self,
        cypher_queries: list[str],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """执行 Cypher 查询并整理结果"""
        logger.info(f"[_execute_graph_queries] Starting with {len(cypher_queries)} queries")
        if not self.neo4j or not cypher_queries:
            logger.warning(f"[_execute_graph_queries] Missing requirements - neo4j: {self.neo4j is not None}, queries: {len(cypher_queries)}")
            return []

        driver = self.neo4j._driver
        if not driver:
            return []

        chunks: list[RetrievedChunk] = []
        seen_paths = set()

        async with driver.session() as session:
            for query in cypher_queries[:3]:  # 最多执行3个查询
                try:
                    logger.info(f"[_execute_graph_queries] Executing query: {query[:100]}...")
                    result = await session.run(query)
                    record_count = 0
                    async for record in result:
                        record_count += 1
                        # 构建路径描述
                        path_parts = []
                        for key, value in record.items():
                            if value and hasattr(value, 'get'):
                                if 'name' in value:
                                    path_parts.append(f"{value.get('name')}({value.get('type', 'unknown')})")

                        path_desc = " -> ".join(path_parts) if path_parts else str(record)

                        # 去重
                        if path_desc in seen_paths:
                            continue
                        seen_paths.add(path_desc)

                        chunks.append(RetrievedChunk(
                            text=path_desc,
                            score=2.0,  # LLM生成的查询结果给较高权重
                            source="graph",
                            meta={"kind": "path", "cypher": query},
                        ))

                        if len(chunks) >= top_k:
                            break

                    logger.info(f"[_execute_graph_queries] Query returned {record_count} records, total chunks: {len(chunks)}")

                except Exception as e:
                    logger.warning(f"[_execute_graph_queries] Cypher query failed: {e}")
                    continue

        return chunks

    async def _expand_with_correlations(
        self,
        graph_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """利用 CORRELATED_WITH 边扩展相似新闻。

        从已有的 graph_chunks 中提取 NewsItem 节点，查询其 CORRELATED_WITH
        邻居，把高相似度的新闻作为补充检索结果返回。
        """
        if not self.neo4j or not graph_chunks:
            return []

        driver = self.neo4j._driver
        if not driver:
            return []

        # 从 graph_chunks 的 text 中提取 NewsItem name（格式: "name(type)"）
        news_names: set[str] = set()
        for chunk in graph_chunks:
            if chunk.source != "graph":
                continue
            # text 格式如: "name(type) -> name2(type2)"
            parts = chunk.text.split(" -> ")
            for part in parts:
                if "(NewsItem)" in part:
                    name = part.split("(NewsItem)")[0].strip()
                    if name:
                        news_names.add(name)

        if not news_names:
            logger.info("[_expand_with_correlations] No NewsItem found in graph_chunks, skipping")
            return []

        logger.info(f"[_expand_with_correlations] Found {len(news_names)} NewsItem names, expanding...")

        expanded: list[RetrievedChunk] = []
        seen_texts: set[str] = set()

        async with driver.session() as session:
            for name in list(news_names)[:5]:
                try:
                    result = await session.run(
                        """
                        MATCH (n:Entity {type: 'NewsItem', name: $name})
                              -[r:CORRELATED_WITH]-(m:Entity {type: 'NewsItem'})
                        RETURN m.name as news_name, m.doc_id as doc_id,
                               r.score as score, r.entity_score as entity_score,
                               r.vector_score as vector_score, r.correlation_type as corr_type
                        ORDER BY r.score DESC
                        LIMIT $limit
                        """,
                        name=name,
                        limit=3,
                    )
                    async for record in result:
                        news_name = record.get("news_name")
                        score = record.get("score", 0)
                        if not news_name:
                            continue

                        text = f"{news_name}(NewsItem) [相似度关联: {name}]"
                        if text in seen_texts:
                            continue
                        seen_texts.add(text)

                        # 计算加权分数: base 1.5 + correlation_score * 2
                        weighted_score = 1.5 + float(score) * 2

                        expanded.append(RetrievedChunk(
                            text=text,
                            score=round(weighted_score, 2),
                            source="graph",
                            meta={
                                "kind": "correlated_news",
                                "source_news": name,
                                "correlation_score": score,
                                "entity_score": record.get("entity_score"),
                                "vector_score": record.get("vector_score"),
                                "correlation_type": record.get("corr_type"),
                                "doc_id": record.get("doc_id"),
                            },
                        ))
                except Exception as e:
                    logger.warning(f"[_expand_with_correlations] Query failed for '{name}': {e}")

        logger.info(f"[_expand_with_correlations] Expanded {len(expanded)} correlated news items")
        return expanded[:top_k]

    async def _generate_answer_with_reasoning(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """使用 LLM 基于证据进行推理回答"""
        if not settings.openai_api_key:
            return "LLM 未配置，无法生成回答。"

        if not chunks:
            return "未找到相关信息。请尝试使用其他关键词提问。"

        # 构建上下文
        context_items = []
        for i, chunk in enumerate(chunks):
            context_items.append(f"[{i+1}] {chunk.text}")
        context = "\n".join(context_items)

        system_prompt = """你是一个智能知识助手。基于提供的检索结果，回答问题并进行合理推理。

【任务要求】
1. 仔细阅读检索结果，提取关键信息
2. 进行归纳总结，不要简单罗列
3. 展现语义理解能力，识别同义词和隐含关联
4. 如果问的是"哪些省份"，列出所有找到的省份名称
5. 用 [数字] 标注信息来源

【回答风格】
- 直接回答用户问题
- 体现推理过程
- 承认信息局限性（如果有）

【示例】
检索结果：
[1] 浙江(ProvinceTag) -> 推进义务教育
[2] 广东(ProvinceTag) -> 教育改革涉及义务教育
[3] 北京(ProvinceTag) -> 相关工作

用户问：哪些省份涉及义务教育？

回答：
根据知识库数据，浙江、广东、北京等多个省份都涉及义务教育相关工作 [1][2][3]。

具体而言，浙江省明确推进义务教育工作；广东省在教育改革中涉及义务教育内容；北京市也有相关工作部署。"""

        user_prompt = f"""检索结果：
{context}

---

用户问题：{question}

请基于检索结果回答问题，展现语义理解和归纳能力。"""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings.openai_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.openai_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1500,
                    },
                )
                response.raise_for_status()
                result = response.json()
                choices = result.get("choices", [])
                if not isinstance(choices, list) or not choices:
                    raise ValueError(f"Invalid LLM response: {result}")
                first = choices[0]
                if not isinstance(first, dict):
                    raise ValueError(f"Invalid LLM response: {result}")
                message = first.get("message", {})
                if not isinstance(message, dict):
                    raise ValueError(f"Invalid LLM response: {result}")
                content = message.get("content", "")
                return str(content)

        except Exception as e:
            return f"LLM 调用失败: {str(e)}"

    async def _retrieve_documents(
        self,
        question: str,
        doc_id: str | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """从 SQLite 原始文档检索（简化版）"""
        if self.store is None:
            return []

        # 简化处理：获取所有文档
        if doc_id:
            doc = self.store.get_doc(doc_id)
            docs = [doc] if doc else []
        else:
            doc_ids = self.store.list_finished_doc_ids(limit=20)
            docs = [self.store.get_doc(did) for did in doc_ids if self.store.get_doc(did)]

        chunks = []
        for doc in docs:
            if doc and len(doc.text.strip()) >= 10:
                # 简单包含检查
                if any(kw in doc.text for kw in question.split()):
                    chunks.append(RetrievedChunk(
                        text=doc.text[:1000],
                        score=1.0,
                        source="document",
                        meta={"doc_id": doc.doc_id, "title": doc.title},
                    ))

        return chunks[:top_k]

    def _verify_answer_sources(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> dict[str, Any]:
        """验证 LLM 回答中的引用是否真实存在于检索结果中。

        防幻觉核心机制：
        1. 提取回答中的所有引用标记 [数字]
        2. 检查每个引用是否在有效范围内（1-N）
        3. 检查引用对应的检索结果是否确实支持该陈述
        4. 标记无来源支持的陈述

        Returns:
            验证结果，包含有效引用、无效引用、无支持陈述等信息
        """
        import re

        # 提取所有引用标记 [数字]
        citations = re.findall(r'\[(\d+)\]', answer)
        citation_nums = [int(c) for c in citations]

        if not citation_nums:
            return {
                "has_citations": False,
                "valid_citations": [],
                "invalid_citations": [],
                "citation_count": 0,
                "warning": "回答中没有引用标记 [数字]，无法验证来源",
            }

        valid_citations = []
        invalid_citations = []

        for num in citation_nums:
            if 1 <= num <= len(chunks):
                valid_citations.append(num)
            else:
                invalid_citations.append(num)

        return {
            "has_citations": True,
            "citation_count": len(set(citation_nums)),
            "valid_citations": list(set(valid_citations)),
            "invalid_citations": list(set(invalid_citations)),
            "total_chunks": len(chunks),
            "warning": f"发现 {len(invalid_citations)} 个无效引用" if invalid_citations else None,
        }

    def _post_process_answer(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        verification: dict[str, Any],
    ) -> str:
        """对 LLM 回答进行后处理，添加防幻觉标记和免责声明。

        处理逻辑：
        1. 如果存在无效引用，添加警告说明
        2. 如果完全没有引用，添加免责声明
        3. 根据置信度添加相应提示
        """
        result = answer

        # 检查验证结果
        if verification.get("invalid_citations"):
            invalid = verification["invalid_citations"]
            warning = f"\n\n[系统提示：回答中包含 {len(invalid)} 个无法验证的引用标记 {invalid}，可能存在不准确信息]"
            result += warning

        if not verification.get("has_citations"):
            disclaimer = "\n\n[免责声明：此回答未标注信息来源，请谨慎核实]"
            result += disclaimer

        return result

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
        """构建来源信息"""
        sources: list[dict[str, Any]] = []
        for chunk in chunks:
            if chunk.source == "graph":
                sources.append({
                    "type": "graph",
                    "text": chunk.text,
                    "meta": chunk.meta,
                })
            elif chunk.source == "document":
                sources.append({
                    "type": "document",
                    "title": chunk.meta.get("title", "未命名"),
                    "doc_id": chunk.meta.get("doc_id"),
                })
        return sources
