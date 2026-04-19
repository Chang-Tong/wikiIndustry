"""Graph QA Service - structured output combining schema, graph, and raw sources."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.settings import settings
from app.services.rag_engine_v2 import AdaptiveRAGEngine, LLMQueryPlan, RAGAnswerV2

logger = logging.getLogger(__name__)


@dataclass
class SQLiteSource:
    """A raw document snippet retrieved from SQLite."""

    doc_id: str
    title: str
    text_snippet: str
    relevance_score: float


@dataclass
class StructuredGraphAnswer:
    """Structured answer combining graph, schema, and raw sources."""

    question: str
    answer: str
    reasoning_process: str
    confidence: str
    schema_snapshot: dict[str, Any] = field(default_factory=dict)
    graph_results: list[dict[str, Any]] = field(default_factory=list)
    sqlite_sources: list[SQLiteSource] = field(default_factory=list)
    query_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "reasoning_process": self.reasoning_process,
            "confidence": self.confidence,
            "schema_snapshot": self.schema_snapshot,
            "graph_results": self.graph_results,
            "sqlite_sources": [
                {
                    "doc_id": s.doc_id,
                    "title": s.title,
                    "text_snippet": s.text_snippet,
                    "relevance_score": s.relevance_score,
                }
                for s in self.sqlite_sources
            ],
            "query_plan": self.query_plan,
        }


class GraphQAService:
    """Service that answers graph questions with structured, multi-source output."""

    def __init__(
        self,
        neo4j_client: Any,
        sqlite_store: Any,
    ) -> None:
        self.rag_engine = AdaptiveRAGEngine(
            neo4j_client=neo4j_client,
            sqlite_store=sqlite_store,
        )
        self.store = sqlite_store

    async def query(
        self,
        question: str,
        *,
        top_k: int = 10,
        include_raw_sources: bool = True,
    ) -> StructuredGraphAnswer:
        """Answer a question using graph + optional raw document sources.

        Returns a structured answer with schema, graph results, and SQLite sources.
        """
        # Step 1: Run adaptive RAG to get graph-based answer
        rag_answer: RAGAnswerV2 = await self.rag_engine.answer(
            question=question,
            top_k=top_k,
        )

        # Step 2: Capture schema snapshot
        schema_snapshot = await self.rag_engine._discover_schema()

        # Step 3: Retrieve raw SQLite sources if requested
        sqlite_sources: list[SQLiteSource] = []
        if include_raw_sources and self.store is not None:
            sqlite_sources = await self._search_sqlite(question, top_k=top_k)

        # Step 4: Build query plan summary
        query_plan = self._summarize_query_plan(rag_answer.query_plans)

        # Step 5: If raw sources exist and we want a richer answer, regenerate
        # the answer incorporating both graph and raw sources
        final_answer = rag_answer.answer
        final_reasoning = rag_answer.reasoning_process
        final_confidence = rag_answer.confidence

        if sqlite_sources and rag_answer.retrieved_data:
            enriched = await self._generate_enriched_answer(
                question=question,
                graph_data=rag_answer.retrieved_data[:top_k],
                sqlite_sources=sqlite_sources,
                existing_reasoning=rag_answer.reasoning_process,
            )
            final_answer = enriched.get("answer", rag_answer.answer)
            final_reasoning = enriched.get("reasoning", rag_answer.reasoning_process)
            final_confidence = enriched.get("confidence", rag_answer.confidence)

        return StructuredGraphAnswer(
            question=question,
            answer=final_answer,
            reasoning_process=final_reasoning,
            confidence=final_confidence,
            schema_snapshot=schema_snapshot,
            graph_results=rag_answer.retrieved_data[:top_k],
            sqlite_sources=sqlite_sources,
            query_plan=query_plan,
        )

    async def _search_sqlite(self, question: str, top_k: int) -> list[SQLiteSource]:
        """Search SQLite docs for relevant raw text using keyword matching."""
        keywords = await self._expand_keywords(question)
        if not keywords:
            return []

        # Simple keyword search against docs table
        doc_ids = self._search_docs_by_keywords(keywords, limit=top_k * 3)
        if not doc_ids:
            return []

        # Score and rank results
        scored: list[tuple[float, SQLiteSource]] = []
        seen: set[str] = set()

        for doc_id, title, text in doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)

            score = self._calculate_text_score(text, keywords)
            # Truncate text to a reasonable snippet
            snippet = text[:500] + "..." if len(text) > 500 else text
            scored.append((
                score,
                SQLiteSource(
                    doc_id=doc_id,
                    title=title,
                    text_snippet=snippet,
                    relevance_score=score,
                ),
            ))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    async def _expand_keywords(self, question: str) -> list[str]:
        """Extract keywords from question using LLM or fallback to basic tokenization."""
        if not settings.openai_api_key or not settings.openai_base_url:
            return self._basic_tokenize(question)

        system = (
            "你是一个检索词提取器。请从用户问题中提取最重要的核心实体、专有名词、"
            "地名、主题词作为检索关键词。直接输出关键词列表，词与词之间用空格分隔。"
            "不要输出任何解释，不要包含'有哪些'、'什么'、'新闻'等停用词。"
        )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{settings.openai_base_url}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {settings.openai_api_key}",
                    },
                    json={
                        "model": settings.openai_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": question},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 100,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    terms = [t.strip() for t in re.split(r"[\s，、,]+", content) if t.strip()]
                    return terms
        except Exception as e:
            logger.warning(f"LLM keyword expansion failed: {e}")

        return self._basic_tokenize(question)

    @staticmethod
    def _basic_tokenize(text: str) -> list[str]:
        """Basic Chinese/English tokenization fallback."""
        tokens = [
            t.strip()
            for t in re.split(r"[\s，。；、,.!?！？:：;；/|（）()【】\[\]\"'""''」『』]+", text)
            if t.strip()
        ]
        return [t for t in tokens if len(t) > 1] or tokens

    def _search_docs_by_keywords(
        self,
        keywords: list[str],
        limit: int,
    ) -> list[tuple[str, str, str]]:
        """Query SQLite docs table for documents matching any keyword.

        Returns list of (doc_id, title, text) tuples.
        """
        if not self.store:
            return []
        try:
            return self.store.search_docs(keywords, limit=limit)
        except Exception as e:
            logger.warning(f"SQLite doc search failed: {e}")
            return []

    @staticmethod
    def _calculate_text_score(text: str, keywords: list[str]) -> float:
        """Calculate a simple relevance score based on keyword matches."""
        score = 0.0
        text_lower = text.lower()
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in text_lower:
                count = text_lower.count(kw_lower)
                score += count * 1.0
        return score

    def _summarize_query_plan(
        self,
        query_plans: list[LLMQueryPlan],
    ) -> dict[str, Any]:
        """Summarize query plans into a serializable dict."""
        if not query_plans:
            return {}
        latest = query_plans[-1]
        return {
            "thinking": latest.thinking,
            "queries": latest.queries,
            "strategy": "direct_analysis" if latest.needs_direct_analysis else "cypher",
            "iterations": len(query_plans),
        }

    async def _generate_enriched_answer(
        self,
        *,
        question: str,
        graph_data: list[dict[str, Any]],
        sqlite_sources: list[SQLiteSource],
        existing_reasoning: str,
    ) -> dict[str, str]:
        """Generate an enriched answer combining graph and raw sources."""
        if not settings.openai_api_key:
            return {"answer": "", "reasoning": existing_reasoning, "confidence": "medium"}

        graph_text = self._format_graph_data(graph_data)
        sources_text = "\n\n".join(
            f"[{i}] {s.title}\n{s.text_snippet}"
            for i, s in enumerate(sqlite_sources)
        )

        prompt = f"""基于以下信息，综合回答问题。

用户问题：{question}

图谱检索结果：
{graph_text}

原文来源：
{sources_text}

要求：
1. 结合图谱中的结构化关系信息和原文中的具体描述
2. 只基于提供的数据回答，不要编造
3. 如果不确定，明确说明
4. 引用来源时标注 [数字]
5. 展示推理过程

请用 JSON 格式输出：
{{
  "reasoning": "你的推理过程...",
  "answer": "最终回答...",
  "confidence": "high|medium|low"
}}"""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{settings.openai_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.openai_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant. Always respond in valid JSON format.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 2000,
                    },
                )
                resp.raise_for_status()
                result = resp.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        return {
                            "answer": str(parsed.get("answer", "")),
                            "reasoning": str(parsed.get("reasoning", existing_reasoning)),
                            "confidence": str(parsed.get("confidence", "medium")),
                        }
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.warning(f"Enriched answer generation failed: {e}")

        return {"answer": "", "reasoning": existing_reasoning, "confidence": "medium"}

    @staticmethod
    def _format_graph_data(data: list[dict[str, Any]]) -> str:
        """Format graph data for LLM prompt."""
        if not data:
            return "无图谱检索结果"
        items = []
        for i, item in enumerate(data):
            if "source" in item and "target" in item:
                src = item.get("source", {})
                tgt = item.get("target", {})
                rel = item.get("relationship", {})
                src_name = src.get("name", "未知") if isinstance(src, dict) else str(src)
                tgt_name = tgt.get("name", "未知") if isinstance(tgt, dict) else str(tgt)
                rel_type = rel.get("type", "关联") if isinstance(rel, dict) else str(rel)
                items.append(f"[{i}] {src_name} --{rel_type}--> {tgt_name}")
            elif "n" in item:
                node = item["n"]
                if isinstance(node, dict):
                    items.append(f"[{i}] 节点: {node.get('name', '未知')} (类型: {node.get('type', '')})")
                else:
                    items.append(f"[{i}] {str(node)}")
            else:
                items.append(f"[{i}] {json.dumps(item, ensure_ascii=False)}")
        return "\n".join(items)

