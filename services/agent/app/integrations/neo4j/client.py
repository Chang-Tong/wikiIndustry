from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.integrations.neo4j.cypher import (
    CONSTRAINTS_CYPHER,
    DELETE_GRAPH_BY_DOC_ID,
    READ_ALL_GRAPH,
    READ_GRAPH_BY_DOC_ID,
    UPSERT_GRAPH_CYPHER,
    UPSERT_REL_CYPHER,
)

logger = logging.getLogger(__name__)


@dataclass
class QueryLog:
    """记录 Neo4j 查询日志"""

    query: str
    parameters: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    result_count: int = 0
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query[:200] + "..." if len(self.query) > 200 else self.query,
            "parameters": self.parameters,
            "duration_ms": round(self.duration_ms, 2),
            "result_count": self.result_count,
            "error": self.error,
        }


def stable_id(*parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GraphNode:
    id: str
    name: str
    type: str
    doc_id: str


@dataclass(frozen=True)
class GraphEdge:
    id: str
    source_id: str
    target_id: str
    type: str
    label: str
    doc_id: str
    evidence: str | None


class Neo4jClient:
    def __init__(self, *, uri: str, user: str, password: str) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: AsyncDriver | None = None
        self.query_logs: list[QueryLog] = []

    def get_query_logs(self) -> list[dict[str, Any]]:
        """获取查询日志"""
        return [log.to_dict() for log in self.query_logs]

    def clear_query_logs(self) -> None:
        """清空查询日志"""
        self.query_logs = []

    async def open(self) -> None:
        self._driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def ensure_constraints(self) -> None:
        driver = self._require_driver()
        async with driver.session() as session:
            for stmt in CONSTRAINTS_CYPHER:
                await session.run(stmt)

    async def upsert_graph(self, *, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        driver = self._require_driver()
        async with driver.session() as session:
            await session.run(
                UPSERT_GRAPH_CYPHER,
                nodes=[{"id": n.id, "name": n.name, "type": n.type, "doc_id": n.doc_id} for n in nodes],
            )
            await session.run(
                UPSERT_REL_CYPHER,
                edges=[
                    {
                        "id": e.id,
                        "source_id": e.source_id,
                        "target_id": e.target_id,
                        "type": e.type,
                        "label": e.label,
                        "doc_id": e.doc_id,
                        "evidence": e.evidence,
                    }
                    for e in edges
                ],
            )

    async def read_graph_by_doc_id(self, *, doc_id: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        log = QueryLog(query=READ_GRAPH_BY_DOC_ID, parameters={"doc_id": doc_id})
        driver = self._require_driver()
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}

        try:
            async with driver.session() as session:
                res = await session.run(READ_GRAPH_BY_DOC_ID, doc_id=doc_id)
                async for record in res:
                    source = record.get("source")
                    rel = record.get("rel")
                    target = record.get("target")
                    corr_rel = record.get("corr_rel")
                    corr_target = record.get("corr_target")

                    # 处理主实体
                    if source is not None:
                        sid = str(source.get("id"))
                        if sid not in nodes:
                            nodes[sid] = GraphNode(
                                id=sid,
                                name=str(source.get("name")),
                                type=str(source.get("type")),
                                doc_id=str(source.get("doc_id")),
                            )

                    # 处理 REL 关系的目标实体
                    if target is not None:
                        tid = str(target.get("id"))
                        if tid not in nodes:
                            nodes[tid] = GraphNode(
                                id=tid,
                                name=str(target.get("name")),
                                type=str(target.get("type")),
                                doc_id=str(target.get("doc_id")),
                            )

                    # 处理 CORRELATED_WITH 关系的目标实体
                    if corr_target is not None:
                        ctid = str(corr_target.get("id"))
                        if ctid not in nodes:
                            nodes[ctid] = GraphNode(
                                id=ctid,
                                name=str(corr_target.get("name")),
                                type=str(corr_target.get("type")),
                                doc_id=str(corr_target.get("doc_id")),
                            )

                    # 处理 REL 边
                    if rel is not None:
                        rid = str(rel.get("id"))
                        if rid not in edges and source is not None and target is not None:
                            edges[rid] = GraphEdge(
                                id=rid,
                                source_id=str(source.get("id")),
                                target_id=str(target.get("id")),
                                type=str(rel.get("type")),
                                label=str(rel.get("label")),
                                doc_id=str(rel.get("doc_id")),
                                evidence=str(rel.get("evidence")) if rel.get("evidence") is not None else None,
                            )

                    # 处理 CORRELATED_WITH 边
                    if corr_rel is not None:
                        crid = str(corr_rel.get("id"))
                        if crid not in edges and source is not None and corr_target is not None:
                            # 提取 similarity score
                            score = corr_rel.get('score', 0)
                            corr_type = corr_rel.get('correlation_type', 'unknown')
                            edges[crid] = GraphEdge(
                                id=crid,
                                source_id=str(source.get("id")),
                                target_id=str(corr_target.get("id")),
                                type=str(corr_rel.get("type")),
                                label=f"相似度: {score:.2f}" if score else "相似关联",
                                doc_id=str(corr_rel.get("doc_id")),
                                evidence=f"score: {score}, type: {corr_type}",
                            )
                log.result_count = len(nodes) + len(edges)
        except Exception as e:
            log.error = str(e)
            raise
        finally:
            log.end_time = time.time()
            self.query_logs.append(log)
            logger.info(f"Neo4j Query: {log.query[:80]}... | Duration: {log.duration_ms:.2f}ms | Results: {log.result_count}")

        return list(nodes.values()), list(edges.values())

    async def read_all_graph(
        self,
        *,
        limit: int = 1000,
        types: list[str] | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """读取所有图谱数据（限制数量避免过大）"""
        log = QueryLog(query=READ_ALL_GRAPH, parameters={"limit": limit, "types": types})
        driver = self._require_driver()
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}

        try:
            async with driver.session() as session:
                res = await session.run(READ_ALL_GRAPH, limit=limit, types=types)
                async for record in res:
                    source = record.get("source")
                    rel = record.get("rel")
                    target = record.get("target")

                    if source is not None:
                        sid = str(source.get("id"))
                        nodes[sid] = GraphNode(
                            id=sid,
                            name=str(source.get("name")),
                            type=str(source.get("type")),
                            doc_id=str(source.get("doc_id")),
                        )
                    if target is not None:
                        tid = str(target.get("id"))
                        nodes[tid] = GraphNode(
                            id=tid,
                            name=str(target.get("name")),
                            type=str(target.get("type")),
                            doc_id=str(target.get("doc_id")),
                        )
                    if rel is not None:
                        rid = str(rel.get("id"))
                        source_id = str(source.get("id")) if source is not None else ""
                        target_id = str(target.get("id")) if target is not None else ""
                        edges[rid] = GraphEdge(
                            id=rid,
                            source_id=source_id,
                            target_id=target_id,
                            type=str(rel.get("type")),
                            label=str(rel.get("label")),
                            doc_id=str(rel.get("doc_id")),
                            evidence=str(rel.get("evidence")) if rel.get("evidence") is not None else None,
                        )
                log.result_count = len(nodes) + len(edges)
        except Exception as e:
            log.error = str(e)
            raise
        finally:
            log.end_time = time.time()
            self.query_logs.append(log)
            logger.info(f"Neo4j Query: {log.query[:80]}... | Duration: {log.duration_ms:.2f}ms | Results: {log.result_count}")

        return list(nodes.values()), list(edges.values())

    async def delete_graph_by_doc_id(self, *, doc_id: str) -> None:
        driver = self._require_driver()
        async with driver.session() as session:
            await session.run(DELETE_GRAPH_BY_DOC_ID, doc_id=doc_id)

    async def get_schema_stats(self) -> dict[str, int]:
        """获取图谱统计信息"""
        driver = self._require_driver()
        stats: dict[str, int] = {
            "total_nodes": 0,
            "total_edges": 0,
            "unique_doc_ids": 0,
        }

        queries: list[tuple[str, dict[str, Any]]] = [
            ("MATCH (e:Entity) RETURN count(e) AS cnt", {}),
            ("MATCH ()-[r:REL]->() RETURN count(r) AS cnt", {}),
            ("MATCH (e:Entity) RETURN count(DISTINCT e.doc_id) AS cnt", {}),
            ("MATCH (e:Entity) RETURN e.type AS type, count(*) AS cnt ORDER BY cnt DESC LIMIT 20", {}),
        ]

        async with driver.session() as session:
            # 统计节点数
            log = QueryLog(query=queries[0][0], parameters=queries[0][1])
            try:
                result = await session.run(queries[0][0])
                record = await result.single()
                if record:
                    stats["total_nodes"] = record["cnt"]
                log.result_count = 1
            except Exception as e:
                log.error = str(e)
            finally:
                log.end_time = time.time()
                self.query_logs.append(log)

            # 统计关系数
            log = QueryLog(query=queries[1][0], parameters=queries[1][1])
            try:
                result = await session.run(queries[1][0])
                record = await result.single()
                if record:
                    stats["total_edges"] = record["cnt"]
                log.result_count = 1
            except Exception as e:
                log.error = str(e)
            finally:
                log.end_time = time.time()
                self.query_logs.append(log)

            # 统计文档数
            log = QueryLog(query=queries[2][0], parameters=queries[2][1])
            try:
                result = await session.run(queries[2][0])
                record = await result.single()
                if record:
                    stats["unique_doc_ids"] = record["cnt"]
                log.result_count = 1
            except Exception as e:
                log.error = str(e)
            finally:
                log.end_time = time.time()
                self.query_logs.append(log)

            # 统计节点类型分布
            log = QueryLog(query=queries[3][0], parameters=queries[3][1])
            type_count = 0
            try:
                result = await session.run(queries[3][0])
                async for record in result:
                    type_name = record["type"]
                    count = record["cnt"]
                    stats[f"nodes_{type_name}"] = count
                    type_count += 1
                log.result_count = type_count
            except Exception as e:
                log.error = str(e)
            finally:
                log.end_time = time.time()
                self.query_logs.append(log)

        return stats

    async def get_province_stats(self) -> dict[str, Any]:
        """获取省份统计信息"""
        driver = self._require_driver()

        # 查询1: 统计所有省份标签数量
        query1 = """
            MATCH (p:Entity {type: 'ProvinceTag'})
            RETURN count(p) AS province_count
        """

        # 查询2: 统计每个省份关联的新闻数量
        query2 = """
            MATCH (n:Entity {type: 'NewsItem'})-[:REL]->(p:Entity {type: 'ProvinceTag'})
            RETURN p.name AS province, count(n) AS news_count
            ORDER BY news_count DESC
        """

        # 查询3: 统计有省份标签的新闻数量
        query3 = """
            MATCH (n:Entity {type: 'NewsItem'})-[:REL]->(p:Entity {type: 'ProvinceTag'})
            RETURN count(DISTINCT n) AS news_with_province
        """

        result = {
            "total_provinces": 0,
            "news_with_province": 0,
            "province_distribution": [],
        }

        async with driver.session() as session:
            # 统计省份标签数量
            log = QueryLog(query=query1, parameters={})
            try:
                res = await session.run(query1)
                record = await res.single()
                if record:
                    result["total_provinces"] = record["province_count"]
                log.result_count = 1
            except Exception as e:
                log.error = str(e)
            finally:
                log.end_time = time.time()
                self.query_logs.append(log)

            # 统计每个省份的新闻数量
            log = QueryLog(query=query2, parameters={})
            provinces = []
            try:
                res = await session.run(query2)
                async for record in res:
                    provinces.append({
                        "province": record["province"],
                        "news_count": record["news_count"]
                    })
                result["province_distribution"] = provinces
                log.result_count = len(provinces)
            except Exception as e:
                log.error = str(e)
            finally:
                log.end_time = time.time()
                self.query_logs.append(log)

            # 统计有省份标签的新闻数量
            log = QueryLog(query=query3, parameters={})
            try:
                res = await session.run(query3)
                record = await res.single()
                if record:
                    result["news_with_province"] = record["news_with_province"]
                log.result_count = 1
            except Exception as e:
                log.error = str(e)
            finally:
                log.end_time = time.time()
                self.query_logs.append(log)

        return result

    def _require_driver(self) -> AsyncDriver:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialized")
        return self._driver

    # ========== 主题标签相关方法 ==========

    async def get_theme_tags(self, limit: int = 100) -> list[dict[str, Any]]:
        """获取所有主题标签及其关联新闻数量。"""
        driver = self._require_driver()
        results: list[dict[str, Any]] = []

        query = """
        MATCH (t:Entity {type: 'ThemeTag'})-[:REL]-(n:Entity {type: 'NewsItem'})
        WITH t, count(DISTINCT n) as news_count
        RETURN t.name as theme, news_count
        ORDER BY news_count DESC
        LIMIT $limit
        """

        async with driver.session() as session:
            result = await session.run(query, limit=limit)
            async for record in result:
                results.append({
                    "theme": record["theme"],
                    "news_count": record["news_count"],
                })

        return results

    async def get_nodes_by_theme(self, theme: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        """获取与指定主题相关的所有节点和边。"""
        driver = self._require_driver()
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}

        # 查询主题相关的所有新闻节点及其关联实体
        query = """
        // 找到主题标签
        MATCH (theme:Entity {type: 'ThemeTag', name: $theme})
        // 找到所有相关新闻
        MATCH (theme)-[:REL]-(news:Entity {type: 'NewsItem'})
        // 找到新闻的所有关联实体
        OPTIONAL MATCH (news)-[r:REL]-(related:Entity)
        RETURN theme, news, r, related
        UNION
        // 同时包含其他与这些新闻相关的主题标签
        MATCH (theme:Entity {type: 'ThemeTag', name: $theme})-[:REL]-(news:Entity {type: 'NewsItem'})
        MATCH (news)-[:REL]-(other_theme:Entity {type: 'ThemeTag'})
        WHERE other_theme.name <> $theme
        RETURN theme, news, null as r, other_theme as related
        """

        async with driver.session() as session:
            result = await session.run(query, theme=theme)
            async for record in result:
                theme_node = record.get("theme")
                news = record.get("news")
                rel = record.get("r")
                related = record.get("related")

                # 添加主题节点
                if theme_node is not None:
                    tid = str(theme_node.get("id"))
                    if tid not in nodes:
                        nodes[tid] = GraphNode(
                            id=tid,
                            name=str(theme_node.get("name")),
                            type=str(theme_node.get("type")),
                            doc_id=str(theme_node.get("doc_id", "")),
                        )

                # 添加新闻节点
                if news is not None:
                    nid = str(news.get("id"))
                    if nid not in nodes:
                        nodes[nid] = GraphNode(
                            id=nid,
                            name=str(news.get("name")),
                            type=str(news.get("type")),
                            doc_id=str(news.get("doc_id")),
                        )

                # 添加关联实体节点
                if related is not None:
                    rid = str(related.get("id"))
                    if rid not in nodes:
                        nodes[rid] = GraphNode(
                            id=rid,
                            name=str(related.get("name")),
                            type=str(related.get("type")),
                            doc_id=str(related.get("doc_id", "")),
                        )

                # 添加边
                if rel is not None and news is not None and related is not None:
                    eid = str(rel.get("id"))
                    if eid not in edges:
                        edges[eid] = GraphEdge(
                            id=eid,
                            source_id=str(news.get("id")),
                            target_id=str(related.get("id")),
                            type=str(rel.get("type")),
                            label=str(rel.get("label")),
                            doc_id=str(rel.get("doc_id")),
                            evidence=str(rel.get("evidence")) if rel.get("evidence") is not None else None,
                        )

        return list(nodes.values()), list(edges.values())

    # ========== 向量相关方法 ==========

    async def set_news_embedding(
        self,
        doc_id: str,
        embedding: list[float],
        model: str = "text-embedding-3-small",
    ) -> bool:
        """Store embedding vector for a news item.

        Args:
            doc_id: Document ID
            embedding: Embedding vector
            model: Embedding model name

        Returns:
            True if successful
        """
        from app.integrations.neo4j.cypher import SET_NEWS_EMBEDDING

        driver = self._require_driver()
        log = QueryLog(
            query=SET_NEWS_EMBEDDING,
            parameters={"doc_id": doc_id, "model": model},
        )

        try:
            async with driver.session() as session:
                result = await session.run(
                    SET_NEWS_EMBEDDING,
                    doc_id=doc_id,
                    embedding=embedding,
                    model=model,
                )
                record = await result.single()
                log.result_count = 1 if record else 0
                return record is not None
        except Exception as e:
            log.error = str(e)
            raise
        finally:
            log.end_time = time.time()
            self.query_logs.append(log)

    async def get_news_with_embeddings(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all news items that have embeddings.

        Args:
            limit: Maximum number of results

        Returns:
            List of news items with embeddings
        """
        from app.integrations.neo4j.cypher import GET_NEWS_WITH_EMBEDDING

        driver = self._require_driver()
        results: list[dict[str, Any]] = []

        async with driver.session() as session:
            result = await session.run(GET_NEWS_WITH_EMBEDDING, limit=limit)
            async for record in result:
                results.append({
                    "doc_id": record["doc_id"],
                    "title": record["title"],
                    "embedding": record["embedding"],
                })

        return results

    async def get_news_without_embeddings(
        self,
        limit: int = 100,
    ) -> list[dict[str, str]]:
        """Get news items that don't have embeddings yet.

        Args:
            limit: Maximum number of results

        Returns:
            List of news items without embeddings
        """
        from app.integrations.neo4j.cypher import GET_NEWS_WITHOUT_EMBEDDING

        driver = self._require_driver()
        results: list[dict[str, str]] = []

        async with driver.session() as session:
            result = await session.run(GET_NEWS_WITHOUT_EMBEDDING, limit=limit)
            async for record in result:
                results.append({
                    "doc_id": record["doc_id"],
                    "title": record["title"],
                })

        return results

    async def vector_similarity_search(
        self,
        embedding: list[float],
        min_score: float = 0.7,
        limit: int = 10,
        exclude_doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find similar news items using vector cosine similarity.

        Uses pure Cypher calculation since Neo4j Community Edition
        doesn't have built-in vector index.

        Args:
            embedding: Query embedding vector
            min_score: Minimum similarity score (0-1)
            limit: Maximum results
            exclude_doc_id: Document ID to exclude

        Returns:
            List of similar news items with scores
        """
        driver = self._require_driver()
        results: list[dict[str, Any]] = []

        # Use dot product and magnitude calculation in Cypher
        query = """
        MATCH (n:Entity {type: 'NewsItem'})
        WHERE n.embedding IS NOT NULL
          AND ($exclude_doc_id IS NULL OR n.doc_id <> $exclude_doc_id)
        WITH n,
             // Calculate cosine similarity using dot product
             reduce(dot = 0.0, i in range(0, size(n.embedding)-1) |
                 dot + n.embedding[i] * $embedding[i]
             ) as dot_product,
             // Calculate magnitudes
             sqrt(reduce(sum = 0.0, x in n.embedding | sum + x^2)) as norm1,
             sqrt(reduce(sum = 0.0, x in $embedding | sum + x^2)) as norm2
        WITH n, dot_product, norm1, norm2,
             CASE WHEN norm1 * norm2 = 0 THEN 0.0
                  ELSE dot_product / (norm1 * norm2)
             END as similarity
        WHERE similarity >= $min_score
        RETURN n.doc_id as doc_id, n.name as title, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """

        log = QueryLog(
            query=query[:200],
            parameters={"min_score": min_score, "limit": limit},
        )

        try:
            async with driver.session() as session:
                result = await session.run(
                    query,
                    embedding=embedding,
                    min_score=min_score,
                    limit=limit,
                    exclude_doc_id=exclude_doc_id,
                )
                async for record in result:
                    results.append({
                        "doc_id": record["doc_id"],
                        "title": record["title"],
                        "similarity": record["similarity"],
                    })
                log.result_count = len(results)
        except Exception as e:
            log.error = str(e)
            raise
        finally:
            log.end_time = time.time()
            self.query_logs.append(log)

        return results
