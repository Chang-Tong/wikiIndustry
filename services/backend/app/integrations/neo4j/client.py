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
    properties: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class GraphEdge:
    id: str
    source_id: str
    target_id: str
    type: str
    label: str
    doc_id: str
    evidence: str | None
    properties: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


class Neo4jClient:
    def __init__(self, *, uri: str, user: str, password: str) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.query_logs: list[QueryLog] = []

    def get_query_logs(self) -> list[dict[str, Any]]:
        """获取查询日志"""
        return [log.to_dict() for log in self.query_logs]

    def clear_query_logs(self) -> None:
        """清空查询日志"""
        self.query_logs = []

    async def open(self) -> None:
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        await self._driver.close()

    def _require_driver(self) -> AsyncDriver:
        return self._driver

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
                                properties=dict(source),
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
                                properties=dict(target),
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
                                properties=dict(corr_target),
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
                                properties=dict(rel),
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
                                properties=dict(corr_rel),
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
        node_limit: int = 25,
        types: list[str] | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """读取所有图谱数据（限制数量避免过大）"""
        log = QueryLog(query=READ_ALL_GRAPH, parameters={"node_limit": node_limit, "types": types})
        driver = self._require_driver()
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}

        try:
            async with driver.session() as session:
                res = await session.run(READ_ALL_GRAPH, node_limit=node_limit, types=types)
                async for record in res:
                    source = record.get("source")
                    rel = record.get("rel")
                    target = record.get("target")

                    if source is not None:
                        sid = str(source.get("id"))
                        if sid not in nodes:
                            nodes[sid] = GraphNode(
                                id=sid,
                                name=str(source.get("name")),
                                type=str(source.get("type")),
                                doc_id=str(source.get("doc_id")),
                                properties=dict(source),
                            )
                    if target is not None:
                        tid = str(target.get("id"))
                        if tid not in nodes:
                            nodes[tid] = GraphNode(
                                id=tid,
                                name=str(target.get("name")),
                                type=str(target.get("type")),
                                doc_id=str(target.get("doc_id")),
                                properties=dict(target),
                            )
                    if rel is not None:
                        rid = str(rel.get("id"))
                        source_id = str(source.get("id")) if source is not None else ""
                        target_id = str(target.get("id")) if target is not None else ""
                        if rid not in edges:
                            edges[rid] = GraphEdge(
                                id=rid,
                                source_id=source_id,
                                target_id=target_id,
                                type=str(rel.get("type")),
                                label=str(rel.get("label")),
                                doc_id=str(rel.get("doc_id")),
                                evidence=str(rel.get("evidence")) if rel.get("evidence") is not None else None,
                                properties=dict(rel),
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

    async def get_schema_summary(self) -> dict[str, Any]:
        """返回供 LLM 使用的轻量级 Schema 摘要。"""
        driver = self._require_driver()
        summary: dict[str, Any] = {
            "node_labels": [],
            "node_types": [],
            "rel_types": [],
            "property_keys": [],
            "samples": [],
        }
        async with driver.session() as session:
            try:
                result = await session.run("CALL db.labels() YIELD label RETURN collect(label) AS labels")
                record = await result.single()
                if record:
                    summary["node_labels"] = record["labels"]
            except Exception:
                pass

            try:
                result = await session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types"
                )
                record = await result.single()
                if record:
                    summary["rel_types"] = record["types"]
            except Exception:
                pass

            try:
                result = await session.run(
                    "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS keys LIMIT 50"
                )
                record = await result.single()
                if record:
                    summary["property_keys"] = record["keys"][:50]
            except Exception:
                pass

            try:
                result = await session.run(
                    "MATCH (n:Entity) RETURN n.type AS type, count(*) AS cnt ORDER BY cnt DESC LIMIT 15"
                )
                types: list[dict[str, Any]] = []
                async for record in result:
                    types.append({"type": record["type"], "count": record["cnt"]})
                summary["node_types"] = types
            except Exception:
                pass

            try:
                result = await session.run(
                    "MATCH ()-[r:REL]->() RETURN r.type AS type, count(*) AS cnt ORDER BY cnt DESC LIMIT 15"
                )
                rel_types: list[dict[str, Any]] = []
                async for record in result:
                    rel_types.append({"type": record["type"], "count": record["cnt"]})
                summary["rel_types_sample"] = rel_types
            except Exception:
                pass

            try:
                result = await session.run(
                    """
                    MATCH (n:Entity)-[r:REL]->(m:Entity)
                    RETURN n.name AS source, n.type AS source_type, r.type AS rel_type,
                           m.name AS target, m.type AS target_type
                    LIMIT 5
                    """
                )
                samples: list[dict[str, Any]] = []
                async for record in result:
                    samples.append(
                        {
                            "source": record["source"],
                            "source_type": record["source_type"],
                            "rel": record["rel_type"],
                            "target": record["target"],
                            "target_type": record["target_type"],
                        }
                    )
                summary["samples"] = samples
            except Exception:
                pass
        return summary

    async def execute_cypher(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """执行任意 Cypher 查询并返回记录列表。"""
        driver = self._require_driver()
        log = QueryLog(query=query, parameters=parameters or {})
        records: list[dict[str, Any]] = []
        try:
            async with driver.session() as session:
                result = await session.run(query, parameters or {})
                async for record in result:
                    records.append(dict(record))
                log.result_count = len(records)
        except Exception as e:
            log.error = str(e)
            raise
        finally:
            log.end_time = time.time()
            self.query_logs.append(log)
        return records

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

    async def get_nodes_by_theme(
        self, theme: str, news_limit: int = 30
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """获取与指定主题相关的所有节点和边。"""
        driver = self._require_driver()
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}

        # 查询主题相关的所有新闻节点及其关联实体，限制新闻数量避免过大
        query = """
        // 找到主题标签
        MATCH (theme:Entity {type: 'ThemeTag', name: $theme})
        // 找到相关新闻并限制数量
        MATCH (theme)-[:REL]-(news:Entity {type: 'NewsItem'})
        WITH theme, news
        LIMIT $news_limit
        // 找到新闻的所有关联实体（包括 REL 和 CORRELATED_WITH）
        OPTIONAL MATCH (news)-[r:REL|CORRELATED_WITH]-(related:Entity)
        RETURN theme, news, r, related
        UNION
        // 同时包含其他与这些新闻相关的主题标签
        MATCH (theme:Entity {type: 'ThemeTag', name: $theme})-[:REL]-(news:Entity {type: 'NewsItem'})
        WITH theme, news
        LIMIT $news_limit
        MATCH (news)-[:REL]-(other_theme:Entity {type: 'ThemeTag'})
        WHERE other_theme.name <> $theme
        RETURN theme, news, null as r, other_theme as related
        """

        async with driver.session() as session:
            result = await session.run(query, theme=theme, news_limit=news_limit)
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

    async def retrieve_subgraph_by_keywords(
        self,
        terms: list[str],
        candidate_limit: int = 15,
        expand_limit: int = 300,
        doc_id: str | None = None,
        rel_types: list[str] | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """基于关键词在Neo4j中检索候选节点并扩展1跳子图。

        避免把全图拉到Python内存，直接在数据库层完成相关子图提取。
        可通过 rel_types 限制优先扩展的关系类型。
        """
        driver = self._require_driver()
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}

        if not terms:
            return [], []

        query = """
        UNWIND $terms AS term
        MATCH (n:Entity)
        WHERE n.name CONTAINS term
          AND ($doc_id IS NULL OR n.doc_id = $doc_id)
        WITH n, count(*) AS hit
        ORDER BY hit DESC
        LIMIT $candidate_limit
        WITH collect(n) AS seeds
        UNWIND seeds AS s
        MATCH (s)-[r]-(m)
        WHERE ($doc_id IS NULL OR m.doc_id = $doc_id)
          AND ($rel_types IS NULL OR r.type IN $rel_types)
        RETURN s, r, m
        LIMIT $expand_limit
        """

        log = QueryLog(
            query=query[:120],
            parameters={"terms": terms, "candidate_limit": candidate_limit, "doc_id": doc_id, "rel_types": rel_types},
        )

        try:
            async with driver.session() as session:
                result = await session.run(
                    query,
                    terms=terms,
                    candidate_limit=candidate_limit,
                    expand_limit=expand_limit,
                    doc_id=doc_id,
                    rel_types=rel_types,
                )
                async for record in result:
                    source = record.get("s")
                    rel = record.get("r")
                    target = record.get("m")

                    if source is not None:
                        sid = str(source.get("id"))
                        if sid not in nodes:
                            nodes[sid] = GraphNode(
                                id=sid,
                                name=str(source.get("name")),
                                type=str(source.get("type")),
                                doc_id=str(source.get("doc_id")),
                                properties=dict(source),
                            )
                    if target is not None:
                        tid = str(target.get("id"))
                        if tid not in nodes:
                            nodes[tid] = GraphNode(
                                id=tid,
                                name=str(target.get("name")),
                                type=str(target.get("type")),
                                doc_id=str(target.get("doc_id")),
                                properties=dict(target),
                            )
                    if rel is not None and source is not None and target is not None:
                        rid = str(rel.get("id"))
                        if rid not in edges:
                            edges[rid] = GraphEdge(
                                id=rid,
                                source_id=str(source.get("id")),
                                target_id=str(target.get("id")),
                                type=str(rel.get("type")),
                                label=str(rel.get("label")),
                                doc_id=str(rel.get("doc_id")),
                                evidence=str(rel.get("evidence")) if rel.get("evidence") is not None else None,
                                properties=dict(rel),
                            )
                log.result_count = len(nodes) + len(edges)
        except Exception as e:
            log.error = str(e)
            raise
        finally:
            log.end_time = time.time()
            self.query_logs.append(log)
            logger.info(
                f"Neo4j keyword subgraph: terms={terms[:3]}... | "
                f"Duration: {log.duration_ms:.2f}ms | Results: {log.result_count}"
            )

        return list(nodes.values()), list(edges.values())

    async def retrieve_subgraph_by_embedding(
        self,
        embedding: list[float],
        min_score: float = 0.55,
        top_n: int = 5,
        expand_limit: int = 200,
        doc_id: str | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """基于向量相似度找种子NewsItem并扩展1跳子图。"""
        driver = self._require_driver()
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}

        query = """
        MATCH (n:Entity {type: 'NewsItem'})
        WHERE n.embedding IS NOT NULL
          AND ($doc_id IS NULL OR n.doc_id = $doc_id)
        WITH n,
             reduce(dot = 0.0, i in range(0, size(n.embedding)-1) |
                 dot + n.embedding[i] * $embedding[i]
             ) as dot_product,
             sqrt(reduce(sum = 0.0, x in n.embedding | sum + x^2)) as norm1,
             sqrt(reduce(sum = 0.0, x in $embedding | sum + x^2)) as norm2
        WITH n, dot_product, norm1, norm2,
             CASE WHEN norm1 * norm2 = 0 THEN 0.0
                  ELSE dot_product / (norm1 * norm2)
             END as similarity
        WHERE similarity >= $min_score
        WITH n, similarity
        ORDER BY similarity DESC
        LIMIT $top_n
        WITH collect(n) AS seeds
        UNWIND seeds AS s
        MATCH (s)-[r]-(m)
        WHERE $doc_id IS NULL OR m.doc_id = $doc_id
        RETURN s, r, m, s.doc_id as seed_doc_id, s.name as seed_title
        LIMIT $expand_limit
        """

        log = QueryLog(
            query=query[:120],
            parameters={"min_score": min_score, "top_n": top_n, "doc_id": doc_id},
        )

        try:
            async with driver.session() as session:
                result = await session.run(
                    query,
                    embedding=embedding,
                    min_score=min_score,
                    top_n=top_n,
                    expand_limit=expand_limit,
                    doc_id=doc_id,
                )
                async for record in result:
                    source = record.get("s")
                    rel = record.get("r")
                    target = record.get("m")

                    if source is not None:
                        sid = str(source.get("id"))
                        if sid not in nodes:
                            nodes[sid] = GraphNode(
                                id=sid,
                                name=str(source.get("name")),
                                type=str(source.get("type")),
                                doc_id=str(source.get("doc_id")),
                                properties=dict(source),
                            )
                    if target is not None:
                        tid = str(target.get("id"))
                        if tid not in nodes:
                            nodes[tid] = GraphNode(
                                id=tid,
                                name=str(target.get("name")),
                                type=str(target.get("type")),
                                doc_id=str(target.get("doc_id")),
                                properties=dict(target),
                            )
                    if rel is not None and source is not None and target is not None:
                        rid = str(rel.get("id"))
                        if rid not in edges:
                            edges[rid] = GraphEdge(
                                id=rid,
                                source_id=str(source.get("id")),
                                target_id=str(target.get("id")),
                                type=str(rel.get("type")),
                                label=str(rel.get("label")),
                                doc_id=str(rel.get("doc_id")),
                                evidence=str(rel.get("evidence")) if rel.get("evidence") is not None else None,
                                properties=dict(rel),
                            )
                log.result_count = len(nodes) + len(edges)
        except Exception as e:
            log.error = str(e)
            raise
        finally:
            log.end_time = time.time()
            self.query_logs.append(log)
            logger.info(
                f"Neo4j embedding subgraph: min_score={min_score} | "
                f"Duration: {log.duration_ms:.2f}ms | Results: {log.result_count}"
            )

        return list(nodes.values()), list(edges.values())
