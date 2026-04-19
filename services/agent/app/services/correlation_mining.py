"""News correlation mining service with vector + entity hybrid similarity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.integrations.neo4j.client import Neo4jClient
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Correlation result between two news items."""

    news_id_1: str
    news_id_2: str
    news_title_1: str
    news_title_2: str
    similarity_score: float  # 0-1, overall correlation (hybrid)
    entity_score: float  # 0-1, entity-based similarity
    vector_score: float  # 0-1, vector-based similarity
    shared_entities: list[dict[str, Any]]  # Common entities
    shared_tags: list[str]  # Common tags
    temporal_proximity: float  # 0-1, how close in time
    correlation_type: str  # "entity", "vector", "hybrid"


class CorrelationMiningService:
    """Service for mining correlations between news items using hybrid similarity."""

    def __init__(
        self,
        neo4j: Neo4jClient,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """Initialize with Neo4j client and optional embedding service.

        Args:
            neo4j: Neo4j client instance
            embedding_service: Embedding service for vector similarity
        """
        self.neo4j = neo4j
        # 使用 EmbeddingService，强制使用 Ollama（通过 settings.require_ollama_embedding 控制）
        self.embedding = embedding_service or EmbeddingService()

    async def generate_embeddings(
        self,
        batch_size: int = 10,
    ) -> dict[str, Any]:
        """Generate embeddings for all news items without embeddings.

        Args:
            batch_size: Number of items to process in each batch

        Returns:
            Stats about generated embeddings
        """
        # Get news items without embeddings
        news_without_emb = await self.neo4j.get_news_without_embeddings(limit=100)

        if not news_without_emb:
            return {"processed": 0, "message": "No news items need embeddings"}

        processed = 0
        errors = []

        # Process in batches
        for i in range(0, len(news_without_emb), batch_size):
            batch = news_without_emb[i:i + batch_size]
            texts = [item["title"] for item in batch]

            try:
                from app.core.settings import settings
                embeddings = await self.embedding.embed(texts)

                # 检查是否所有 embedding 都为空（当强制 Ollama 时可能表示失败）
                if not any(embeddings):
                    if settings.require_ollama_embedding:
                        raise RuntimeError("Ollama embedding 返回全部为空")
                    logger.warning(f"Batch {i//batch_size + 1}: all embeddings empty")

                for item, emb in zip(batch, embeddings):
                    if emb:
                        await self.neo4j.set_news_embedding(
                            doc_id=item["doc_id"],
                            embedding=emb,
                            model=settings.ollama_embedding_model if settings.require_ollama_embedding else "text-embedding-3-small",
                        )
                        processed += 1

            except Exception as e:
                from app.core.settings import settings
                # 强制使用 Ollama 时，embedding 失败应该直接抛出异常
                if settings.require_ollama_embedding:
                    logger.error(f"强制使用 Ollama embedding 失败: {e}")
                    raise
                error_msg = f"Batch {i//batch_size + 1} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        return {
            "processed": processed,
            "total": len(news_without_emb),
            "errors": errors,
            "model": settings.ollama_embedding_model if settings.require_ollama_embedding else "text-embedding-3-small",
            "source": "ollama" if settings.require_ollama_embedding else "api",
            "message": f"Generated embeddings for {processed}/{len(news_without_emb)} news items",
        }

    async def find_correlations(
        self,
        doc_id: str | None = None,
        min_score: float = 0.3,
        limit: int = 100,
        use_vector: bool = True,
    ) -> list[CorrelationResult]:
        """Find correlations between news items using hybrid similarity.

        Combines entity-based and vector-based similarity:
        - Entity similarity: Jaccard similarity of shared entities
        - Vector similarity: Cosine similarity of embeddings
        - Hybrid score: weighted combination (0.6 * entity + 0.4 * vector)

        Args:
            doc_id: Optional specific document to find correlations for
            min_score: Minimum hybrid similarity score (0-1)
            limit: Maximum number of results
            use_vector: Whether to include vector similarity (if available)

        Returns:
            List of correlation results sorted by hybrid score
        """
        logger.info(
            f"Finding correlations for doc_id={doc_id}, "
            f"min_score={min_score}, use_vector={use_vector}"
        )

        # Get entity-based correlations
        entity_correlations = await self._find_entity_correlations(doc_id, min_score=0.05)
        logger.info(f"Entity correlations found: {len(entity_correlations)}")

        if not entity_correlations:
            logger.warning("No entity correlations found. Check if news items share common entities.")

        if not use_vector:
            # Return only entity-based results
            return [
                CorrelationResult(
                    news_id_1=c["news_id_1"],
                    news_id_2=c["news_id_2"],
                    news_title_1=c["news_title_1"],
                    news_title_2=c["news_title_2"],
                    similarity_score=c["entity_score"],
                    entity_score=c["entity_score"],
                    vector_score=0.0,
                    shared_entities=c["shared_entities"],
                    shared_tags=[],
                    temporal_proximity=0.5,
                    correlation_type="entity",
                )
                for c in entity_correlations
                if c["entity_score"] >= min_score
            ][:limit]

        # Get vector-based correlations (use lower threshold to get more candidates)
        vector_correlations = await self._find_vector_correlations(doc_id, min_score=0.3)
        logger.info(f"Vector correlations found: {len(vector_correlations)}")

        # Merge and calculate hybrid scores
        merged = self._merge_correlations(entity_correlations, vector_correlations)
        logger.info(f"Merged correlations before filtering: {len(merged)}")
        if merged:
            scores = [r.similarity_score for r in merged]
            logger.info(f"Hybrid score range: {min(scores):.3f} - {max(scores):.3f}")

        # Filter by hybrid score and sort
        results_before_filter = len(merged)
        results = [
            r for r in merged
            if r.similarity_score >= min_score
        ]
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        logger.info(f"After filtering by min_score={min_score}: {len(results)} / {results_before_filter} correlations kept")
        if results:
            scores = [r.similarity_score for r in results]
            logger.info(f"Final score range: {min(scores):.3f} - {max(scores):.3f}")

        return results[:limit]

    async def _find_entity_correlations(
        self,
        doc_id: str | None,
        min_score: float,
    ) -> list[dict[str, Any]]:
        """Find entity-based correlations.

        Returns list of correlation dicts with entity_score.
        """
        driver = self.neo4j._driver
        if not driver:
            raise RuntimeError("Neo4j driver not initialized")

        results: list[dict[str, Any]] = []

        # First check how many news items exist
        count_query = "MATCH (n:Entity {type: 'NewsItem'}) RETURN count(n) as count"
        async with driver.session() as session:
            count_result = await session.run(count_query)
            count_record = await count_result.single()
            news_count = count_record["count"] if count_record else 0
            logger.info(f"_find_entity_correlations: found {news_count} NewsItem entities, doc_id={doc_id}, min_score={min_score}")

            if news_count < 2:
                logger.warning("Less than 2 news items in database. Cannot compute correlations.")
                return results

        async with driver.session() as session:
            if doc_id:
                # Find correlations for specific document
                # Note: We find all pairs and then filter to ensure consistent ID ordering
                query = """
                MATCH (target:Entity {doc_id: $doc_id, type: 'NewsItem'})
                MATCH (target)-[:REL]-(e1:Entity)
                MATCH (other:Entity {type: 'NewsItem'})-[:REL]-(e2:Entity)
                WHERE other.doc_id <> $doc_id
                  AND e1.name = e2.name
                  AND e1.type = e2.type
                  AND e1.type <> 'NewsItem'
                WITH target, other,
                     collect(DISTINCT {name: e1.name, type: e1.type}) as shared_list,
                     count(DISTINCT e1.name) as shared_count
                // Get total entity counts for Jaccard
                MATCH (target)-[:REL]-(t1:Entity)
                MATCH (other)-[:REL]-(t2:Entity)
                WHERE t1.type <> 'NewsItem' AND t2.type <> 'NewsItem'
                WITH target, other, shared_list, shared_count,
                     count(DISTINCT t1.name) as target_entities,
                     count(DISTINCT t2.name) as other_entities
                WITH target, other, shared_list, shared_count,
                     CASE
                       WHEN target_entities + other_entities - shared_count = 0 THEN 0.0
                       ELSE shared_count * 1.0 / (target_entities + other_entities - shared_count)
                     END as jaccard
                WHERE jaccard >= $min_score
                // Return with consistent ID ordering (smaller ID first)
                RETURN CASE WHEN target.doc_id < other.doc_id THEN target.doc_id ELSE other.doc_id END as id1,
                       CASE WHEN target.doc_id < other.doc_id THEN target.name ELSE other.name END as title1,
                       CASE WHEN target.doc_id < other.doc_id THEN other.doc_id ELSE target.doc_id END as id2,
                       CASE WHEN target.doc_id < other.doc_id THEN other.name ELSE target.name END as title2,
                       jaccard as entity_score, shared_list, shared_count
                ORDER BY jaccard DESC
                """
                result = await session.run(query, doc_id=doc_id, min_score=min_score)
            else:
                # Simplified query to find correlations
                query = """
                // Find all pairs of news items that share at least one entity
                MATCH (n1:Entity {type: 'NewsItem'})-[:REL]-(e:Entity)-[:REL]-(n2:Entity {type: 'NewsItem'})
                WHERE n1.doc_id < n2.doc_id
                  AND e.type <> 'NewsItem'
                WITH n1, n2, count(DISTINCT e) as shared_count, collect(DISTINCT e.name)[0..5] as shared_names
                WHERE shared_count >= 1
                // Get total entity counts for each news
                OPTIONAL MATCH (n1)-[:REL]-(t1:Entity)
                WHERE t1.type <> 'NewsItem'
                WITH n1, n2, shared_count, shared_names, count(DISTINCT t1) as n1_entities
                OPTIONAL MATCH (n2)-[:REL]-(t2:Entity)
                WHERE t2.type <> 'NewsItem'
                WITH n1, n2, shared_count, shared_names, n1_entities, count(DISTINCT t2) as n2_entities
                // Calculate Jaccard similarity
                WITH n1, n2, shared_count, shared_names, n1_entities, n2_entities,
                     CASE
                       WHEN n1_entities + n2_entities - shared_count <= 0 THEN 0.0
                       ELSE shared_count * 1.0 / (n1_entities + n2_entities - shared_count)
                     END as jaccard
                WHERE jaccard >= $min_score
                RETURN n1.doc_id as id1, n1.name as title1,
                       n2.doc_id as id2, n2.name as title2,
                       jaccard as entity_score,
                       [x in shared_names | {name: x, type: 'shared'}] as shared_list,
                       shared_count
                ORDER BY jaccard DESC
                LIMIT 1000
                """
                result = await session.run(query, min_score=min_score)
                logger.info(f"Entity correlation query executed with min_score={min_score}")

            count = 0
            async for record in result:
                count += 1
                results.append({
                    "news_id_1": record["id1"],
                    "news_id_2": record["id2"],
                    "news_title_1": record["title1"],
                    "news_title_2": record["title2"],
                    "entity_score": record["entity_score"],
                    "shared_entities": record["shared_list"],
                    "shared_count": record["shared_count"],
                })

            logger.info(f"Entity correlation query returned {count} records")

            # Debug: log first few results
            if results:
                logger.info(f"First entity correlation: {results[0]['news_id_1']} vs {results[0]['news_id_2']}, score={results[0]['entity_score']:.3f}, shared={results[0]['shared_count']}")
            else:
                # Debug: check why no results
                debug_query = """
                MATCH (n:Entity {type: 'NewsItem'})
                OPTIONAL MATCH (n)-[:REL]-(e:Entity)
                RETURN count(DISTINCT n) as news_count, count(e) as entity_rel_count
                """
                debug_result = await session.run(debug_query)
                debug_record = await debug_result.single()
                if debug_record:
                    logger.warning(f"DEBUG: news_count={debug_record['news_count']}, entity_rel_count={debug_record['entity_rel_count']}")

        return results

    async def _find_vector_correlations(
        self,
        doc_id: str | None,
        min_score: float,
    ) -> list[dict[str, Any]]:
        """Find vector-based correlations.

        Returns list of correlation dicts with vector_score.
        """
        results: list[dict[str, Any]] = []

        # Get all news with embeddings
        news_with_emb = await self.neo4j.get_news_with_embeddings(limit=100)
        logger.info(f"_find_vector_correlations: found {len(news_with_emb)} news with embeddings, min_score={min_score}")

        if len(news_with_emb) < 2:
            logger.warning("Less than 2 news items have embeddings. Cannot compute vector correlations.")
            return results

        # Build lookup map
        emb_map = {item["doc_id"]: item for item in news_with_emb}

        if doc_id:
            # Find correlations for specific document
            if doc_id not in emb_map:
                return results

            source = emb_map[doc_id]
            source_emb = source["embedding"]

            for target in news_with_emb:
                if target["doc_id"] == doc_id:
                    continue

                similarity = EmbeddingService.cosine_similarity(
                    source_emb, target["embedding"]
                )

                if similarity >= min_score:
                    results.append({
                        "news_id_1": doc_id,
                        "news_id_2": target["doc_id"],
                        "news_title_1": source["title"],
                        "news_title_2": target["title"],
                        "vector_score": similarity,
                    })
        else:
            # Find all pairwise correlations
            for i, item1 in enumerate(news_with_emb):
                for item2 in news_with_emb[i + 1:]:
                    similarity = EmbeddingService.cosine_similarity(
                        item1["embedding"], item2["embedding"]
                    )

                    if similarity >= min_score:
                        # Ensure consistent ID ordering (smaller ID first)
                        id1, id2 = item1["doc_id"], item2["doc_id"]
                        title1, title2 = item1["title"], item2["title"]
                        if id1 > id2:
                            id1, id2 = id2, id1
                            title1, title2 = title2, title1
                        results.append({
                            "news_id_1": id1,
                            "news_id_2": id2,
                            "news_title_1": title1,
                            "news_title_2": title2,
                            "vector_score": similarity,
                        })

        logger.info(f"Vector correlations meeting threshold {min_score}: {len(results)}")
        return results

    def _merge_correlations(
        self,
        entity_correlations: list[dict[str, Any]],
        vector_correlations: list[dict[str, Any]],
    ) -> list[CorrelationResult]:
        """Merge entity and vector correlations into hybrid scores.

        Hybrid score = 0.6 * entity_score + 0.4 * vector_score

        Args:
            entity_correlations: List of entity-based correlations
            vector_correlations: List of vector-based correlations

        Returns:
            List of merged CorrelationResult with hybrid scores
        """
        # Build lookup maps
        entity_map = {
            (c["news_id_1"], c["news_id_2"]): c
            for c in entity_correlations
        }
        vector_map = {
            (c["news_id_1"], c["news_id_2"]): c
            for c in vector_correlations
        }

        # Collect all unique pairs
        all_pairs = set(entity_map.keys()) | set(vector_map.keys())

        results: list[CorrelationResult] = []

        for id1, id2 in all_pairs:
            entity_c = entity_map.get((id1, id2))
            vector_c = vector_map.get((id1, id2))

            # Get basic info
            if entity_c:
                title1 = entity_c["news_title_1"]
                title2 = entity_c["news_title_2"]
                shared = entity_c.get("shared_entities", [])
            elif vector_c:
                title1 = vector_c["news_title_1"]
                title2 = vector_c["news_title_2"]
                shared = []
            else:
                continue

            # Calculate scores
            entity_score = entity_c["entity_score"] if entity_c else 0.0
            vector_score = vector_c["vector_score"] if vector_c else 0.0

            # Determine correlation type and calculate final score
            if entity_c and vector_c:
                corr_type = "hybrid"
                # Weight: 60% entity, 40% vector
                hybrid_score = 0.6 * entity_score + 0.4 * vector_score
            elif vector_c:
                corr_type = "vector"
                hybrid_score = vector_score
            else:
                corr_type = "entity"
                hybrid_score = entity_score  # Use entity_score directly, not 0.6 * entity_score

            results.append(CorrelationResult(
                news_id_1=id1,
                news_id_2=id2,
                news_title_1=title1,
                news_title_2=title2,
                similarity_score=round(hybrid_score, 3),
                entity_score=round(entity_score, 3),
                vector_score=round(vector_score, 3),
                shared_entities=shared,
                shared_tags=[],
                temporal_proximity=0.5,
                correlation_type=corr_type,
            ))

        return results

    async def calculate_similarity_matrix(
        self,
        doc_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Calculate hybrid similarity matrix for news items.

        Args:
            doc_ids: Optional list of document IDs

        Returns:
            Similarity matrix as dict with entity, vector, and hybrid matrices
        """
        driver = self.neo4j._driver
        if not driver:
            raise RuntimeError("Neo4j driver not initialized")

        # Get news items
        async with driver.session() as session:
            if doc_ids:
                query = """
                MATCH (n:Entity {type: 'NewsItem'})
                WHERE n.doc_id IN $doc_ids
                RETURN n.doc_id as doc_id, n.name as title, n.embedding as embedding
                ORDER BY n.doc_id
                """
                result = await session.run(query, doc_ids=doc_ids)
            else:
                query = """
                MATCH (n:Entity {type: 'NewsItem'})
                RETURN n.doc_id as doc_id, n.name as title, n.embedding as embedding
                ORDER BY n.doc_id
                LIMIT 30
                """
                result = await session.run(query)

            news_items = []
            async for record in result:
                news_items.append({
                    "doc_id": record["doc_id"],
                    "title": record["title"],
                    "embedding": record["embedding"],
                })

        n = len(news_items)
        entity_matrix = [[0.0] * n for _ in range(n)]
        vector_matrix = [[0.0] * n for _ in range(n)]
        hybrid_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    entity_matrix[i][j] = 1.0
                    vector_matrix[i][j] = 1.0
                    hybrid_matrix[i][j] = 1.0
                elif i < j:
                    # Calculate entity similarity (Jaccard)
                    entity_sim = await self._calc_jaccard_similarity(
                        news_items[i]["doc_id"],
                        news_items[j]["doc_id"],
                    )

                    # Calculate vector similarity
                    emb1 = news_items[i].get("embedding")
                    emb2 = news_items[j].get("embedding")
                    if emb1 and emb2:
                        vector_sim = EmbeddingService.cosine_similarity(emb1, emb2)
                    else:
                        vector_sim = 0.0

                    # Hybrid score
                    hybrid_sim = 0.6 * entity_sim + 0.4 * vector_sim

                    entity_matrix[i][j] = entity_matrix[j][i] = round(entity_sim, 3)
                    vector_matrix[i][j] = vector_matrix[j][i] = round(vector_sim, 3)
                    hybrid_matrix[i][j] = hybrid_matrix[j][i] = round(hybrid_sim, 3)

        return {
            "items": [{"doc_id": item["doc_id"], "title": item["title"]} for item in news_items],
            "entity_matrix": entity_matrix,
            "vector_matrix": vector_matrix,
            "hybrid_matrix": hybrid_matrix,
        }

    async def _calc_jaccard_similarity(self, doc_id1: str, doc_id2: str) -> float:
        """Calculate Jaccard similarity between two documents based on shared entities."""
        driver = self.neo4j._driver
        if not driver:
            return 0.0

        query = """
        // Get entities for doc1 by name
        MATCH (n1:Entity {doc_id: $doc_id1})-[:REL]-(e1:Entity)
        WHERE e1.type <> 'NewsItem'
        WITH n1, collect(DISTINCT e1.name) as entities1
        // Get entities for doc2 by name
        MATCH (n2:Entity {doc_id: $doc_id2})-[:REL]-(e2:Entity)
        WHERE e2.type <> 'NewsItem'
        WITH entities1, n2, collect(DISTINCT e2.name) as entities2
        // Calculate intersection
        WITH entities1, entities2,
             [x IN entities1 WHERE x IN entities2] as shared
        // Calculate Jaccard
        WITH entities1, entities2, shared,
             size(entities1) + size(entities2) - size(shared) as union_size
        RETURN CASE WHEN union_size = 0 THEN 0.0
               ELSE size(shared) * 1.0 / union_size
               END as jaccard
        """

        async with driver.session() as session:
            result = await session.run(query, doc_id1=doc_id1, doc_id2=doc_id2)
            record = await result.single()
            return record["jaccard"] if record else 0.0

    async def create_correlation_edges(
        self,
        min_score: float = 0.2,
        use_vector: bool = True,
    ) -> dict[str, Any]:
        """Create CORRELATED_WITH edges in Neo4j based on hybrid similarity.

        Args:
            min_score: Minimum hybrid similarity to create edge
            use_vector: Whether to include vector similarity in calculation

        Returns:
            Stats about created edges
        """
        driver = self.neo4j._driver
        if not driver:
            raise RuntimeError("Neo4j driver not initialized")

        from app.core.settings import settings

        # Check if Ollama is required
        if settings.require_ollama_embedding and use_vector:
            logger.info("强制使用 Ollama embedding 进行向量化...")

        # First, ensure all embeddings are generated
        try:
            emb_stats = await self.generate_embeddings()
            logger.info(f"Embedding generation: {emb_stats}")
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}. Falling back to entity-only.")
            emb_stats = {"processed": 0, "total": 0, "errors": [str(e)]}

        # Check if we have any embeddings
        news_with_emb = await self.neo4j.get_news_with_embeddings(limit=10)
        has_embeddings = len(news_with_emb) >= 2
        logger.info(f"News with embeddings: {len(news_with_emb)}, use_vector={use_vector}")

        # Check total news items
        news_without_emb = await self.neo4j.get_news_without_embeddings(limit=1000)
        total_news = len(news_with_emb) + len(news_without_emb)
        logger.info(f"Total news items: {total_news}")

        if use_vector and not has_embeddings:
            logger.warning(
                "没有可用的 embeddings，将仅使用实体相似度创建边"
            )
            use_vector = False

        # Get correlations (hybrid or entity-only)
        correlations = await self.find_correlations(
            min_score=min_score,
            limit=1000,
            use_vector=use_vector,
        )

        logger.info(f"找到 {len(correlations)} 个相似度 >= {min_score} 的关联")

        if not correlations:
            logger.warning("No correlations found. Cannot create edges.")

            # 详细诊断信息
            diagnostics = {
                "news_count": 0,
                "news_with_embeddings": len(news_with_emb),
                "news_without_embeddings": len(news_without_emb),
                "news_with_entities": 0,
                "embeddings_generated": emb_stats.get("processed", 0),
                "min_score": min_score,
            }

            async with driver.session() as session:
                # Check news count
                debug_result = await session.run("""
                    MATCH (n:Entity {type: 'NewsItem'})
                    RETURN count(n) as count, collect(n.doc_id)[0..5] as sample_ids
                """)
                record = await debug_result.single()
                if record:
                    diagnostics["news_count"] = record["count"]
                    logger.info(f"DEBUG: Found {record['count']} NewsItem nodes, samples: {record['sample_ids']}")

                # Check if they have REL relationships
                debug_result2 = await session.run("""
                    MATCH (n:Entity {type: 'NewsItem'})-[:REL]-(e:Entity)
                    RETURN count(DISTINCT n) as news_with_entities
                """)
                record2 = await debug_result2.single()
                if record2:
                    diagnostics["news_with_entities"] = record2["news_with_entities"]
                    logger.info(f"DEBUG: {record2['news_with_entities']} NewsItem nodes have REL relationships")

            # 分析原因并给出建议
            reasons = []
            suggestions = []

            if diagnostics["news_count"] < 2:
                reasons.append("数据库中新闻数量不足2条")
                suggestions.append("请先上传至少2条新闻数据")

            if diagnostics["news_with_entities"] < 2:
                reasons.append("新闻没有关联的实体（OneKE 提取可能失败）")
                suggestions.append("检查 OneKE 服务是否正常运行")

            if use_vector and diagnostics["news_with_embeddings"] < 2:
                reasons.append(f"只有 {diagnostics['news_with_embeddings']} 条新闻有 embedding")
                suggestions.append("先调用 POST /api/v1/correlations/embeddings 生成 embedding")

            if min_score > 0.5:
                reasons.append(f"相似度阈值 min_score={min_score} 过高")
                suggestions.append("尝试降低 min_score 到 0.2 或更低")

            if not reasons:
                reasons.append("新闻之间没有共享实体或语义相似度不足")
                suggestions.append("上传更多相关主题的新闻，或降低 min_score 阈值")

            error_message = " | ".join(reasons) if reasons else "Unknown reason"
            suggestion_message = "; ".join(suggestions) if suggestions else "Check logs for details"

            return {
                "created_edges": 0,
                "min_score": min_score,
                "embeddings_generated": emb_stats.get("processed", 0),
                "embeddings_total": emb_stats.get("total", 0),
                "use_vector": use_vector,
                "has_embeddings": has_embeddings,
                "correlations_found": 0,
                "emb_errors": emb_stats.get("errors", []),
                "diagnostics": diagnostics,
                "reasons": reasons,
                "message": f"未找到相似度关联: {error_message}",
                "suggestions": suggestion_message,
            }

        created_count = 0

        async with driver.session() as session:
            for corr in correlations:
                # 使用双向边表示相似度关系（无方向）
                query = """
                MATCH (n1:Entity {type: 'NewsItem', doc_id: $doc_id1})
                MATCH (n2:Entity {type: 'NewsItem', doc_id: $doc_id2})
                WHERE n1.doc_id < n2.doc_id
                MERGE (n1)-[r:CORRELATED_WITH]-(n2)
                SET r.id = n1.id + '_corr_' + n2.id,
                    r.type = 'CORRELATED_WITH',
                    r.label = '相似关联',
                    r.doc_id = n1.doc_id + '_' + n2.doc_id,
                    r.score = $score,
                    r.entity_score = $entity_score,
                    r.vector_score = $vector_score,
                    r.correlation_type = $corr_type,
                    r.created_at = datetime()
                RETURN r
                """

                result = await session.run(
                    query,
                    doc_id1=corr.news_id_1,
                    doc_id2=corr.news_id_2,
                    score=corr.similarity_score,
                    entity_score=corr.entity_score,
                    vector_score=corr.vector_score,
                    corr_type=corr.correlation_type,
                )
                record = await result.single()
                if record:
                    created_count += 1

        logger.info(f"Created {created_count} correlation edges")

        return {
            "created_edges": created_count,
            "min_score": min_score,
            "embeddings_generated": emb_stats.get("processed", 0),
            "embeddings_total": emb_stats.get("total", 0),
            "use_vector": use_vector,
            "has_embeddings": has_embeddings,
            "correlations_found": len(correlations),
            "emb_errors": emb_stats.get("errors", []),
        }
