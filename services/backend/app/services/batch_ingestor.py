"""Batch document ingest service for Agent-facing bulk uploads."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from app.core.settings import settings
from app.domain.extraction.models import ExtractionResult
from app.integrations.neo4j.client import Neo4jClient
from app.integrations.oneke.client import OneKEClient
from app.services.graph_builder import GraphBuilder
from app.services.json_processor import NewsItem, extract_structured_metadata
from app.services.schema_registry import SchemaRegistry
from app.store.sqlite import SqliteStore

logger = logging.getLogger(__name__)


class BatchIngestService:
    """Service for batch-ingesting documents into the knowledge graph."""

    def __init__(self) -> None:
        self.store = SqliteStore(settings.sqlite_path)
        self.registry = SchemaRegistry(self.store)
        self.oneke = OneKEClient(
            base_url=settings.oneke_base_url,
            openai_base_url=settings.openai_base_url,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            schema_registry=self.registry,
        )
        self.neo4j = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        self.graph_builder = GraphBuilder()

    async def ingest(
        self,
        documents: list[dict[str, Any]],
        *,
        schema_name: str = "MOE_News",
        mode: str = "incremental",
    ) -> dict[str, Any]:
        """Ingest a batch of documents and build the knowledge graph.

        Args:
            documents: List of document dicts with at least 'title' and 'content'.
            schema_name: OneKE extraction schema name.
            mode: 'incremental' or 'overwrite'.

        Returns:
            Ingestion result summary.
        """
        job_id = str(uuid.uuid4())
        logger.info(f"[{job_id}] Starting batch ingest - Mode: {mode}, Docs: {len(documents)}")

        total_entities = 0
        total_relations = 0
        processed_count = 0
        errors: list[str] = []

        # Clear existing data if overwrite mode
        if mode == "overwrite":
            logger.warning(f"[{job_id}] Overwrite mode - clearing existing data")
            try:
                self.store.clear_all()
                logger.info(f"[{job_id}] Cleared SQLite data")
            except Exception as e:
                logger.error(f"[{job_id}] Failed to clear SQLite: {e}")

        try:
            neo4j_start = time.time()
            await self.neo4j.open()
            logger.info(f"[{job_id}] Neo4j connected in {time.time() - neo4j_start:.2f}s")

            if mode == "overwrite":
                try:
                    from neo4j import AsyncDriver
                    driver = self.neo4j._driver
                    if driver is None:
                        raise RuntimeError("Neo4j driver not initialized")
                    driver_typed: AsyncDriver = driver
                    async with driver_typed.session() as session:
                        result = await session.run("MATCH (n) DETACH DELETE n")
                        summary = await result.consume()
                        logger.info(f"[{job_id}] Cleared {summary.counters.nodes_deleted} Neo4j nodes")
                except Exception as e:
                    logger.error(f"[{job_id}] Failed to clear Neo4j: {e}")

            for doc in documents:
                item = self._dict_to_news_item(doc)
                item_job_id = str(uuid.uuid4())
                doc_id = item.generate_id()
                try:
                    self.store.create_job(job_id=item_job_id, doc_id=doc_id)
                    self.store.create_doc(
                        doc_id=doc_id,
                        title=item.title or "未命名文档",
                        text=item.to_oneke_text(),
                    )

                    text = item.to_oneke_text()
                    extraction = await self._extract(text, schema_name)
                    metadata = extract_structured_metadata(item)
                    nodes, edges = self.graph_builder.build_from_extraction(
                        extraction=extraction,
                        doc_id=doc_id,
                        metadata=metadata,
                    )
                    await self.neo4j.upsert_graph(nodes=nodes, edges=edges)

                    total_entities += len(nodes)
                    total_relations += len(edges)
                    processed_count += 1
                    self.store.update_job(job_id=item_job_id, status="finished", error=None)

                except Exception as e:
                    error_msg = f"Failed to process '{item.title[:50]}...': {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"[{job_id}] {error_msg}")
                    self.store.update_job(job_id=item_job_id, status="failed", error=error_msg)

            # Auto correlation mining
            if processed_count > 0:
                try:
                    logger.info(f"[{job_id}] Starting auto correlation mining...")
                    from app.services.correlation_mining import CorrelationMiningService

                    corr_service = CorrelationMiningService(self.neo4j)
                    corr_result = await corr_service.create_correlation_edges(
                        min_score=0.05,
                        use_vector=True,
                    )
                    logger.info(
                        f"[{job_id}] Correlation mining: "
                        f"{corr_result.get('created_edges', 0)} edges, "
                        f"{corr_result.get('embeddings_generated', 0)} embeddings"
                    )
                except Exception as e:
                    logger.warning(f"[{job_id}] Correlation mining failed (non-critical): {e}")

        except Exception as e:
            logger.exception(f"[{job_id}] Batch ingest failed")
            errors.append(str(e))
        finally:
            await self.neo4j.close()

        return {
            "job_id": job_id,
            "status": "completed" if not errors else "completed_with_errors",
            "total_items": len(documents),
            "processed_items": processed_count,
            "extracted_entities": total_entities,
            "extracted_relations": total_relations,
            "errors": errors,
        }

    @staticmethod
    def _dict_to_news_item(doc: dict[str, Any]) -> NewsItem:
        """Convert a flat document dict to NewsItem."""
        return NewsItem(
            title=str(doc.get("title", "")),
            site=str(doc.get("site", doc.get("source", ""))),
            channel=str(doc.get("channel", "")),
            date=str(doc.get("date", "")),
            tag=str(doc.get("tag", "")),
            summary=str(doc.get("summary", "")),
            content=str(doc.get("content", "")),
            link=str(doc.get("link", "")),
            source_id=str(doc.get("source_id", "")),
        )

    async def _extract(self, text: str, schema_name: str) -> ExtractionResult:
        """Extract entities and relations from text using OneKE."""
        if not settings.oneke_base_url:
            raise RuntimeError("ONEKE_BASE_URL not configured")
        if not settings.require_real_oneke:
            raise RuntimeError("REQUIRE_REAL_ONEKE must be true")
        return await self.oneke.extract(text=text, schema_name=schema_name)
