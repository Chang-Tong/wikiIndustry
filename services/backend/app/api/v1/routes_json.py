"""API routes for JSON news data ingestion and processing."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.domain.extraction.models import ExtractionResult
from app.integrations.neo4j.client import Neo4jClient
from app.integrations.oneke.client import OneKEClient
from app.services.json_processor import JSONNewsProcessor, NewsItem, extract_structured_metadata
from app.services.graph_builder import GraphBuilder
from app.store.sqlite import SqliteStore
from app.core.settings import settings

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()


class JSONIngestRequest(BaseModel):
    """Request to ingest JSON data directly."""

    json_data: dict[str, Any] = Field(..., description="JSON news data")
    schema_name: str = Field(default="MOE_News", description="Extraction schema name")


class JSONIngestResponse(BaseModel):
    """Response from JSON ingestion."""

    job_id: str
    status: str
    total_items: int
    processed_items: int
    extracted_entities: int
    extracted_relations: int
    correlation_edges: int = Field(default=0, description="Number of correlation edges created")
    embeddings_generated: int = Field(default=0, description="Number of embeddings generated")
    errors: list[str] = []


class JobStatusResponse(BaseModel):
    """Job status response."""

    job_id: str
    status: str  # processing, completed, failed
    progress: dict[str, Any] = {}


# In-memory job storage (replace with Redis/DB in production)
_job_store: dict[str, dict[str, Any]] = {}


@router.post("/json/upload", response_model=JSONIngestResponse)
async def upload_json_file(
    file: UploadFile = File(..., description="JSON news data file"),
    schema_name: str = Form(default="MOE_News"),
    mode: str = Form(default="incremental", description="Upload mode: 'incremental' or 'overwrite'"),
) -> JSONIngestResponse:
    """Upload JSON file and build knowledge graph.

    Args:
        file: JSON file containing news data
        schema_name: OneKE extraction schema name

    Returns:
        Ingestion result with job ID
    """
    # Validate file type
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JSON files are supported",
        )

    try:
        content = await file.read()
        processor = JSONNewsProcessor()
        batch = processor.process_bytes(content)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process JSON: {str(e)}",
        )

    # Validate mode
    if mode not in ("incremental", "overwrite"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mode must be 'incremental' or 'overwrite'",
        )

    # Process items and build graph
    return await _process_batch(batch, schema_name, mode=mode)


@router.post("/json/ingest", response_model=JSONIngestResponse)
async def ingest_json_data(
    request: JSONIngestRequest,
) -> JSONIngestResponse:
    """Ingest JSON data directly and build knowledge graph.

    Args:
        request: JSON ingest request

    Returns:
        Ingestion result with job ID
    """
    try:
        processor = JSONNewsProcessor()
        batch = processor.process(request.json_data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process JSON: {str(e)}",
        )

    return await _process_batch(batch, request.schema_name)


async def _process_batch(
    batch: Any,
    schema_name: str,
    mode: str = "incremental",
    store: SqliteStore | None = None,
) -> JSONIngestResponse:
    """Process news batch and build knowledge graph."""
    job_id = str(uuid.uuid4())

    logger.info(f"[{job_id}] Starting batch processing - Mode: {mode}, Items: {batch.total_count}")

    _job_store[job_id] = {
        "status": "processing",
        "total": batch.total_count,
        "processed": 0,
        "errors": [],
    }

    # Create store if not provided
    if store is None:
        store = SqliteStore(settings.sqlite_path)

    oneke = OneKEClient(
        base_url=settings.oneke_base_url,
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )

    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    graph_builder = GraphBuilder()
    total_entities = 0
    total_relations = 0
    processed_count = 0
    errors: list[str] = []

    # Clear SQLite data first if overwrite mode
    if mode == "overwrite":
        logger.warning(f"[{job_id}] Overwrite mode - clearing existing data")
        try:
            store.clear_all()
            logger.info(f"[{job_id}] Cleared SQLite docs and jobs tables")
        except Exception as e:
            logger.error(f"[{job_id}] Failed to clear SQLite data: {e}")

    try:
        if mode == "overwrite":
            try:
                from neo4j import AsyncDriver
                driver = neo4j._driver
                if driver is None:
                    raise RuntimeError("Neo4j driver not initialized")
                driver_typed: AsyncDriver = driver
                async with driver_typed.session() as session:
                    result = await session.run("MATCH (n) DETACH DELETE n")
                    summary = await result.consume()
                    logger.info(f"[{job_id}] Cleared {summary.counters.nodes_deleted} Neo4j nodes")
            except Exception as e:
                logger.error(f"[{job_id}] Failed to clear Neo4j data: {e}")

        for item in batch.items:
            item_job_id = str(uuid.uuid4())
            doc_id = item.generate_id()
            try:
                # Create job record for this item
                store.create_job(job_id=item_job_id, doc_id=doc_id)

                # Store original document in SQLite for RAG
                store.create_doc(
                    doc_id=doc_id,
                    title=item.title or "未命名文档",
                    text=item.to_oneke_text(),
                )

                # Extract entities and relations
                extraction = await _extract_from_item(
                    item=item,
                    oneke=oneke,
                    schema_name=schema_name,
                )

                # Build graph
                metadata = extract_structured_metadata(item)
                nodes, edges = graph_builder.build_from_extraction(
                    extraction=extraction,
                    doc_id=doc_id,
                    metadata=metadata,
                )

                # Store in Neo4j
                await neo4j.upsert_graph(nodes=nodes, edges=edges)

                total_entities += len(nodes)
                total_relations += len(edges)
                processed_count += 1

                # Update job progress
                _job_store[job_id]["processed"] = processed_count
                store.update_job(job_id=item_job_id, status="finished", error=None)

            except Exception as e:
                error_msg = f"Failed to process '{item.title[:50]}...': {str(e)}"
                errors.append(error_msg)
                _job_store[job_id]["errors"].append(error_msg)
                store.update_job(job_id=item_job_id, status="failed", error=error_msg)

        _job_store[job_id]["status"] = "completed"

        # Step 3: 自动构建相似度边（增量模式 - 只处理新数据）
        if processed_count > 0:
            try:
                logger.info(f"[{job_id}] Starting auto correlation mining for {processed_count} new items...")
                from app.services.correlation_mining import CorrelationMiningService

                correlation_service = CorrelationMiningService(neo4j)
                corr_result = await correlation_service.create_correlation_edges(
                    min_score=0.3,
                    use_vector=True,
                )
                logger.info(
                    f"[{job_id}] Auto correlation mining completed: "
                    f"{corr_result.get('created_edges', 0)} edges created, "
                    f"{corr_result.get('embeddings_generated', 0)} embeddings generated"
                )
                # 将相似度信息添加到响应中
                _job_store[job_id]["correlation_edges"] = corr_result.get("created_edges", 0)
                _job_store[job_id]["embeddings_generated"] = corr_result.get("embeddings_generated", 0)
            except Exception as e:
                logger.warning(f"[{job_id}] Auto correlation mining failed (non-critical): {e}")
                # 相似度构建失败不应影响主流程
                _job_store[job_id]["correlation_error"] = str(e)

    except Exception as e:
        _job_store[job_id]["status"] = "failed"
        errors.append(str(e))

    return JSONIngestResponse(
        job_id=job_id,
        status=_job_store[job_id]["status"],
        total_items=batch.total_count,
        processed_items=processed_count,
        extracted_entities=total_entities,
        extracted_relations=total_relations,
        correlation_edges=_job_store[job_id].get("correlation_edges", 0),
        embeddings_generated=_job_store[job_id].get("embeddings_generated", 0),
        errors=errors,
    )


async def _extract_from_item(
    item: NewsItem,
    oneke: OneKEClient,
    schema_name: str,
) -> ExtractionResult:
    """Extract entities and relations from news item - 强制使用 OneKE.

    Args:
        item: News item
        oneke: OneKE client
        schema_name: Schema name

    Returns:
        Extraction result

    Raises:
        RuntimeError: 如果 OneKE 未配置或提取失败
    """
    text = item.to_oneke_text()

    # 强制使用 OneKE，如果未配置则报错
    if not settings.oneke_base_url:
        raise RuntimeError("ONEKE_BASE_URL 未配置，无法使用 OneKE 服务")

    if not settings.require_real_oneke:
        raise RuntimeError("REQUIRE_REAL_ONEKE 必须设置为 true 以强制使用 OneKE")

    # 调用 OneKE 服务，失败时抛出异常（不 fallback）
    try:
        return await oneke.extract(text=text, schema_name=schema_name)
    except Exception as e:
        raise RuntimeError(f"OneKE 提取失败: {str(e)}") from e


@router.get("/json/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get job processing status.

    Args:
        job_id: Job ID

    Returns:
        Job status
    """
    if job_id not in _job_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    job = _job_store[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress={
            "total": job.get("total", 0),
            "processed": job.get("processed", 0),
            "errors": job.get("errors", []),
        },
    )
