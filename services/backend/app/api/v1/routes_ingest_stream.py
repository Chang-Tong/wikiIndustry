"""API routes for batch document ingestion (Agent-facing)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.batch_ingestor import BatchIngestService

router = APIRouter()


class DocumentItem(BaseModel):
    """Single document for batch ingestion."""

    title: str = Field(..., min_length=1, description="Document title")
    content: str = Field(..., min_length=1, description="Document body/content")
    site: str | None = Field(default=None, description="Source site")
    channel: str | None = Field(default=None, description="Channel/category")
    date: str | None = Field(default=None, description="Publication date")
    tag: str | None = Field(default=None, description="Tags")
    summary: str | None = Field(default=None, description="Summary")
    link: str | None = Field(default=None, description="URL/link")
    source_id: str | None = Field(default=None, description="Original source ID")


class BatchIngestRequest(BaseModel):
    """Request to batch ingest documents."""

    documents: list[DocumentItem] = Field(..., min_length=1, description="Documents to ingest")
    schema_name: str = Field(default="MOE_News", description="Extraction schema name")
    mode: str = Field(default="incremental", description="'incremental' or 'overwrite'")


class BatchIngestResponse(BaseModel):
    """Response from batch ingestion."""

    job_id: str
    status: str
    total_items: int
    processed_items: int
    extracted_entities: int
    extracted_relations: int
    errors: list[str]


@router.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(request: BatchIngestRequest) -> BatchIngestResponse:
    """Batch ingest documents and build the knowledge graph.

    Each document is processed through OneKE extraction and stored in Neo4j/SQLite.
    """
    if request.mode not in ("incremental", "overwrite"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mode must be 'incremental' or 'overwrite'",
        )

    service = BatchIngestService()
    docs = [d.model_dump() for d in request.documents]
    result = await service.ingest(
        documents=docs,
        schema_name=request.schema_name,
        mode=request.mode,
    )
    return BatchIngestResponse(
        job_id=result["job_id"],
        status=result["status"],
        total_items=result["total_items"],
        processed_items=result["processed_items"],
        extracted_entities=result["extracted_entities"],
        extracted_relations=result["extracted_relations"],
        errors=result["errors"],
    )
