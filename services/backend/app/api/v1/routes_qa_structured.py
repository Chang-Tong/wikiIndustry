"""API routes for structured graph QA with multi-source output."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.settings import settings
from app.integrations.neo4j.client import Neo4jClient
from app.services.graph_qa_service import GraphQAService
from app.store.sqlite import SqliteStore

router = APIRouter()


class QueryGraphRequest(BaseModel):
    """Request for structured graph QA."""

    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=10, ge=1, le=50, description="Max results to retrieve")
    include_raw_sources: bool = Field(
        default=True, description="Include SQLite raw document sources"
    )


class QueryGraphResponse(BaseModel):
    """Structured graph QA response."""

    question: str
    answer: str
    reasoning_process: str
    confidence: str
    schema_snapshot: dict[str, Any]
    graph_results: list[dict[str, Any]]
    sqlite_sources: list[dict[str, Any]]
    query_plan: dict[str, Any]


@router.post("/qa/query-graph-structured", response_model=QueryGraphResponse)
async def query_graph_structured(request: QueryGraphRequest) -> QueryGraphResponse:
    """Answer a question with structured output combining graph schema, Neo4j results,
    SQLite raw sources, and LLM reasoning.
    """
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    store = SqliteStore(settings.sqlite_path)
    service = GraphQAService(neo4j_client=neo4j, sqlite_store=store)

    try:
        await neo4j.open()
        result = await service.query(
            question=request.question,
            top_k=request.top_k,
            include_raw_sources=request.include_raw_sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    finally:
        await neo4j.close()

    return QueryGraphResponse(**result.to_dict())
