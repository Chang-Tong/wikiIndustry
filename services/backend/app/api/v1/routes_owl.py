"""API routes for OWL/RDF export."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

from app.integrations.neo4j.client import Neo4jClient
from app.services.owl_exporter import OWLExporter

logger = logging.getLogger(__name__)

router = APIRouter()


class OWLExportStatsResponse(BaseModel):
    """OWL export stats response."""

    total_nodes: int = Field(..., description="Total nodes in graph")
    total_edges: int = Field(..., description="Total edges in graph")
    node_types: dict[str, int] = Field(default_factory=dict, description="Node type distribution")
    relation_types: dict[str, int] = Field(default_factory=dict, description="Relation type distribution")


class OWLExportRequest(BaseModel):
    """OWL export request."""

    doc_id: str | None = Field(default=None, description="Specific document to export")
    format: str = Field(default="owl", description="Export format: owl or turtle")
    include_individuals: bool = Field(default=True, description="Include instance data")


@router.get("/export/owl/stats", response_model=OWLExportStatsResponse)
async def get_owl_export_stats(
    request: Request,
) -> OWLExportStatsResponse:
    """Get statistics about exportable graph data.

    Args:
        request: FastAPI request object

    Returns:
        Export stats
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    try:
        exporter = OWLExporter(neo4j)
        stats = await exporter.get_export_stats()

        return OWLExportStatsResponse(
            total_nodes=stats.get("total_nodes", 0),
            total_edges=stats.get("total_edges", 0),
            node_types=stats.get("node_types", {}),
            relation_types=stats.get("relation_types", {}),
        )

    except Exception as e:
        logger.error(f"Failed to get export stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get export stats: {str(e)}",
        )


@router.get("/export/owl")
async def export_owl(
    request: Request,
    doc_id: str | None = Query(default=None, description="Specific document to export"),
    format: str = Query(default="owl", description="Export format: owl or turtle"),
    download: bool = Query(default=False, description="Trigger file download"),
) -> Response:
    """Export knowledge graph to OWL/RDF format.

    Args:
        request: FastAPI request object
        doc_id: Optional specific document to export
        format: Export format (owl or turtle)
        download: Whether to trigger file download

    Returns:
        OWL/XML or Turtle content
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    try:
        exporter = OWLExporter(neo4j)

        if format.lower() == "turtle":
            content = await exporter.export_to_turtle(doc_id=doc_id)
            media_type = "text/turtle"
            filename = "knowledge_graph.ttl"
        else:
            content = await exporter.export_to_owl(doc_id=doc_id)
            media_type = "application/rdf+xml"
            filename = "knowledge_graph.owl"

        headers = {"Content-Type": media_type}
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'

        return Response(
            content=content,
            media_type=media_type,
            headers=headers,
        )

    except Exception as e:
        logger.error(f"Failed to export OWL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export OWL: {str(e)}",
        )


@router.post("/export/owl")
async def export_owl_post(
    request: Request,
    export_request: OWLExportRequest,
) -> Response:
    """Export knowledge graph to OWL/RDF format (POST).

    Args:
        request: FastAPI request object
        export_request: Export configuration

    Returns:
        OWL/XML or Turtle content
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    try:
        exporter = OWLExporter(neo4j)

        if export_request.format.lower() == "turtle":
            content = await exporter.export_to_turtle(doc_id=export_request.doc_id)
            media_type = "text/turtle"
            filename = "knowledge_graph.ttl"
        else:
            content = await exporter.export_to_owl(
                doc_id=export_request.doc_id,
                include_individuals=export_request.include_individuals,
            )
            media_type = "application/rdf+xml"
            filename = "knowledge_graph.owl"

        headers = {
            "Content-Type": media_type,
            "Content-Disposition": f'attachment; filename="{filename}"',
        }

        return Response(
            content=content,
            media_type=media_type,
            headers=headers,
        )

    except Exception as e:
        logger.error(f"Failed to export OWL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export OWL: {str(e)}",
        )
