"""API routes for correlation mining between news items."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.integrations.neo4j.client import Neo4jClient
from app.services.correlation_mining import CorrelationMiningService

logger = logging.getLogger(__name__)

router = APIRouter()


class CorrelationResponse(BaseModel):
    """Correlation result response with hybrid similarity."""

    news_id_1: str = Field(..., description="First news item ID")
    news_id_2: str = Field(..., description="Second news item ID")
    news_title_1: str = Field(..., description="First news item title")
    news_title_2: str = Field(..., description="Second news item title")
    similarity_score: float = Field(..., description="Hybrid similarity score (0-1)", ge=0, le=1)
    entity_score: float = Field(default=0.0, description="Entity-based similarity (0-1)", ge=0, le=1)
    vector_score: float = Field(default=0.0, description="Vector-based similarity (0-1)", ge=0, le=1)
    shared_entities: list[dict[str, Any]] = Field(default_factory=list, description="Common entities")
    shared_tags: list[str] = Field(default_factory=list, description="Common tags")
    correlation_type: str = Field(..., description="Type: entity, vector, or hybrid")


class CorrelationListResponse(BaseModel):
    """List of correlations response."""

    correlations: list[CorrelationResponse] = Field(default_factory=list)
    total: int = Field(..., description="Total number of correlations")
    min_score: float = Field(..., description="Minimum score filter applied")


class SimilarityMatrixResponse(BaseModel):
    """Similarity matrix response with entity, vector, and hybrid matrices."""

    items: list[dict[str, str]] = Field(default_factory=list, description="News items")
    entity_matrix: list[list[float]] = Field(default_factory=list, description="Entity-based similarity matrix")
    vector_matrix: list[list[float]] = Field(default_factory=list, description="Vector-based similarity matrix")
    hybrid_matrix: list[list[float]] = Field(default_factory=list, description="Hybrid similarity matrix")


class CreateCorrelationEdgesResponse(BaseModel):
    """Response for creating correlation edges."""

    created_edges: int = Field(..., description="Number of edges created")
    embeddings_generated: int = Field(default=0, description="Number of embeddings generated")
    embeddings_total: int = Field(default=0, description="Total number of items needing embeddings")
    min_score: float = Field(..., description="Minimum score used")
    use_vector: bool = Field(default=True, description="Whether vector similarity was used")
    has_embeddings: bool = Field(default=False, description="Whether embeddings are available")
    correlations_found: int = Field(default=0, description="Number of correlations found before filtering")
    emb_errors: list[str] = Field(default_factory=list, description="Errors during embedding generation")
    message: str = Field(..., description="Status message")


class GenerateEmbeddingsResponse(BaseModel):
    """Response for generating embeddings."""

    processed: int = Field(..., description="Number of embeddings generated")
    total: int = Field(..., description="Total news items processed")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")
    message: str = Field(..., description="Status message")


@router.get("/correlations", response_model=CorrelationListResponse)
async def get_correlations(
    request: Request,
    doc_id: str | None = Query(default=None, description="Filter by specific document ID"),
    min_score: float = Query(default=0.3, ge=0, le=1, description="Minimum hybrid similarity score"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum results to return"),
    use_vector: bool = Query(default=True, description="Include vector similarity if available"),
) -> CorrelationListResponse:
    """Get correlations between news items using hybrid similarity.

    Combines entity-based and vector-based similarity:
    - Entity similarity: Jaccard similarity of shared entities
    - Vector similarity: Cosine similarity of text embeddings
    - Hybrid score: 0.6 * entity + 0.4 * vector

    Args:
        request: FastAPI request object
        doc_id: Optional specific document to find correlations for
        min_score: Minimum hybrid similarity score (0-1)
        limit: Maximum number of results
        use_vector: Whether to include vector similarity

    Returns:
        List of correlation results with entity_score, vector_score, and hybrid score
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    try:
        service = CorrelationMiningService(neo4j)
        results = await service.find_correlations(
            doc_id=doc_id,
            min_score=min_score,
            limit=limit,
            use_vector=use_vector,
        )

        correlations = [
            CorrelationResponse(
                news_id_1=r.news_id_1,
                news_id_2=r.news_id_2,
                news_title_1=r.news_title_1,
                news_title_2=r.news_title_2,
                similarity_score=r.similarity_score,
                entity_score=r.entity_score,
                vector_score=r.vector_score,
                shared_entities=r.shared_entities,
                shared_tags=r.shared_tags,
                correlation_type=r.correlation_type,
            )
            for r in results
        ]

        return CorrelationListResponse(
            correlations=correlations,
            total=len(correlations),
            min_score=min_score,
        )

    except Exception as e:
        logger.error(f"Failed to get correlations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get correlations: {str(e)}",
        )


@router.post("/correlations/embeddings", response_model=GenerateEmbeddingsResponse)
async def generate_embeddings(
    request: Request,
    batch_size: int = Query(default=10, ge=1, le=50, description="Batch size for embedding generation"),
) -> GenerateEmbeddingsResponse:
    """Generate embeddings for all news items without embeddings.

    This is a prerequisite for vector-based similarity search.
    Requires Ollama to be running if REQUIRE_OLLAMA_EMBEDDING=true.

    Args:
        request: FastAPI request object
        batch_size: Number of items to process in each batch

    Returns:
        Stats about generated embeddings
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    try:
        service = CorrelationMiningService(neo4j)
        result = await service.generate_embeddings(batch_size=batch_size)

        return GenerateEmbeddingsResponse(
            processed=result.get("processed", 0),
            total=result.get("total", 0),
            errors=result.get("errors", []),
            message=result.get("message", "Completed"),
        )

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}",
        )


@router.get("/correlations/matrix", response_model=SimilarityMatrixResponse)
async def get_similarity_matrix(
    request: Request,
    doc_ids: list[str] | None = Query(default=None, description="Specific document IDs"),
) -> SimilarityMatrixResponse:
    """Get similarity matrix for news items.

    Returns three matrices:
    - entity_matrix: Jaccard similarity based on shared entities
    - vector_matrix: Cosine similarity based on text embeddings
    - hybrid_matrix: Weighted combination (0.6 * entity + 0.4 * vector)

    Args:
        request: FastAPI request object
        doc_ids: Optional list of specific document IDs to include

    Returns:
        Similarity matrices with items list
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    try:
        service = CorrelationMiningService(neo4j)
        matrix_data = await service.calculate_similarity_matrix(doc_ids=doc_ids)

        return SimilarityMatrixResponse(
            items=matrix_data.get("items", []),
            entity_matrix=matrix_data.get("entity_matrix", []),
            vector_matrix=matrix_data.get("vector_matrix", []),
            hybrid_matrix=matrix_data.get("hybrid_matrix", []),
        )

    except Exception as e:
        logger.error(f"Failed to get similarity matrix: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get similarity matrix: {str(e)}",
        )


@router.post("/correlations/build-edges", response_model=CreateCorrelationEdgesResponse)
async def build_correlation_edges(
    request: Request,
    min_score: float = Query(default=0.3, ge=0, le=1, description="Minimum hybrid similarity to create edge"),
    use_vector: bool = Query(default=True, description="Whether to use vector similarity (if embeddings available)"),
) -> CreateCorrelationEdgesResponse:
    """Create CORRELATED_WITH edges in Neo4j based on hybrid similarity.

    This operation:
    1. Generates embeddings for all news items (if needed and use_vector=True)
    2. Calculates hybrid similarity (entity + vector, or entity-only if no embeddings)
    3. Creates edges between news items with similarity >= min_score

    Args:
        request: FastAPI request object
        min_score: Minimum hybrid similarity score to create an edge
        use_vector: Whether to use vector similarity (falls back to entity-only if no embeddings)

    Returns:
        Stats about created edges and embeddings
    """
    neo4j: Neo4jClient | None = request.app.state.neo4j
    if neo4j is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neo4j not available",
        )

    try:
        service = CorrelationMiningService(neo4j)
        result = await service.create_correlation_edges(
            min_score=min_score,
            use_vector=use_vector,
        )

        # Build appropriate message
        if result.get("created_edges", 0) == 0:
            msg = result.get("message", "未创建任何边")
        elif result.get("has_embeddings"):
            msg = f"创建 {result['created_edges']} 条相似度边 (hybrid: entity + vector)"
        else:
            msg = f"创建 {result['created_edges']} 条相似度边 (entity-only, 无 embeddings)"

        if result.get("embeddings_generated", 0) > 0:
            msg += f", 新生成 {result['embeddings_generated']} 个 embeddings"

        return CreateCorrelationEdgesResponse(
            created_edges=result["created_edges"],
            embeddings_generated=result.get("embeddings_generated", 0),
            embeddings_total=result.get("embeddings_total", 0),
            min_score=result["min_score"],
            use_vector=result.get("use_vector", use_vector),
            has_embeddings=result.get("has_embeddings", False),
            correlations_found=result.get("correlations_found", 0),
            emb_errors=result.get("emb_errors", []),
            message=msg,
        )

    except Exception as e:
        logger.error(f"Failed to create correlation edges: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create correlation edges: {str(e)}",
        )
