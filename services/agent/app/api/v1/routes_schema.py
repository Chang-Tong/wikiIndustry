"""API routes for extraction schema management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.settings import settings
from app.services.schema_registry import ExtractionSchema, SchemaRegistry
from app.store.sqlite import SqliteStore

router = APIRouter()


class RelationTypeDef(BaseModel):
    """Relation type definition (subject-relation-object triple)."""

    subject: str = Field(..., min_length=1, description="Subject entity type")
    relation: str = Field(..., min_length=1, description="Relation verb/phrase")
    object: str = Field(..., min_length=1, description="Object entity type")


class SchemaCreateRequest(BaseModel):
    """Request to create an extraction schema."""

    schema_name: str = Field(..., min_length=1, description="Unique schema name")
    entity_types: list[str] = Field(..., min_length=1, description="List of entity types")
    relation_types: list[RelationTypeDef] = Field(default_factory=list, description="List of relation triples")
    instruction: str | None = Field(default=None, description="Custom extraction instruction override")


class SchemaUpdateRequest(BaseModel):
    """Request to update an extraction schema."""

    schema_name: str | None = Field(default=None, description="Schema name")
    entity_types: list[str] | None = Field(default=None, min_length=1, description="List of entity types")
    relation_types: list[RelationTypeDef] | None = Field(default=None, description="List of relation triples")
    instruction: str | None = Field(default=None, description="Custom extraction instruction override")


class SchemaResponse(BaseModel):
    """Schema response."""

    schema_id: str
    schema_name: str
    entity_types: list[str]
    relation_types: list[dict[str, str]]
    instruction: str | None


class SchemaListResponse(BaseModel):
    """List of schemas response."""

    schemas: list[SchemaResponse]
    total: int


def _get_registry() -> SchemaRegistry:
    store = SqliteStore(settings.sqlite_path)
    return SchemaRegistry(store)


def _schema_to_response(schema: ExtractionSchema) -> SchemaResponse:
    return SchemaResponse(**schema.to_dict())


@router.get("/schemas", response_model=SchemaListResponse)
async def list_schemas(limit: int = 100) -> SchemaListResponse:
    """List all extraction schemas."""
    registry = _get_registry()
    schemas = registry.list_all(limit=limit)
    return SchemaListResponse(
        schemas=[_schema_to_response(s) for s in schemas],
        total=len(schemas),
    )


@router.post("/schemas", response_model=SchemaResponse, status_code=status.HTTP_201_CREATED)
async def create_schema(request: SchemaCreateRequest) -> SchemaResponse:
    """Create a new extraction schema."""
    registry = _get_registry()

    # Check for duplicate name
    existing = registry.get_by_name(request.schema_name)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Schema '{request.schema_name}' already exists",
        )

    relation_types = [r.model_dump() for r in request.relation_types]
    schema = registry.create(
        schema_name=request.schema_name,
        entity_types=request.entity_types,
        relation_types=relation_types,
        instruction=request.instruction,
    )
    return _schema_to_response(schema)


@router.get("/schemas/{schema_id}", response_model=SchemaResponse)
async def get_schema(schema_id: str) -> SchemaResponse:
    """Get an extraction schema by ID."""
    registry = _get_registry()
    schema = registry.get_by_id(schema_id)
    if schema is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schema not found",
        )
    return _schema_to_response(schema)


@router.put("/schemas/{schema_id}", response_model=SchemaResponse)
async def update_schema(schema_id: str, request: SchemaUpdateRequest) -> SchemaResponse:
    """Update an extraction schema."""
    registry = _get_registry()
    existing = registry.get_by_id(schema_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schema not found",
        )

    relation_types = None
    if request.relation_types is not None:
        relation_types = [r.model_dump() for r in request.relation_types]

    updated = registry.update(
        schema_id=schema_id,
        schema_name=request.schema_name,
        entity_types=request.entity_types,
        relation_types=relation_types,
        instruction=request.instruction,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update schema",
        )
    return _schema_to_response(updated)


@router.delete("/schemas/{schema_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_schema(schema_id: str) -> None:
    """Delete an extraction schema."""
    registry = _get_registry()
    deleted = registry.delete(schema_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schema not found",
        )
