from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)


class ExtractedRelation(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    type: str = Field(min_length=1)
    evidence: str | None = None


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    engine: str

