"""Schema registry for configurable OneKE extraction schemas."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from app.store.sqlite import SqliteStore


@dataclass(frozen=True)
class ExtractionSchema:
    """OneKE extraction schema configuration."""

    schema_id: str
    schema_name: str
    entity_types: list[str]
    relation_types: list[dict[str, str]]
    instruction: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_name": self.schema_name,
            "entity_types": self.entity_types,
            "relation_types": self.relation_types,
            "instruction": self.instruction,
        }


# Default schema matching the original hardcoded values in OneKEClient
DEFAULT_ENTITY_TYPES = [
    "Organization",
    "Policy",
    "Project",
    "Time",
    "Location",
    "Person",
    "Industry",
    "Theme",
    "Meeting",
    "Other",
]

DEFAULT_RELATION_TYPES = [
    {"subject": "Organization", "relation": "发布", "object": "Policy"},
    {"subject": "Organization", "relation": "启动", "object": "Project"},
    {"subject": "Organization", "relation": "召开", "object": "Meeting"},
    {"subject": "Policy", "relation": "属于", "object": "Industry"},
    {"subject": "Policy", "relation": "涉及", "object": "Location"},
    {"subject": "Policy", "relation": "时间", "object": "Time"},
    {"subject": "Meeting", "relation": "召开于", "object": "Location"},
    {"subject": "Meeting", "relation": "时间", "object": "Time"},
    {"subject": "Project", "relation": "覆盖", "object": "Location"},
    {"subject": "Organization", "relation": "位于", "object": "Location"},
    {"subject": "Organization", "relation": "参与", "object": "Project"},
    {"subject": "Person", "relation": "出席", "object": "Meeting"},
    {"subject": "Person", "relation": "发布", "object": "Policy"},
]

DEFAULT_SCHEMA_NAME = "MOE_News"


class SchemaRegistry:
    """Registry for managing extraction schemas backed by SQLite."""

    def __init__(self, store: SqliteStore) -> None:
        self.store = store
        self._default_ensured: bool = False
        self._ensure_default_schema()

    def _ensure_default_schema(self) -> None:
        """Create the default schema if it doesn't exist."""
        if self._default_ensured:
            return
        existing = self.store.get_schema_by_name(DEFAULT_SCHEMA_NAME)
        if existing is None:
            self.store.create_schema(
                schema_id=str(uuid.uuid4()),
                schema_name=DEFAULT_SCHEMA_NAME,
                entity_types=json.dumps(DEFAULT_ENTITY_TYPES, ensure_ascii=False),
                relation_types=json.dumps(DEFAULT_RELATION_TYPES, ensure_ascii=False),
                instruction=None,
            )
        self._default_ensured = True

    @staticmethod
    def _serialize_entity_types(entity_types: list[str]) -> str:
        return json.dumps(entity_types, ensure_ascii=False)

    @staticmethod
    def _serialize_relation_types(
        relation_types: list[dict[str, str]],
    ) -> str:
        return json.dumps(relation_types, ensure_ascii=False)

    @staticmethod
    def _parse_entity_types(raw: str) -> list[str]:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"entity_types must be a JSON list, got {type(parsed).__name__}")
        return [str(item) for item in parsed]

    @staticmethod
    def _parse_relation_types(raw: str) -> list[dict[str, str]]:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"relation_types must be a JSON list, got {type(parsed).__name__}")
        out: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError(f"relation_types items must be objects, got {type(item).__name__}")
            out.append({
                "subject": str(item.get("subject", "")),
                "relation": str(item.get("relation", "")),
                "object": str(item.get("object", "")),
            })
        return out

    def _row_to_schema(self, row: Any) -> ExtractionSchema:
        return ExtractionSchema(
            schema_id=row.schema_id,
            schema_name=row.schema_name,
            entity_types=self._parse_entity_types(row.entity_types),
            relation_types=self._parse_relation_types(row.relation_types),
            instruction=row.instruction,
        )

    def create(
        self,
        schema_name: str,
        entity_types: list[str],
        relation_types: list[dict[str, str]],
        instruction: str | None = None,
    ) -> ExtractionSchema:
        """Create a new extraction schema."""
        schema_id = str(uuid.uuid4())
        row = self.store.create_schema(
            schema_id=schema_id,
            schema_name=schema_name,
            entity_types=self._serialize_entity_types(entity_types),
            relation_types=self._serialize_relation_types(relation_types),
            instruction=instruction,
        )
        return self._row_to_schema(row)

    def get_by_name(self, schema_name: str) -> ExtractionSchema | None:
        """Get schema by name."""
        row = self.store.get_schema_by_name(schema_name)
        if row is None:
            return None
        return self._row_to_schema(row)

    def get_by_id(self, schema_id: str) -> ExtractionSchema | None:
        """Get schema by ID."""
        row = self.store.get_schema_by_id(schema_id)
        if row is None:
            return None
        return self._row_to_schema(row)

    def list_all(self, *, limit: int = 100) -> list[ExtractionSchema]:
        """List all schemas ordered by updated_at desc."""
        rows = self.store.list_schemas(limit=limit)
        return [self._row_to_schema(row) for row in rows]

    def update(
        self,
        schema_id: str,
        *,
        schema_name: str | None = None,
        entity_types: list[str] | None = None,
        relation_types: list[dict[str, str]] | None = None,
        instruction: str | None = None,
    ) -> ExtractionSchema | None:
        """Update an existing schema."""
        existing = self.store.get_schema_by_id(schema_id)
        if existing is None:
            return None

        self.store.update_schema(
            schema_id=schema_id,
            schema_name=schema_name,
            entity_types=self._serialize_entity_types(entity_types) if entity_types is not None else None,
            relation_types=self._serialize_relation_types(relation_types) if relation_types is not None else None,
            instruction=instruction,
        )
        return self.get_by_id(schema_id)

    def delete(self, schema_id: str) -> bool:
        """Delete a schema. Returns True if deleted."""
        return self.store.delete_schema(schema_id)
