"""MCP tool implementations for WikiProject Agent capabilities."""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.types import TextContent

from app.core.settings import settings
from app.integrations.neo4j.client import Neo4jClient
from app.services.batch_ingestor import BatchIngestService
from app.services.graph_qa_service import GraphQAService
from app.services.schema_registry import SchemaRegistry
from app.store.sqlite import SqliteStore

logger = logging.getLogger(__name__)

# ── Tool handlers ──────────────────────────────────────────────────────────


async def ingest_documents(arguments: dict[str, Any]) -> list[TextContent]:
    """Batch ingest documents into the knowledge graph.

    Input: {documents: [{title, content, ...}], schema_name?, mode?}
    Output: {job_id, status, total_items, processed_items, extracted_entities, extracted_relations, errors}
    """
    documents = arguments.get("documents", [])
    if not documents:
        return [TextContent(type="text", text=json.dumps({"error": "No documents provided"}, ensure_ascii=False))]

    schema_name = arguments.get("schema_name", "MOE_News")
    mode = arguments.get("mode", "incremental")

    if mode not in ("incremental", "overwrite"):
        return [TextContent(type="text", text=json.dumps({"error": "mode must be 'incremental' or 'overwrite'"}, ensure_ascii=False))]

    service = BatchIngestService()
    result = await service.ingest(
        documents=documents,
        schema_name=schema_name,
        mode=mode,
    )
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def query_graph(arguments: dict[str, Any]) -> list[TextContent]:
    """Answer a question using the knowledge graph with structured output.

    Input: {question, top_k?, include_raw_sources?}
    Output: {question, answer, reasoning_process, confidence, schema_snapshot, graph_results, sqlite_sources, query_plan}
    """
    question = arguments.get("question", "")
    if not question:
        return [TextContent(type="text", text=json.dumps({"error": "question is required"}, ensure_ascii=False))]

    top_k = arguments.get("top_k", 10)
    include_raw_sources = arguments.get("include_raw_sources", True)

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
            question=question,
            top_k=top_k,
            include_raw_sources=include_raw_sources,
        )
        return [TextContent(type="text", text=json.dumps(result.to_dict(), ensure_ascii=False))]
    except Exception as e:
        logger.exception("query_graph failed")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]
    finally:
        await neo4j.close()


async def configure_extraction_schema(arguments: dict[str, Any]) -> list[TextContent]:
    """Create or update an extraction schema for OneKE.

    Input: {schema_name, entity_types, relation_types: [{subject, relation, object}], instruction?}
    Output: {schema_id, schema_name, entity_types, relation_types, instruction}
    """
    schema_name = arguments.get("schema_name", "")
    entity_types = arguments.get("entity_types", [])
    relation_types = arguments.get("relation_types", [])

    if not schema_name or not entity_types:
        return [TextContent(type="text", text=json.dumps({"error": "schema_name and entity_types are required"}, ensure_ascii=False))]

    store = SqliteStore(settings.sqlite_path)
    registry = SchemaRegistry(store)

    existing = registry.get_by_name(schema_name)
    if existing is not None:
        updated = registry.update(
            schema_id=existing.schema_id,
            entity_types=entity_types,
            relation_types=relation_types,
            instruction=arguments.get("instruction"),
        )
        if updated:
            return [TextContent(type="text", text=json.dumps(updated.to_dict(), ensure_ascii=False))]
        return [TextContent(type="text", text=json.dumps({"error": "Failed to update schema"}, ensure_ascii=False))]

    schema = registry.create(
        schema_name=schema_name,
        entity_types=entity_types,
        relation_types=relation_types,
        instruction=arguments.get("instruction"),
    )
    return [TextContent(type="text", text=json.dumps(schema.to_dict(), ensure_ascii=False))]


async def list_schemas(_arguments: dict[str, Any]) -> list[TextContent]:
    """List all available extraction schemas.

    Output: {schemas: [...], total: N}
    """
    store = SqliteStore(settings.sqlite_path)
    registry = SchemaRegistry(store)
    schemas = registry.list_all()
    result = {
        "schemas": [s.to_dict() for s in schemas],
        "total": len(schemas),
    }
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def get_schema(arguments: dict[str, Any]) -> list[TextContent]:
    """Get a specific schema by name.

    Input: {schema_name}
    Output: schema dict or error
    """
    schema_name = arguments.get("schema_name", "")
    if not schema_name:
        return [TextContent(type="text", text=json.dumps({"error": "schema_name is required"}, ensure_ascii=False))]

    store = SqliteStore(settings.sqlite_path)
    registry = SchemaRegistry(store)
    schema = registry.get_by_name(schema_name)
    if schema is None:
        return [TextContent(type="text", text=json.dumps({"error": f"Schema '{schema_name}' not found"}, ensure_ascii=False))]
    return [TextContent(type="text", text=json.dumps(schema.to_dict(), ensure_ascii=False))]


# ── Tool dispatch map ──────────────────────────────────────────────────────

TOOL_HANDLERS: dict[str, Any] = {
    "ingest_documents": ingest_documents,
    "query_graph": query_graph,
    "configure_extraction_schema": configure_extraction_schema,
    "list_schemas": list_schemas,
    "get_schema": get_schema,
}
