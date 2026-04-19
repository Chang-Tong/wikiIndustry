"""MCP Server entry point for WikiProject Agent capabilities.

Usage:
    uv run python -m agent_mcp.server
    # or after pip install:
    wiki-agent
"""

from __future__ import annotations

import asyncio
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from agent_mcp.tools import TOOL_HANDLERS

logger = logging.getLogger(__name__)

# Tool definitions exposed to MCP clients
TOOLS: list[Tool] = [
    Tool(
        name="ingest_documents",
        description="Batch ingest documents into the knowledge graph. Each document is processed through OneKE extraction and stored in Neo4j/SQLite. Mode can be 'incremental' (add to existing) or 'overwrite' (clear first).",
        inputSchema={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Document title"},
                            "content": {"type": "string", "description": "Document body/content"},
                            "site": {"type": "string", "description": "Source site"},
                            "channel": {"type": "string", "description": "Channel/category"},
                            "date": {"type": "string", "description": "Publication date"},
                            "tag": {"type": "string", "description": "Tags"},
                            "summary": {"type": "string", "description": "Summary"},
                            "link": {"type": "string", "description": "URL/link"},
                        },
                        "required": ["title", "content"],
                    },
                    "description": "List of documents to ingest",
                },
                "schema_name": {
                    "type": "string",
                    "default": "MOE_News",
                    "description": "OneKE extraction schema name",
                },
                "mode": {
                    "type": "string",
                    "enum": ["incremental", "overwrite"],
                    "default": "incremental",
                    "description": "Ingest mode: incremental or overwrite",
                },
            },
            "required": ["documents"],
        },
    ),
    Tool(
        name="query_graph",
        description="Answer a question using the knowledge graph. Returns structured output including graph schema snapshot, Neo4j graph results, SQLite raw document sources, LLM reasoning, and confidence.",
        inputSchema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to answer"},
                "top_k": {"type": "integer", "default": 10, "description": "Max results to retrieve"},
                "include_raw_sources": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include SQLite raw document sources",
                },
            },
            "required": ["question"],
        },
    ),
    Tool(
        name="configure_extraction_schema",
        description="Create or update a OneKE extraction schema. Defines entity types and relation triples (subject-relation-object) that guide information extraction.",
        inputSchema={
            "type": "object",
            "properties": {
                "schema_name": {"type": "string", "description": "Unique schema name"},
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entity types, e.g. ['Organization', 'Person', 'Policy']",
                },
                "relation_types": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "relation": {"type": "string"},
                            "object": {"type": "string"},
                        },
                        "required": ["subject", "relation", "object"],
                    },
                    "description": "List of relation triples",
                },
                "instruction": {
                    "type": "string",
                    "description": "Custom extraction instruction override",
                },
            },
            "required": ["schema_name", "entity_types"],
        },
    ),
    Tool(
        name="list_schemas",
        description="List all available extraction schemas.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="get_schema",
        description="Get a specific extraction schema by name.",
        inputSchema={
            "type": "object",
            "properties": {
                "schema_name": {"type": "string", "description": "Schema name to look up"},
            },
            "required": ["schema_name"],
        },
    ),
]

app = Server("wikiProject-agent")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return [TextContent(type="text", text='{"error": "Unknown tool"}')]
    return await handler(arguments)


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
