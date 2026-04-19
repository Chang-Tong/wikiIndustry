"""Agent MCP tools test suite - simulating source-code sharing scenario.

Each tool handler is tested 10 times, followed by an end-to-end integration test.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure temp-agent uses its own isolated DB
os.environ["SQLITE_PATH"] = os.path.join(tempfile.gettempdir(), "test_agent_mcp.db")

from agent_mcp.tools import (
    configure_extraction_schema,
    get_schema,
    ingest_documents,
    list_schemas,
    query_graph,
)


# ── Helper ────────────────────────────────────────────────────────────────


def _parse_text(content: list) -> dict[str, Any]:
    """Extract JSON from TextContent MCP response."""
    text = content[0].text if content else "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


# ═══════════════════════════════════════════════════════════════════════════
#  1. list_schemas  × 10
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_list_schemas_10_times() -> None:
    """Call list_schemas 10 times; default schema must always be present."""
    for i in range(10):
        result = await list_schemas({})
        data = _parse_text(result)
        assert "schemas" in data, f"Iteration {i}: missing 'schemas' key"
        assert "total" in data, f"Iteration {i}: missing 'total' key"
        assert data["total"] >= 1, f"Iteration {i}: no default schema found"
        assert any(
            s["schema_name"] == "MOE_News" for s in data["schemas"]
        ), f"Iteration {i}: default MOE_News schema missing"
        print(f"  [list_schemas] iteration {i+1}/10 OK  total={data['total']}")


# ═══════════════════════════════════════════════════════════════════════════
#  2. get_schema  × 10
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_get_schema_10_times() -> None:
    """Call get_schema 10 times with the default schema name."""
    for i in range(10):
        result = await get_schema({"schema_name": "MOE_News"})
        data = _parse_text(result)
        assert "error" not in data, f"Iteration {i}: got error {data}"
        assert data.get("schema_name") == "MOE_News", f"Iteration {i}: wrong schema"
        assert "entity_types" in data, f"Iteration {i}: missing entity_types"
        assert "relation_types" in data, f"Iteration {i}: missing relation_types"
        print(f"  [get_schema] iteration {i+1}/10 OK  entities={len(data['entity_types'])}")


# ═══════════════════════════════════════════════════════════════════════════
#  3. configure_extraction_schema  × 10 (create) + 10 (update)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_configure_schema_create_10_times() -> None:
    """Create 10 distinct schemas."""
    created_ids: list[str] = []
    for i in range(10):
        name = f"TestSchema_{i:03d}"
        result = await configure_extraction_schema(
            {
                "schema_name": name,
                "entity_types": ["Person", "Organization", f"CustomType{i}"],
                "relation_types": [
                    {
                        "subject": "Person",
                        "relation": f"works_at_{i}",
                        "object": "Organization",
                    }
                ],
                "instruction": f"Custom instruction for schema {i}",
            }
        )
        data = _parse_text(result)
        assert "error" not in data, f"Iteration {i}: create failed: {data}"
        assert data["schema_name"] == name
        created_ids.append(data["schema_id"])
        print(f"  [configure_schema create] iteration {i+1}/10 OK  id={data['schema_id'][:8]}...")

    # Verify all 10 are listed
    result = await list_schemas({})
    data = _parse_text(result)
    names = {s["schema_name"] for s in data["schemas"]}
    for i in range(10):
        assert f"TestSchema_{i:03d}" in names, f"Schema TestSchema_{i:03d} not found in list"


@pytest.mark.asyncio
async def test_configure_schema_update_10_times() -> None:
    """Create one schema then update it 10 times."""
    # Create base schema
    result = await configure_extraction_schema(
        {
            "schema_name": "UpdateTarget",
            "entity_types": ["A"],
            "relation_types": [],
        }
    )
    data = _parse_text(result)
    schema_id = data["schema_id"]

    for i in range(10):
        result = await configure_extraction_schema(
            {
                "schema_name": "UpdateTarget",
                "entity_types": [f"Type_v{i}"],
                "relation_types": [
                    {"subject": f"Type_v{i}", "relation": f"rel_{i}", "object": f"Type_v{i}"}
                ],
            }
        )
        data = _parse_text(result)
        assert "error" not in data, f"Iteration {i}: update failed: {data}"
        assert data["schema_id"] == schema_id, f"Iteration {i}: schema_id changed unexpectedly"
        assert f"Type_v{i}" in data["entity_types"]
        print(f"  [configure_schema update] iteration {i+1}/10 OK")


# ═══════════════════════════════════════════════════════════════════════════
#  4. ingest_documents  × 10  (mocked OneKE + Neo4j)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_ingest_documents_10_times_mocked() -> None:
    """Ingest 10 batches of documents with mocked external services."""
    mock_extraction = MagicMock()
    mock_extraction.entities = []
    mock_extraction.relations = []
    mock_extraction.engine = "oneke"

    with patch(
        "app.integrations.oneke.client.OneKEClient.extract",
        new_callable=AsyncMock,
        return_value=mock_extraction,
    ), patch(
        "app.integrations.neo4j.client.Neo4jClient.upsert_graph",
        new_callable=AsyncMock,
    ), patch(
        "app.integrations.neo4j.client.Neo4jClient.open",
        new_callable=AsyncMock,
    ), patch(
        "app.integrations.neo4j.client.Neo4jClient.close",
        new_callable=AsyncMock,
    ):
        for i in range(10):
            docs = [
                {
                    "title": f"Test News {i}_{j}",
                    "content": f"This is test content batch {i} doc {j}. " * 20,
                    "date": "2024-01-01",
                }
                for j in range(3)
            ]
            result = await ingest_documents(
                {
                    "documents": docs,
                    "schema_name": "MOE_News",
                    "mode": "incremental",
                }
            )
            data = _parse_text(result)
            assert "error" not in data, f"Iteration {i}: ingest failed: {data}"
            assert data["status"] in ("completed", "completed_with_errors")
            assert data["total_items"] == 3
            print(f"  [ingest_documents] iteration {i+1}/10 OK  processed={data['processed_items']}")


# ═══════════════════════════════════════════════════════════════════════════
#  5. query_graph  × 10  (mocked Neo4j + LLM)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_query_graph_10_times_mocked() -> None:
    """Query graph 10 times with mocked Neo4j and LLM responses."""
    mock_schema = {
        "labels": ["Entity"],
        "relationship_types": ["REL"],
        "node_samples": {},
        "type_distribution": {"Entity": 100},
    }

    with patch(
        "app.services.rag_engine_v2.AdaptiveRAGEngine._discover_schema",
        new_callable=AsyncMock,
        return_value=mock_schema,
    ), patch(
        "app.services.rag_engine_v2.AdaptiveRAGEngine._generate_query_plan",
        new_callable=AsyncMock,
    ) as mock_plan, patch(
        "app.services.rag_engine_v2.AdaptiveRAGEngine._execute_queries",
        new_callable=AsyncMock,
    ), patch(
        "app.services.rag_engine_v2.AdaptiveRAGEngine._generate_final_answer",
        new_callable=AsyncMock,
    ) as mock_answer, patch(
        "app.integrations.neo4j.client.Neo4jClient.open",
        new_callable=AsyncMock,
    ), patch(
        "app.integrations.neo4j.client.Neo4jClient.close",
        new_callable=AsyncMock,
    ), patch(
        "app.services.graph_qa_service.GraphQAService._search_sqlite",
        new_callable=AsyncMock,
        return_value=[],
    ):
        from app.services.rag_engine_v2 import LLMQueryPlan, RAGAnswerV2

        mock_plan.return_value = LLMQueryPlan(
            thinking="Mock plan",
            queries=["MATCH (n) RETURN n LIMIT 5"],
            needs_direct_analysis=False,
            follow_up_needed=False,
        )
        mock_answer.return_value = RAGAnswerV2(
            answer="Mock answer",
            reasoning_process="Mock reasoning",
            sources=[],
            confidence="high",
            query_plans=[mock_plan.return_value],
        )

        for i in range(10):
            result = await query_graph(
                {
                    "question": f"第{i}次测试问题？",
                    "top_k": 5,
                    "include_raw_sources": False,
                }
            )
            data = _parse_text(result)
            assert "error" not in data, f"Iteration {i}: query failed: {data}"
            assert "answer" in data
            assert "schema_snapshot" in data
            assert "graph_results" in data
            print(f"  [query_graph] iteration {i+1}/10 OK  answer={data['answer'][:20]}...")


# ═══════════════════════════════════════════════════════════════════════════
#  6. End-to-end integration test
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_end_to_end_integration() -> None:
    """Full workflow: configure schema → ingest docs → list schemas → get schema → query graph."""
    mock_extraction = MagicMock()
    mock_extraction.entities = []
    mock_extraction.relations = []
    mock_extraction.engine = "oneke"

    mock_schema = {
        "labels": ["Entity"],
        "relationship_types": ["REL"],
        "node_samples": {},
        "type_distribution": {"Entity": 50},
    }

    with patch(
        "app.integrations.oneke.client.OneKEClient.extract",
        new_callable=AsyncMock,
        return_value=mock_extraction,
    ), patch(
        "app.integrations.neo4j.client.Neo4jClient.upsert_graph",
        new_callable=AsyncMock,
    ), patch(
        "app.integrations.neo4j.client.Neo4jClient.open",
        new_callable=AsyncMock,
    ), patch(
        "app.integrations.neo4j.client.Neo4jClient.close",
        new_callable=AsyncMock,
    ), patch(
        "app.services.rag_engine_v2.AdaptiveRAGEngine._discover_schema",
        new_callable=AsyncMock,
        return_value=mock_schema,
    ), patch(
        "app.services.rag_engine_v2.AdaptiveRAGEngine._generate_query_plan",
        new_callable=AsyncMock,
    ) as mock_plan, patch(
        "app.services.rag_engine_v2.AdaptiveRAGEngine._execute_queries",
        new_callable=AsyncMock,
    ), patch(
        "app.services.rag_engine_v2.AdaptiveRAGEngine._generate_final_answer",
        new_callable=AsyncMock,
    ) as mock_answer, patch(
        "app.services.graph_qa_service.GraphQAService._search_sqlite",
        new_callable=AsyncMock,
        return_value=[],
    ):
        from app.services.rag_engine_v2 import LLMQueryPlan, RAGAnswerV2

        mock_plan.return_value = LLMQueryPlan(
            thinking="Integration test plan",
            queries=[],
            needs_direct_analysis=True,
            follow_up_needed=False,
        )
        mock_answer.return_value = RAGAnswerV2(
            answer="Integration test answer",
            reasoning_process="Full pipeline verified",
            sources=[],
            confidence="high",
            query_plans=[mock_plan.return_value],
        )

        print("\n  === Integration Test Start ===")

        # Step 1: Configure custom schema
        print("  [1/5] Configuring custom schema...")
        result = await configure_extraction_schema(
            {
                "schema_name": "IntegrationSchema",
                "entity_types": ["Company", "Product", "Person"],
                "relation_types": [
                    {"subject": "Company", "relation": "发布", "object": "Product"},
                    {"subject": "Person", "relation": "任职于", "object": "Company"},
                ],
                "instruction": "Extract tech news entities and relations",
            }
        )
        data = _parse_text(result)
        assert "error" not in data, f"Schema creation failed: {data}"
        schema_id = data["schema_id"]
        print(f"      Created schema: {schema_id[:16]}...")

        # Step 2: Ingest documents
        print("  [2/5] Ingesting documents...")
        result = await ingest_documents(
            {
                "documents": [
                    {"title": "腾讯发布新游戏", "content": "腾讯公司今日发布了一款新游戏..."},
                    {"title": "阿里财报", "content": "阿里巴巴集团发布季度财报..."},
                ],
                "schema_name": "IntegrationSchema",
                "mode": "incremental",
            }
        )
        data = _parse_text(result)
        assert "error" not in data, f"Ingest failed: {data}"
        assert data["total_items"] == 2
        print(f"      Ingested {data['processed_items']}/{data['total_items']} docs")

        # Step 3: List schemas
        print("  [3/5] Listing schemas...")
        result = await list_schemas({})
        data = _parse_text(result)
        assert data["total"] >= 2  # default + IntegrationSchema
        assert any(s["schema_name"] == "IntegrationSchema" for s in data["schemas"])
        print(f"      Found {data['total']} schemas")

        # Step 4: Get schema
        print("  [4/5] Getting schema detail...")
        result = await get_schema({"schema_name": "IntegrationSchema"})
        data = _parse_text(result)
        assert data["schema_name"] == "IntegrationSchema"
        assert len(data["entity_types"]) == 3
        assert len(data["relation_types"]) == 2
        print(f"      Schema has {len(data['entity_types'])} entity types, {len(data['relation_types'])} relations")

        # Step 5: Query graph
        print("  [5/5] Querying graph...")
        result = await query_graph(
            {
                "question": "腾讯发布了什么？",
                "top_k": 5,
                "include_raw_sources": False,
            }
        )
        data = _parse_text(result)
        assert "error" not in data, f"Query failed: {data}"
        assert "answer" in data
        assert "schema_snapshot" in data
        assert "query_plan" in data
        print(f"      Answer: {data['answer']}")
        print(f"      Confidence: {data['confidence']}")

        print("  === Integration Test PASSED ===\n")
