from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.settings import settings
from app.main import app


def test_graph_contract_empty_when_neo4j_disabled() -> None:
    settings.neo4j_disabled = True
    settings.sqlite_path = ":memory:"
    with TestClient(app) as client:
        res = client.get("/api/v1/graph", params={"doc_id": "x"})
        assert res.status_code == 200
        body = res.json()
        assert "elements" in body
        assert "nodes" in body["elements"]
        assert "edges" in body["elements"]
        assert isinstance(body["elements"]["nodes"], list)
        assert isinstance(body["elements"]["edges"], list)

