from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.settings import settings
from app.main import app


def test_healthz() -> None:
    settings.neo4j_disabled = True
    settings.sqlite_path = ":memory:"
    with TestClient(app) as client:
        res = client.get("/healthz")
        assert res.status_code == 200
        assert res.json()["ok"] is True

