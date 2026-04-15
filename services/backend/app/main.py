from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.settings import settings
from app.integrations.neo4j.client import Neo4jClient
from app.store.sqlite import SqliteStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store = SqliteStore(settings.sqlite_path)
    app.state.store = store
    app.state.extract_semaphore = asyncio.Semaphore(2)

    neo4j_client: Neo4jClient | None = None
    if not settings.neo4j_disabled:
        neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        last_error: Exception | None = None
        for _ in range(30):
            try:
                await neo4j_client.open()
                await neo4j_client.ensure_constraints()
                last_error = None
                break
            except Exception as e:
                last_error = e
                await asyncio.sleep(2)
        if last_error is not None:
            raise last_error

    app.state.neo4j = neo4j_client
    yield
    if neo4j_client is not None:
        await neo4j_client.close()


app = FastAPI(title="wiki demo backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    return {"ok": True, "env": settings.app_env}
