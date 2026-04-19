from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import json
import logging
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.settings import settings
from app.domain.extraction.models import ExtractionResult
from app.integrations.neo4j.client import Neo4jClient
from app.services.batch_ingestor import BatchIngestService
from app.services.graph_builder import GraphBuilder
from app.services.json_processor import extract_structured_metadata
from app.services.schema_registry import SchemaRegistry
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


@app.websocket("/ws/ingest")
async def websocket_ingest(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time document ingestion.

    Clients send JSON documents one at a time. The server processes each
    document through OneKE extraction and Neo4j upsert, then sends back
    progress updates.
    """
    await websocket.accept()
    logger = logging.getLogger(__name__)

    # Initialize services per connection (reusing app state if available)
    store: SqliteStore = getattr(websocket.app.state, "store", None) or SqliteStore(settings.sqlite_path)
    neo4j: Neo4jClient | None = getattr(websocket.app.state, "neo4j", None)
    registry = SchemaRegistry(store)
    graph_builder = GraphBuilder()

    service = BatchIngestService()
    # Use the service's oneke client (already configured with schema_registry)
    oneke = service.oneke

    if neo4j is None:
        await websocket.send_json({"type": "error", "message": "Neo4j not available"})
        await websocket.close()
        return

    await websocket.send_json({"type": "connected", "message": "Ready to ingest. Send documents as JSON."})

    try:
        while True:
            message = await websocket.receive_text()
            try:
                doc = json.loads(message)
            except json.JSONDecodeError as e:
                await websocket.send_json({"type": "error", "message": f"Invalid JSON: {e}"})
                continue

            schema_name = doc.get("schema_name", "MOE_News")
            item = BatchIngestService._dict_to_news_item(doc)
            doc_id = item.generate_id()
            item_job_id = str(uuid.uuid4())
            item_text = item.to_oneke_text()

            try:
                # Store raw doc
                store.create_doc(
                    doc_id=doc_id,
                    title=item.title or "未命名文档",
                    text=item_text,
                )
                store.create_job(job_id=item_job_id, doc_id=doc_id)

                # OneKE extraction
                extraction = await oneke.extract(text=item_text, schema_name=schema_name)

                # Build graph
                metadata = extract_structured_metadata(item)
                nodes, edges = graph_builder.build_from_extraction(
                    extraction=extraction,
                    doc_id=doc_id,
                    metadata=metadata,
                )
                await neo4j.upsert_graph(nodes=nodes, edges=edges)

                store.update_job(job_id=item_job_id, status="finished", error=None)

                await websocket.send_json({
                    "type": "progress",
                    "doc_id": doc_id,
                    "title": item.title,
                    "entities": len(nodes),
                    "relations": len(edges),
                    "status": "finished",
                })

            except Exception as e:
                logger.exception(f"WebSocket ingest failed for doc {doc_id}")
                store.update_job(job_id=item_job_id, status="failed", error=str(e))
                await websocket.send_json({
                    "type": "progress",
                    "doc_id": doc_id,
                    "title": item.title,
                    "status": "failed",
                    "error": str(e),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


