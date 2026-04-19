from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes_correlation import router as correlation_router
from app.api.v1.routes_docs import router as docs_router
from app.api.v1.routes_extract import router as extract_router
from app.api.v1.routes_graph import router as graph_router
from app.api.v1.routes_ingest_stream import router as ingest_router
from app.api.v1.routes_json import router as json_router
from app.api.v1.routes_owl import router as owl_router
from app.api.v1.routes_qa_structured import router as qa_structured_router
from app.api.v1.routes_schema import router as schema_router

api_router = APIRouter()
api_router.include_router(docs_router, tags=["docs"])
api_router.include_router(extract_router, tags=["extract"])
api_router.include_router(graph_router, tags=["graph"])
api_router.include_router(ingest_router, tags=["ingest"])
api_router.include_router(json_router, tags=["json"])
api_router.include_router(correlation_router, tags=["correlation"])
api_router.include_router(owl_router, tags=["export"])
api_router.include_router(schema_router, tags=["schema"])
api_router.include_router(qa_structured_router, tags=["qa"])

