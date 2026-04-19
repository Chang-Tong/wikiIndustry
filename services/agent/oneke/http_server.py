from __future__ import annotations

import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)
    schema_name: str | None = None
    extraction: dict[str, Any] | None = None


app = FastAPI(title="OneKE HTTP Adapter")


def _llm_config() -> tuple[str, str, str]:
    base_url = (os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    return base_url, api_key, model


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {"entities": [], "relations": []}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"entities": [], "relations": []}
    candidate = text[start : end + 1].strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"entities": [], "relations": []}


def _normalize_result(payload: dict[str, Any]) -> dict[str, Any]:
    entities: list[dict[str, str]] = []
    relations: list[dict[str, str]] = []

    for row in payload.get("entities", []):
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        type_ = row.get("type")
        if isinstance(name, str) and name.strip():
            entities.append({"name": name.strip(), "type": type_.strip() if isinstance(type_, str) and type_.strip() else "Entity"})

    for row in payload.get("relations", []):
        if not isinstance(row, dict):
            continue
        source = row.get("source")
        target = row.get("target")
        if not isinstance(source, str) or not source.strip():
            continue
        if not isinstance(target, str) or not target.strip():
            continue
        type_ = row.get("type")
        evidence = row.get("evidence")
        relations.append(
            {
                "source": source.strip(),
                "target": target.strip(),
                "type": type_.strip() if isinstance(type_, str) and type_.strip() else "关联",
                "evidence": evidence.strip() if isinstance(evidence, str) and evidence.strip() else "",
            }
        )
    return {"entities": entities, "relations": relations}


async def _oneke_extract(text: str, schema_name: str) -> dict[str, Any]:
    base_url, api_key, model = _llm_config()
    if not api_key or not model:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY 或 OPENAI_MODEL 未配置")

    system = "你是 OneKE 风格的模式引导信息抽取助手，只输出严格 JSON。"
    user = (
        "你需要根据 schema 进行信息抽取。\n"
        f"schema_name={schema_name}\n"
        "输出 JSON 结构必须是：\n"
        "{\n"
        '  "entities": [{"name": "...", "type": "..."}],\n'
        '  "relations": [{"source": "...", "target": "...", "type": "...", "evidence": "..."}]\n'
        "}\n"
        "约束：\n"
        "- 只输出 JSON，不要 Markdown\n"
        "- evidence 尽量是原文中的短句\n"
        "- entities.type 尽量精确\n\n"
        f"文本：\n{text[:18000]}"
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    parsed = _extract_json_object(content if isinstance(content, str) else "")
    return _normalize_result(parsed)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
async def extract_root(payload: ExtractRequest) -> dict[str, Any]:
    schema_name = payload.schema_name or str((payload.extraction or {}).get("output_schema") or "MOE_News")
    return await _oneke_extract(payload.text, schema_name)


@app.post("/api/v1/oneke/extract")
async def extract_api(payload: ExtractRequest) -> dict[str, Any]:
    schema_name = payload.schema_name or str((payload.extraction or {}).get("output_schema") or "MOE_News")
    return await _oneke_extract(payload.text, schema_name)
