from __future__ import annotations

import io
import re
from typing import Any, cast

import httpx


class RAGFlowClient:
    def __init__(self, base_url: str, *, api_key: str = "", dataset_name: str = "wikiProject-kb") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.dataset_name = dataset_name.strip() or "wikiProject-kb"

    def _enabled(self) -> bool:
        return bool(self.base_url and self.api_key)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _api_url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    async def _ensure_dataset(self, client: httpx.AsyncClient) -> dict[str, Any]:
        params: dict[str, str | int | bool] = {
            "page": 1,
            "page_size": 30,
            "orderby": "create_time",
            "desc": True,
            "name": self.dataset_name,
        }
        list_resp = await client.get(self._api_url("/datasets"), params=params, headers=self._headers())
        list_resp.raise_for_status()
        list_data = cast(dict[str, Any], list_resp.json())
        if list_data.get("code") != 0:
            raise RuntimeError(f"ragflow_list_datasets_failed: {list_data.get('message')}")

        datasets = list_data.get("data")
        if isinstance(datasets, list):
            for ds in datasets:
                if not isinstance(ds, dict):
                    continue
                if ds.get("name") == self.dataset_name and isinstance(ds.get("id"), str):
                    return ds

        create_resp = await client.post(
            self._api_url("/datasets"),
            headers=self._headers(),
            json={"name": self.dataset_name},
        )
        create_resp.raise_for_status()
        create_data = cast(dict[str, Any], create_resp.json())
        if create_data.get("code") != 0:
            raise RuntimeError(f"ragflow_create_dataset_failed: {create_data.get('message')}")
        created = create_data.get("data")
        if not isinstance(created, dict) or not isinstance(created.get("id"), str):
            raise RuntimeError("ragflow_create_dataset_invalid_response")
        return created

    def _safe_name(self, value: str) -> str:
        cleaned = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", value).strip("_")
        return cleaned[:80] if cleaned else "untitled"

    async def _remove_existing_docs(self, client: httpx.AsyncClient, *, dataset_id: str, doc_id: str) -> None:
        prefix = f"{doc_id}__"
        params: dict[str, str | int | bool] = {
            "keywords": doc_id,
            "page": 1,
            "page_size": 100,
            "orderby": "create_time",
            "desc": True,
        }
        docs_resp = await client.get(
            self._api_url(f"/datasets/{dataset_id}/documents"),
            params=params,
            headers=self._headers(),
        )
        docs_resp.raise_for_status()
        docs_data = cast(dict[str, Any], docs_resp.json())
        if docs_data.get("code") != 0:
            return
        raw_docs = docs_data.get("data", {}).get("docs")
        if not isinstance(raw_docs, list):
            return

        hit_ids: list[str] = []
        for row in raw_docs:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            rid = row.get("id")
            if isinstance(name, str) and isinstance(rid, str) and name.startswith(prefix):
                hit_ids.append(rid)
        if not hit_ids:
            return

        rm_resp = await client.request(
            "DELETE",
            self._api_url(f"/datasets/{dataset_id}/documents"),
            headers=self._headers(),
            json={"ids": hit_ids, "delete_all": False},
        )
        rm_resp.raise_for_status()

    async def ingest_text(self, *, doc_id: str, title: str, text: str) -> dict[str, Any]:
        if not self._enabled():
            return {"enabled": False}

        async with httpx.AsyncClient(timeout=180.0) as client:
            dataset = await self._ensure_dataset(client)
            dataset_id = cast(str, dataset["id"])

            await self._remove_existing_docs(client, dataset_id=dataset_id, doc_id=doc_id)

            display_name = f"{doc_id}__{self._safe_name(title)}.txt"
            content = f"标题：{title}\n文档ID：{doc_id}\n\n{text}".encode("utf-8")
            files = {"file": (display_name, io.BytesIO(content), "text/plain")}
            upload_resp = await client.post(
                self._api_url(f"/datasets/{dataset_id}/documents"),
                headers=self._headers(),
                files=files,
            )
            upload_resp.raise_for_status()
            upload_data = cast(dict[str, Any], upload_resp.json())
            if upload_data.get("code") != 0:
                raise RuntimeError(f"ragflow_upload_failed: {upload_data.get('message')}")

            docs = upload_data.get("data")
            if not isinstance(docs, list):
                raise RuntimeError("ragflow_upload_invalid_response")
            document_ids = [row.get("id") for row in docs if isinstance(row, dict) and isinstance(row.get("id"), str)]
            if not document_ids:
                raise RuntimeError("ragflow_upload_no_document_id")

            parse_resp = await client.post(
                self._api_url(f"/datasets/{dataset_id}/chunks"),
                headers=self._headers(),
                json={"document_ids": document_ids},
            )
            parse_resp.raise_for_status()
            parse_data = cast(dict[str, Any], parse_resp.json())
            if parse_data.get("code") != 0:
                raise RuntimeError(f"ragflow_parse_failed: {parse_data.get('message')}")

            return {
                "enabled": True,
                "dataset_id": dataset_id,
                "dataset_name": self.dataset_name,
                "document_ids": document_ids,
            }

    async def query(
        self,
        *,
        query: str,
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        if not self._enabled():
            return {"enabled": False}

        async with httpx.AsyncClient(timeout=180.0) as client:
            dataset = await self._ensure_dataset(client)
            dataset_id = cast(str, dataset["id"])

            payload: dict[str, Any] = {
                "question": query,
                "dataset_ids": [dataset_id],
                "document_ids": [],
                "page": 1,
                "page_size": max(top_k * 4, 30),
                "top_k": max(top_k * 4, 30),
                "similarity_threshold": 0.1,
                "vector_similarity_weight": 0.3,
                "keyword": True,
                "use_kg": False,
                "toc_enhance": False,
            }
            resp = await client.post(self._api_url("/retrieval"), headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = cast(dict[str, Any], resp.json())
            if data.get("code") != 0:
                raise RuntimeError(f"ragflow_retrieval_failed: {data.get('message')}")

            raw_chunks = data.get("data", {}).get("chunks")
            if not isinstance(raw_chunks, list):
                return {"chunks": []}

            out_chunks: list[dict[str, Any]] = []
            prefix = f"{doc_id}__" if doc_id else ""
            for row in raw_chunks:
                if not isinstance(row, dict):
                    continue
                content = row.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                doc_name = row.get("document_name")
                if prefix and isinstance(doc_name, str) and not doc_name.startswith(prefix):
                    continue
                out_chunks.append(
                    {
                        "text": content.strip(),
                        "score": row.get("similarity"),
                        "meta": {
                            "dataset_id": dataset_id,
                            "document_id": row.get("document_id"),
                            "document_name": doc_name,
                        },
                    }
                )
                if len(out_chunks) >= top_k:
                    break

            return {"chunks": out_chunks}
