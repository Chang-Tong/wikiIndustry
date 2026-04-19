from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class DocRow:
    doc_id: str
    title: str
    text: str
    created_at: str


JobStatus = Literal["queued", "running", "finished", "failed"]


@dataclass(frozen=True)
class JobRow:
    job_id: str
    doc_id: str
    status: JobStatus
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class SchemaRow:
    schema_id: str
    schema_name: str
    entity_types: str
    relation_types: str
    instruction: str | None
    created_at: str
    updated_at: str


class SqliteStore:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS docs (
                  doc_id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  text TEXT NOT NULL,
                  created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  job_id TEXT PRIMARY KEY,
                  doc_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  error TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(doc_id) REFERENCES docs(doc_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schemas (
                  schema_id TEXT PRIMARY KEY,
                  schema_name TEXT NOT NULL UNIQUE,
                  entity_types TEXT NOT NULL,
                  relation_types TEXT NOT NULL,
                  instruction TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )

    def create_doc(self, doc_id: str, title: str, text: str) -> DocRow:
        created_at = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO docs (doc_id, title, text, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, title, text, created_at),
            )
        return DocRow(doc_id=doc_id, title=title, text=text, created_at=created_at)

    def upsert_doc(self, doc_id: str, title: str, text: str) -> DocRow:
        created_at = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO docs (doc_id, title, text, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, title, text, created_at),
            )
        return DocRow(doc_id=doc_id, title=title, text=text, created_at=created_at)

    def get_doc(self, doc_id: str) -> DocRow | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        return DocRow(
            doc_id=str(row["doc_id"]),
            title=str(row["title"]),
            text=str(row["text"]),
            created_at=str(row["created_at"]),
        )

    def create_job(self, job_id: str, doc_id: str) -> JobRow:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, doc_id, status, error, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, doc_id, "queued", None, now, now),
            )
        return JobRow(
            job_id=job_id,
            doc_id=doc_id,
            status="queued",
            error=None,
            created_at=now,
            updated_at=now,
        )

    def update_job(self, job_id: str, *, status: JobStatus, error: str | None) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?",
                (status, error, now, job_id),
            )

    def get_job(self, job_id: str) -> JobRow | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        status = self._parse_status(str(row["status"]))
        return JobRow(
            job_id=str(row["job_id"]),
            doc_id=str(row["doc_id"]),
            status=status,
            error=str(row["error"]) if row["error"] is not None else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def list_jobs(self, *, limit: int = 50) -> list[JobRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out: list[JobRow] = []
        for row in rows:
            status = self._parse_status(str(row["status"]))
            out.append(
                JobRow(
                    job_id=str(row["job_id"]),
                    doc_id=str(row["doc_id"]),
                    status=status,
                    error=str(row["error"]) if row["error"] is not None else None,
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                )
            )
        return out

    def list_finished_doc_ids(self, *, limit: int = 200) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, MAX(updated_at) AS latest
                FROM jobs
                WHERE status = 'finished'
                GROUP BY doc_id
                ORDER BY latest DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [str(row["doc_id"]) for row in rows]

    def clear_all(self) -> None:
        """Clear all data from docs and jobs tables."""
        with self._connect() as conn:
            conn.execute("DELETE FROM docs")
            conn.execute("DELETE FROM jobs")
            conn.commit()

    def to_debug_dict(self) -> dict[str, Any]:
        return {"sqlite_path": self.path}

    def search_docs(
        self,
        keywords: list[str],
        *,
        limit: int = 50,
    ) -> list[tuple[str, str, str]]:
        """Search docs by keywords (OR match on title or text).

        Returns list of (doc_id, title, text) tuples ordered by created_at desc.
        """
        if not keywords:
            return []
        conn = self._connect()
        try:
            conditions = " OR ".join(["title LIKE ? OR text LIKE ?"] * len(keywords))
            params: list[str] = []
            for kw in keywords:
                like = f"%{kw}%"
                params.extend([like, like])
            query = f"""
                SELECT doc_id, title, text FROM docs
                WHERE {conditions}
                ORDER BY created_at DESC
                LIMIT ?
            """
            params.append(str(limit))
            rows = conn.execute(query, params).fetchall()
            return [(str(r["doc_id"]), str(r["title"]), str(r["text"])) for r in rows]
        finally:
            conn.close()

    def _parse_status(self, status: str) -> JobStatus:
        if status not in ("queued", "running", "finished", "failed"):
            return "failed"
        return cast(JobStatus, status)

    # ── Schema registry ─────────────────────────────────────────────

    def create_schema(
        self,
        schema_id: str,
        schema_name: str,
        entity_types: str,
        relation_types: str,
        instruction: str | None = None,
    ) -> SchemaRow:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO schemas (schema_id, schema_name, entity_types, relation_types, instruction, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (schema_id, schema_name, entity_types, relation_types, instruction, now, now),
            )
        return SchemaRow(
            schema_id=schema_id,
            schema_name=schema_name,
            entity_types=entity_types,
            relation_types=relation_types,
            instruction=instruction,
            created_at=now,
            updated_at=now,
        )

    def get_schema_by_name(self, schema_name: str) -> SchemaRow | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM schemas WHERE schema_name = ?", (schema_name,)).fetchone()
        if row is None:
            return None
        return SchemaRow(
            schema_id=str(row["schema_id"]),
            schema_name=str(row["schema_name"]),
            entity_types=str(row["entity_types"]),
            relation_types=str(row["relation_types"]),
            instruction=str(row["instruction"]) if row["instruction"] is not None else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def get_schema_by_id(self, schema_id: str) -> SchemaRow | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM schemas WHERE schema_id = ?", (schema_id,)).fetchone()
        if row is None:
            return None
        return SchemaRow(
            schema_id=str(row["schema_id"]),
            schema_name=str(row["schema_name"]),
            entity_types=str(row["entity_types"]),
            relation_types=str(row["relation_types"]),
            instruction=str(row["instruction"]) if row["instruction"] is not None else None,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def list_schemas(self, *, limit: int = 100) -> list[SchemaRow]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM schemas ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out: list[SchemaRow] = []
        for row in rows:
            out.append(
                SchemaRow(
                    schema_id=str(row["schema_id"]),
                    schema_name=str(row["schema_name"]),
                    entity_types=str(row["entity_types"]),
                    relation_types=str(row["relation_types"]),
                    instruction=str(row["instruction"]) if row["instruction"] is not None else None,
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                )
            )
        return out

    def update_schema(
        self,
        schema_id: str,
        *,
        schema_name: str | None = None,
        entity_types: str | None = None,
        relation_types: str | None = None,
        instruction: str | None = None,
    ) -> None:
        now = utc_now_iso()
        fields: list[str] = []
        values: list[Any] = []
        if schema_name is not None:
            fields.append("schema_name = ?")
            values.append(schema_name)
        if entity_types is not None:
            fields.append("entity_types = ?")
            values.append(entity_types)
        if relation_types is not None:
            fields.append("relation_types = ?")
            values.append(relation_types)
        if instruction is not None:
            fields.append("instruction = ?")
            values.append(instruction)
        if not fields:
            return
        fields.append("updated_at = ?")
        values.append(now)
        values.append(schema_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE schemas SET {', '.join(fields)} WHERE schema_id = ?",
                values,
            )

    def delete_schema(self, schema_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM schemas WHERE schema_id = ?", (schema_id,))
            return cursor.rowcount > 0
