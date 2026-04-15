from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.domain.extraction.models import ExtractionResult, ExtractedEntity, ExtractedRelation

NEWS_ENTITY_TYPES = [
    "Organization",
    "Policy",
    "Project",
    "Time",
    "Location",
    "Person",
    "Industry",
    "Theme",
    "Meeting",
    "Other",
]

NEWS_RELATION_TYPES = [
    ("Organization", "发布", "Policy"),
    ("Organization", "启动", "Project"),
    ("Organization", "召开", "Meeting"),
    ("Policy", "属于", "Industry"),
    ("Policy", "涉及", "Location"),
    ("Policy", "时间", "Time"),
    ("Meeting", "召开于", "Location"),
    ("Meeting", "时间", "Time"),
    ("Project", "覆盖", "Location"),
    ("Organization", "位于", "Location"),
    ("Organization", "参与", "Project"),
    ("Person", "出席", "Meeting"),
    ("Person", "发布", "Policy"),
]


class OneKEClient:
    def __init__(
        self,
        base_url: str,
        *,
        openai_base_url: str = "",
        openai_api_key: str = "",
        openai_model: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.openai_base_url = openai_base_url.rstrip("/")
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model

    async def extract(self, *, text: str, schema_name: str | None = None) -> ExtractionResult:
        if not self.base_url:
            if self.openai_api_key and self.openai_model:
                return await self._extract_llm(text=text, schema_name=schema_name)
            return self._extract_demo(text=text)

        payload = self._build_remote_payload(text=text, schema_name=schema_name)
        data = await self._request_remote(payload=payload)
        entities, relations = self._normalize_remote_result(data=data)
        return ExtractionResult(entities=entities, relations=relations, engine="oneke")

    def _build_remote_payload(self, *, text: str, schema_name: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"text": text}
        if schema_name:
            payload["schema_name"] = schema_name

        entity_types_str = ", ".join(NEWS_ENTITY_TYPES)
        relation_examples = "; ".join([f"{s}-{r}->{t}" for s, r, t in NEWS_RELATION_TYPES[:5]])

        payload["extraction"] = {
            "task": "Base",
            "instruction": (
                f"Extract key information from the news/policy text. "
                f"Entity types: {entity_types_str}. "
                f"Relation examples: {relation_examples}..."
            ),
            "output_schema": schema_name or "MOE_News",
            "mode": "customized",
            "entity_types": NEWS_ENTITY_TYPES,
        }
        return payload

    async def _request_remote(self, *, payload: dict[str, Any]) -> object:
        errors: list[str] = []
        endpoints = ["/extract", "/api/extract", "/v1/extract"]
        limits = httpx.Limits(max_keepalive_connections=0, max_connections=10)
        async with httpx.AsyncClient(
            timeout=180.0,
            limits=limits,
            http1=True,
            http2=False,
        ) as client:
            for ep in endpoints:
                url = f"{self.base_url}{ep}"
                try:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as e:
                    errors.append(f"{ep}:{repr(e)}")
        raise RuntimeError("; ".join(errors))

    def _normalize_remote_result(
        self, *, data: object
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        if isinstance(data, dict):
            direct_entities = data.get("entities")
            direct_relations = data.get("relations")
            if isinstance(direct_entities, list) or isinstance(direct_relations, list):
                return (
                    self._build_entities(direct_entities),
                    self._build_relations(direct_relations),
                )

            for key in ("result", "data", "output"):
                nested = data.get(key)
                if isinstance(nested, dict):
                    entities = nested.get("entities")
                    relations = nested.get("relations")
                    if isinstance(entities, list) or isinstance(relations, list):
                        return (
                            self._build_entities(entities),
                            self._build_relations(relations),
                        )

        return ([], [])

    def _build_entities(self, rows: object) -> list[ExtractedEntity]:
        out: list[ExtractedEntity] = []
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            type_ = row.get("type")
            if not isinstance(name, str) or not name.strip():
                continue
            if not isinstance(type_, str) or not type_.strip():
                type_ = "Entity"
            out.append(ExtractedEntity(name=name.strip(), type=type_.strip()))
        return out

    def _build_relations(self, rows: object) -> list[ExtractedRelation]:
        out: list[ExtractedRelation] = []
        if not isinstance(rows, list):
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            source = row.get("source")
            target = row.get("target")
            type_ = row.get("type")
            evidence = row.get("evidence")
            if not isinstance(source, str) or not source.strip():
                continue
            if not isinstance(target, str) or not target.strip():
                continue
            if not isinstance(type_, str) or not type_.strip():
                type_ = "关联"
            if not isinstance(evidence, str):
                evidence = None
            out.append(
                ExtractedRelation(
                    source=source.strip(),
                    target=target.strip(),
                    type=type_.strip(),
                    evidence=evidence.strip() if isinstance(evidence, str) and evidence.strip() else None,
                )
            )
        return out

    async def _extract_llm(self, *, text: str, schema_name: str | None) -> ExtractionResult:
        base_url = self.openai_base_url or "https://api.openai.com/v1"
        base_url = base_url.rstrip("/")

        snippet = text.strip()
        if len(snippet) > 12000:
            snippet = snippet[:12000]

        schema_hint = schema_name or "MOE_News"

        system = (
            "你是 OneKE 风格的模式引导信息抽取助手。"
            "只输出严格 JSON（不要 Markdown，不要解释）。"
        )
        user = (
            f"从下面文本中抽取实体与关系，抽取粒度偏新闻/政策/机构/人物/项目/会议/地点/时间/设备。\n"
            f"schema_name={schema_hint}\n\n"
            "输出 JSON 结构必须是：\n"
            '{\n'
            '  "entities": [{"name": "...", "type": "..."}],\n'
            '  "relations": [{"source": "...", "target": "...", "type": "...", "evidence": "..."}]\n'
            "}\n\n"
            "约束：\n"
            "- entities.type 建议使用：Organization, Person, Policy, Project, Meeting, Location, Time, Device, Other\n"
            "- relations.type 使用中文动词短语即可，例如：召开于、发布、指出、参与、启动、支持、覆盖、配置、培训、强调\n"
            "- evidence 必须是原文中的一句短引文（<=120字）\n\n"
            f"文本：\n{snippet}"
        )

        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        payload: dict[str, Any] = {
            "model": self.openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = self._extract_json_object(content)

        entities = [ExtractedEntity(**e) for e in parsed.get("entities", [])]
        relations = [ExtractedRelation(**r) for r in parsed.get("relations", [])]
        return ExtractionResult(entities=entities, relations=relations, engine="oneke-llm")

    def _extract_json_object(self, text: str) -> dict[str, Any]:
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

        candidate = text[start : end + 1]
        candidate = candidate.strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {"entities": [], "relations": []}

    def _extract_demo(self, *, text: str) -> ExtractionResult:
        text = (text or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        entities: dict[str, ExtractedEntity] = {}
        relations: list[ExtractedRelation] = []

        pattern = re.compile(r"^(?P<a>[^，,。;；]+?)\s+(?P<rel>是|属于|位于|包含|负责)\s+(?P<b>.+?)$")
        for ln in lines[:200]:
            m = pattern.match(ln)
            if not m:
                continue
            a = m.group("a").strip()
            b = m.group("b").strip().rstrip("。.")
            rel = m.group("rel").strip()
            entities.setdefault(a, ExtractedEntity(name=a, type="Entity"))
            entities.setdefault(b, ExtractedEntity(name=b, type="Entity"))
            relations.append(
                ExtractedRelation(source=a, target=b, type=rel, evidence=ln[:2000])
            )

        if relations:
            return ExtractionResult(
                entities=list(entities.values()),
                relations=relations,
                engine="demo",
            )

        org_hints = [
            "教育部",
            "教育部办公厅",
            "国务院",
            "中国高等教育学会",
            "中国教师发展基金会",
            "腾讯公司",
            "新疆生产建设兵团",
        ]
        org_suffix = r"(部|厅|委|委员会|学会|协会|基金会|公司|集团|中心|大学|学院|中学|小学|学校|办公室)"
        org_pat = re.compile(rf"(?P<org>[\u4e00-\u9fffA-Za-z0-9·（）()]{{2,40}}{org_suffix})")
        title_pat = re.compile(r"[《「『\"](?P<title>[^》」』\"]{{2,60}})[》」』\"]")
        time_pat = re.compile(
            r"(?P<t>\d{{4}}年\d{{1,2}}月\d{{1,2}}日|\d{{4}}年\d{{1,2}}月|\d{{4}}-\d{{1,2}}-\d{{1,2}}|\d{{4}}/\d{{1,2}}/\d{{1,2}}|近日|日前|今年|本周|上周|上月)"
        )
        loc_pat = re.compile(r"在(?P<loc>[^，。；;]{{2,20}}?)(召开|启动|举行|举办|发布|印发)")
        meeting_pat = re.compile(r"(?P<m>[\u4e00-\u9fffA-Za-z0-9·（）()\"\"]{{2,40}}(会议|工作会|理事会|论坛|发布会))")
        device_pat = re.compile(r"(AED|自动体外除颤器)")
        project_pat = re.compile(r"(?P<p>[\u4e00-\u9fffA-Za-z0-9·（）()\"\"]{{2,40}}(公益项目|项目|行动|计划|工程))")

        def add_entity(name: str, etype: str) -> None:
            key = f"{etype}:{name}"
            if key not in entities:
                entities[key] = ExtractedEntity(name=name, type=etype)

        def add_relation(source: str, target: str, rtype: str, evidence: str) -> None:
            relations.append(
                ExtractedRelation(
                    source=source,
                    target=target,
                    type=rtype,
                    evidence=evidence[:2000] if evidence else None,
                )
            )

        def guess_orgs(s: str) -> list[str]:
            found = set()
            for h in org_hints:
                if h in s:
                    found.add(h)
            for m in org_pat.finditer(s):
                found.add(m.group("org"))
            return list(found)

        sentences = re.split(r"[。；;]\s*", text)
        for raw in sentences:
            s = raw.strip()
            if not s:
                continue

            orgs = guess_orgs(s)
            titles = [m.group("title") for m in title_pat.finditer(s)]
            times = [m.group("t") for m in time_pat.finditer(s)]
            locs = [m.group("loc") for m in loc_pat.finditer(s)]
            meetings = [m.group("m") for m in meeting_pat.finditer(s)]
            devices = [m.group(1) for m in device_pat.finditer(s)]
            projects = [m.group("p") for m in project_pat.finditer(s)]

            for org in orgs:
                add_entity(org, "Organization")
            for t in times:
                add_entity(t, "Time")
            for loc in locs:
                add_entity(loc, "Location")
            for d in devices:
                add_entity(d, "Device")
            for mname in meetings:
                add_entity(mname, "Meeting")
            for pname in projects:
                add_entity(pname, "Project")
            for title in titles:
                add_entity(title, "Policy")

            if "印发" in s or "发布" in s or "印发了" in s:
                for org in orgs:
                    for title in titles:
                        add_relation(org, title, "发布", s)

            if "启动" in s:
                for org in orgs:
                    for pname in projects:
                        add_relation(org, pname, "启动", s)

            if "召开" in s or "举行" in s or "举办" in s:
                for org in orgs:
                    for mname in meetings:
                        add_relation(org, mname, "召开", s)
                for mname in meetings:
                    for loc in locs:
                        add_relation(mname, loc, "召开于", s)
                    for t in times:
                        add_relation(mname, t, "时间", s)

            if "配置" in s or "配备" in s:
                for pname in projects:
                    for d in devices:
                        add_relation(pname, d, "配置", s)

            if "覆盖" in s:
                for pname in projects:
                    for loc in locs:
                        add_relation(pname, loc, "覆盖", s)

            if "培训" in s:
                for pname in projects:
                    add_entity("师资", "Other")
                    add_relation(pname, "师资", "培训", s)

        return ExtractionResult(
            entities=list(entities.values()),
            relations=relations,
            engine="demo",
        )
