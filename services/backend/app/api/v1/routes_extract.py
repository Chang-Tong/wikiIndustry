from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
from io import BytesIO
from typing import Any
from uuid import uuid4

import httpx
from fastapi import APIRouter, BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.core.settings import settings
from app.domain.extraction.models import ExtractionResult
from app.integrations.neo4j.client import GraphEdge, GraphNode, Neo4jClient, stable_id
from app.integrations.oneke.client import OneKEClient
from app.integrations.ragflow.client import RAGFlowClient
from app.store.sqlite import SqliteStore

router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StartExtractRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    schema_name: str | None = None


class StartExtractResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    doc_id: str
    status: str
    error: str | None


class UploadExtractResponse(BaseModel):
    doc_id: str
    job_id: str


class BatchJobItem(BaseModel):
    batch_no: int
    doc_id: str
    job_id: str
    item_count: int


class UploadBatchExtractResponse(BaseModel):
    total_items: int
    batch_size: int
    total_batches: int
    batches: list[BatchJobItem]


class SplitPreviewItem(BaseModel):
    id: str
    title: str
    source: str
    category: str
    sub_category: str
    url: str
    date: str
    summary: str


class SplitPreviewResponse(BaseModel):
    total_items: int
    categories: dict[str, int]
    items: list[SplitPreviewItem]


class StructuredNewsItem(BaseModel):
    id: str
    title: str
    source: str
    category: str
    sub_category: str
    url: str
    date: str
    summary: str
    tags: dict[str, list[str]]


class StructuredExportResponse(BaseModel):
    total_items: int
    categories: dict[str, int]
    items: list[StructuredNewsItem]
    cypher: str | None = None


class LocalOneKEExtractRequest(BaseModel):
    text: str = Field(min_length=1)
    schema_name: str | None = None


def _extract_text_from_docx_bytes(data: bytes) -> str:
    docx = importlib.import_module("docx")
    document_factory = getattr(docx, "Document")
    doc = document_factory(BytesIO(data))
    parts: list[str] = []

    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)

    for table in doc.tables:
        for row in table.rows:
            cells = [(c.text or "").strip() for c in row.cells]
            line = "\t".join([c for c in cells if c])
            if line:
                parts.append(line)

    return "\n".join(parts).strip()


async def _read_upload_as_text(file: UploadFile) -> str:
    data = await file.read()
    filename = (file.filename or "").lower()

    if filename.endswith(".docx"):
        return _extract_text_from_docx_bytes(data)

    if filename.endswith(".txt") or filename.endswith(".md"):
        return data.decode("utf-8", errors="replace").strip()

    ctype = (file.content_type or "").lower()
    if ctype.startswith("text/"):
        return data.decode("utf-8", errors="replace").strip()

    raise HTTPException(status_code=400, detail="unsupported_file_type")


def _parse_moe_news_bundle(text: str) -> list[dict[str, str]]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not any("标题：" in ln and ln.startswith("【") for ln in lines):
        return []

    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    title_re = re.compile(r"^【(?P<idx>\d+)】标题：(?P<title>.+)$")
    kv_re = re.compile(r"^(?P<k>所属网站|子栏目|网址|日期|摘要)：(?P<v>.*)$")
    category_count_re = re.compile(r"^(?P<cat>.+?)（\d+条）$")
    known_categories = {
        "国务院要闻",
        "教育部文件",
        "教育要闻",
        "其他部门文件",
        "时政要闻",
        "政策解读",
        "北京要闻",
        "教育新闻",
        "其他文件",
        "黑龙江省教育厅_教育厅",
        "黑龙江省教育厅_时政要闻",
    }
    current_category = ""

    def normalize_category(s: str) -> str:
        return re.sub(r"^[一二三四五六七八九十]+、", "", s.strip())

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        if current.get("title"):
            items.append(current)
        current = None

    for ln in lines:
        cm = category_count_re.match(ln)
        if cm:
            c = normalize_category(cm.group("cat"))
            if c:
                current_category = c
            continue
        maybe_category = normalize_category(ln)
        if maybe_category in known_categories:
            current_category = maybe_category
            continue
        m = title_re.match(ln)
        if m:
            flush()
            current = {
                "idx": m.group("idx"),
                "title": m.group("title").strip(),
                "摘要": "",
                "一级分类": current_category,
            }
            continue
        if current is None:
            continue
        m2 = kv_re.match(ln)
        if m2:
            key = m2.group("k")
            val = m2.group("v").strip()
            current[key] = val
            continue
        if "摘要" in current and current["摘要"]:
            current["摘要"] = f"{current['摘要']}\n{ln}".strip()
        else:
            current["摘要"] = ln

    flush()
    return items


def _build_news_batch_text(items: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for it in items:
        idx = (it.get("idx") or "").strip()
        title = (it.get("title") or "").strip()
        if title:
            if idx:
                lines.append(f"【{idx}】标题：{title}")
            else:
                lines.append(f"标题：{title}")
        for key in ("所属网站", "子栏目", "网址", "日期"):
            val = (it.get(key) or "").strip()
            if val:
                lines.append(f"{key}：{val}")
        top_category = (it.get("一级分类") or "").strip()
        if top_category:
            lines.append(f"一级分类：{top_category}")
        summary = (it.get("摘要") or "").strip()
        if summary:
            lines.append(f"摘要：{summary}")
        lines.append("")
    return "\n".join(lines).strip()


def _preview_item_from_bundle(it: dict[str, str]) -> SplitPreviewItem:
    idx = (it.get("idx") or "").strip()
    top = (it.get("一级分类") or "").strip()
    id_prefix = top if top else "未分类"
    item_id = f"{id_prefix}-{idx}" if idx else id_prefix
    return SplitPreviewItem(
        id=item_id,
        title=(it.get("title") or "").strip(),
        source=(it.get("所属网站") or "").strip(),
        category=top,
        sub_category=(it.get("子栏目") or "").strip(),
        url=(it.get("网址") or "").strip(),
        date=(it.get("日期") or "").strip(),
        summary=(it.get("摘要") or "").strip(),
    )


def _structured_item_from_bundle(it: dict[str, str]) -> StructuredNewsItem:
    idx = (it.get("idx") or "").strip()
    title = (it.get("title") or "").strip()
    source = (it.get("所属网站") or "").strip()
    category = (it.get("一级分类") or "").strip() or "未分类"
    sub_category = (it.get("子栏目") or "").strip()
    url = (it.get("网址") or "").strip()
    date = (it.get("日期") or "").strip()
    summary = (it.get("摘要") or "").strip()
    content_tag = _guess_news_type(title=title, summary=summary)
    tags = {
        "province": _extract_province_tags(title=title, summary=summary),
        "city": _extract_city_tags(title=title, summary=summary),
        "industry": _extract_industry_tags(title=title, summary=summary),
        "content": [content_tag] if content_tag else [],
        "theme": _extract_theme_tags(title=title, summary=summary),
        "department": _guess_departments(site=source, title=title, summary=summary),
    }
    return StructuredNewsItem(
        id=f"{category}-{idx}" if idx else category,
        title=title,
        source=source,
        category=category,
        sub_category=sub_category,
        url=url,
        date=date,
        summary=summary,
        tags=tags,
    )


def _build_import_cypher(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("WITH " + json.dumps(rows, ensure_ascii=False) + " AS rows")
    lines.append("UNWIND rows AS row")
    lines.append("MERGE (n:News {id: row.id})")
    lines.append(
        "SET n.title=row.title, n.source=row.source, n.category=row.category, "
        "n.sub_category=row.sub_category, n.url=row.url, n.date=row.date, n.summary=row.summary"
    )
    lines.append("MERGE (c:Category {name: row.category})")
    lines.append("MERGE (n)-[:属于]->(c)")
    lines.append(
        "FOREACH (x IN CASE WHEN row.sub_category IS NULL OR row.sub_category = '' THEN [] ELSE [row.sub_category] END | "
        "MERGE (sc:SubCategory {name:x}) MERGE (n)-[:子栏目]->(sc))"
    )
    lines.append("FOREACH (x IN coalesce(row.tags.province, []) | MERGE (t:Tag {name:x, kind:'省份'}) MERGE (n)-[:包含]->(t))")
    lines.append("FOREACH (x IN coalesce(row.tags.city, []) | MERGE (t:Tag {name:x, kind:'地市'}) MERGE (n)-[:包含]->(t))")
    lines.append("FOREACH (x IN coalesce(row.tags.industry, []) | MERGE (t:Tag {name:x, kind:'行业'}) MERGE (n)-[:包含]->(t))")
    lines.append("FOREACH (x IN coalesce(row.tags.content, []) | MERGE (t:Tag {name:x, kind:'内容'}) MERGE (n)-[:包含]->(t))")
    lines.append("FOREACH (x IN coalesce(row.tags.theme, []) | MERGE (t:Tag {name:x, kind:'主题'}) MERGE (n)-[:包含]->(t))")
    lines.append("FOREACH (x IN coalesce(row.tags.department, []) | MERGE (t:Tag {name:x, kind:'部门'}) MERGE (n)-[:包含]->(t));")
    lines.append("MATCH (a:News)-[:包含]->(t:Tag)<-[:包含]-(b:News) WHERE a.id < b.id")
    lines.append("WITH a,b,count(DISTINCT t) AS common")
    lines.append("MATCH (a)-[:包含]->(ta:Tag)")
    lines.append("WITH a,b,common,count(DISTINCT ta) AS ca")
    lines.append("MATCH (b)-[:包含]->(tb:Tag)")
    lines.append("WITH a,b,common,ca,count(DISTINCT tb) AS cb")
    lines.append(
        "WITH a,b,toFloat(common) / CASE WHEN (ca + cb - common)=0 THEN 1 ELSE (ca + cb - common) END AS score"
    )
    lines.append("WHERE score >= 0.2")
    lines.append("MERGE (a)-[r:相似]->(b)")
    lines.append("SET r.score = score;")
    return "\n".join(lines)


def _extract_json_from_text(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def _guess_news_type(*, title: str, summary: str) -> str:
    s = f"{title}\n{summary}"
    if any(k in s for k in ("解读", "问答", "答记者问", "释义")):
        return "解读"
    if any(k in s for k in ("规划", "计划纲要", "总体方案", "专项规划", "五年规划")):
        return "规划文件"
    if any(k in s for k in ("通知", "通告")):
        return "通知"
    if any(k in s for k in ("印发", "办法", "条例", "方案", "意见", "规定", "指引", "要点", "细则", "法")):
        return "政策法规"
    if any(k in s for k in ("公告", "公示", "通报", "声明")):
        return "公告"
    return "新闻动态"


def _guess_departments(*, site: str, title: str, summary: str) -> list[str]:
    org_suffix = r"(部|厅|委|委员会|学会|协会|基金会|公司|集团|中心|大学|学院|中学|小学|学校|办公室|银行|医院)"
    org_pat = re.compile(rf"(?P<org>[\u4e00-\u9fffA-Za-z0-9·（）()]{2,40}{org_suffix})")
    found: list[str] = []
    if site:
        found.append(site)
    for m in org_pat.finditer(f"{title}\n{summary}"):
        org = m.group("org").strip()
        if org and org not in found:
            found.append(org)
    return found[:8]


def _extract_tags(*, title: str, summary: str, limit: int = 8) -> list[str]:
    s = re.sub(r"\s+", " ", f"{title} {summary}".strip())
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9·]", "", s)
    stop = {
        "工作",
        "会议",
        "召开",
        "举行",
        "举办",
        "推进",
        "开展",
        "加强",
        "提高",
        "全省",
        "全市",
        "有关",
        "关于",
        "重点",
        "进一步",
        "持续",
        "全面",
        "深入",
        "落实",
        "强调",
        "要求",
        "指出",
        "表示",
        "一行",
        "方面",
        "领域",
        "建设",
        "发展",
        "问题",
        "任务",
        "措施",
        "机制",
    }
    grams: dict[str, int] = {}
    for n in (2, 3, 4):
        for i in range(0, max(0, len(s) - n + 1)):
            g = s[i : i + n]
            if any(ch.isascii() for ch in g):
                continue
            if g in stop:
                continue
            grams[g] = grams.get(g, 0) + 1
    ranked = sorted(grams.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    tags: list[str] = []
    for g, _ in ranked:
        if any(g in t or t in g for t in tags):
            continue
        tags.append(g)
        if len(tags) >= limit:
            break
    if not tags:
        tags = [t for t in title.split() if t][:limit]
    return tags


def _unique_keep_order(items: list[str], limit: int = 10) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        k = x.strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
        if len(out) >= limit:
            break
    return out


def _extract_industry_tags(*, title: str, summary: str) -> list[str]:
    s = f"{title}\n{summary}"
    taxonomy: dict[str, tuple[str, ...]] = {
        "教育": ("教育", "学校", "高校", "中小学", "职教", "师资", "招生", "学位"),
        "数字经济": ("数字化", "数据要素", "算力", "人工智能", "大模型", "云计算", "平台"),
        "先进制造": ("制造业", "工业", "工厂", "装备", "机器人", "产业链", "供应链"),
        "新能源": ("新能源", "储能", "光伏", "风电", "氢能", "充电桩"),
        "生物医药": ("医药", "医疗", "生物", "药品", "医院", "健康"),
        "农业农村": ("农业", "乡村", "种业", "粮食", "农产品"),
        "交通物流": ("交通", "物流", "港口", "航运", "铁路", "机场"),
        "文旅体育": ("文化", "旅游", "文旅", "体育", "赛事"),
        "金融": ("金融", "银行", "保险", "基金", "信贷", "融资"),
        "环保双碳": ("双碳", "碳达峰", "碳中和", "环保", "减排", "生态"),
    }
    out: list[str] = []
    for tag, kws in taxonomy.items():
        if any(k in s for k in kws):
            out.append(tag)
    if not out:
        if "政策" in s or "规定" in s:
            out.append("公共治理")
    return _unique_keep_order(out, limit=8)


def _extract_province_tags(*, title: str, summary: str) -> list[str]:
    s = f"{title}\n{summary}"
    province_alias: dict[str, tuple[str, ...]] = {
        "北京市": ("北京市", "北京"),
        "天津市": ("天津市", "天津"),
        "上海市": ("上海市", "上海"),
        "重庆市": ("重庆市", "重庆"),
        "河北省": ("河北省", "河北"),
        "山西省": ("山西省", "山西"),
        "内蒙古自治区": ("内蒙古自治区", "内蒙古"),
        "辽宁省": ("辽宁省", "辽宁"),
        "吉林省": ("吉林省", "吉林"),
        "黑龙江省": ("黑龙江省", "黑龙江"),
        "江苏省": ("江苏省", "江苏"),
        "浙江省": ("浙江省", "浙江"),
        "安徽省": ("安徽省", "安徽"),
        "福建省": ("福建省", "福建"),
        "江西省": ("江西省", "江西"),
        "山东省": ("山东省", "山东"),
        "河南省": ("河南省", "河南"),
        "湖北省": ("湖北省", "湖北"),
        "湖南省": ("湖南省", "湖南"),
        "广东省": ("广东省", "广东"),
        "广西壮族自治区": ("广西壮族自治区", "广西"),
        "海南省": ("海南省", "海南"),
        "四川省": ("四川省", "四川"),
        "贵州省": ("贵州省", "贵州"),
        "云南省": ("云南省", "云南"),
        "西藏自治区": ("西藏自治区", "西藏"),
        "陕西省": ("陕西省", "陕西"),
        "甘肃省": ("甘肃省", "甘肃"),
        "青海省": ("青海省", "青海"),
        "宁夏回族自治区": ("宁夏回族自治区", "宁夏"),
        "新疆维吾尔自治区": ("新疆维吾尔自治区", "新疆"),
        "香港特别行政区": ("香港特别行政区", "香港"),
        "澳门特别行政区": ("澳门特别行政区", "澳门"),
        "台湾省": ("台湾", "台湾省"),
    }
    out: list[str] = []
    for canon, aliases in province_alias.items():
        if any(a in s for a in aliases):
            out.append(canon)
    return _unique_keep_order(out, limit=8)


def _extract_city_tags(*, title: str, summary: str) -> list[str]:
    s = f"{title}\n{summary}"
    city_pat = re.compile(r"(?P<city>[\u4e00-\u9fff]{2,12}(市|州|盟|地区|新区|区|县))")
    stop = {
        "教育部",
        "国务院",
        "自治区",
        "特别行政区",
        "工作会议",
        "常务会议",
        "发布会",
    }
    out: list[str] = []
    for m in city_pat.finditer(s):
        c = m.group("city").strip()
        if c in stop:
            continue
        if len(c) < 2 or len(c) > 12:
            continue
        if c.endswith("会议") or c.endswith("方案") or c.endswith("行动"):
            continue
        out.append(c)
    return _unique_keep_order(out, limit=10)


def _extract_theme_tags(*, title: str, summary: str) -> list[str]:
    s = f"{title}\n{summary}"
    taxonomy: dict[str, tuple[str, ...]] = {
        "项目建设": ("项目建设", "项目落地", "开工", "建设"),
        "资金补贴": ("资金", "补贴", "奖补", "财政支持", "专项资金"),
        "监管要求": ("监管", "合规", "检查", "督导", "风险防控"),
        "人事任免": ("任命", "免去", "任免", "履新", "任职"),
        "数据发布": ("数据发布", "统计", "同比", "环比", "监测数据", "报告显示"),
        "招生考试": ("招生", "考试", "录取", "学位"),
        "安全教育": ("安全教育", "应急", "急救", "AED", "校园安全"),
        "政策发布": ("印发", "发布", "通知", "规定", "办法"),
        "会议活动": ("会议", "论坛", "发布会", "召开", "举办"),
    }
    out: list[str] = []
    for tag, kws in taxonomy.items():
        if any(k in s for k in kws):
            out.append(tag)
    if not out:
        out = _extract_tags(title=title, summary=summary, limit=8)
    return _unique_keep_order(out, limit=10)


async def _llm_semantic_meta_by_idx(items: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    if not settings.openai_api_key or not settings.openai_model:
        return {}

    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    system = "你是新闻/公告/政策摘要的语义分析助手。只输出严格 JSON（不要 Markdown，不要解释）。"

    out: dict[str, dict[str, Any]] = {}
    chunk_size = 12
    for i in range(0, len(items), chunk_size):
        chunk = items[i : i + chunk_size]
        payload_items = [
            {
                "idx": it.get("idx", ""),
                "title": it.get("title", ""),
                "site": it.get("所属网站", ""),
                "category": it.get("子栏目", ""),
                "date": it.get("日期", ""),
                "summary": (it.get("摘要", "") or "")[:1200],
            }
            for it in chunk
        ]
        user = (
            "对每条内容，给出：\n"
            "- content_type（公告/政策法规/新闻动态/通知/解读/规划文件）\n"
            "- industry_tags（行业标签数组）\n"
            "- provinces（省份标签数组，省级行政区）\n"
            "- cities（地市标签数组，城市/区县/片区）\n"
            "- theme_tags（主题标签数组）\n"
            "- departments（相关部门/机构数组）\n"
            "返回 JSON 数组，每个元素必须包含 idx/content_type/industry_tags/provinces/cities/theme_tags/departments。\n"
            f"输入：{json.dumps(payload_items, ensure_ascii=False)}"
        )
        req: dict[str, Any] = {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=req)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _extract_json_from_text(content)
        if not isinstance(parsed, list):
            continue
        for row in parsed:
            if not isinstance(row, dict):
                continue
            idx = str(row.get("idx") or "").strip()
            if not idx:
                continue
            content_type = str(
                row.get("content_type") or row.get("news_type") or "新闻动态"
            ).strip()
            departments_any = row.get("departments")
            departments_list: list[Any] = departments_any if isinstance(departments_any, list) else []
            industry_any = row.get("industry_tags")
            industry_list: list[Any] = industry_any if isinstance(industry_any, list) else []
            provinces_any = row.get("provinces")
            provinces_list: list[Any] = provinces_any if isinstance(provinces_any, list) else []
            cities_any = row.get("cities")
            cities_list: list[Any] = cities_any if isinstance(cities_any, list) else []
            theme_any = row.get("theme_tags") or row.get("tags")
            theme_list: list[Any] = theme_any if isinstance(theme_any, list) else []
            out[idx] = {
                "content_type": content_type,
                "departments": [str(x).strip() for x in departments_list if str(x).strip()][:10],
                "industry_tags": [str(x).strip() for x in industry_list if str(x).strip()][:10],
                "provinces": [str(x).strip() for x in provinces_list if str(x).strip()][:10],
                "cities": [str(x).strip() for x in cities_list if str(x).strip()][:10],
                "theme_tags": [str(x).strip() for x in theme_list if str(x).strip()][:10],
            }

    return out


def _upsert_node(nodes: dict[str, GraphNode], *, node_id: str, name: str, type_: str, doc_id: str) -> None:
    if node_id in nodes:
        return
    nodes[node_id] = GraphNode(id=node_id, name=name, type=type_, doc_id=doc_id)


def _upsert_edge(
    edges: dict[str, GraphEdge],
    *,
    edge_id: str,
    source_id: str,
    target_id: str,
    type_: str,
    label: str,
    doc_id: str,
    evidence: str | None,
) -> None:
    if edge_id in edges:
        return
    edges[edge_id] = GraphEdge(
        id=edge_id,
        source_id=source_id,
        target_id=target_id,
        type=type_,
        label=label,
        doc_id=doc_id,
        evidence=evidence,
    )


async def _run_extract_job(*, job_id: str, doc_id: str, schema_name: str | None, app: FastAPI) -> None:
    store: SqliteStore = app.state.store
    neo4j: Neo4jClient | None = app.state.neo4j

    doc = store.get_doc(doc_id)
    if doc is None:
        store.update_job(job_id, status="failed", error="doc_not_found")
        logger.error("extract_job_failed job_id=%s doc_id=%s reason=doc_not_found", job_id, doc_id)
        return

    store.update_job(job_id, status="running", error=None)
    logger.warning(
        "extract_job_started job_id=%s doc_id=%s title=%s text_len=%s schema_name=%s",
        job_id,
        doc_id,
        doc.title,
        len(doc.text),
        schema_name or "",
    )
    await asyncio.sleep(0)

    ragflow_url = settings.ragflow_base_url.strip()
    oneke_url = settings.oneke_base_url.strip()
    if settings.require_real_ragflow:
        if not ragflow_url:
            store.update_job(job_id, status="failed", error="RAGFLOW_BASE_URL 未配置，请填写真实 RAGFlow 服务地址")
            logger.error("extract_job_failed job_id=%s doc_id=%s reason=ragflow_base_url_missing", job_id, doc_id)
            return
        if not settings.ragflow_api_key.strip():
            store.update_job(job_id, status="failed", error="RAGFLOW_API_KEY 未配置，请填写真实 RAGFlow API Key")
            logger.error("extract_job_failed job_id=%s doc_id=%s reason=ragflow_api_key_missing", job_id, doc_id)
            return
        if ragflow_url.startswith("http://localhost:8000/api/v1/ragflow"):
            store.update_job(job_id, status="failed", error="RAGFLOW_BASE_URL 指向本地 mock，请改为真实 RAGFlow 服务地址")
            logger.error("extract_job_failed job_id=%s doc_id=%s reason=ragflow_base_url_mock", job_id, doc_id)
            return
    if settings.require_real_oneke:
        if not oneke_url:
            store.update_job(job_id, status="failed", error="ONEKE_BASE_URL 未配置，请填写真实 OneKE 服务地址")
            logger.error("extract_job_failed job_id=%s doc_id=%s reason=oneke_base_url_missing", job_id, doc_id)
            return
        if oneke_url.startswith("http://localhost:8000/api/v1/oneke"):
            store.update_job(job_id, status="failed", error="ONEKE_BASE_URL 指向本地 mock，请改为真实 OneKE 服务地址")
            logger.error("extract_job_failed job_id=%s doc_id=%s reason=oneke_base_url_mock", job_id, doc_id)
            return

    ragflow = RAGFlowClient(
        settings.ragflow_base_url,
        api_key=settings.ragflow_api_key,
        dataset_name=settings.ragflow_dataset_name,
    )
    try:
        await ragflow.ingest_text(doc_id=doc.doc_id, title=doc.title, text=doc.text)
        logger.warning("ragflow_ingest_done job_id=%s doc_id=%s", job_id, doc_id)
    except Exception as e:
        if settings.require_real_ragflow and settings.ragflow_base_url.strip():
            store.update_job(job_id, status="failed", error=f"ragflow_ingest_failed: {repr(e)}")
            logger.exception("ragflow_ingest_failed job_id=%s doc_id=%s", job_id, doc_id)
            return
        logger.exception("ragflow_ingest_failed job_id=%s doc_id=%s", job_id, doc_id)

    oneke = OneKEClient(
        settings.oneke_base_url,
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )
    bundle_items = _parse_moe_news_bundle(doc.text)
    logger.warning(
        "document_parsed job_id=%s doc_id=%s bundle_items=%s mode=%s",
        job_id,
        doc_id,
        len(bundle_items),
        settings.classification_mode,
    )

    if neo4j is None:
        store.update_job(job_id, status="finished", error=None)
        logger.warning("extract_job_finished job_id=%s doc_id=%s neo4j=disabled", job_id, doc_id)
        return

    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}
    name_to_id: dict[str, str] = {}

    async def upsert_extraction_into_graph(*, extracted: ExtractionResult) -> None:
        for ent in extracted.entities:
            nid = stable_id(doc_id, ent.type, ent.name)
            name_to_id.setdefault(ent.name, nid)
            _upsert_node(nodes, node_id=nid, name=ent.name, type_=ent.type, doc_id=doc_id)

        for rel in extracted.relations:
            sid = name_to_id.get(rel.source) or stable_id(doc_id, "Entity", rel.source)
            tid = name_to_id.get(rel.target) or stable_id(doc_id, "Entity", rel.target)

            _upsert_node(nodes, node_id=sid, name=rel.source, type_="Entity", doc_id=doc_id)
            _upsert_node(nodes, node_id=tid, name=rel.target, type_="Entity", doc_id=doc_id)

            eid = stable_id(doc_id, rel.type, sid, tid, rel.evidence or "")
            _upsert_edge(
                edges,
                edge_id=eid,
                source_id=sid,
                target_id=tid,
                type_=rel.type,
                label=rel.type,
                doc_id=doc_id,
                evidence=rel.evidence,
            )

    async def extract_or_fail(*, text: str) -> ExtractionResult:
        try:
            return await oneke.extract(text=text, schema_name=schema_name)
        except Exception as e:
            raise RuntimeError(repr(e)) from e

    try:
        await neo4j.delete_graph_by_doc_id(doc_id=doc_id)
        logger.warning("neo4j_graph_deleted job_id=%s doc_id=%s", job_id, doc_id)
        if bundle_items:
            semantic_by_idx: dict[str, dict[str, Any]] = {}
            if settings.classification_mode.lower() == "llm":
                semantic_by_idx = await _llm_semantic_meta_by_idx(bundle_items)
                if not semantic_by_idx:
                    raise RuntimeError("LLM 分类为空：请检查 OPENAI 配置或分类提示词")
                logger.warning(
                    "llm_classification_done job_id=%s doc_id=%s classified_items=%s",
                    job_id,
                    doc_id,
                    len(semantic_by_idx),
                )
            news_item_ids: list[str] = []
            idx_to_news_id: dict[str, str] = {}
            entity_to_news: dict[str, set[str]] = {}
            news_to_entities: dict[str, set[str]] = {}
            news_to_departments: dict[str, set[str]] = {}
            news_to_theme_tags: dict[str, set[str]] = {}
            news_to_industry_tags: dict[str, set[str]] = {}
            news_to_provinces: dict[str, set[str]] = {}
            news_to_cities: dict[str, set[str]] = {}
            news_to_content_type: dict[str, str] = {}
            news_to_label_keys: dict[str, set[str]] = {}
            label_key_to_news: dict[str, set[str]] = {}
            label_key_to_node_id: dict[str, str] = {}

            total_items = len(bundle_items)
            for item_idx, item in enumerate(bundle_items, start=1):
                idx = item.get("idx", "")
                title = item.get("title", "").strip()
                site = item.get("所属网站", "").strip()
                top_category = item.get("一级分类", "").strip()
                sub_category = item.get("子栏目", "").strip()
                url = item.get("网址", "").strip()
                date = item.get("日期", "").strip()
                summary = item.get("摘要", "").strip()

                news_name = f"【{idx}】{title}" if idx and title else (title or idx or "news")
                news_id = stable_id(doc_id, "NewsItem", idx, title, url)
                news_item_ids.append(news_id)
                if idx:
                    idx_to_news_id[idx] = news_id
                _upsert_node(nodes, node_id=news_id, name=news_name, type_="NewsItem", doc_id=doc_id)

                if site:
                    site_id = stable_id(doc_id, "Organization", site)
                    _upsert_node(nodes, node_id=site_id, name=site, type_="Organization", doc_id=doc_id)
                    _upsert_edge(
                        edges,
                        edge_id=stable_id(doc_id, "来自", news_id, site_id),
                        source_id=news_id,
                        target_id=site_id,
                        type_="来自",
                        label="来自",
                        doc_id=doc_id,
                        evidence=None,
                    )
                if top_category:
                    top_cat_id = stable_id(doc_id, "Category", top_category)
                    _upsert_node(nodes, node_id=top_cat_id, name=top_category, type_="Category", doc_id=doc_id)
                    _upsert_edge(
                        edges,
                        edge_id=stable_id(doc_id, "一级分类", news_id, top_cat_id),
                        source_id=news_id,
                        target_id=top_cat_id,
                        type_="一级分类",
                        label="一级分类",
                        doc_id=doc_id,
                        evidence=None,
                    )
                if sub_category:
                    sub_cat_id = stable_id(doc_id, "SubCategory", sub_category)
                    _upsert_node(nodes, node_id=sub_cat_id, name=sub_category, type_="SubCategory", doc_id=doc_id)
                    _upsert_edge(
                        edges,
                        edge_id=stable_id(doc_id, "子栏目", news_id, sub_cat_id),
                        source_id=news_id,
                        target_id=sub_cat_id,
                        type_="子栏目",
                        label="子栏目",
                        doc_id=doc_id,
                        evidence=None,
                    )
                if url:
                    url_id = stable_id(doc_id, "URL", url)
                    _upsert_node(nodes, node_id=url_id, name=url, type_="URL", doc_id=doc_id)
                    _upsert_edge(
                        edges,
                        edge_id=stable_id(doc_id, "网址", news_id, url_id),
                        source_id=news_id,
                        target_id=url_id,
                        type_="网址",
                        label="网址",
                        doc_id=doc_id,
                        evidence=None,
                    )
                if date:
                    date_id = stable_id(doc_id, "Time", date)
                    _upsert_node(nodes, node_id=date_id, name=date, type_="Time", doc_id=doc_id)
                    _upsert_edge(
                        edges,
                        edge_id=stable_id(doc_id, "日期", news_id, date_id),
                        source_id=news_id,
                        target_id=date_id,
                        type_="日期",
                        label="日期",
                        doc_id=doc_id,
                        evidence=None,
                    )

                if settings.classification_mode.lower() == "llm":
                    meta = semantic_by_idx.get(idx)
                    if not meta:
                        raise RuntimeError(f"LLM 分类缺失：idx={idx}")
                else:
                    meta = {
                        "content_type": _guess_news_type(title=title, summary=summary),
                        "departments": _guess_departments(site=site, title=title, summary=summary),
                        "industry_tags": _extract_industry_tags(title=title, summary=summary),
                        "provinces": _extract_province_tags(title=title, summary=summary),
                        "cities": _extract_city_tags(title=title, summary=summary),
                        "theme_tags": _extract_theme_tags(title=title, summary=summary),
                    }
                content_type = str(
                    meta.get("content_type") or meta.get("news_type") or "新闻动态"
                ).strip() or "新闻动态"
                departments = [str(x).strip() for x in (meta.get("departments") or []) if str(x).strip()]
                industry_tags = [
                    str(x).strip() for x in (meta.get("industry_tags") or []) if str(x).strip()
                ]
                provinces = [str(x).strip() for x in (meta.get("provinces") or []) if str(x).strip()]
                cities = [str(x).strip() for x in (meta.get("cities") or []) if str(x).strip()]
                theme_tags = [str(x).strip() for x in (meta.get("theme_tags") or []) if str(x).strip()]
                if not industry_tags and settings.classification_mode.lower() != "llm":
                    industry_tags = _extract_industry_tags(title=title, summary=summary)
                if not provinces and settings.classification_mode.lower() != "llm":
                    provinces = _extract_province_tags(title=title, summary=summary)
                if not cities and settings.classification_mode.lower() != "llm":
                    cities = _extract_city_tags(title=title, summary=summary)
                if not theme_tags and settings.classification_mode.lower() != "llm":
                    theme_tags = _extract_theme_tags(title=title, summary=summary)

                label_keys: set[str] = set()

                def add_label_link(*, dim: str, value: str, node_type: str, edge_type: str) -> None:
                    v = value.strip()
                    if not v:
                        return
                    k = f"{dim}:{v}"
                    label_keys.add(k)
                    nid = stable_id(doc_id, node_type, v)
                    label_key_to_node_id[k] = nid
                    label_key_to_news.setdefault(k, set()).add(news_id)
                    _upsert_node(nodes, node_id=nid, name=v, type_=node_type, doc_id=doc_id)
                    _upsert_edge(
                        edges,
                        edge_id=stable_id(doc_id, edge_type, news_id, nid),
                        source_id=news_id,
                        target_id=nid,
                        type_=edge_type,
                        label=edge_type,
                        doc_id=doc_id,
                        evidence=None,
                    )

                type_id = stable_id(doc_id, "ContentType", content_type)
                _upsert_node(nodes, node_id=type_id, name=content_type, type_="ContentType", doc_id=doc_id)
                _upsert_edge(
                    edges,
                    edge_id=stable_id(doc_id, "内容类型", news_id, type_id),
                    source_id=news_id,
                    target_id=type_id,
                    type_="内容类型",
                    label="内容类型",
                    doc_id=doc_id,
                    evidence=None,
                )
                content_key = f"内容类型:{content_type}"
                label_keys.add(content_key)
                label_key_to_node_id[content_key] = type_id
                label_key_to_news.setdefault(content_key, set()).add(news_id)

                dept_set: set[str] = set()
                for dep in departments[:10]:
                    dept_set.add(dep)
                    dep_id = stable_id(doc_id, "Organization", dep)
                    _upsert_node(nodes, node_id=dep_id, name=dep, type_="Organization", doc_id=doc_id)
                    _upsert_edge(
                        edges,
                        edge_id=stable_id(doc_id, "相关部门", news_id, dep_id),
                        source_id=news_id,
                        target_id=dep_id,
                        type_="相关部门",
                        label="相关部门",
                        doc_id=doc_id,
                        evidence=None,
                    )
                news_to_departments[news_id] = dept_set
                news_to_content_type[news_id] = content_type

                industry_set: set[str] = set()
                for tag in _unique_keep_order(industry_tags, limit=10):
                    industry_set.add(tag)
                    add_label_link(dim="行业", value=tag, node_type="IndustryTag", edge_type="行业标签")
                news_to_industry_tags[news_id] = industry_set

                province_set: set[str] = set()
                for tag in _unique_keep_order(provinces, limit=10):
                    province_set.add(tag)
                    add_label_link(dim="省份", value=tag, node_type="ProvinceTag", edge_type="省份标签")
                news_to_provinces[news_id] = province_set

                city_set: set[str] = set()
                for tag in _unique_keep_order(cities, limit=10):
                    city_set.add(tag)
                    add_label_link(dim="地市", value=tag, node_type="CityTag", edge_type="地市标签")
                news_to_cities[news_id] = city_set

                theme_set: set[str] = set()
                for tag in _unique_keep_order(theme_tags, limit=10):
                    theme_set.add(tag)
                    add_label_link(dim="主题", value=tag, node_type="ThemeTag", edge_type="主题标签")
                news_to_theme_tags[news_id] = theme_set
                news_to_label_keys[news_id] = label_keys

                if item_idx == 1:
                    await neo4j.upsert_graph(nodes=list(nodes.values()), edges=list(edges.values()))
                    logger.warning(
                        "neo4j_seed_upsert job_id=%s doc_id=%s nodes=%s edges=%s",
                        job_id,
                        doc_id,
                        len(nodes),
                        len(edges),
                    )

                if summary:
                    extracted = await extract_or_fail(text=summary)
                    await upsert_extraction_into_graph(extracted=extracted)

                    ent_set: set[str] = set()
                    for ent in extracted.entities:
                        ent_id = name_to_id.get(ent.name) or stable_id(doc_id, ent.type, ent.name)
                        ent_set.add(ent_id)
                        entity_to_news.setdefault(ent_id, set()).add(news_id)
                        _upsert_edge(
                            edges,
                            edge_id=stable_id(doc_id, "提及", news_id, ent_id),
                            source_id=news_id,
                            target_id=ent_id,
                            type_="提及",
                            label="提及",
                            doc_id=doc_id,
                            evidence=None,
                        )
                    news_to_entities[news_id] = ent_set

                if idx and "完全相同" in summary:
                    m = re.search(r"【(?P<ref>\d+)】内容完全相同", summary)
                    if m:
                        ref_idx = m.group("ref")
                        ref_id = idx_to_news_id.get(ref_idx)
                        if not ref_id:
                            continue
                        _upsert_edge(
                            edges,
                            edge_id=stable_id(doc_id, "重复", news_id, ref_id),
                            source_id=news_id,
                            target_id=ref_id,
                            type_="重复",
                            label="重复",
                            doc_id=doc_id,
                            evidence=summary[:2000],
                        )

                if item_idx % 8 == 0 or item_idx == total_items:
                    await neo4j.upsert_graph(nodes=list(nodes.values()), edges=list(edges.values()))
                    logger.warning(
                        "neo4j_progress_upsert job_id=%s doc_id=%s processed=%s/%s nodes=%s edges=%s",
                        job_id,
                        doc_id,
                        item_idx,
                        total_items,
                        len(nodes),
                        len(edges),
                    )

            for a, b in zip(news_item_ids, news_item_ids[1:], strict=False):
                _upsert_edge(
                    edges,
                    edge_id=stable_id(doc_id, "下一条", a, b),
                    source_id=a,
                    target_id=b,
                    type_="下一条",
                    label="下一条",
                    doc_id=doc_id,
                    evidence=None,
                )

            for ent_id, news_ids in entity_to_news.items():
                if len(news_ids) < 2:
                    continue
                ordered = [nid for nid in news_item_ids if nid in news_ids]
                for a, b in zip(ordered, ordered[1:], strict=False):
                    ent_node = nodes.get(ent_id)
                    _upsert_edge(
                        edges,
                        edge_id=stable_id(doc_id, "关联(共现)", a, b, ent_id),
                        source_id=a,
                        target_id=b,
                        type_="关联(共现)",
                        label="关联(共现)",
                        doc_id=doc_id,
                        evidence=ent_node.name if ent_node else None,
                    )

            label_key_list = list(label_key_to_news.keys())
            for i in range(len(label_key_list)):
                k1 = label_key_list[i]
                n1 = label_key_to_news.get(k1, set())
                if len(n1) < 2:
                    continue
                nid1 = label_key_to_node_id.get(k1)
                if not nid1:
                    continue
                for j in range(i + 1, len(label_key_list)):
                    k2 = label_key_list[j]
                    n2 = label_key_to_news.get(k2, set())
                    if len(n2) < 2:
                        continue
                    nid2 = label_key_to_node_id.get(k2)
                    if not nid2:
                        continue
                    overlap = n1 & n2
                    if len(overlap) < 2:
                        continue
                    score = len(overlap) / max(1, min(len(n1), len(n2)))
                    if score < 0.34:
                        continue
                    _upsert_edge(
                        edges,
                        edge_id=stable_id(doc_id, "标签相似", nid1, nid2, f"{score:.2f}"),
                        source_id=nid1,
                        target_id=nid2,
                        type_="标签相似",
                        label=f"标签相似{score:.2f}",
                        doc_id=doc_id,
                        evidence=f"共现摘要数={len(overlap)}",
                    )

            degrees: dict[str, int] = {nid: 0 for nid in news_item_ids}
            candidates: list[tuple[float, str, str, str]] = []
            for i in range(len(news_item_ids)):
                a = news_item_ids[i]
                theme_a = news_to_theme_tags.get(a, set())
                ind_a = news_to_industry_tags.get(a, set())
                p_a = news_to_provinces.get(a, set())
                c_a = news_to_cities.get(a, set())
                labels_a = news_to_label_keys.get(a, set())
                deps_a = news_to_departments.get(a, set())
                type_a = news_to_content_type.get(a, "")
                ent_a = news_to_entities.get(a, set())
                for j in range(i + 1, len(news_item_ids)):
                    b = news_item_ids[j]
                    theme_b = news_to_theme_tags.get(b, set())
                    ind_b = news_to_industry_tags.get(b, set())
                    p_b = news_to_provinces.get(b, set())
                    c_b = news_to_cities.get(b, set())
                    labels_b = news_to_label_keys.get(b, set())
                    deps_b = news_to_departments.get(b, set())
                    type_b = news_to_content_type.get(b, "")
                    ent_b = news_to_entities.get(b, set())

                    label_overlap = labels_a & labels_b
                    theme_overlap = theme_a & theme_b
                    ind_overlap = ind_a & ind_b
                    p_overlap = p_a & p_b
                    c_overlap = c_a & c_b
                    dep_overlap = deps_a & deps_b
                    ent_overlap = ent_a & ent_b

                    score = 0.0
                    if labels_a and labels_b:
                        score += 0.30 * (len(label_overlap) / max(1, len(labels_a | labels_b)))
                    if theme_a and theme_b:
                        score += 0.35 * (len(theme_overlap) / max(1, len(theme_a | theme_b)))
                    if ind_a and ind_b:
                        score += 0.15 * (len(ind_overlap) / max(1, len(ind_a | ind_b)))
                    if p_a and p_b:
                        score += 0.10 * (len(p_overlap) / max(1, len(p_a | p_b)))
                    if c_a and c_b:
                        score += 0.08 * (len(c_overlap) / max(1, len(c_a | c_b)))
                    if type_a and type_a == type_b:
                        score += 0.08
                    if dep_overlap:
                        score += 0.05
                    if ent_overlap:
                        score += min(0.18, 0.04 * len(ent_overlap))
                    if score < 0.23:
                        continue

                    evidence_parts: list[str] = []
                    if theme_overlap:
                        evidence_parts.append("主题=" + "、".join(sorted(theme_overlap)[:4]))
                    if ind_overlap:
                        evidence_parts.append("行业=" + "、".join(sorted(ind_overlap)[:3]))
                    if p_overlap:
                        evidence_parts.append("省份=" + "、".join(sorted(p_overlap)[:2]))
                    if c_overlap:
                        evidence_parts.append("地市=" + "、".join(sorted(c_overlap)[:2]))
                    if dep_overlap:
                        evidence_parts.append("部门=" + "、".join(sorted(dep_overlap)[:2]))
                    if ent_overlap:
                        ent_names = [nodes[eid].name for eid in sorted(ent_overlap)[:3] if eid in nodes]
                        if ent_names:
                            evidence_parts.append("实体=" + "、".join(ent_names))
                    evidence = " | ".join(evidence_parts) if evidence_parts else "多维标签相似"
                    candidates.append((score, a, b, evidence))

            candidates.sort(key=lambda x: x[0], reverse=True)
            for score, a, b, evidence in candidates:
                if degrees.get(a, 0) >= 6 or degrees.get(b, 0) >= 6:
                    continue
                degrees[a] = degrees.get(a, 0) + 1
                degrees[b] = degrees.get(b, 0) + 1
                source_id, target_id = (a, b) if a < b else (b, a)
                label = f"相似{score:.2f}"
                _upsert_edge(
                    edges,
                    edge_id=stable_id(doc_id, "摘要相似推荐", source_id, target_id, label, evidence),
                    source_id=source_id,
                    target_id=target_id,
                    type_="摘要相似推荐",
                    label=label,
                    doc_id=doc_id,
                    evidence=evidence,
                )
        else:
            extracted = await extract_or_fail(text=doc.text)
            await upsert_extraction_into_graph(extracted=extracted)
    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e))
        logger.exception("extract_job_failed job_id=%s doc_id=%s", job_id, doc_id)
        return

    try:
        await neo4j.upsert_graph(nodes=list(nodes.values()), edges=list(edges.values()))
        logger.warning(
            "neo4j_final_upsert_done job_id=%s doc_id=%s nodes=%s edges=%s",
            job_id,
            doc_id,
            len(nodes),
            len(edges),
        )
    except Exception as e:
        store.update_job(job_id, status="failed", error=str(e))
        logger.exception("neo4j_final_upsert_failed job_id=%s doc_id=%s", job_id, doc_id)
        return

    store.update_job(job_id, status="finished", error=None)
    logger.warning("extract_job_finished job_id=%s doc_id=%s nodes=%s edges=%s", job_id, doc_id, len(nodes), len(edges))


async def _run_extract_job_gated(*, job_id: str, doc_id: str, schema_name: str | None, app: FastAPI) -> None:
    sem = getattr(app.state, "extract_semaphore", None)
    if sem is None:
        await _run_extract_job(job_id=job_id, doc_id=doc_id, schema_name=schema_name, app=app)
        return
    async with sem:
        await _run_extract_job(job_id=job_id, doc_id=doc_id, schema_name=schema_name, app=app)


@router.post("/extract", response_model=StartExtractResponse)
async def start_extract(payload: StartExtractRequest, background: BackgroundTasks, request: Request) -> StartExtractResponse:
    store: SqliteStore = request.app.state.store
    doc = store.get_doc(payload.doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="doc_not_found")

    job_id = uuid4().hex
    store.create_job(job_id=job_id, doc_id=payload.doc_id)
    asyncio.create_task(
        _run_extract_job_gated(
            job_id=job_id,
            doc_id=payload.doc_id,
            schema_name=payload.schema_name,
            app=request.app,
        )
    )
    return StartExtractResponse(job_id=job_id)


@router.post("/extract/upload", response_model=UploadExtractResponse)
async def upload_and_extract(
    background: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    title: str = Form("untitled"),
    schema_name: str | None = Form(None),
    replace: bool = Form(True),
) -> UploadExtractResponse:
    store: SqliteStore = request.app.state.store

    text = await _read_upload_as_text(file)
    if not text:
        raise HTTPException(status_code=400, detail="empty_text")

    source_key = (file.filename or title).strip() or "untitled"
    doc_id = stable_id("doc", source_key) if replace else uuid4().hex
    store.upsert_doc(doc_id=doc_id, title=title, text=text)

    job_id = uuid4().hex
    store.create_job(job_id=job_id, doc_id=doc_id)
    asyncio.create_task(
        _run_extract_job_gated(
            job_id=job_id,
            doc_id=doc_id,
            schema_name=schema_name,
            app=request.app,
        )
    )

    return UploadExtractResponse(doc_id=doc_id, job_id=job_id)


@router.post("/extract/preview-split", response_model=SplitPreviewResponse)
async def preview_split(
    file: UploadFile = File(...),
    limit: int = Form(100),
) -> SplitPreviewResponse:
    text = await _read_upload_as_text(file)
    if not text:
        raise HTTPException(status_code=400, detail="empty_text")
    items = _parse_moe_news_bundle(text)
    if not items:
        return SplitPreviewResponse(total_items=0, categories={}, items=[])
    counts: dict[str, int] = {}
    for it in items:
        c = (it.get("一级分类") or "").strip() or "未分类"
        counts[c] = counts.get(c, 0) + 1
    safe_limit = max(1, min(limit, 500))
    preview_items = [_preview_item_from_bundle(it) for it in items[:safe_limit]]
    return SplitPreviewResponse(total_items=len(items), categories=counts, items=preview_items)


@router.post("/extract/preview-structured", response_model=StructuredExportResponse)
async def preview_structured(
    file: UploadFile = File(...),
    limit: int = Form(100),
    include_cypher: bool = Form(False),
) -> StructuredExportResponse:
    text = await _read_upload_as_text(file)
    if not text:
        raise HTTPException(status_code=400, detail="empty_text")
    items = _parse_moe_news_bundle(text)
    if not items:
        return StructuredExportResponse(total_items=0, categories={}, items=[], cypher=None)
    counts: dict[str, int] = {}
    for it in items:
        c = (it.get("一级分类") or "").strip() or "未分类"
        counts[c] = counts.get(c, 0) + 1
    safe_limit = max(1, min(limit, 500))
    structured = [_structured_item_from_bundle(it) for it in items[:safe_limit]]
    cypher = _build_import_cypher([x.model_dump() for x in structured]) if include_cypher else None
    return StructuredExportResponse(total_items=len(items), categories=counts, items=structured, cypher=cypher)


@router.post("/extract/upload-batch", response_model=UploadBatchExtractResponse)
async def upload_and_extract_batch(
    background: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    title: str = Form("untitled"),
    schema_name: str | None = Form(None),
    batch_size: int = Form(20),
    replace: bool = Form(True),
) -> UploadBatchExtractResponse:
    store: SqliteStore = request.app.state.store

    text = await _read_upload_as_text(file)
    if not text:
        raise HTTPException(status_code=400, detail="empty_text")

    items = _parse_moe_news_bundle(text)
    if not items:
        doc_id = stable_id("doc", (file.filename or title).strip() or "untitled") if replace else uuid4().hex
        store.upsert_doc(doc_id=doc_id, title=title, text=text)
        job_id = uuid4().hex
        store.create_job(job_id=job_id, doc_id=doc_id)
        asyncio.create_task(
            _run_extract_job_gated(
                job_id=job_id,
                doc_id=doc_id,
                schema_name=schema_name,
                app=request.app,
            )
        )
        return UploadBatchExtractResponse(
            total_items=1,
            batch_size=1,
            total_batches=1,
            batches=[BatchJobItem(batch_no=1, doc_id=doc_id, job_id=job_id, item_count=1)],
        )

    safe_batch_size = max(1, min(batch_size, 200))
    batches: list[BatchJobItem] = []
    source_key = (file.filename or title).strip() or "untitled"
    for i in range(0, len(items), safe_batch_size):
        part = items[i : i + safe_batch_size]
        batch_no = (i // safe_batch_size) + 1
        doc_text = _build_news_batch_text(part)
        base_doc = stable_id("doc-batch", source_key, str(batch_no), str(safe_batch_size))
        doc_id = base_doc if replace else uuid4().hex
        batch_title = f"{title}-batch-{batch_no}"
        store.upsert_doc(doc_id=doc_id, title=batch_title, text=doc_text)
        job_id = uuid4().hex
        store.create_job(job_id=job_id, doc_id=doc_id)
        asyncio.create_task(
            _run_extract_job_gated(
                job_id=job_id,
                doc_id=doc_id,
                schema_name=schema_name,
                app=request.app,
            )
        )
        batches.append(BatchJobItem(batch_no=batch_no, doc_id=doc_id, job_id=job_id, item_count=len(part)))

    return UploadBatchExtractResponse(
        total_items=len(items),
        batch_size=safe_batch_size,
        total_batches=len(batches),
        batches=batches,
    )


@router.get("/extract/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, request: Request) -> JobStatusResponse:
    store: SqliteStore = request.app.state.store
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    return JobStatusResponse(job_id=job.job_id, doc_id=job.doc_id, status=job.status, error=job.error)


@router.post("/oneke/extract")
async def local_oneke_extract(payload: LocalOneKEExtractRequest) -> dict[str, object]:
    primary = OneKEClient(
        "",
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
    )
    try:
        extracted = await primary.extract(text=payload.text, schema_name=payload.schema_name)
    except Exception:
        fallback = OneKEClient("")
        extracted = await fallback.extract(text=payload.text, schema_name=payload.schema_name)
    return {
        "entities": [e.model_dump() for e in extracted.entities],
        "relations": [r.model_dump() for r in extracted.relations],
    }
