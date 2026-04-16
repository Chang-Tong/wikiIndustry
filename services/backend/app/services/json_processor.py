"""JSON news data processor for knowledge graph construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class NewsItem:
    """Single news article extracted from JSON."""

    title: str
    site: str
    channel: str
    date: str
    tag: str
    summary: str
    content: str
    link: str
    source_id: str = ""  # 原始来源标识

    def to_oneke_text(self) -> str:
        """Convert news item to OneKE extraction format."""
        parts = [
            f"【标题】{self.title}",
            f"【来源】{self.site}/{self.channel}",
            f"【日期】{self.date}",
            f"【标签】{self.tag}",
        ]
        if self.summary:
            parts.append(f"【摘要】{self.summary}")
        if self.content:
            parts.append(f"【正文】{self.content[:2000]}...")  # 限制长度
        return "\n".join(parts)

    def generate_id(self) -> str:
        """Generate unique document ID for this news item."""
        content = f"{self.title}:{self.date}:{self.site}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:32]


@dataclass
class ProcessedNewsBatch:
    """Batch of processed news items with metadata."""

    items: list[NewsItem]
    crawl_time: str
    start_date: str
    end_date: str
    total_count: int


class JSONNewsProcessor:
    """Processor for JSON news data format."""

    def __init__(self) -> None:
        """Initialize processor."""
        self.validation_errors: list[str] = []

    def validate(self, data: dict[str, Any]) -> bool:
        """Validate JSON structure."""
        self.validation_errors = []

        if not isinstance(data, dict):
            self.validation_errors.append("Root must be an object")
            return False

        # Check required fields
        if "news" not in data:
            self.validation_errors.append("Missing 'news' field")
            return False

        if not isinstance(data["news"], list):
            self.validation_errors.append("'news' must be an array")
            return False

        # Validate news structure
        for i, site_group in enumerate(data["news"]):
            if not isinstance(site_group, dict):
                self.validation_errors.append(f"news[{i}] must be an object")
                continue

            if "site" not in site_group:
                self.validation_errors.append(f"news[{i}] missing 'site' field")

            if "data" not in site_group or not isinstance(site_group.get("data"), list):
                self.validation_errors.append(f"news[{i}] missing or invalid 'data' field")

        return len(self.validation_errors) == 0

    def process(self, data: dict[str, Any]) -> ProcessedNewsBatch:
        """Process JSON data into news items.

        Args:
            data: Parsed JSON object

        Returns:
            ProcessedNewsBatch with extracted items
        """
        if not self.validate(data):
            raise ValueError(f"Invalid JSON: {'; '.join(self.validation_errors)}")

        items: list[NewsItem] = []

        for site_group in data.get("news", []):
            site_name = site_group.get("site", "")

            for news_data in site_group.get("data", []):
                if not isinstance(news_data, dict):
                    continue

                item = NewsItem(
                    title=news_data.get("title", ""),
                    site=news_data.get("site", site_name),
                    channel=news_data.get("channel", ""),
                    date=news_data.get("date", ""),
                    tag=news_data.get("tag", ""),
                    summary=news_data.get("summary", ""),
                    content=news_data.get("content", ""),
                    link=news_data.get("link", ""),
                    source_id=f"{site_name}_{news_data.get('date', '')}",
                )

                # Skip items without title
                if not item.title:
                    continue

                items.append(item)

        return ProcessedNewsBatch(
            items=items,
            crawl_time=data.get("crawl_time", ""),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            total_count=len(items),
        )

    def process_file(self, file_path: str) -> ProcessedNewsBatch:
        """Process JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            ProcessedNewsBatch with extracted items
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.process(data)

    def process_bytes(self, content: bytes) -> ProcessedNewsBatch:
        """Process JSON bytes.

        Args:
            content: JSON file content as bytes

        Returns:
            ProcessedNewsBatch with extracted items
        """
        data = json.loads(content.decode("utf-8"))
        return self.process(data)


def _extract_province(text: str) -> str | None:
    """Extract province from arbitrary text.

    Examples:
        "黑龙江省教育厅" -> "黑龙江省"
        "北京市教育委员会" -> "北京市"
        "教育部" -> None
    """
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
        "台湾省": ("台湾省", "台湾"),
    }

    for canon, aliases in province_alias.items():
        if any(a and a in text for a in aliases):
            return canon

    return None


def extract_structured_metadata(item: NewsItem) -> dict[str, Any]:
    """Extract structured metadata from news item for graph building.

    Args:
        item: News item

    Returns:
        Dictionary with structured metadata
    """
    # Extract province from multiple fields, not only site.
    merged_text = " ".join(
        [
            item.site or "",
            item.channel or "",
            item.title or "",
            item.summary or "",
            (item.content or "")[:1000],
        ]
    )
    province = _extract_province(merged_text)

    return {
        "title": item.title,
        "site": item.site,
        "channel": item.channel,
        "date": item.date,
        "tag": item.tag,
        "summary": item.summary,
        "url": item.link,
        "doc_id": item.generate_id(),
        "province": province,  # Add extracted province
    }
