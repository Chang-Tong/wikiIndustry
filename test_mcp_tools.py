#!/usr/bin/env python3
"""Test MCP tools directly with detailed terminal output."""

import asyncio
import json
import sys
from pathlib import Path

# Add both backend and agent to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "services" / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "services" / "agent"))

from agent_mcp.tools import (
    ingest_documents,
    query_graph,
    configure_extraction_schema,
    list_schemas,
    get_schema,
)


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_json(label: str, data: dict) -> None:
    print(f"\n📦 {label}:")
    print(json.dumps(data, ensure_ascii=False, indent=2))


async def test_list_schemas() -> None:
    print_header("TEST 1: list_schemas")
    print("调用 list_schemas() - 列出所有提取模式")
    result = await list_schemas({})
    for item in result:
        data = json.loads(item.text)
        print_json("list_schemas 输出", data)


async def test_get_schema() -> None:
    print_header("TEST 2: get_schema")
    print("调用 get_schema(schema_name='MOE_News') - 获取默认模式")
    result = await get_schema({"schema_name": "MOE_News"})
    for item in result:
        data = json.loads(item.text)
        print_json("get_schema 输出", data)


async def test_configure_extraction_schema() -> None:
    print_header("TEST 3: configure_extraction_schema")
    print("调用 configure_extraction_schema() - 创建/更新提取模式")
    result = await configure_extraction_schema({
        "schema_name": "HLJ_Test_Schema",
        "entity_types": ["Organization", "Person", "Policy", "Location"],
        "relation_types": [
            {"subject": "Person", "relation": "发布", "object": "Policy"},
            {"subject": "Organization", "relation": "位于", "object": "Location"},
        ],
        "instruction": "从黑龙江省教育新闻中提取组织和人物信息",
    })
    for item in result:
        data = json.loads(item.text)
        print_json("configure_extraction_schema 输出", data)


async def test_ingest_documents() -> None:
    print_header("TEST 4: ingest_documents")
    print("调用 ingest_documents() - 批量导入 heilongjiang_news.json")

    # Load the heilongjiang news JSON
    hlj_path = PROJECT_ROOT / "heilongjiang_news.json"
    if not hlj_path.exists():
        print(f"❌ 错误: 找不到文件 {hlj_path}")
        return

    with open(hlj_path, "r", encoding="utf-8") as f:
        hlj_data = json.load(f)

    # Convert to MCP document format
    documents = []
    for site_group in hlj_data.get("news", []):
        for item in site_group.get("data", []):
            doc = {
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "site": item.get("site", "黑龙江省教育厅"),
                "channel": item.get("channel", ""),
                "date": item.get("date", ""),
                "tag": item.get("tag", ""),
                "summary": item.get("summary", ""),
                "link": item.get("link", ""),
            }
            if doc["title"] and doc["content"]:
                documents.append(doc)

    print(f"📄 加载了 {len(documents)} 条新闻文档")
    if len(documents) == 0:
        print("❌ 错误: 没有有效的文档")
        return

    # Only ingest first 3 docs to save time
    test_docs = documents[:3]
    print(f"🧪 将导入前 {len(test_docs)} 条进行测试")
    for i, doc in enumerate(test_docs, 1):
        print(f"  [{i}] {doc['title'][:60]}...")

    result = await ingest_documents({
        "documents": test_docs,
        "schema_name": "MOE_News",
        "mode": "incremental",
    })
    for item in result:
        data = json.loads(item.text)
        print_json("ingest_documents 输出", data)


async def test_query_graph() -> None:
    print_header("TEST 5: query_graph")
    print("调用 query_graph() - 图谱问答")

    test_questions = [
        "黑龙江省最近有什么教育新闻？",
        "哈尔滨工业大学有什么动态？",
    ]

    for question in test_questions:
        print(f"\n❓ 问题: {question}")
        result = await query_graph({
            "question": question,
            "top_k": 5,
            "include_raw_sources": True,
        })
        for item in result:
            data = json.loads(item.text)
            if "error" in data:
                print(f"❌ 错误: {data['error']}")
            else:
                print(f"\n💡 回答: {data.get('answer', '无回答')[:200]}...")
                print(f"📊 置信度: {data.get('confidence', 'unknown')}")
                print(f"🔍 推理过程: {data.get('reasoning_process', '无')[:150]}...")
                print(f"📈 查询计划: {len(data.get('query_plan', {}).get('queries', []))} 个 Cypher 查询")
                print_json("完整输出", data)


async def main() -> None:
    print("=" * 60)
    print("  MCP Tools 测试 - 使用 heilongjiang_news.json")
    print("=" * 60)

    # Test non-destructive tools first
    await test_list_schemas()
    await test_get_schema()
    await test_configure_extraction_schema()

    # Test ingestion
    await test_ingest_documents()

    # Test query (need data first)
    await test_query_graph()

    print("\n" + "=" * 60)
    print("  MCP Tools 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
