# WikiProject 技术交接文档

> 最后更新：2026-04-20
> 版本：删除 v2/RAGFlow 后，保留 v1 + CORRELATED_WITH 增强

---

## 一、项目概述

WikiProject 是一个**知识图谱 + RAG 问答系统**，用于处理中文新闻/政策文档：

1. **上传 JSON 新闻** → OneKE 提取实体/关系
2. **构建图谱** → 存入 Neo4j
3. **计算相似度** → Ollama Embedding + entity 重叠 → CORRELATED_WITH 边
4. **问答** → LLM 生成 Cypher 查询图谱 → 自动扩展相似新闻 → 生成回答

### 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | FastAPI (Python 3.11+) |
| 前端 | React 18 + TypeScript + AntV G6 |
| 图数据库 | Neo4j 5 |
| 原始文档存储 | SQLite |
| NLP 提取 | OneKE |
| Embedding | Ollama (`qwen3-embedding:0.6b`) |
| LLM | DeepSeek API (OpenAI-compatible) |
| Agent MCP | `mcp` Python SDK |

---

## 二、系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   前端 UI    │────▶│  FastAPI    │────▶│   OneKE 服务    │
│ (React+G6)  │◄────│  (backend)  │◄────│  (实体/关系提取) │
└─────────────┘     └──────┬──────┘     └─────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐ ┌──────────┐ ┌──────────┐
        │  Neo4j  │ │  SQLite  │ │ DeepSeek │
        │ (图谱)   │ │ (原文档) │ │  (LLM)   │
        └─────────┘ └──────────┘ └──────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐ ┌──────────┐ ┌──────────┐
        │  Ollama │ │ Agent MCP│ │  router  │
        │(Embedding)│ │  Server  │ │(FastAPI) │
        └─────────┘ └──────────┘ └──────────┘
```

---

## 三、函数调用链（数据流详解）

### 3.1 数据上传流程

```
前端上传 JSON
  │ POST /api/v1/json/upload
  ▼
routes_json.py: upload_json_file()
  │ 解析 JSON → NewsItem 列表
  ▼
_process_batch() (后台任务)
  │
  ├──▶ SqliteStore.create_doc()      # 原文档存入 SQLite
  │
  ├──▶ OneKEClient.extract()         # 调用 OneKE 服务提取实体/关系
  │      │
  │      └──▶ HTTP POST → OneKE 服务
  │
  ├──▶ GraphBuilder.build_from_extraction()
  │      │
  │      ├── 创建 NewsItem 节点 (中心节点)
  │      ├── 创建 Entity 节点 (OneKE 提取的实体)
  │      ├── 创建 REL 边 (实体间关系)
  │      └── 创建元数据边 (site, date, tag, province...)
  │
  ├──▶ Neo4jClient.upsert_graph()    # 写入 Neo4j
  │
  └──▶ CorrelationMiningService.create_correlation_edges()
         │
         ├──▶ generate_embeddings()
         │      │
         │      └──▶ EmbeddingService.embed() → Ollama API
         │           (为每条 NewsItem 标题生成 embedding)
         │
         ├──▶ _find_entity_correlations()   # Jaccard 相似度
         ├──▶ _find_vector_correlations()    # 余弦相似度
         ├──▶ _merge_correlations()          # hybrid = 0.6*entity + 0.4*vector
         └──▶ 写入 CORRELATED_WITH 边到 Neo4j
```

### 3.2 问答流程（Graph QA）

```
用户提问
  │ POST /api/v1/qa/ask-graph
  ▼
routes_docs.py: qa_ask_graph()
  │
  └──▶ RAGEngine.answer()              # v1 引擎（唯一引擎）
         │
         ├──▶ _generate_cypher_queries() # LLM 生成 Cypher 查询
         │      │
         │      └──▶ HTTP POST → DeepSeek API
         │           Prompt 包含：图谱 Schema + 查询示例 + CORRELATED_WITH 说明
         │
         ├──▶ _execute_graph_queries()   # 执行 Cypher 查询
         │      │
         │      └──▶ Neo4j session.run()
         │           返回 GraphNode/GraphEdge → RetrievedChunk
         │
         ├──▶ _expand_with_correlations() # 【新增】CORRELATED_WITH 扩展
         │      │
         │      ├── 从 graph_chunks 中提取 NewsItem name
         │      ├── MATCH (n:NewsItem)-[r:CORRELATED_WITH]-(m:NewsItem)
         │      ├── 按 r.score DESC 取 top 3
         │      └── 包装为 correlated_news 类型的 RetrievedChunk
         │           加权分数 = 1.5 + score * 2
         │
         ├──▶ _retrieve_documents()      # SQLite 文档检索（fallback）
         │
         ├──▶ _merge_chunks()            # 合并 + 去重 + 排序
         │
         ├──▶ _generate_answer_with_reasoning()  # LLM 生成回答
         │      │
         │      └──▶ HTTP POST → DeepSeek API
         │           Prompt 包含：检索结果 + 用户问题 + 推理要求
         │
         ├──▶ _verify_answer_sources()   # 防幻觉验证
         │      ├── 提取 [数字] 引用
         │      └── 检查是否在有效范围内
         │
         └──▶ _post_process_answer()     # 添加免责声明
```

### 3.3 相似度分析流程

```
前端点击"构建相似度关联"
  │ POST /api/v1/correlations/build-edges
  ▼
routes_correlation.py: build_correlation_edges()
  │
  └──▶ CorrelationMiningService.create_correlation_edges()
         │
         ├──▶ generate_embeddings()
         │      │
         │      ├── 查询无 embedding 的 NewsItem
         │      ├── EmbeddingService.embed() → Ollama API
         │      └── Neo4jClient.set_news_embedding()
         │
         ├──▶ find_correlations()
         │      │
         │      ├── _find_entity_correlations()   # Jaccard
         │      │      MATCH NewsItem-[:REL]-Entity-[:REL]-NewsItem
         │      │      计算共享实体比例
         │      │
         │      ├── _find_vector_correlations()   # Cosine
         │      │      读取已存 embedding，计算余弦相似度
         │      │
         │      └── _merge_correlations()
         │           hybrid_score = 0.6 * entity + 0.4 * vector
         │
         └──▶ 写入 CORRELATED_WITH 边
              MERGE (n1)-[r:CORRELATED_WITH]-(n2)
              SET r.score, r.entity_score, r.vector_score, r.correlation_type
```

---

## 四、组件配合说明

### 4.1 核心模块职责

| 模块 | 文件路径 | 职责 |
|------|---------|------|
| **RAGEngine** | `app/services/rag_engine.py` | 唯一问答引擎。LLM 生成 Cypher → 执行查询 → 扩展相似新闻 → LLM 生成回答 → 防幻觉验证 |
| **CorrelationMiningService** | `app/services/correlation_mining.py` | 相似度挖掘。Ollama 生成 embedding → entity Jaccard + vector cosine → hybrid 合并 → 写 CORRELATED_WITH 边 |
| **GraphBuilder** | `app/services/graph_builder.py` | 从 OneKE 提取结果构建图谱节点和边。创建 NewsItem 中心节点 + Entity 节点 + REL 边 |
| **Neo4jClient** | `app/integrations/neo4j/client.py` | Neo4j 异步驱动封装。upsert_graph, read_graph, get_schema_stats, set_news_embedding |
| **OneKEClient** | `app/integrations/oneke/client.py` | OneKE HTTP 客户端。发送 text → 返回 ExtractedEntity + ExtractedRelation |
| **EmbeddingService** | `app/services/embedding_service.py` | Ollama Embedding 封装。强制使用 Ollama，失败时报错 |
| **SqliteStore** | `app/store/sqlite.py` | 原始文档存储。create_doc, get_doc, search_docs, list_finished_doc_ids |
| **SchemaRegistry** | `app/services/schema_registry.py` | OneKE Schema 管理。create/get/update/delete 提取模式 |
| **BatchIngestService** | `app/services/batch_ingestor.py` | Agent/MCP 批量导入服务。封装完整的导入流程 |

### 4.2 数据模型

#### Neo4j 图谱 Schema

```
节点标签：Entity（所有节点都是这个标签）
  - type 属性区分类型：NewsItem, Organization, Person, Policy, ThemeTag, ProvinceTag, CityTag, Time, Location, Event, Technology, Entity
  - name: 显示名称
  - doc_id: 关联的文档 ID
  - embedding: 向量（仅 NewsItem 有）
  - embedding_model: 使用的模型名

关系类型：
  - REL: OneKE 提取的关系（白色实线，有向）
  - CORRELATED_WITH: 相似度计算的关系（红色虚线 #FF375F，双向）
    - score: hybrid 相似度分数
    - entity_score: Jaccard 相似度
    - vector_score: 余弦相似度
    - correlation_type: "entity" | "vector" | "hybrid"
```

#### SQLite Schema

```sql
-- docs: 原始文档
CREATE TABLE docs (doc_id TEXT PRIMARY KEY, title TEXT, text TEXT, created_at TIMESTAMP)

-- jobs: 处理任务状态
CREATE TABLE jobs (job_id TEXT PRIMARY KEY, doc_id TEXT, status TEXT, error TEXT, created_at TIMESTAMP)

-- schemas: OneKE 提取模式
CREATE TABLE schemas (schema_id TEXT PRIMARY KEY, schema_name TEXT, entity_types TEXT, relation_types TEXT, instruction TEXT)
```

---

## 五、MCP 接口服务详细用法

### 5.1 什么是 MCP

MCP (Model Context Protocol) 是 Anthropic 推出的开放协议，允许 AI Agent 通过标准化接口调用外部工具。

本项目中的 MCP Server 位于 `services/agent/agent_mcp/`，封装了 WikiProject 的核心能力为 5 个 MCP Tool。

### 5.2 启动 MCP Server

```bash
cd services/agent

# 方式一：直接运行
python -m agent_mcp.server

# 方式二：通过 entry point（pip install 后）
wiki-agent

# 方式三：通过 uv
uv run python -m agent_mcp.server
```

MCP Server 使用 **stdio 传输**（标准输入输出），通过 JSON-RPC 2.0 与 MCP Client 通信。

### 5.3 暴露的 Tools

#### Tool 1: `ingest_documents` — 批量文档导入

**用途：** 将文档批量导入知识图谱（OneKE 提取 → Neo4j 存储 → 自动相似度计算）

**输入参数：**

```json
{
  "documents": [
    {
      "title": "教育部发布新政策",
      "content": "教育部今日发布...",
      "site": "教育部官网",
      "channel": "政策发布",
      "date": "2026-04-20",
      "tag": "教育政策",
      "summary": "摘要",
      "link": "https://..."
    }
  ],
  "schema_name": "MOE_News",
  "mode": "incremental"
}
```

**必填：** `documents`（至少包含 `title` 和 `content`）

**可选：**
- `schema_name`: 提取模式名，默认 `"MOE_News"`
- `mode`: `"incremental"`（追加）或 `"overwrite"`（清空后重建）

**输出：**

```json
{
  "job_id": "uuid",
  "status": "completed",
  "total_items": 10,
  "processed_items": 10,
  "extracted_entities": 45,
  "extracted_relations": 32,
  "errors": []
}
```

**内部流程：**

```
ingest_documents()
  └──▶ BatchIngestService.ingest()
         ├── 每个文档：
         │    ├── SqliteStore.create_doc()     # 存原文
         │    ├── OneKEClient.extract()        # 提取实体/关系
         │    ├── GraphBuilder.build()         # 构建图谱
         │    └── Neo4jClient.upsert_graph()   # 写入 Neo4j
         └── 全部完成后：
              └── CorrelationMiningService.create_correlation_edges()
                   # 自动计算相似度并创建 CORRELATED_WITH 边
```

#### Tool 2: `query_graph` — 图谱问答

**用途：** 基于知识图谱回答问题，返回结构化输出

**输入参数：**

```json
{
  "question": "教育部发布了哪些政策？",
  "top_k": 10,
  "include_raw_sources": true
}
```

**必填：** `question`

**可选：**
- `top_k`: 最大检索结果数，默认 10
- `include_raw_sources`: 是否包含 SQLite 原始文档，默认 true

**输出：**

```json
{
  "question": "教育部发布了哪些政策？",
  "answer": "根据知识库数据，教育部近期发布了...",
  "reasoning_process": "检索到 3 条相关政策...",
  "confidence": "high",
  "schema_snapshot": {
    "labels": ["Entity"],
    "relationship_types": ["REL", "CORRELATED_WITH"],
    "node_samples": {...},
    "type_distribution": {"NewsItem": 71, "Organization": 86, ...}
  },
  "graph_results": [
    {"source": {...}, "relationship": {...}, "target": {...}}
  ],
  "sqlite_sources": [
    {"doc_id": "...", "title": "...", "text_snippet": "...", "relevance_score": 0.85}
  ],
  "query_plan": {
    "thinking": "...",
    "queries": ["MATCH ..."],
    "strategy": "cypher",
    "iterations": 1
  }
}
```

> ⚠️ **注意：** MCP 中的 `query_graph` 目前仍调用 `GraphQAService`，它底层使用的是 `AdaptiveRAGEngine`（v2）。如果需要统一使用 v1，需要修改 `agent_mcp/tools.py` 中的 `query_graph` 函数，将 `GraphQAService` 替换为 `RAGEngine`。

#### Tool 3: `configure_extraction_schema` — 配置提取模式

**用途：** 创建或更新 OneKE 的实体/关系提取模式

**输入参数：**

```json
{
  "schema_name": "Custom_News",
  "entity_types": ["Organization", "Person", "Policy", "Location"],
  "relation_types": [
    {"subject": "Person", "relation": "发布", "object": "Policy"},
    {"subject": "Organization", "relation": "位于", "object": "Location"}
  ],
  "instruction": "从新闻中提取组织和人物信息"
}
```

**必填：** `schema_name`, `entity_types`

**可选：** `relation_types`, `instruction`

**输出：**

```json
{
  "schema_id": "uuid",
  "schema_name": "Custom_News",
  "entity_types": ["Organization", "Person", "Policy", "Location"],
  "relation_types": [{"subject": "Person", "relation": "发布", "object": "Policy"}],
  "instruction": "从新闻中提取组织和人物信息"
}
```

**行为：**
- 如果 `schema_name` 已存在 → **更新**现有模式
- 如果 `schema_name` 不存在 → **创建**新模式

#### Tool 4: `list_schemas` — 列出所有模式

**输入：** 无参数（空对象 `{}`）

**输出：**

```json
{
  "schemas": [
    {"schema_id": "...", "schema_name": "MOE_News", "entity_types": [...], ...},
    {"schema_id": "...", "schema_name": "Custom_News", "entity_types": [...], ...}
  ],
  "total": 2
}
```

#### Tool 5: `get_schema` — 获取特定模式

**输入参数：**

```json
{"schema_name": "MOE_News"}
```

**输出：** 单个 schema 对象，或 `{"error": "Schema 'MOE_News' not found"}`

### 5.4 MCP Server 架构

```
┌─────────────────────────────────────────┐
│           MCP Client (AI Agent)          │
│  (Claude Desktop / Cursor / 自定义 Agent)│
└─────────────────┬───────────────────────┘
                  │ stdio (JSON-RPC 2.0)
                  ▼
┌─────────────────────────────────────────┐
│       agent_mcp/server.py               │
│  - Server("wikiProject-agent")          │
│  - @list_tools() → 返回 5 个 Tool 定义   │
│  - @call_tool() → 路由到 TOOL_HANDLERS   │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────┐   ┌────────────────┐
│  TOOL_HANDLERS│   │  app/ 服务层    │
│  (tools.py)   │──▶│  (与 backend    │
│               │   │   共享代码)     │
└──────────────┘   └────────────────┘
```

### 5.5 接入 Claude Desktop 示例

在 Claude Desktop 配置文件 `claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "wikiproject": {
      "command": "python",
      "args": [
        "-m",
        "agent_mcp.server"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "neo4j_password",
        "ONEKE_BASE_URL": "http://localhost:8010",
        "OPENAI_BASE_URL": "https://api.deepseek.com/v1",
        "OPENAI_API_KEY": "sk-...",
        "OPENAI_MODEL": "deepseek-chat",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
        "OLLAMA_EMBEDDING_MODEL": "qwen3-embedding:0.6b"
      }
    }
  }
}
```

> 注意：需要设置 `REQUIRE_REAL_ONEKE=true` 和 `REQUIRE_OLLAMA_EMBEDDING=true`

---

## 六、关键 API 端点

### 6.1 数据上传

```bash
# 上传 JSON 文件
curl -X POST http://localhost:8000/api/v1/json/upload \
  -F "file=@news_data.json" \
  -F "schema_name=MOE_News" \
  -F "mode=incremental"

# 查看任务状态
curl http://localhost:8000/api/v1/json/jobs/{job_id}
```

### 6.2 图谱问答

```bash
# Graph QA（主流程）
curl -X POST http://localhost:8000/api/v1/qa/ask-graph \
  -H "Content-Type: application/json" \
  -d '{"question": "哈工大有什么新闻？", "top_k": 10}'
```

### 6.3 图谱查询

```bash
# 获取图谱数据
curl "http://localhost:8000/api/v1/graph?node_limit=50"

# 按主题过滤
curl "http://localhost:8000/api/v1/graph/theme/义务教育?news_limit=20"

# 获取主题列表
curl http://localhost:8000/api/v1/graph/themes

# 省份统计
curl http://localhost:8000/api/v1/graph/provinces
```

### 6.4 相似度分析

```bash
# 生成 embedding（为所有无 embedding 的新闻）
curl -X POST "http://localhost:8000/api/v1/correlations/embeddings?batch_size=10"

# 构建相似度边
curl -X POST "http://localhost:8000/api/v1/correlations/build-edges?min_score=0.05&use_vector=true"

# 查询相似度
curl "http://localhost:8000/api/v1/correlations?min_score=0.3&limit=5"

# 相似度矩阵
curl "http://localhost:8000/api/v1/correlations/matrix"
```

---

## 七、部署启动

### 7.1 环境配置

复制 `.env.example` 为 `.env`，填入：

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

# OneKE（强制使用）
ONEKE_BASE_URL=http://localhost:8010
REQUIRE_REAL_ONEKE=true

# LLM (DeepSeek)
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=deepseek-chat

# Ollama Embedding（强制使用）
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
REQUIRE_OLLAMA_EMBEDDING=true

# Frontend
FRONTEND_ORIGIN=http://localhost:5173
```

### 7.2 启动方式

```bash
# 方式一：Docker 全量启动（生产-like）
./start-all.sh

# 方式二：本地开发模式（热重载）
./start-all.sh --dev

# 方式三：手动启动
# 1. 启动 Neo4j + OneKE（Docker）
docker compose up -d neo4j oneke

# 2. 启动后端
cd services/backend
source .venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 3. 启动前端
cd services/frontend
npm run dev

# 4. 启动 MCP Server（可选）
cd services/agent
python -m agent_mcp.server
```

### 7.3 依赖检查

```bash
# Neo4j
curl http://localhost:7474

# OneKE
curl http://localhost:8010/health

# Backend
curl http://localhost:8000/healthz

# Frontend
curl http://localhost:5173
```

---

## 八、常见问题

### Q: 问答时返回 "未找到相关信息"

**排查步骤：**
1. 检查 Neo4j 是否有数据：`curl http://localhost:8000/api/v1/graph`
2. 检查 OneKE 是否成功提取实体
3. 检查 LLM API Key 是否配置正确：`curl http://localhost:8000/api/v1/debug/settings`

### Q: CORRELATED_WITH 边没有生成

**排查步骤：**
1. 检查 Ollama 是否运行：`curl http://localhost:11434`
2. 检查模型是否已下载：`ollama list`
3. 手动调用生成 embedding：`POST /correlations/embeddings`
4. 手动调用建边：`POST /correlations/build-edges`

### Q: MCP Server 无法连接

**排查步骤：**
1. 确保环境变量已设置（NEO4J_URI, OPENAI_API_KEY 等）
2. 确保 `services/agent` 目录在 Python path 中
3. 检查依赖：`pip install mcp>=1.6.0`

---

## 九、文件变更记录

本次交接涉及的主要变更：

| 变更 | 说明 |
|------|------|
| 删除 `rag_engine_v2.py` | AdaptiveRAGEngine（v2 引擎）已移除 |
| 删除 `graph_qa_service.py` | 基于 v2 的结构化问答服务已移除 |
| 删除 `routes_qa_structured.py` | `/qa/query-graph-structured` 路由已移除 |
| 删除 `integrations/ragflow/` | RAGFlow 客户端已移除 |
| 增强 `rag_engine.py` | v1 引擎增加 `_expand_with_correlations()` |
| 清理 `routes_docs.py` | 删除所有 RAGFlow 相关路由 |
| 清理 `routes_extract.py` | 删除 RAGFlow 索引逻辑 |
| 清理 `settings.py` | 删除 RAGFlow 配置项 |
| 新增 `test_rag_correlation_expand.py` | v1 扩展功能单元测试（已删，v2 专用） |

---

*文档结束。如有疑问，请查看 `CLAUDE.md` 和代码注释。*
