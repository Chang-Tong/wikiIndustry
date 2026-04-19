# WikiProject Agent

WikiProject Agent 是将后端能力封装为 AI Agent 可调用的原子能力单元。

## 特点

- **开箱即用**：内置 Neo4j + OneKE 容器编排，无需单独配置基础设施
- **只需改 LLM Key**：.env 中所有服务地址已预设好，仅需填入 `OPENAI_API_KEY`
- **MCP Server**：AI Agent 通过 Model Context Protocol 调用
- **REST API**：FastAPI 服务，可作为常规后端使用

## 快速开始

### 前置条件

- Python 3.10+
- Docker（用于运行 Neo4j 和 OneKE）
- [uv](https://github.com/astral-sh/uv)（推荐，或 pip）

### 1. 初始化

```bash
./scripts/setup.sh
```

此脚本会：
- 检查 Docker 是否安装
- 生成 `.env` 配置文件（从 `.env.example` 复制）
- 检查 uv 是否安装

### 2. 配置 LLM（唯一需要改的地方）

```bash
# 编辑 .env，填入你的 API Key
OPENAI_API_KEY=sk-你的API密钥
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

> 其他配置（Neo4j、OneKE 地址）已预设好，通常不需要修改。

### 3. 启动依赖服务（Neo4j + OneKE）

```bash
./scripts/start-deps.sh
```

此脚本会：
- 启动 Neo4j 容器（端口 7474/7687）
- 启动 OneKE 容器（端口 8010）
- 等待两个服务健康就绪

### 4. 启动 Agent MCP Server

```bash
# 方式一：直接运行
uv run python -m agent_mcp.server

# 方式二：通过入口脚本
uv run wiki-agent
```

### 5. 停止依赖

```bash
./scripts/stop-deps.sh
```

## 原子能力（MCP Tools）

| Tool | 说明 |
|------|------|
| `ingest_documents` | 批量/增量文档上传，自动抽取实体关系并入图谱 |
| `query_graph` | 图谱问答，返回结构化 JSON（schema + graph + sqlite 原文 + LLM 回答） |
| `configure_extraction_schema` | 配置 OneKE 抽取 schema（实体类型 + 关系三元组） |
| `list_schemas` | 列出所有可用 schema |
| `get_schema` | 获取指定 schema 详情 |

## 检查服务状态

```bash
./scripts/check-deps.sh
```

## Docker 方式运行（可选）

```bash
# 构建镜像
docker build -t wiki-agent .

# 运行（需要指定 .env）
docker run --env-file .env wiki-agent
```

## 项目结构

```
.
├── docker-compose.yml       # Neo4j + OneKE 容器编排
├── Dockerfile               # Agent 自身的容器化
├── pyproject.toml           # Python 依赖配置
├── .env.example             # 环境配置模板
├── README.md                # 本文件
├── agent_mcp/               # MCP Server 实现
│   ├── __init__.py
│   ├── server.py            # MCP Server 入口
│   └── tools.py             # 原子能力工具
├── app/                     # 核心服务（内嵌自 backend）
│   ├── core/settings.py
│   ├── services/
│   ├── integrations/
│   └── store/
├── oneke/                   # OneKE HTTP 服务（内嵌）
│   ├── Dockerfile
│   ├── http_server.py
│   └── requirements.txt
├── scripts/                 # 便捷脚本
│   ├── setup.sh             # 初始化
│   ├── start-deps.sh        # 启动依赖
│   ├── stop-deps.sh         # 停止依赖
│   └── check-deps.sh        # 健康检查
└── tests/                   # 测试
    └── test_agent_mcp.py
```

## 环境变量说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | — | **必填** LLM API Key |
| `OPENAI_BASE_URL` | — | LLM API Base URL |
| `OPENAI_MODEL` | — | LLM 模型名 |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j 连接地址 |
| `NEO4J_USER` | `neo4j` | Neo4j 用户名 |
| `NEO4J_PASSWORD` | `neo4j_password` | Neo4j 密码 |
| `ONEKE_BASE_URL` | `http://localhost:8010` | OneKE 服务地址 |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama 地址（可选） |
