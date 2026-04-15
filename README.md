# WikiProject - 知识图谱驱动的 RAG 系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/React-18+-61DAFB.svg?logo=react" alt="React 18+">
  <img src="https://img.shields.io/badge/Neo4j-5.x-008CC1.svg?logo=neo4j" alt="Neo4j 5.x">
  <img src="https://img.shields.io/badge/AntV%20G6-5.x-1677FF.svg" alt="AntV G6">
  <img src="https://img.shields.io/badge/OneKE-实体提取-orange.svg" alt="OneKE">
</p>

<p align="center">
  <b>智能知识管理 · 语义检索 · 防幻觉问答</b>
</p>

---

## 项目简介

WikiProject 是一个基于**知识图谱**和**RAG（检索增强生成）**技术的智能文档问答系统。系统能够从非结构化文档中提取实体和关系构建知识图谱，结合语义相似度计算，提供可靠的问答服务。

### 核心特性

- **智能实体提取**：集成 OneKE 模型，自动识别文档中的实体和关系
- **混合相似度计算**：结合 Jaccard 实体相似度 + 余弦向量相似度（权重 6:4）
- **防幻觉 RAG**：强制引用来源、答案验证、置信度评估
- **交互式图谱可视化**：基于 AntV G6 的力导向布局，支持缩放/拖拽/点击
- **多数据源融合**：图谱数据 + 原始文档 + 外部搜索（可扩展）

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端层 (Frontend)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   数据上传   │  │  图谱可视化  │  │      智能问答        │  │
│  │  (Upload)   │  │   (Graph)   │  │        (QA)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                          React + TypeScript                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        后端层 (Backend)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  JSON 处理   │  │  RAG 引擎    │  │    相似度计算       │  │
│  │  (OneKE)    │  │  (LLM)      │  │  (Entity+Vector)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                          FastAPI (Python)                     │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Neo4j 图谱   │    │  SQLite 文档   │    │  Ollama/Deep  │
│  (图数据库)    │    │  (原始数据)    │    │   Embedding   │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## 功能演示

### 1. 数据上传与图谱构建

```bash
curl -X POST http://localhost:8000/api/v1/json/upload \
  -F "file=@test_data.json" \
  -F "schema_name=MOE_News"
```

响应示例：
```json
{
  "job_id": "b4260709-f5ac-49b1-a6ff-526e2db4e447",
  "status": "completed",
  "total_items": 5,
  "processed_items": 5,
  "extracted_entities": 88,
  "extracted_relations": 129,
  "engine": "oneke"
}
```

### 2. 混合相似度计算

```bash
curl http://localhost:8000/api/v1/correlations?min_score=0.3&limit=5
```

响应示例：
```json
{
  "correlations": [
    {
      "news_id_1": "news_001",
      "news_id_2": "news_002",
      "similarity_score": 0.85,
      "entity_score": 0.80,
      "vector_score": 0.72,
      "correlation_type": "hybrid",
      "shared_entities": ["教育部", "教育信息化"]
    }
  ]
}
```

### 3. 防幻觉 RAG 问答

```bash
curl -X POST http://localhost:8000/api/v1/qa/rag \
  -H "Content-Type: application/json" \
  -d '{
    "question": "教育部发布了哪些教育信息化政策？",
    "doc_id": "news_001"
  }'
```

响应示例：
```json
{
  "answer": "教育部发布了《教育信息化2.0行动计划》[1]和《教育信息化十年发展规划》[2]。\n\n【依据】\n- [1] 来自知识图谱：教育部 --[发布]--> 《教育信息化2.0行动计划》\n- [2] 来自文档：《教育部政策解读》\n\n【说明】\n本回答基于检索到的知识库数据，置信度：高",
  "sources": [...],
  "confidence": {
    "level": "high",
    "avg_score": 4.2,
    "chunks_count": 5
  }
}
```

---

## 快速开始

### 环境要求

- Docker & Docker Compose
- 8GB+ 内存（OneKE 模型需要）
- DeepSeek API Key（或 OpenAI API Key）
- **Ollama**（强制使用，用于文本向量化）
  - 安装 Ollama: https://ollama.com
  - 下载模型: `ollama pull qwen3-embedding:0.6b`

### 一键启动

```bash
# 克隆项目
git clone <repository-url>
cd wikiProject

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key

# 启动所有服务
./start.sh

# 查看服务状态
docker compose ps
```

### 访问服务

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端界面 | http://localhost:5173 | 数据上传/图谱展示/智能问答 |
| 后端 API | http://localhost:8000 | RESTful API |
| API 文档 | http://localhost:8000/docs | Swagger UI |
| Neo4j Browser | http://localhost:7474 | 图数据库管理界面 |

### 上传测试数据

```bash
curl -X POST http://localhost:8000/api/v1/json/upload \
  -F "file=@test_data.json" \
  -F "schema_name=MOE_News"
```

---

## 技术栈

### 后端

| 技术 | 用途 |
|------|------|
| **FastAPI** | Web 框架，自动生成 OpenAPI 文档 |
| **Neo4j** | 图数据库，存储实体和关系 |
| **SQLite** | 轻量级文档存储 |
| **OneKE** | 实体关系抽取（NLP） |
| **DeepSeek/OpenAI** | 大语言模型，用于问答生成 |
| **Ollama** | 本地 Embedding 模型服务（强制使用） |

### 前端

| 技术 | 用途 |
|------|------|
| **React 18** | UI 框架 |
| **TypeScript** | 类型安全 |
| **AntV G6** | 图谱可视化（力导向布局）|
| **CSS Variables** | Apple Design 风格样式系统 |

---

## 核心设计

### 1. 混合相似度计算

```
Hybrid Score = 0.6 × Entity Score + 0.4 × Vector Score

Entity Score: Jaccard 相似度
              = |共享实体| / (|实体A| + |实体B| - |共享实体|)

Vector Score: 余弦相似度
              = (A·B) / (||A|| × ||B||)
```

### 2. 防幻觉 RAG

**System Prompt 约束：**
- 只能使用提供的上下文信息
- 每个关键事实必须用 `[数字]` 标注来源
- 信息不足时明确说明无法回答

**答案验证：**
- ✅ **引用检查**：自动提取回答中的 `[数字]` 引用标记
- ✅ **来源验证**：验证每个引用是否在检索结果的有效范围内
- ✅ **无效引用标记**：发现无效引用时添加警告提示
- ✅ **置信度调整**：存在无效引用时自动降低置信度为 "low"
- ✅ **无引用提醒**：回答中无引用时添加免责声明

### 3. 新闻文档切分策略

- **不切分**：每条新闻作为完整检索单元
- **原因**：新闻本身短小、语义完整，切分会破坏上下文
- **实现**：`services/backend/app/services/rag_engine.py`

---

## API 端点

### 文档管理

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/json/upload` | 上传 JSON 文件构建图谱 |
| GET  | `/api/v1/json/jobs/{job_id}` | 查询上传任务状态 |

### 图谱操作

| 方法 | 端点 | 描述 |
|------|------|------|
| GET  | `/api/v1/graph` | 获取图谱数据 |
| POST | `/api/v1/graph/clear` | 清空图谱 |
| GET  | `/api/v1/graph/stats` | 获取图谱统计 |

### 相似度计算

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/correlations/embeddings` | 生成 embeddings |
| GET  | `/api/v1/correlations` | 查询相似度 |
| GET  | `/api/v1/correlations/matrix` | 相似度矩阵 |
| POST | `/api/v1/correlations/build-edges` | 创建关联边 |

### 智能问答

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/qa/rag` | RAG 问答（带防幻觉）|
| POST | `/api/v1/qa/ask-graph` | 图谱问答（直接 LLM）|

---

## 项目结构

```
wikiProject/
├── services/
│   ├── backend/              # FastAPI 后端
│   │   ├── app/
│   │   │   ├── api/v1/       # API 路由
│   │   │   ├── integrations/ # Neo4j/OneKE 集成
│   │   │   └── services/     # RAG/Embedding/相似度服务
│   │   └── Dockerfile
│   ├── frontend/             # React 前端
│   │   ├── src/
│   │   │   ├── App.tsx       # 主应用组件
│   │   │   └── styles.css    # Apple Design 样式
│   │   └── Dockerfile
│   └── oneke-official/       # OneKE 实体提取服务
├── docker-compose.yml        # Docker Compose 配置
├── start.sh                  # 一键启动脚本
├── dev.sh                    # 开发模式启动
├── CLAUDE.md                 # 详细技术文档
└── README.md                 # 本文件
```

---

## 配置说明

### 环境变量 (.env)

```bash
# LLM 配置（用于问答）
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-your-deepseek-api-key
OPENAI_MODEL=deepseek-chat

# OneKE 配置（强制使用真实服务）
ONEKE_BASE_URL=http://oneke:8000
REQUIRE_REAL_ONEKE=true

# Neo4j 配置
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

# Ollama Embedding（强制使用）
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
REQUIRE_OLLAMA_EMBEDDING=true  # 强制使用 Ollama，禁用 fallback
```

**⚠️ 重要：** 当 `REQUIRE_OLLAMA_EMBEDDING=true` 时，系统会强制使用 Ollama 进行向量化，不会回退到 OpenAI API 或本地 TF-IDF。请确保：
1. Ollama 服务已启动：`ollama serve`
2. 模型已下载：`ollama pull qwen3-embedding:0.6b`

---

## 开发指南

### 本地开发模式

```bash
# 启动数据库和 OneKE（后台运行）
docker compose up -d neo4j oneke

# 启动后端（热重载）
cd services/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 启动前端（热重载）
cd services/frontend
npm install
npm run dev
```

### 运行测试

```bash
# 测试 OneKE 服务
curl http://localhost:8010/health

# 测试 Embedding 生成
curl -X POST http://localhost:8000/api/v1/correlations/embeddings

# 测试 RAG 问答
curl -X POST http://localhost:8000/api/v1/qa/rag \
  -d '{"question": "教育部发布了哪些政策？"}'
```

---

## 最近更新

- **2025-04-12**: 强制使用 Ollama 进行向量化（禁用 fallback）
- **2025-04-12**: 向量语义相似度 + 防幻觉 RAG
- **2025-04-12**: 前端可视化库迁移到 AntV G6
- **2025-04-11**: 强制使用 OneKE 构建图谱（禁止模拟数据）
- **2025-04-11**: Apple Design 风格前端重构

详见 [CLAUDE.md](./CLAUDE.md) 完整变更记录。

---

## 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 许可证

MIT License - 详见 [LICENSE](./LICENSE) 文件

---

<p align="center">
  Built with ❤️ by WikiProject Team
</p>
