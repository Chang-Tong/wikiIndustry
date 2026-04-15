## Summary

目标：做一个“知识库 + 在线大模型”的可演示 Demo，其中：
- RAG：使用 RAGFlow 负责文档入库、切分、向量化与检索
- 关系挖掘：使用 OneKE 做关系/三元组抽取
- 图谱存储：Neo4j
- 可视化：前端用 Cytoscape.js

本计划聚焦“项目结构 + 服务边界 + 数据流 + API 形状 + 可落地的目录组织”，便于后续直接进入实现。

## Current State Analysis

仓库当前状态：目录为空（尚未初始化任何代码/配置）。

## Proposed Changes

### 1) 总体架构（单仓库多服务，一键启动）

采用 Monorepo + docker-compose 编排的方式，包含 5 类组件：
- frontend：React + Vite + Cytoscape.js（展示与交互）
- backend：FastAPI（对外 API、协调 RAGFlow/OneKE/Neo4j，提供统一鉴权与数据模型）
- neo4j：图数据库
- ragflow：RAGFlow 服务（作为外部服务，由 compose 管理）
- oneke：OneKE 服务（优先采用 OneKE 自带的 dockerized 形态；backend 通过 HTTP/SDK 调用）

核心原则：
- 前端永远只调用 backend（不直接访问 Neo4j/RAGFlow/OneKE），避免 CORS、权限、协议耦合
- backend 负责：任务编排、状态管理、数据落库（Neo4j）与统一的数据返回格式（适配 Cytoscape）
- 先把“文档 → 抽取 → 入图 → 可视化”闭环跑通，再把 RAG 问答面板接上（RAGFlow 已经在同一闭环里负责 ingestion/index）

### 2) 推荐目录结构（落到文件层面）

仓库根目录建议如下：

/
  README.md
  .env.example
  docker-compose.yml
  infra/
    neo4j/
      init.cypher
    ragflow/
      README.md
    oneke/
      README.md
  services/
    backend/
      pyproject.toml
      uv.lock 或 poetry.lock（按选型落地）
      app/
        main.py
        core/
          settings.py
          logging.py
        api/
          v1/
            routes_docs.py
            routes_extract.py
            routes_graph.py
            routes_rag.py
        domain/
          docs/
            models.py
            service.py
          extraction/
            models.py
            service.py
            schemas.py
          graph/
            models.py
            service.py
        integrations/
          neo4j/
            client.py
            cypher.py
          ragflow/
            client.py
          oneke/
            client.py
        store/
          sqlite.py（任务/文档元数据的轻量存储）
          files.py（上传/缓存文件）
      tests/
        test_health.py
        test_graph_contract.py
      Dockerfile
    frontend/
      package.json
      vite.config.ts
      src/
        main.tsx
        pages/
          Upload.tsx
          Graph.tsx
        components/
          GraphView.tsx
          SidePanel.tsx
        api/
          client.ts
          graph.ts
          docs.ts
        styles/
      Dockerfile
  packages/
    shared/
      graph-contract/
        schema.ts（前后端共享的 Cytoscape JSON contract）

说明：
- services/backend 按“domain + integrations”分层：业务模型在 domain，外部系统调用在 integrations，API 路由只做入参校验与调用 service
- packages/shared 放“契约类”内容，保证前后端对节点/边字段一致（尤其是 Cytoscape 需要的 data/id/source/target）
- sqlite 仅用于 demo 的任务状态与文档元数据（doc_id、ragflow_doc_id、extract_job_id、时间戳等）；图谱事实在 Neo4j

### 3) 数据流（端到端）

以“上传 Markdown/纯文本，按文档全量抽取”为主线：

1. 前端上传文档
   - POST /api/v1/docs
   - backend 保存原文（文件或 DB），返回 doc_id

2. backend 触发 RAGFlow 入库
   - backend -> RAGFlow：创建数据集/上传文档/触发切分向量化（具体调用以 RAGFlow API 为准）
   - backend 保存 ragflow_dataset_id / ragflow_doc_id 到 sqlite

3. backend 调用 OneKE 做关系抽取
   - backend -> OneKE：传入全文 + 抽取 schema（实体类型、关系类型）
   - OneKE -> backend：返回结构化结果（entities, relations, evidence）
   - backend 将抽取结果写入 Neo4j，并写入溯源信息（doc_id、原文片段/offset、模型信息）

4. 前端获取图数据并用 Cytoscape 展示
   - GET /api/v1/graph?doc_id=...
   - backend -> Neo4j：按 doc_id 查询子图（节点、边、属性）
   - backend -> frontend：返回 Cytoscape elements JSON（nodes/edges）

可选增强（同一结构下后续加）：
- 图上点选节点/边 → GET /api/v1/graph/evidence?id=... 显示证据文本
- 图上搜索实体 → GET /api/v1/graph/search?q=...
- RAG 问答 → POST /api/v1/rag/ask（由 backend 调用 RAGFlow 检索并用在线大模型生成回答）

### 4) API 设计（契约先行，便于前后端并行）

最小可用 API：
- POST /api/v1/docs
  - 入参：text、title（可选）
  - 出参：doc_id

- POST /api/v1/extract
  - 入参：doc_id、schema_name（可选）
  - 出参：job_id

- GET /api/v1/extract/{job_id}
  - 出参：status、progress、error（可选）

- GET /api/v1/graph
  - 入参：doc_id
  - 出参：{ elements: { nodes: [...], edges: [...] } }

约定 Cytoscape contract（共享 schema）：
- node: { data: { id, label, type, properties..., doc_id } }
- edge: { data: { id, source, target, label, type, properties..., doc_id } }

### 5) Neo4j 模型（最小可用 + 可演进）

节点：
- (:Entity {id, name, type, doc_id, ...})

关系：
- (:Entity)-[:REL {id, type, doc_id, evidence, ...}]->(:Entity)

索引与约束（init.cypher）：
- Entity(id) 唯一约束
- Entity(doc_id) 普通索引（按文档查子图）
- REL(doc_id) 普通索引（按文档查边）

### 6) 配置与密钥管理

根目录提供：
- .env.example（不放真实密钥）
- docker-compose.yml 从 .env 读取：
  - OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
  - NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
  - RAGFLOW_BASE_URL / RAGFLOW_API_KEY（如需要）
  - ONEKE_BASE_URL（或 OneKE 容器内部地址）

### 7) docker-compose 编排

docker-compose.yml 计划包含：
- neo4j（映射 7474/7687）
- backend（依赖 neo4j、ragflow、oneke；暴露 8000）
- frontend（暴露 5173 或 80）
- ragflow（以官方推荐方式起服务；若官方需要额外依赖，如 ES/MinIO/MySQL，则按官方 compose 拆到 infra/ragflow）
- oneke（以 OneKE 官方 docker 方式起；需要访问在线大模型的环境变量）

为了控制复杂度：
- 第一阶段允许 ragflow/oneke 以“外部已部署服务”方式接入，只要 backend 通过 BASE_URL 调用即可
- 第二阶段再把它们纳入统一 compose（如果官方部署链路复杂，就用 infra/ragflow、infra/oneke 单独维护）

## Assumptions & Decisions

已确定偏好（来自讨论）：
- 单仓库多服务 + docker-compose
- 后端 FastAPI
- 在线大模型走 OpenAI 兼容 API
- Demo 展示重点：关系抽取 + 图谱（RAGFlow 负责入库/切分/向量化，为后续问答预留）
- 文档抽取粒度：按文档全量
- 首期数据源：Markdown/纯文本
- 前端：React + Vite + Cytoscape.js

待实现时的默认决策（本计划先锁定，后续如需可调整）：
- backend 用轻量 sqlite 存任务状态（避免引入 Redis/Celery 复杂度）
- 抽取流程采用“后台任务 + 轮询 job 状态”，保证前端体验
- Neo4j 写入采用 merge 幂等策略（同 doc_id 下重复抽取不会无限膨胀）

## Verification

落地后验证方式（以可执行为准）：
- docker-compose up 后：
  - frontend 能访问并成功调用 backend health
  - 上传文本 → 返回 doc_id
  - 触发抽取 → job 进入 running → finished
  - 图谱接口返回 nodes/edges 且 Cytoscape 能正确渲染
- 后端单测（最少 2 个）：
  - API contract：/graph 返回结构符合 shared schema
  - Neo4j 写入幂等：同一 doc_id 重复写入节点/边数量不爆炸

