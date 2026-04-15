# WikiProject 改进优先级清单

> 自动记录自 Claude Code 会话，按优先级降序排列。

## P0 - 性能与稳定性（最高优先级）

1. **增量相似度计算**
   - 当前问题：每次 JSON 上传后，`create_correlation_edges()` 对全库重新计算相似度。数据量增长后上传会越来越慢。
   - 目标：改为只给本次上传的 `doc_id` 关联节点计算 `CORRELATED_WITH` 边。
   - 相关文件：`services/backend/app/services/correlation_mining.py`、`routes_json.py`

2. **`start-all.sh` 开发模式进程残留**
   - 当前问题：`--dev` 用 `&` 后台启动 uvicorn / npm，异常退出时进程可能残留为孤儿进程。
   - 目标：把 PID 写入文件，增加 `trap EXIT` 清理逻辑，确保异常退出也杀进程。
   - 相关文件：`start-all.sh`

## P1 - 数据一致性

3. **Overwrite 模式未清理向量/嵌入缓存**
   - 当前问题：覆盖上传仅清空了 Neo4j 和 SQLite，但如果有嵌入缓存文件（未来）会不一致。
   - 目标：确认 `store.clear_all()` 已彻底清空 SQLite；如有 Ollama 嵌入缓存也一并清理。
   - 相关文件：`routes_json.py`

4. **Job 状态内存存储，重启丢失**
   - 当前问题：`_job_store` 是内存 dict，后端重启后用户查不到历史 job。
   - 目标：把 job 状态持久化到 SQLite（加 `jobs_status` 表）。
   - 相关文件：`routes_json.py`、`app/store/sqlite.py`

## P2 - 代码结构与可维护性

5. **前端单文件 `App.tsx` 过于庞大**
   - 当前问题：一个文件承载了 Graph、Upload、QA 三个大 Tab 及所有状态。
   - 目标：拆分为 `GraphTab.tsx`、`UploadTab.tsx`、`QATab.tsx`，保持 `App.tsx` 只做路由/布局。
   - 相关文件：`services/frontend/src/App.tsx`

## P3 - 测试与长期效率

6. **缺少端到端集成测试**
   - 当前问题：验证靠手工 curl 和子集数据测试，回归成本高。
   - 目标：增加 `pytest` 集成测试流水线：上传 JSON → 检查节点数 → 调用 `/qa/ask-graph`。
   - 相关文件：`services/backend/tests/`
