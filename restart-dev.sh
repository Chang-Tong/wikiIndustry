#!/bin/bash
# 快速重启本地开发环境（Neo4j + OneKE + Backend）
# 适用于：合盖唤醒后连接断开、或者 backend 报 Neo4j driver 错误

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔄 重启开发环境..."

# 1. 确保 Docker 依赖在运行
echo "  → 启动 Neo4j + OneKE..."
COMPOSE="docker compose"
if ! docker compose version >/dev/null 2>&1; then
    COMPOSE="docker-compose"
fi
${COMPOSE} up -d neo4j oneke

# 2. 等待 Neo4j 就绪
echo -n "  → 等待 Neo4j 就绪"
for i in {1..30}; do
    if curl -s http://localhost:7474 >/dev/null 2>&1; then
        echo " ✓"
        break
    fi
    echo -n "."
    sleep 2
done

# 3. 杀掉旧后端并重启
echo "  → 重启后端..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 1

export $(grep -v '^#' .env | xargs)
cd "$SCRIPT_DIR/services/backend"
source .venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &

echo ""
echo "✅ 后端已重启，稍等 3 秒..."
sleep 3
curl -s http://localhost:8000/healthz | python3 -m json.tool || true
echo ""
echo "🚀 开发环境恢复完成！"
echo "   前端: http://localhost:5173"
echo "   后端: http://localhost:8000"
