#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "[start-deps] 启动依赖服务 (neo4j + oneke)..."

# 确保 .env 存在
if [ ! -f .env ]; then
    echo "ERROR: .env 不存在。请先运行 ./scripts/setup.sh"
    exit 1
fi

docker compose up -d neo4j oneke

echo "[start-deps] 等待服务就绪..."

# 等待 neo4j
for i in {1..30}; do
    if curl -s http://localhost:7474 &>/dev/null; then
        echo "  Neo4j ready ✓"
        break
    fi
    echo "  Neo4j 启动中... ($i/30)"
    sleep 2
done

# 等待 oneke
for i in {1..30}; do
    if curl -s http://localhost:8010/health &>/dev/null; then
        echo "  OneKE ready ✓"
        break
    fi
    echo "  OneKE 启动中... ($i/30)"
    sleep 2
done

echo ""
echo "[start-deps] 所有依赖已就绪！"
echo "  Neo4j Browser: http://localhost:7474"
echo "  OneKE API:     http://localhost:8010"
echo ""
echo "现在可以启动 Agent:"
echo "  uv run python -m agent_mcp.server"
echo ""
