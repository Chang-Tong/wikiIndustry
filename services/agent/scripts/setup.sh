#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=================================="
echo "  WikiProject Agent 初始化"
echo "=================================="

# 检查 Docker
echo "[1/3] 检查 Docker..."
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker 未安装。请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
if ! docker info &>/dev/null; then
    echo "ERROR: Docker 守护进程未运行。请启动 Docker。"
    exit 1
fi
echo "  Docker OK"

# 生成 .env
echo "[2/3] 生成 .env 配置文件..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  已创建 .env，请编辑填入你的 OPENAI_API_KEY"
else
    echo "  .env 已存在，跳过"
fi

# 检查 uv
echo "[3/3] 检查 uv..."
if ! command -v uv &>/dev/null; then
    echo "  WARNING: uv 未安装。推荐安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  你也可以用 pip: pip install uv"
else
    echo "  uv OK"
fi

echo ""
echo "=================================="
echo "  初始化完成！"
echo "=================================="
echo ""
echo "下一步:"
echo "  1. 编辑 .env，填入 OPENAI_API_KEY"
echo "  2. ./scripts/start-deps.sh   # 启动 neo4j + oneke"
echo "  3. uv run python -m agent_mcp.server  # 启动 MCP Server"
echo ""
