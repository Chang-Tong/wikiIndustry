#!/bin/bash

# 知识图谱系统 - 本地开发一键启动
# 同时启动前端和后端，支持热重载

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  知识图谱系统 - 开发模式      ${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}创建默认 .env 文件...${NC}"
    cat > .env << 'EOF'
# Neo4j 配置 (Docker 模式)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

# OneKE 配置
ONEKE_BASE_URL=http://localhost:8010
REQUIRE_REAL_ONEKE=true

# LLM 配置 (DeepSeek)
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=your-deepseek-api-key
OPENAI_MODEL=deepseek-chat

# 前端配置
FRONTEND_ORIGIN=http://localhost:5173
EOF
    echo -e "${YELLOW}请编辑 .env 文件，填入你的 DeepSeek API Key${NC}"
fi

# 加载环境变量
export $(grep -v '^#' .env | xargs)

PID_BACKEND="$SCRIPT_DIR/.backend.pid"
PID_FRONTEND="$SCRIPT_DIR/.frontend.pid"

_kill_by_pidfile() {
    local pidfile="$1"
    local name="$2"
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${YELLOW}停止残留 ${name} 进程 (PID: $pid)...${NC}"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}

cleanup() {
    echo ""
    echo -e "${YELLOW}正在停止服务...${NC}"
    _kill_by_pidfile "$PID_BACKEND" "后端"
    _kill_by_pidfile "$PID_FRONTEND" "前端"
    exit 0
}
trap cleanup INT TERM EXIT

# 启动前清理可能残留的进程
_kill_by_pidfile "$PID_BACKEND" "后端"
_kill_by_pidfile "$PID_FRONTEND" "前端"

echo -e "${BLUE}启动依赖服务...${NC}"
echo -e "确保 ${CYAN}Neo4j${NC} 和 ${CYAN}OneKE${NC} 已启动:"
echo "  docker-compose up -d neo4j oneke"
echo ""

# 检查 Neo4j 是否运行
echo -n "检查 Neo4j 连接..."
if ! curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo -e " ${RED}未启动${NC}"
    echo -e "${YELLOW}正在启动 Neo4j...${NC}"
    docker-compose up -d neo4j
    sleep 5
else
    echo -e " ${GREEN}已连接${NC}"
fi

# 检查 OneKE 是否运行
echo -n "检查 OneKE 连接..."
if ! curl -s http://localhost:8010/health > /dev/null 2>&1; then
    echo -e " ${RED}未启动${NC}"
    echo -e "${YELLOW}正在启动 OneKE...${NC}"
    docker-compose up -d oneke
    sleep 5
else
    echo -e " ${GREEN}已连接${NC}"
fi

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}  启动后端服务...${NC}"
echo -e "${BLUE}================================${NC}"
cd "$SCRIPT_DIR/services/backend"
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null || true

# 在后台启动后端
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo $BACKEND_PID > "$PID_BACKEND"

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}  启动前端服务...${NC}"
echo -e "${BLUE}================================${NC}"
cd "$SCRIPT_DIR/services/frontend"
npm install 2>/dev/null || true

# 在后台启动前端
npm run dev &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$PID_FRONTEND"
cd "$SCRIPT_DIR"

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  所有服务已启动!              ${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "服务地址:"
echo -e "  前端界面: ${CYAN}http://localhost:5173${NC}"
echo -e "  后端 API: ${CYAN}http://localhost:8000${NC}"
echo -e "  API 文档: ${CYAN}http://localhost:8000/docs${NC}"
echo ""
echo -e "按 ${YELLOW}Ctrl+C${NC} 停止所有服务"
echo ""

# 等待用户中断
wait
