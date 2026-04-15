#!/bin/bash

# WikiProject - 全量服务一键启动脚本
# 启动顺序: Neo4j -> OneKE -> Backend -> Frontend

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE="docker compose"
if ! docker compose version >/dev/null 2>&1; then
    COMPOSE="docker-compose"
fi

# 用法
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --dev     本地开发模式: Docker 只启动 Neo4j + OneKE, 后端/前端用本地进程启动(支持热重载)"
    echo "  --stop    停止所有服务"
    echo "  --status  查看服务状态"
    echo "  --help    显示帮助"
    exit 0
}

# 检查 Docker
_check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}错误: Docker 未运行${NC}"
        echo "请先启动 Docker Desktop"
        exit 1
    fi
}

# 等待服务健康
_wait_for() {
    local name="$1"
    local url="$2"
    local max_wait="${3:-60}"
    local interval="${4:-2}"

    echo -e "${BLUE}等待 ${name} 就绪...${NC}"
    echo -n "  等待中"
    local waited=0
    while [ "$waited" -lt "$max_wait" ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo ""
            echo -e "  ${GREEN}${name} 已就绪!${NC}"
            return 0
        fi
        echo -n "."
        sleep "$interval"
        waited=$((waited + interval))
    done
    echo ""
    echo -e "  ${YELLOW}${name} 启动超时 (${max_wait}s), 请检查日志: ${COMPOSE} logs ${name,,}${NC}"
    return 1
}

# 停止所有
do_stop() {
    echo -e "${BLUE}停止所有服务...${NC}"
    # 停止本地前后端进程
    local pidfile
    for pidfile in "$SCRIPT_DIR/.backend.pid" "$SCRIPT_DIR/.frontend.pid"; do
        if [ -f "$pidfile" ]; then
            local pid
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "  ${YELLOW}停止本地进程 (PID: $pid)...${NC}"
                kill "$pid" 2>/dev/null || true
                sleep 1
                kill -9 "$pid" 2>/dev/null || true
            fi
            rm -f "$pidfile"
        fi
    done
    # 兜底
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true
    # 停止 Docker
    ${COMPOSE} down
    echo -e "${GREEN}已停止${NC}"
}

# 查看状态
do_status() {
    echo -e "${BLUE}服务状态:${NC}"
    ${COMPOSE} ps
    echo ""
    echo -e "${CYAN}快速检查:${NC}"
    for svc in "Neo4j@http://localhost:7474" "OneKE@http://localhost:8010/health" "Backend@http://localhost:8000/healthz" "Frontend@http://localhost:5173"; do
        name="${svc%%@*}"
        url="${svc##*@}"
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e "  ${GREEN}✓${NC} $name: $url"
        else
            echo -e "  ${RED}✗${NC} $name: 未响应"
        fi
    done
}

# 启动 Docker 模式
do_start_docker() {
    _check_docker

    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  WikiProject - Docker 全量启动 ${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""

    mkdir -p data/neo4j data/sqlite

    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}创建默认 .env 文件...${NC}"
        cat > .env << 'EOF'
# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

# OneKE
ONEKE_BASE_URL=http://oneke:8000
REQUIRE_REAL_ONEKE=true

# LLM (DeepSeek)
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=your-deepseek-api-key
OPENAI_MODEL=deepseek-chat

# Frontend
FRONTEND_ORIGIN=http://localhost:5173

# Ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
REQUIRE_OLLAMA_EMBEDDING=true
EOF
        echo -e "${YELLOW}注意: 请编辑 .env 填入真实的 API Key${NC}"
    fi

    echo -e "${BLUE}步骤 1/5: 构建镜像...${NC}"
    ${COMPOSE} build

    echo ""
    echo -e "${BLUE}步骤 2/5: 启动基础服务 (Neo4j + OneKE)...${NC}"
    ${COMPOSE} up -d neo4j oneke

    echo ""
    _wait_for "Neo4j" "http://localhost:7474" 60 2
    _wait_for "OneKE" "http://localhost:8010/health" 120 2

    echo ""
    echo -e "${BLUE}步骤 3/5: 启动后端...${NC}"
    ${COMPOSE} up -d backend
    _wait_for "Backend" "http://localhost:8000/healthz" 60 2

    echo ""
    echo -e "${BLUE}步骤 4/5: 启动前端...${NC}"
    ${COMPOSE} up -d frontend
    _wait_for "Frontend" "http://localhost:5173" 60 2

    echo ""
    echo -e "${BLUE}步骤 5/5: 打印系统状态...${NC}"
    ${COMPOSE} ps

    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}  所有服务已启动!              ${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "访问地址:"
    echo -e "  前端界面:      ${CYAN}http://localhost:5173${NC}"
    echo -e "  后端 API:      ${CYAN}http://localhost:8000${NC}"
    echo -e "  API 文档:      ${CYAN}http://localhost:8000/docs${NC}"
    echo -e "  Neo4j Browser: ${CYAN}http://localhost:7474${NC}"
    echo ""
    echo -e "常用命令:"
    echo -e "  查看全部日志:  ${YELLOW}${COMPOSE} logs -f${NC}"
    echo -e "  查看后端日志:  ${YELLOW}${COMPOSE} logs -f backend${NC}"
    echo -e "  停止所有服务:  ${YELLOW}${COMPOSE} down${NC}"
    echo -e "  查看状态:      ${YELLOW}$0 --status${NC}"
    echo ""
}

# 启动开发模式
do_start_dev() {
    _check_docker

    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  WikiProject - 本地开发模式   ${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""

    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}创建默认 .env 文件...${NC}"
        cat > .env << 'EOF'
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password
ONEKE_BASE_URL=http://localhost:8010
REQUIRE_REAL_ONEKE=true
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_API_KEY=your-deepseek-api-key
OPENAI_MODEL=deepseek-chat
FRONTEND_ORIGIN=http://localhost:5173
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
REQUIRE_OLLAMA_EMBEDDING=true
EOF
        echo -e "${YELLOW}注意: 请编辑 .env 填入真实的 API Key${NC}"
    fi

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

    cleanup_dev() {
        echo ""
        echo -e "${YELLOW}停止本地服务...${NC}"
        _kill_by_pidfile "$PID_BACKEND" "后端"
        _kill_by_pidfile "$PID_FRONTEND" "前端"
        exit 0
    }

    # 启动前清理可能残留的进程
    _kill_by_pidfile "$PID_BACKEND" "后端"
    _kill_by_pidfile "$PID_FRONTEND" "前端"

    echo -e "${BLUE}启动 Docker 依赖 (Neo4j + OneKE)...${NC}"
    ${COMPOSE} up -d neo4j oneke

    _wait_for "Neo4j" "http://localhost:7474" 60 2
    _wait_for "OneKE" "http://localhost:8010/health" 120 2

    echo ""
    echo -e "${BLUE}启动本地后端 (热重载)...${NC}"
    cd "$SCRIPT_DIR/services/backend"
    python3 -m venv .venv 2>/dev/null || true
    source .venv/bin/activate
    pip install -q -r requirements.txt 2>/dev/null || true
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$PID_BACKEND"
    cd "$SCRIPT_DIR"
    _wait_for "Backend" "http://localhost:8000/healthz" 60 2

    echo ""
    echo -e "${BLUE}启动本地前端 (热重载)...${NC}"
    cd "$SCRIPT_DIR/services/frontend"
    npm install 2>/dev/null || true
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$PID_FRONTEND"
    cd "$SCRIPT_DIR"
    _wait_for "Frontend" "http://localhost:5173" 60 2

    echo ""
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}  开发模式已启动!              ${NC}"
    echo -e "${GREEN}================================${NC}"
    echo ""
    echo -e "访问地址:"
    echo -e "  前端界面: ${CYAN}http://localhost:5173${NC}"
    echo -e "  后端 API: ${CYAN}http://localhost:8000${NC}"
    echo -e "  API 文档: ${CYAN}http://localhost:8000/docs${NC}"
    echo ""
    echo -e "按 ${YELLOW}Ctrl+C${NC} 停止本地服务 (Docker 依赖保持运行)"
    echo ""

    trap cleanup_dev INT TERM EXIT
    wait
}

# 主入口
case "${1:-}" in
    --dev)
        do_start_dev
        ;;
    --stop)
        do_stop
        ;;
    --status)
        do_status
        ;;
    --help|-h)
        usage
        ;;
    "")
        do_start_docker
        ;;
    *)
        echo -e "${RED}未知选项: $1${NC}"
        usage
        ;;
esac
