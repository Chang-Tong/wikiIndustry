#!/bin/bash

# WikiProject - 停止所有服务（Docker + 本地进程）

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${YELLOW}正在停止 WikiProject 服务...${NC}"

# 停止本地前后端进程（通过 PID 文件）
_kill_by_pidfile() {
    local pidfile="$1"
    local name="$2"
    if [ -f "$pidfile" ]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${YELLOW}停止本地 ${name} 进程 (PID: $pid)...${NC}"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
}

_kill_by_pidfile "$SCRIPT_DIR/.backend.pid" "后端"
_kill_by_pidfile "$SCRIPT_DIR/.frontend.pid" "前端"

# 兜底：停止本地 uvicorn 和 vite 进程
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

# 停止 Docker 服务
COMPOSE="docker compose"
if ! docker compose version >/dev/null 2>&1; then
    COMPOSE="docker-compose"
fi

if docker info >/dev/null 2>&1; then
    echo -e "  ${YELLOW}停止 Docker 服务...${NC}"
    ${COMPOSE} down 2>/dev/null || true
else
    echo -e "  ${YELLOW}Docker 未运行，跳过容器停止${NC}"
fi

echo -e "${GREEN}所有服务已停止${NC}"
