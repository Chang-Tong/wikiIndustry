#!/bin/bash

# 停止所有知识图谱服务

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}正在停止知识图谱服务...${NC}"

# 停止 Docker 服务
docker-compose down 2>/dev/null || true

# 停止本地 Python 进程
pkill -f "uvicorn app.main:app" 2>/dev/null || true

# 停止本地 Node 进程
pkill -f "vite" 2>/dev/null || true

echo -e "${GREEN}所有服务已停止${NC}"
