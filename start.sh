#!/bin/bash

# 知识图谱系统一键启动脚本
# 启动前端 + 后端 + Neo4j

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  知识图谱系统 - 一键启动      ${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}错误: Docker 未运行${NC}"
    echo "请先启动 Docker Desktop"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建必要的目录
mkdir -p data/neo4j data/sqlite

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}创建默认 .env 文件...${NC}"
    cat > .env << 'EOF'
# Neo4j 配置
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

# OneKE 配置
ONEKE_BASE_URL=http://oneke:8000
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

echo -e "${BLUE}步骤 1/4: 构建 Docker 镜像...${NC}"
docker-compose build

echo ""
echo -e "${BLUE}步骤 2/4: 启动服务...${NC}"
docker-compose up -d neo4j oneke

# 等待 Neo4j 启动
echo ""
echo -e "${BLUE}步骤 3/4: 等待 Neo4j 启动...${NC}"
echo -n "等待中"
for i in {1..30}; do
    if curl -s http://localhost:7474 > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}Neo4j 已就绪!${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# 等待 OneKE 启动
echo ""
echo -e "${BLUE}等待 OneKE 启动...${NC}"
echo -n "等待中"
for i in {1..60}; do
    if curl -s http://localhost:8010/health > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}OneKE 已就绪!${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo -e "${BLUE}步骤 4/4: 启动后端和前端...${NC}"
docker-compose up -d backend frontend

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  所有服务已启动!              ${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "服务地址:"
echo -e "  前端界面: ${BLUE}http://localhost:5173${NC}"
echo -e "  后端 API: ${BLUE}http://localhost:8000${NC}"
echo -e "  Neo4j Browser: ${BLUE}http://localhost:7474${NC}"
echo ""
echo -e "查看日志: ${YELLOW}docker-compose logs -f${NC}"
echo -e "停止服务: ${YELLOW}docker-compose down${NC}"
echo ""
