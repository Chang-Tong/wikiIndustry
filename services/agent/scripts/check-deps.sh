#!/usr/bin/env bash
set -euo pipefail

echo "[check-deps] 检查依赖服务状态..."

# Neo4j
if curl -s http://localhost:7474 &>/dev/null; then
    echo "  Neo4j     ✓ http://localhost:7474"
else
    echo "  Neo4j     ✗ 未运行"
fi

# OneKE
if curl -s http://localhost:8010/health &>/dev/null; then
    echo "  OneKE     ✓ http://localhost:8010"
else
    echo "  OneKE     ✗ 未运行"
fi

# LLM (简单 ping)
if [ -n "${OPENAI_BASE_URL:-}" ]; then
    echo "  LLM URL   → $OPENAI_BASE_URL"
fi
