#!/bin/bash
# 开发时阻止 Mac 合盖睡眠，保持 Neo4j + 后端连接不断
# 用法: ./keep-alive.sh
# 按 Ctrl+C 取消

echo "☕ 保持系统唤醒中...（Neo4j 和 Backend 不会断）"
echo "   按 Ctrl+C 停止"
echo ""

# -d: 阻止显示器睡眠
# -i: 阻止系统空闲睡眠
# -s: 只在 AC 电源下生效（推荐插电用）
caffeinate -dis python3 -c "import signal, time; signal.pause()"
