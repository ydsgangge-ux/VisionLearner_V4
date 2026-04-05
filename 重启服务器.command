#!/bin/bash
# ============================================================
#  VisionLearner macOS/Linux 重启服务器
# ============================================================

cd "$(dirname "$0")"

# 优先使用 python3
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

echo "========================================"
echo "  停止现有服务器..."
echo "========================================"

# 查找并终止占用端口的进程
PORT=5000
PID=$(lsof -ti :$PORT 2>/dev/null)
if [ -n "$PID" ]; then
    kill -9 $PID 2>/dev/null
    echo "  已终止端口 $PORT 上的进程 (PID: $PID)"
else
    echo "  端口 $PORT 上没有运行的服务"
fi

sleep 2

echo "========================================"
echo "  启动 VisionLearner Web 服务器"
echo "========================================"
echo ""
echo "  测试界面访问地址:"
echo "    - http://localhost:$PORT/test      (测试界面)"
echo "    - http://localhost:$PORT/simple    (简单测试)"
echo "    - http://localhost:$PORT/         (主界面)"
echo ""
echo "  按 Ctrl+C 停止服务器"
echo "========================================"
echo ""

$PYTHON restart_server.py
