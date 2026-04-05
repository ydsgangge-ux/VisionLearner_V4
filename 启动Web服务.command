#!/bin/bash
# ============================================================
#  VisionLearner macOS/Linux 启动 Web 服务
# ============================================================

cd "$(dirname "$0")"

# 优先使用 python3
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

$PYTHON main.py --web --port 5000
