#!/bin/bash
# ============================================================
#  VisionLearner macOS/Linux 启动命令行模式
# ============================================================

cd "$(dirname "$0")"

# 优先使用 python3
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

$PYTHON main.py
