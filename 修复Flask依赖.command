#!/bin/bash
# ============================================================
#  VisionLearner macOS/Linux 修复 Flask 依赖
# ============================================================

cd "$(dirname "$0")"

# 优先使用 python3
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

echo ""
echo "============================================================"
echo "          VisionLearner Web 服务依赖修复"
echo "============================================================"
echo ""

echo "[1/2] 安装 Flask 和 CORS..."
$PYTHON -m pip install flask flask-cors --quiet 2>/dev/null

if [ $? -ne 0 ]; then
    echo "  [FAIL] 安装失败，请检查网络连接"
    echo "  [CMD] 手动运行: $PYTHON -m pip install flask flask-cors"
    read -p "按回车键退出..."
    exit 1
fi

echo "  [OK] Flask 安装成功"

echo ""
echo "[2/2] 验证安装..."
$PYTHON -c "import flask; import flask_cors; print('[OK] 验证通过')"

if [ $? -ne 0 ]; then
    echo "  [FAIL] 验证失败"
    read -p "按回车键退出..."
    exit 1
fi

echo ""
echo "============================================================"
echo "                     修复完成！"
echo "============================================================"
echo ""
echo "现在可以启动 Web 服务了："
echo "  $PYTHON main.py --web"
echo ""

read -p "是否立即启动？(Y/N): " choice
if [[ "$choice" == "Y" || "$choice" == "y" ]]; then
    $PYTHON main.py --web
fi
