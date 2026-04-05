#!/bin/bash
# ============================================================
#  VisionLearner macOS/Linux GitHub 清理脚本
# ============================================================

echo ""
echo "================================"
echo "  GitHub Upload Cleanup Script"
echo "================================"
echo ""

read -p "Continue? (Y/N): " confirm
if [[ "$confirm" != "Y" && "$confirm" != "y" ]]; then
    exit 0
fi

echo ""
echo "[1/4] Deleting test files..."
find . -maxdepth 1 -name "test_*.py" -delete 2>/dev/null
find . -maxdepth 1 -name "test_*.html" -delete 2>/dev/null
find . -maxdepth 1 -name "TEST_UI_README.md" -delete 2>/dev/null
find . -maxdepth 1 -name "test_ui.bat" -delete 2>/dev/null
find . -maxdepth 1 -name "verify_fixes.py" -delete 2>/dev/null
find . -maxdepth 1 -name "analyze_data.py" -delete 2>/dev/null
find . -maxdepth 1 -name "simple_test.html" -delete 2>/dev/null
echo "  [OK]"

echo ""
echo "[2/4] Deleting check scripts..."
find . -maxdepth 1 -name "check_*.py" -delete 2>/dev/null
echo "  [OK]"

echo ""
echo "[3/4] Deleting outdated files..."
find . -maxdepth 1 -name "requirements_full.txt" -delete 2>/dev/null
find . -maxdepth 1 -name "quiz_lines.txt" -delete 2>/dev/null
rm -rf "更新" 2>/dev/null
echo "  [OK]"

echo ""
echo "[4/4] Cleaning cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -maxdepth 1 -name "*.pyc" -delete 2>/dev/null
echo "  [OK]"

echo ""
echo "================================"
echo "  Cleanup Complete!"
echo "================================"
echo ""
echo "Protected files:"
echo "  - Core code (.py)"
echo "  - Web UI (visionlearner_ui.html)"
echo "  - Documents (*.md)"
echo "  - Startup scripts (.command, .bat)"
echo "  - Skills (skills/)"
echo "  - Screenshots (screenshots/)"
echo ""
echo "Safe from upload (.gitignore):"
echo "  - .env (API keys)"
echo "  - learning_data/ (personal data)"
echo "  - cache/ (LLM cache)"
echo ""

read -p "按回车键退出..."
