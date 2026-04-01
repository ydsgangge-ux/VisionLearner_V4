@echo off
chcp 65001 >nul
title VisionLearner GitHub Clean

echo.
echo ================================
echo   GitHub Upload Cleanup Script
echo ================================
echo.

set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" exit /b 0

echo.
echo [1/4] Deleting test files...
del /q test_*.py 2>nul
del /q test_*.html 2>nul
del /q TEST_UI_README.md 2>nul
del /q test_ui.bat 2>nul
del /q verify_fixes.py 2>nul
del /q analyze_data.py 2>nul
del /q simple_test.html 2>nul
echo   [OK]

echo.
echo [2/4] Deleting check scripts...
del /q check_*.py 2>nul
echo   [OK]

echo.
echo [3/4] Deleting outdated files...
del /q requirements_full.txt 2>nul
del /q quiz_lines.txt 2>nul
rmdir /s /q 更新 2>nul
echo   [OK]

echo.
echo [4/4] Cleaning cache...
rmdir /s /q __pycache__ 2>nul
del /q *.pyc 2>nul
echo   [OK]

echo.
echo ================================
echo   Cleanup Complete!
echo ================================
echo.
echo Protected files:
echo   - Core code (.py)
echo   - Web UI (visionlearner_ui.html)
echo   - Documents (*.md)
echo   - Startup scripts (*.bat)
echo   - Skills (skills/)
echo   - Screenshots (screenshots/)
echo.
echo Safe from upload (.gitignore):
echo   - .env (API keys)
echo   - learning_data/ (personal data)
echo   - cache/ (LLM cache)
echo.
pause