@echo off
chcp 65001 >nul
echo ========================================
echo Starting VisionLearner with Debug Logs
echo ========================================
echo.
echo Server will start on http://localhost:5000
echo.
echo After server starts:
echo   1. Open test_ui.html in your browser
echo   2. Try to select a goal
echo   3. Watch this window for error details
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

python test_with_logging.py

pause
