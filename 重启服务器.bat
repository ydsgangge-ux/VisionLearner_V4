@echo off
chcp 65001 > nul
echo ========================================
echo 停止现有服务器...
echo ========================================
taskkill /F /IM python.exe 2>nul

timeout /t 2 /nobreak > nul

echo ========================================
echo 启动 VisionLearner Web 服务器
echo ========================================
echo.
echo 测试界面访问地址:
echo   - http://localhost:5000/test      (测试界面)
echo   - http://localhost:5000/simple    (简单测试)
echo   - http://localhost:5000/         (主界面)
echo.
echo 按 Ctrl+C 停止服务器
echo ========================================
echo.

cd /d "%~dp0"
python restart_server.py
pause
