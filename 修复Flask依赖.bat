@echo off
chcp 65001 >nul
title VisionLearner 依赖修复

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║          VisionLearner Web 服务依赖修复                   ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

echo [1/2] 安装 Flask 和 CORS...
python -m pip install flask flask-cors --quiet

if errorlevel 1 (
    echo [FAIL] 安装失败，请检查网络连接
    echo [CMD] 手动运行：pip install flask flask-cors
    pause
    exit /b 1
)

echo [OK] Flask 安装成功

echo.
echo [2/2] 验证安装...
python -c "import flask; import flask_cors; print('[OK] 验证通过')"

if errorlevel 1 (
    echo [FAIL] 验证失败
    pause
    exit /b 1
)

echo.
echo ════════════════════════════════════════════════════════════
echo                     修复完成！
echo ════════════════════════════════════════════════════════════
echo.
echo 现在可以启动 Web 服务了：
echo   python main.py --web
echo.

set /p choice="是否立即启动？(Y/N): "
if /i "%choice%"=="Y" goto start
if /i "%choice%"=="y" goto start
exit /b 0

:start
python main.py --web
pause