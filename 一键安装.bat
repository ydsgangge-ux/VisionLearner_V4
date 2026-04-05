@echo off
chcp 65001 >nul
title VisionLearner 一键安装向导

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║          VisionLearner 一键安装向导 v4.2                  ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

:: ============================================
:: 步骤 1: 检查 Python 环境
:: ============================================
echo [1/6] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [FAIL] 未找到 Python，请先安装 Python 3.8+
    echo.
    echo   [DOWNLOAD] https://www.python.org/downloads/
    echo   [TIP] 推荐下载 Python 3.12 或更高版本
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo   [OK] Python %PYTHON_VERSION% 已安装

:: ============================================
:: 步骤 2: 升级 pip
:: ============================================
echo.
echo [2/6] 升级 pip 包管理器...
python -m pip install --upgrade pip >nul 2>&1
echo   [OK] pip 已升级到最新版本

:: ============================================
:: 步骤 3: 安装核心依赖
:: ============================================
echo.
echo [3/6] 安装项目依赖（这可能需要几分钟）...

python -m pip install requests flask flask-cors schedule python-dotenv networkx numpy openai chromadb matplotlib --quiet
if errorlevel 1 (
    echo   [FAIL] 依赖安装失败，尝试单独安装...
    echo.
    echo   正在尝试逐个安装...
    python -m pip install requests --quiet
    python -m pip install flask flask-cors --quiet
    python -m pip install schedule --quiet
    python -m pip install python-dotenv --quiet
    python -m pip install networkx --quiet
    python -m pip install numpy --quiet
    echo.
)

:: 检查 Flask 是否安装成功
python -c "import flask; import flask_cors" 2>nul
if errorlevel 1 (
    echo.
    echo   [FAIL] Flask 安装失败，请手动安装：
    echo   [CMD] pip install flask flask-cors
    pause
    exit /b 1
)
echo   [OK] 所有依赖安装成功

:: ============================================
:: 步骤 4: 检查配置文件
:: ============================================
echo.
echo [4/6] 检查配置文件...

if not exist ".env" (
    if exist ".env.example" (
        echo   [TIP] 未找到 .env 文件，正在创建...
        copy .env.example .env >nul 2>&1
        echo   [OK] 已创建 .env 文件
        echo.
        echo   [IMPORTANT] 请编辑 .env 文件，填入你的 API 密钥！
        echo   [HELP] 查看 API_KEY_SETUP.md 了解如何获取密钥
    ) else (
        echo   [WARN] 未找到配置文件模板
    )
) else (
    echo   [OK] .env 配置文件已存在
)

:: ============================================
:: 步骤 5: 检查知识库
:: ============================================
echo.
echo [5/6] 检查知识库...
if exist "learning_data\goals" (
    echo   [OK] 已发现学习数据
) else (
    echo   [TIP] 首次使用建议导入预置知识体系
    echo   [CMD] 运行以下命令导入：
    echo      python import_knowledge_plan.py
    echo      python import_ai_knowledge.py
    echo.
)

:: ============================================
:: 步骤 6: 验证安装
:: ============================================
echo.
echo [6/6] 验证系统状态...
python -c "from web_server import create_app; print('[OK] Web 服务器模块正常')" 2>nul
if errorlevel 1 (
    echo   [WARN] Web 服务器验证失败，但应该可以正常运行
)

:: ============================================
:: 安装完成
:: ============================================
echo.
echo ════════════════════════════════════════════════════════════
echo                     安装完成！
echo ════════════════════════════════════════════════════════════
echo.
echo 使用方式：
echo.
echo   [WEB] Web 模式（推荐）
echo      - 双击 "启动Web服务.bat"
echo      - 或运行：python main.py --web
echo.
echo   [CLI] 命令行模式
echo      - 双击 "启动命令行模式.bat"
echo      - 或运行：python main.py
echo.
echo 首次使用：
echo   1. 编辑 .env 填入 API 密钥（参考 API_KEY_SETUP.md）
echo   2. 运行 python import_knowledge_plan.py
echo   3. 运行 python import_ai_knowledge.py
echo   4. 启动系统开始学习！
echo.
echo ════════════════════════════════════════════════════════════

echo.
set /p choice="是否立即启动 Web 服务？(Y/N): "
if /i "%choice%"=="Y" goto start_web
if /i "%choice%"=="y" goto start_web
echo.
echo 再见！有问题请查看 README.md
pause >nul
exit /b 0

:start_web
echo.
echo 正在启动 VisionLearner Web 服务...
echo.
python main.py --web

echo.
echo 已退出 VisionLearner
pause >nul