@echo off
chcp 65001 > nul
echo Starting VisionLearner Web Server...
cd /d "%~dp0"
python restart_server.py
pause
