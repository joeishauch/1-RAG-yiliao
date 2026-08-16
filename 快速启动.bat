@echo off
REM 以脚本所在目录为工作目录（无需改绝对路径）
cd /d "%~dp0"
python app_launcher.py
