@echo off
setlocal
REM 以脚本所在目录为工作目录（无需改绝对路径）
cd /d "%~dp0"

REM Python 解释器：改为你自己的 python 路径
set "PYTHON=D:\python\conda\envs\xiangmu-2\python.exe"

echo ========================================
echo 智能分诊系统 - 服务启动脚本
echo ========================================
echo.

echo [1/2] 正在启动 API 服务（端口 8012）...
start "API 服务" cmd /k "%PYTHON% main.py"

timeout /t 5 /nobreak >nul

echo [2/2] 正在启动 Web UI 服务（端口 7860）...
start "Web UI 服务" cmd /k "%PYTHON% webUI.py"

echo.
echo ========================================
echo 服务启动完成！
echo API: http://localhost:8012
echo Web UI: http://localhost:7860
echo ========================================
pause
