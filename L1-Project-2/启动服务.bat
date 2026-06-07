@echo off
echo ========================================
echo 智能分诊系统 - 服务启动脚本
echo ========================================
echo.

cd /d "D:\ai\聚客\02聚客AI大模型第六期\10-项目2_基于LangGraph实现智能分诊系统\项目2_基于LangGraph实现智能分诊系统\L1-Project-2"

echo [1/2] 正在启动 API 服务（端口 8012）...
start "API 服务" cmd /k "D:\python\conda\envs\xiangmu-2\python.exe main.py"

timeout /t 5 /nobreak >nul

echo [2/2] 正在启动 Web UI 服务（端口 7860）...
start "Web UI 服务" cmd /k "D:\python\conda\envs\xiangmu-2\python.exe webUI.py"

echo.
echo ========================================
echo 服务启动完成！
echo API: http://localhost:8012
echo Web UI: http://localhost:7860
echo ========================================
pause
