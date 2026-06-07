@echo off
echo 设置环境变量...

:: 设置 DeepSeek API Key
setx DEEPSEEK_API_KEY "你的DeepSeek API Key"
echo DEEPSEEK_API_KEY 已设置

:: 设置 Qwen/DashScope API Key（用于 Embedding）
setx DASHSCOPE_API_KEY "你的DashScope API Key"
echo DASHSCOPE_API_KEY 已设置

:: 设置其他可选环境变量
setx DEEPSEEK_BASE_URL "https://api.deepseek.com/v1"
echo DEEPSEEK_BASE_URL 已设置

setx DEEPSEEK_MODEL "deepseek-chat"
echo DEEPSEEK_MODEL 已设置

echo.
echo 环境变量设置完成！
echo 请重启命令提示符或 PyCharm 使设置生效。
pause