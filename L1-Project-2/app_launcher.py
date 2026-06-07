import sys
import time
import subprocess
import threading
import webbrowser
import signal
from pathlib import Path

# 项目路径
PROJECT_PATH = Path(__file__).parent
PYTHON_PATH = r"D:\python\conda\envs\xiangmu-2\python.exe"

# 服务进程
api_process = None
webui_process = None

def start_api_service():
    """启动 API 服务"""
    global api_process
    try:
        api_process = subprocess.Popen(
            [PYTHON_PATH, "main.py"],
            cwd=str(PROJECT_PATH),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        print(f"API 服务已启动 (PID: {api_process.pid})")
    except Exception as e:
        print(f"启动 API 服务失败: {e}")

def start_webui_service():
    """启动 Web UI 服务"""
    global webui_process
    try:
        webui_process = subprocess.Popen(
            [PYTHON_PATH, "webUI.py"],
            cwd=str(PROJECT_PATH),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        print(f"Web UI 服务已启动 (PID: {webui_process.pid})")
    except Exception as e:
        print(f"启动 Web UI 服务失败: {e}")

def stop_services():
    """停止所有服务"""
    global api_process, webui_process
    
    print("正在停止服务...")
    
    if api_process and api_process.poll() is None:
        api_process.terminate()
        try:
            api_process.wait(timeout=5)
        except:
            api_process.kill()
        print("API 服务已停止")
    
    if webui_process and webui_process.poll() is None:
        webui_process.terminate()
        try:
            webui_process.wait(timeout=5)
        except:
            webui_process.kill()
        print("Web UI 服务已停止")

def signal_handler(sig, frame):
    """处理退出信号"""
    print("\n收到退出信号...")
    stop_services()
    sys.exit(0)

def main():
    print("=" * 50)
    print("智能分诊系统 - 服务启动器")
    print("=" * 50)
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 启动 API 服务
        print("\n[1/2] 正在启动 API 服务...")
        start_api_service()
        time.sleep(8)
        
        # 启动 Web UI 服务
        print("[2/2] 正在启动 Web UI 服务...")
        start_webui_service()
        time.sleep(3)
        
        # 打开浏览器
        print("\n服务启动完成！")
        print("API: http://localhost:8012")
        print("Web UI: http://localhost:7860")
        print("\n正在打开浏览器...")
        webbrowser.open("http://localhost:7860")
        
        print("\n按 Ctrl+C 停止服务")
        print("=" * 50)
        
        # 保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n")
    finally:
        stop_services()

if __name__ == "__main__":
    main()
