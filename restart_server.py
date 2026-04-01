#!/usr/bin/env python3
"""
启动 Web 服务器（使用最新代码）
"""
import sys
import time

def start_server():
    print("="*60)
    print("启动 VisionLearner Web 服务器")
    print("="*60)

    from main import LearningSystem
    from web_server import run_server

    print("\n[1] 初始化 LearningSystem...")
    system = LearningSystem()
    print("[OK] 系统初始化完成\n")

    print("[2] 启动 Flask 服务器...")
    run_server(system, port=5000, host="0.0.0.0", debug=False)

if __name__ == "__main__":
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n\n[INFO] 服务器已停止")
    except Exception as e:
        print(f"\n[ERROR] 服务器启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
