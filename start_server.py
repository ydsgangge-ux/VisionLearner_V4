import sys
sys.stdout.reconfigure(encoding='utf-8')

print("正在启动 VisionLearner 服务器...")
print("Python版本:", sys.version.split()[0])

try:
    print("\n1. 导入 main 模块...")
    import main
    print("   ✓ main 模块导入成功")
except Exception as e:
    print(f"   ✗ main 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n2. 创建 LearningSystem 实例...")
    system = main.LearningSystem(data_dir="./learning_data")
    print("   ✓ LearningSystem 创建成功")
except Exception as e:
    print(f"   ✗ LearningSystem 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. 创建 Flask 应用...")
    from web_server import create_app
    app = create_app(system)
    print("   ✓ Flask 应用创建成功")
except Exception as e:
    print(f"   ✗ Flask 应用创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("服务器启动成功!")
print("="*50)
print("访问地址: http://localhost:5000")
print("按 Ctrl+C 停止服务器")
print("="*50 + "\n")

# 启动服务器
try:
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
except KeyboardInterrupt:
    print("\n\n服务器已停止")
