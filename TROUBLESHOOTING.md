# VisionLearner 故障排除指南

## 问题：切换目标时出现500错误

### 可能原因和解决方案

#### 1. 浏览器缓存问题 ⚠️ 最常见

**症状**：
- 点击目标切换按钮无反应
- 浏览器控制台显示500错误
- API日志显示GBK编码错误

**解决方案**：
1. **清除浏览器缓存**
   - Chrome/Edge: 按 `Ctrl + Shift + Delete`
   - 选择"缓存的图片和文件"
   - 点击"清除数据"

2. **强制刷新页面**
   - 按 `Ctrl + F5` (Windows)
   - 或 `Cmd + Shift + R` (Mac)

3. **清除特定站点缓存**
   - 打开浏览器开发者工具 (F12)
   - 右键点击刷新按钮
   - 选择"清空缓存并硬性重新加载"

#### 2. 端口冲突

**症状**：
- Web服务器启动失败
- 提示端口被占用

**解决方案**：
```bash
# Windows: 查找占用5000端口的进程
netstat -ano | findstr :5000

# 终止进程（将PID替换为实际进程ID）
taskkill /PID <进程ID> /F

# 或使用其他端口启动
python main.py --web --port 5001
```

#### 3. 数据损坏

**症状**：
- 所有目标都无法切换
- 加载进度时出错

**解决方案**：

**方法A：重置系统状态**
```bash
# 删除系统状态文件（不删除学习数据）
del learning_data\system\state.json

# 重新启动
python main.py --web
```

**方法B：恢复备份**
```bash
# 如果有备份，恢复备份
copy backup\* learning_data\
```

#### 4. 编码问题

**症状**：
- 控制台显示GBK编码错误
- 中文输出乱码

**解决方案**：
确保所有Python文件使用UTF-8编码保存。如果使用VSCode：
1. 打开文件
2. 右下角查看编码
3. 点击并选择"通过编码保存"
4. 选择UTF-8

### 完整重启流程

如果以上方法都无效，请按以下步骤操作：

```bash
# 1. 停止所有Python进程
taskkill /F /IM python.exe

# 2. 清理浏览器缓存
# (在浏览器中操作)

# 3. 重置系统状态
del learning_data\system\state.json

# 4. 重新启动服务器
python main.py --web --port 5000

# 5. 打开浏览器并强制刷新
# 访问 http://localhost:5000
# 按 Ctrl + F5 强制刷新
```

### 调试技巧

#### 查看浏览器控制台

1. 按 `F12` 打开开发者工具
2. 切换到 "Console" 标签
3. 查看是否有红色错误信息
4. 注意API调用的时间戳和错误详情

#### 查看服务器日志

在命令行中启动服务器时，注意查看输出：
- `[INFO]` - 正常信息
- `[ERROR]` - 错误信息
- `500` - HTTP 500错误

#### 测试API

使用curl或Postman测试API：

```bash
# 测试状态
curl http://localhost:5000/api/status

# 测试目标列表
curl http://localhost:5000/api/goals

# 测试目标切换
curl -X POST http://localhost:5000/api/goal/select \
  -H "Content-Type: application/json" \
  -d "{\"goal_id\": \"goal_xxx\"}"
```

### 联系支持

如果问题仍然存在，请提供以下信息：
1. 完整的错误消息
2. 浏览器控制台的截图
3. 服务器启动日志
4. 使用的操作系统和浏览器版本

---

**最后更新**: 2026-03-06
**版本**: v4.0
