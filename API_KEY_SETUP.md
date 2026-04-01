# API 密钥配置指南

## 🚨 当前问题

系统报错：`401 Client Error: Unauthorized for url: https://api.deepseek.com/v1/chat/completions`

**原因**：`.env` 文件中的 API 密钥被替换成了占位符，系统找不到有效的 API 密钥。

---

## ✅ 解决方案

### 方式1：配置讯飞星火（推荐，国内免费）

1. 注册账号：https://console.xfyun.cn/
2. 创建应用，获取 API Password
3. 编辑 `.env` 文件，填入你的密钥：

```bash
# ── 讯飞星火（已配置，使用此）────────────────────────────
SPARK_API_PASSWORD=你的讯飞星火API密码
SPARK_API_URL=https://spark-api-open.xf-yun.com/v1/chat/completions
```

4. 重启系统：
```bash
python main.py --web
```

### 方式2：配置 DeepSeek

1. 注册账号：https://platform.deepseek.com/
2. 获取 API Key
3. 编辑 `.env` 文件：

```bash
DEEPSEEK_API_KEY=sk-你的deepseek密钥
```

4. 重启系统

### 方式3：配置豆包

1. 注册账号：https://www.volcengine.com/
2. 获取 API Key
3. 编辑 `.env` 文件：

```bash
DOUBAO_API_KEY=你的豆包API密钥
```

4. 重启系统

---

## 📝 .env 文件模板

创建或编辑 `.env` 文件：

```bash
# VisionLearner v4.0 配置

# ── 讯飞星火（推荐，国内免费）────────────────────────────
SPARK_API_PASSWORD=你的API密码格式:apikey
SPARK_API_URL=https://spark-api-open.xf-yun.com/v1/chat/completions

# ── 国内LLM（豆包）────────────────────────────
DOUBAO_API_KEY=你的豆包API密钥

# ── 国际LLM（可选）──────────────────────────────────
# OPENROUTER_API_KEY=你的OpenRouter密钥
DEEPSEEK_API_KEY=sk-你的deepseek密钥
# OPENAI_API_KEY=你的OpenAI密钥

# ── 免费云端LLM（暂时禁用）─────────────────────────
# GROQ_API_KEY=你的Groq密钥
# GEMINI_API_KEY=你的Gemini密钥

# ── 国内免费LLM（暂时禁用）──────────────────────────
# SILICONFLOW_API_KEY=你的硅基流动密钥

# ── Telegram Bot（方向A，可选）───────────────────────
TELEGRAM_BOT_TOKEN=你的Bot Token
TELEGRAM_CHAT_ID=你的Chat ID

# ── 系统配置（可选）─────────────────────────────────
# DATA_DIR=./learning_data
# SKILLS_DIR=./skills
```

---

## 🔧 LLM 优先级

系统会按以下顺序自动选择 LLM：

1. **讯飞星火** (SPARK_API_PASSWORD) - 国内免费，推荐
2. **豆包** (DOUBAO_API_KEY) - 国内
3. **DeepSeek** (DEEPSEEK_API_KEY) - 国际
4. **OpenRouter** (OPENROUTER_API_KEY) - 国际
5. **OpenAI** (OPENAI_API_KEY) - 国际

---

## ✅ 验证配置

配置完成后，重启系统，查看日志：

```bash
python main.py --web
```

应该看到类似输出：

```
Auto-detecting available LLM...
   OK Found 讯飞星火（国内免费） API Key
OK LLM client initialized: 讯飞星火（国内免费） / lite
```

如果看到：

```
❌ LLM调用失败: 401 Client Error: Unauthorized
```

说明 API 密钥无效或过期，请检查：

1. 密钥是否正确复制（没有多余空格）
2. 密钥是否已过期或被禁用
3. API 账户是否有余额

---

## 💡 快速测试

测试 LLM 是否配置成功：

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('SPARK_API_PASSWORD:', os.getenv('SPARK_API_PASSWORD')[:10] + '...' if os.getenv('SPARK_API_PASSWORD') else 'Not configured')
print('DEEPSEEK_API_KEY:', os.getenv('DEEPSEEK_API_KEY')[:10] + '...' if os.getenv('DEEPSEEK_API_KEY') else 'Not configured')
"
```

---

## 🆘 常见问题

### Q: 讯飞星火的 API Password 格式是什么？

A: 格式是 `appid:apikey`，例如：
```
12345678:abcdef1234567890abcdef
```

### Q: 系统还是报 401 错误怎么办？

A: 检查以下几点：
1. 确认 `.env` 文件在项目根目录
2. 确认密钥没有多余空格或换行
3. 确认密钥格式正确
4. 检查 API 账户是否正常

### Q: 可以同时配置多个 LLM 吗？

A: 可以！系统会自动选择第一个可用的 LLM。建议配置讯飞星火作为主要 LLM。

---

## 📞 获取帮助

如果还有问题，可以：

1. 查看系统日志：`python main.py --web`
2. 检查 `.env` 文件内容
3. 查看 `TROUBLESHOOTING.md`

---

**重要提示**：为了安全，请勿将 `.env` 文件提交到 GitHub 或其他公开仓库！
