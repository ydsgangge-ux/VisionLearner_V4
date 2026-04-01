#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速配置 API 密钥
"""

from pathlib import Path
import os

def setup_api_keys():
    """交互式配置 API 密钥"""

    print("\n" + "=" * 60)
    print("  VisionLearner API 密钥配置")
    print("=" * 60)

    env_path = Path(".env")

    # 读取现有配置
    existing_config = {}
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    existing_config[key] = value

    print("\n选择要配置的 LLM：")
    print("1. 讯飞星火（推荐，国内免费）")
    print("2. DeepSeek（国际）")
    print("3. 豆包（国内）")
    print("4. 查看当前配置")
    print("0. 退出")

    choice = input("\n请输入选项（0-4）: ").strip()

    if choice == "0":
        print("退出。")
        return
    elif choice == "1":
        # 讯飞星火
        print("\n--- 讯飞星火配置 ---")
        print("1. 注册：https://console.xfyun.cn/")
        print("2. 格式：appid:apikey（例如：12345678:abcdef123456）")

        current_value = existing_config.get("SPARK_API_PASSWORD", "")
        if current_value and not current_value.startswith("your_"):
            print(f"当前配置：{current_value[:10]}...")
        else:
            print("当前未配置")

        new_value = input("请输入 API Password（留空保持当前）: ").strip()

        if new_value:
            existing_config["SPARK_API_PASSWORD"] = new_value
            existing_config["SPARK_API_URL"] = "https://spark-api-open.xf-yun.com/v1/chat/completions"
            print("✅ 讯飞星火已配置")

    elif choice == "2":
        # DeepSeek
        print("\n--- DeepSeek 配置 ---")
        print("1. 注册：https://platform.deepseek.com/")
        print("2. 格式：sk-xxxxx（例如：sk-1234567890abcdef）")

        current_value = existing_config.get("DEEPSEEK_API_KEY", "")
        if current_value and not current_value.startswith("your_"):
            print(f"当前配置：{current_value[:10]}...")
        else:
            print("当前未配置")

        new_value = input("请输入 API Key（留空保持当前）: ").strip()

        if new_value:
            if not new_value.startswith("sk-"):
                new_value = f"sk-{new_value}"
            existing_config["DEEPSEEK_API_KEY"] = new_value
            print("✅ DeepSeek 已配置")

    elif choice == "3":
        # 豆包
        print("\n--- 豆包配置 ---")
        print("1. 注册：https://www.volcengine.com/")
        print("2. 获取 API Key")

        current_value = existing_config.get("DOUBAO_API_KEY", "")
        if current_value and not current_value.startswith("your_"):
            print(f"当前配置：{current_value[:10]}...")
        else:
            print("当前未配置")

        new_value = input("请输入 API Key（留空保持当前）: ").strip()

        if new_value:
            existing_config["DOUBAO_API_KEY"] = new_value
            print("✅ 豆包已配置")

    elif choice == "4":
        # 查看当前配置
        print("\n--- 当前配置 ---")
        print(f"讯飞星火: {'已配置' if existing_config.get('SPARK_API_PASSWORD') and not existing_config.get('SPARK_API_PASSWORD', '').startswith('your_') else '未配置'}")
        print(f"DeepSeek: {'已配置' if existing_config.get('DEEPSEEK_API_KEY') and not existing_config.get('DEEPSEEK_API_KEY', '').startswith('your_') else '未配置'}")
        print(f"豆包: {'已配置' if existing_config.get('DOUBAO_API_KEY') and not existing_config.get('DOUBAO_API_KEY', '').startswith('your_') else '未配置'}")
        return

    # 写入配置
    if choice in ["1", "2", "3"]:
        env_content = """# VisionLearner v4.0 配置

# ── 讯飞星火（已配置，使用此）────────────────────────────
SPARK_API_PASSWORD={}
SPARK_API_URL=https://spark-api-open.xf-yun.com/v1/chat/completions

# ── 国内LLM（豆包）────────────────────────────
DOUBAO_API_KEY={}

# ── 国际LLM（可选）──────────────────────────────────
# OPENROUTER_API_KEY=你的OpenRouter密钥
DEEPSEEK_API_KEY={}
# OPENAI_API_KEY=你的OpenAI密钥

# ── 免费云端LLM（暂时禁用）─────────────────────────
# GROQ_API_KEY=你的Groq密钥
# GEMINI_API_KEY=你的Gemini密钥

# ── 国内免费LLM（暂时禁用）──────────────────────────
# SILICONFLOW_API_KEY=你的硅基流动密钥

# ── Telegram Bot（方向A，可选）───────────────────────
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# ── 系统配置（可选）─────────────────────────────────
# DATA_DIR=./learning_data
# SKILLS_DIR=./skills
""".format(
            existing_config.get("SPARK_API_PASSWORD", "your_password_here"),
            existing_config.get("DOUBAO_API_KEY", "your_doubao_api_key_here"),
            existing_config.get("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")
        )

        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)

        print(f"\n✅ 配置已保存到 {env_path}")
        print("\n下一步：")
        print("1. 重启系统：python main.py --web")
        print("2. 查看日志确认 LLM 已加载")

if __name__ == "__main__":
    setup_api_keys()
