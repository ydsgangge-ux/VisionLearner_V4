# llm_client.py - 统一大模型客户端（免费优先）
"""
支持的免费/低成本LLM：
1. Ollama       - 本地运行，完全免费（推荐首选）
2. Groq         - 免费tier，速度极快（llama3, gemma2, mixtral）
3. Google Gemini- 免费tier，每分钟60次请求
4. OpenRouter   - 聚合平台，有大量免费模型
5. DeepSeek     - 低价，中文能力强
6. 豆包/Doubao  - 国内，有免费额度
7. OpenAI       - 备用，需付费

优先级：Ollama > Groq > Gemini > OpenRouter > DeepSeek > Doubao > OpenAI
"""

import json
import os
import time
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, field
from datetime import datetime

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 如果没有安装python-dotenv，直接跳过


# ========== 配置 ==========

@dataclass
class LLMProviderConfig:
    name: str
    base_url: str
    api_key_env: str          # 环境变量名
    default_model: str
    free_models: List[str]    # 免费/低成本模型列表
    max_tokens: int = 4096
    supports_stream: bool = True
    is_local: bool = False     # 是否本地运行（无需key）
    notes: str = ""
    timeout: int = 60         # 请求超时时间（秒），本地模型可能需要更长

PROVIDERS: Dict[str, LLMProviderConfig] = {
    "ollama": LLMProviderConfig(
        name="Ollama（本地）",
        base_url="http://localhost:11434/v1",
        api_key_env="",
        default_model="qwen2.5:7b",
        free_models=["qwen2.5:7b", "qwen2.5:14b", "llama3.2:3b", "llama3.1:8b",
                     "deepseek-r1:7b", "gemma2:9b", "mistral:7b", "phi3.5:3.8b"],
        is_local=True,
        timeout=300,  # 本地模型可能需要更长时间（5分钟）
        notes="需要先安装Ollama: https://ollama.ai，然后 ollama pull qwen2.5:7b"
    ),
    "groq": LLMProviderConfig(
        name="Groq（免费tier）",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="llama-3.1-8b-instant",
        free_models=["llama-3.1-8b-instant", "llama-3.3-70b-versatile",
                     "gemma2-9b-it", "mixtral-8x7b-32768", "deepseek-r1-distill-llama-70b"],
        notes="免费tier：每天免费，速度极快。注册: https://console.groq.com"
    ),
    "gemini": LLMProviderConfig(
        name="Google Gemini（免费tier）",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-2.0-flash",
        free_models=["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"],
        notes="免费tier：每分钟60次请求。注册: https://aistudio.google.com"
    ),
    "openrouter": LLMProviderConfig(
        name="OpenRouter（含免费模型）",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="meta-llama/llama-3.1-8b-instruct:free",
        free_models=[
            "meta-llama/llama-3.1-8b-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "qwen/qwen-2.5-7b-instruct:free",
            "deepseek/deepseek-r1:free",
        ],
        notes="聚合平台，大量免费模型。注册: https://openrouter.ai"
    ),
    "deepseek": LLMProviderConfig(
        name="DeepSeek（低价）",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
        free_models=["deepseek-chat", "deepseek-reasoner"],
        notes="中文能力强，价格极低。注册: https://platform.deepseek.com"
    ),
    "doubao": LLMProviderConfig(
        name="豆包（国内）",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key_env="DOUBAO_API_KEY",
        default_model="doubao-seed-1-8-251228",
        free_models=["doubao-seed-1-8-251228", "doubao-pro-4k", "doubao-lite-4k"],
        notes="字节跳动，国内访问稳定。注册: https://console.volcengine.com/ark。支持多模态图片识别"
    ),
    "spark": LLMProviderConfig(
        name="讯飞星火（国内免费）",
        base_url="https://spark-api-open.xf-yun.com/v1",
        api_key_env="SPARK_API_PASSWORD",
        default_model="lite",
        free_models=["lite", "generalv3", "pro-128k", "4.0Ultra"],
        notes="讯飞星火，国内免费。注册: https://console.xfyun.cn，配置 SPARK_API_PASSWORD 即可"
    ),
    "siliconflow": LLMProviderConfig(
        name="硅基流动（国内免费）",
        base_url="https://api.siliconflow.cn/v1",
        api_key_env="SILICONFLOW_API_KEY",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        free_models=["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
                     "deepseek-ai/DeepSeek-V2.5", "THUDM/glm-4-9b-chat",
                     "meta-llama/Meta-Llama-3.1-8B-Instruct"],
        notes="硅基流动，中文最强，注册送额度。注册: https://siliconflow.cn"
    ),
    "openai": LLMProviderConfig(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        free_models=["gpt-4o-mini"],
        notes="需付费，但效果最好"
    ),
}

# 自动检测顺序（免费优先）
AUTO_DETECT_ORDER = ["ollama", "spark", "doubao", "siliconflow", "groq", "gemini", "openrouter", "deepseek", "openai"]


# ========== 缓存层 ==========

class LLMCache:
    """磁盘缓存，避免重复调用"""

    def __init__(self, cache_dir: str = "./cache/llm"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, provider: str, model: str, messages: List[Dict]) -> str:
        content = json.dumps({"p": provider, "m": model, "msgs": messages}, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, provider: str, model: str, messages: List[Dict]) -> Optional[str]:
        path = self.cache_dir / f"{self._key(provider, model, messages)}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data.get("response")
            except Exception:
                return None
        return None

    def set(self, provider: str, model: str, messages: List[Dict], response: str) -> None:
        path = self.cache_dir / f"{self._key(provider, model, messages)}.json"
        try:
            path.write_text(
                json.dumps({"provider": provider, "model": model,
                            "response": response, "cached_at": datetime.now().isoformat()},
                           ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass


# ========== 主客户端 ==========

class LLMClient:
    """
    统一LLM客户端 - 免费优先，自动降级

    使用示例：
        client = LLMClient()                    # 自动选择可用provider
        client = LLMClient(provider="groq")     # 指定provider
        client = LLMClient(provider="ollama", model="qwen2.5:7b")

        response = client.chat("解释一下机器学习")
        mindmap = client.generate_mindmap("Python编程基础")
    """

    def __init__(self,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.7,
                 use_cache: bool = True):

        self.temperature = temperature
        self.cache = LLMCache() if use_cache else None
        self._stats = {"calls": 0, "cache_hits": 0, "errors": 0, "tokens_used": 0}

        # 选择provider
        if provider:
            self.provider_name = provider
            self.config = PROVIDERS.get(provider, PROVIDERS["ollama"])
        else:
            self.provider_name, self.config = self._auto_detect()

        # 选择模型
        self.model = model or self.config.default_model

        print(f"OK LLM client initialized: {self.config.name} / {self.model}")
        if self.config.is_local:
            print(f"   Tip - Local execution, completely free")
        else:
            print(f"   Tip - {self.config.notes}")

    def _auto_detect(self) -> tuple:
        """自动检测可用的provider（免费优先）"""
        print("Auto-detecting available LLM...")

        for name in AUTO_DETECT_ORDER:
            config = PROVIDERS[name]

            if config.is_local:
                # Ollama：检测本地服务
                try:
                    r = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if r.status_code == 200:
                        models = [m["name"] for m in r.json().get("models", [])]
                        if models:
                            # 找到最优先的已安装模型
                            for preferred in config.free_models:
                                if preferred in models:
                                    print(f"   OK Found {config.name} with model {preferred}")
                                    return name, config
                except Exception:
                    pass

            else:
                # 云端API：检测 API Key
                key = os.getenv(config.api_key_env, "")
                if key:
                    print(f"   OK Found {config.name} API Key")
                    return name, config

        # 降级到mock
        print("   WARNING - No LLM detected, using mock mode (configure real LLM recommended)")
        return "mock", LLMProviderConfig(
            name="Mock", base_url="", api_key_env="",
            default_model="mock", free_models=["mock"], is_local=True,
            notes="No real LLM, returns demo content"
        )

    def _get_api_key(self) -> str:
        if self.config.is_local or not self.config.api_key_env:
            return "ollama"  # ollama不需要真实key，但openai兼容接口需要非空
        return os.getenv(self.config.api_key_env, "")

    def chat(self,
             prompt: str,
             system: str = "",
             max_tokens: int = 2048,
             use_cache: bool = True) -> str:
        """
        基础对话接口

        Args:
            prompt: 用户输入
            system: 系统提示
            max_tokens: 最大token数
            use_cache: 是否使用缓存

        Returns:
            模型回复文本
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self._call(messages, max_tokens=max_tokens, use_cache=use_cache)

    def chat_with_history(self,
                          messages: List[Dict[str, str]],
                          max_tokens: int = 2048) -> str:
        """带历史记录的多轮对话"""
        return self._call(messages, max_tokens=max_tokens, use_cache=False)

    def chat_with_tools(self,
                     messages: List[Dict],
                     tools: List[Dict],
                     system: str = "",
                     max_tokens: int = 2048) -> Dict:
        """
        支持工具调用的对话（OpenAI Function Calling 格式）

        Args:
            messages: 消息列表（可能包含 tool_calls）
            tools: 工具列表（OpenAI 格式）
            system: 系统提示
            max_tokens: 最大token数

        Returns:
            {"content": str, "tool_calls": List[Dict]}
        """
        import logging
        log = logging.getLogger(__name__)

        # 添加 system 消息
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # 尝试用 OpenAI 兼容格式调用
        try:
            payload = {
                "model": self.model,
                "messages": full_messages,
                "tools": tools if tools else None,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }

            response = self._http_call_with_tools(payload)

            # 解析返回
            if not response:
                return {"content": "", "tool_calls": []}

            choices = response.get("choices", [])
            if not choices:
                return {"content": "", "tool_calls": []}

            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            # 兼容不同提供商的格式
            if tool_calls:
                return {"content": content, "tool_calls": tool_calls}
            else:
                return {"content": content, "tool_calls": []}

        except Exception as e:
            log.warning(f"chat_with_tools 失败: {e}")
            # 降级到普通对话
            content = self._call(full_messages, max_tokens=max_tokens, use_cache=False)
            return {"content": content, "tool_calls": []}

    def chat_with_image(self,
                        prompt: str,
                        image_b64: str,
                        system: str = "",
                        max_tokens: int = 2048) -> str:
        """
        多模态对话 - 支持图片识别

        Args:
            prompt: 用户输入
            image_b64: base64编码的图片数据
            system: 系统提示
            max_tokens: 最大token数

        Returns:
            模型回复文本
        """
        print(f"\n{'='*60}")
        print(f"[DEBUG] ===== 图片识别开始 =====")
        print(f"[DEBUG] 当前 provider: {self.provider_name}")
        print(f"[DEBUG] 当前 model: {self.model}")
        print(f"[DEBUG] 图片数据大小: {len(image_b64)} 字节")
        print(f"[DEBUG] prompt: {prompt[:100]}...")
        print(f"{'='*60}\n")

        # 豆包使用特殊的API格式
        if self.provider_name == "doubao":
            print("[DEBUG] 使用豆包 provider 进行图片识别")
            result = self._call_doubao_image(prompt, image_b64, system, max_tokens)
            print(f"[DEBUG] 豆包识别完成，结果长度: {len(result)} 字符")
            return result

        # 如果当前provider不是豆包但豆包可用,自动切换到豆包进行图片识别
        if self.provider_name != "doubao" and os.getenv("DOUBAO_API_KEY"):
            print(f"[INFO] 当前provider '{self.provider_name}' 不支持图片识别，自动切换到豆包进行图片识别")
            original_provider = self.provider_name
            original_model = self.model
            try:
                # 临时切换到豆包
                doubao_config = PROVIDERS["doubao"]
                self.provider_name = "doubao"
                self.config = doubao_config
                self.model = doubao_config.default_model
                print(f"[DEBUG] 已切换到豆包: {self.model}")
                result = self._call_doubao_image(prompt, image_b64, system, max_tokens)
                print(f"[DEBUG] 豆包识别完成，结果长度: {len(result)} 字符")
                print(f"[INFO] 图片识别完成，恢复原 provider: {original_provider}")
                return result
            finally:
                # 恢复原provider
                self.provider_name = original_provider
                self.config = PROVIDERS[original_provider]
                self.model = original_model

        # 其他provider不支持图片识别,使用OpenAI格式尝试
        print(f"[WARN] provider '{self.provider_name}' 不支持图片识别，尝试 OpenAI 格式")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        # 构建多模态消息（OpenAI格式）
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
        messages.append({"role": "user", "content": content})

        result = self._call(messages, max_tokens=max_tokens, use_cache=False)
        print(f"[DEBUG] OpenAI 格式调用完成，结果长度: {len(result)} 字符")
        return result

    def _call_doubao_image(self, prompt: str, image_b64: str,
                           system: str, max_tokens: int) -> str:
        """调用豆包的多模态API（使用responses.create）"""
        api_key = self._get_api_key()
        print(f"[DEBUG] 豆包 API Key: {api_key[:10]}...")
        print(f"[DEBUG] 豆包 base_url: {self.config.base_url}")
        print(f"[DEBUG] 豆包 model: {self.model}")

        # 使用OpenAI SDK调用豆包的多模态API
        try:
            from openai import OpenAI
            print("[DEBUG] OpenAI SDK 导入成功")

            client = OpenAI(
                base_url=self.config.base_url,
                api_key=api_key,
            )
            print("[DEBUG] OpenAI client 创建成功")

            # 构建内容（顺序很重要：先图片后文本）
            content = [
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_b64}"
                },
                {
                    "type": "input_text",
                    "text": prompt
                }
            ]
            print(f"[DEBUG] 构建请求内容完成")

            print(f"[DEBUG] 开始调用 client.responses.create()...")
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            print(f"[DEBUG] API 调用成功")

            # 解析响应
            if hasattr(response, 'output') and response.output:
                print(f"[DEBUG] 响应包含 output 字段，条目数: {len(response.output)}")
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and isinstance(item.content, list):
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    result = content_item.text
                                    print(f"[DEBUG] 成功提取文本，长度: {len(result)}")
                                    return result

            raise Exception("无法解析响应数据")

        except ImportError:
            # 如果没有安装openai库，使用requests降级方案
            print("[WARN] OpenAI SDK 未安装，使用 requests 降级方案")
            return self._call_doubao_image_fallback(prompt, image_b64, system, max_tokens)
        except Exception as e:
            print(f"[ERROR] 豆包 API 调用失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"豆包图片识别失败: {str(e)}")

    def _call_doubao_image_fallback(self, prompt: str, image_b64: str,
                                    system: str, max_tokens: int) -> str:
        """使用requests降级方案（当OpenAI SDK不可用时）"""
        api_key = self._get_api_key()

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}"
        }

        # 构建豆包特有的多模态内容格式（顺序很重要：先图片后文本）
        content = [
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            {"type": "input_text", "text": prompt}
        ]

        payload = {
            "model": self.model,
            "input": [
                {"role": "user", "content": content}
            ]
        }

        url = f"{self.config.base_url}/responses"

        session = requests.Session()
        try:
            r = session.post(url, headers=headers, json=payload, timeout=self.config.timeout, verify=True)

            if r.status_code != 200:
                raise Exception(f"HTTP {r.status_code}: {r.text}")

            data = r.json()

            # 解析responses.create的响应格式
            if "output" in data:
                for item in data["output"]:
                    if item.get("type") == "message":
                        for content_item in item.get("content", []):
                            if "text" in content_item:
                                return content_item["text"]

            raise Exception("无法解析响应数据")

        except Exception as e:
            raise Exception(f"豆包图片识别失败（降级方案）: {str(e)}")

    def generate_json(self,
                      prompt: str,
                      system: str = "",
                      schema_hint: str = "",
                      max_tokens: int = 3000) -> Optional[Dict]:
        """
        生成JSON结构化输出

        Args:
            prompt: 描述你要什么
            system: 系统提示
            schema_hint: JSON结构提示（如 '{"title": "...", "nodes": [...]}' ）
        """
        full_system = (system + "\n\n" if system else "") + \
            "请严格以JSON格式回复，不要包含任何markdown代码块标记或其他文字。"
        if schema_hint:
            full_system += f"\n期望的JSON结构示例：{schema_hint}"

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt}
        ]

        raw = self._call(messages, max_tokens=max_tokens)

        # 清洗JSON
        cleaned = self._extract_json(raw)
        if cleaned:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

        # 二次尝试：让模型修正
        fix_prompt = f"以下内容解析JSON失败，请只返回合法的JSON，不要有任何其他文字：\n{raw}"
        raw2 = self._call([{"role": "user", "content": fix_prompt}], max_tokens=max_tokens, use_cache=False)
        cleaned2 = self._extract_json(raw2)
        if cleaned2:
            try:
                return json.loads(cleaned2)
            except Exception:
                pass

        return None

    def generate_mindmap(self,
                         topic: str,
                         depth: int = 3,
                         style: str = "balanced",
                         context: str = "") -> Optional[Dict]:
        """
        生成思维导图JSON结构

        Returns:
            {
              "title": "主题",
              "description": "简介",
              "nodes": [
                {
                  "id": "1",
                  "title": "子主题",
                  "description": "说明",
                  "importance": 0.8,
                  "difficulty": 0.5,
                  "node_type": "concept",
                  "estimated_minutes": 30,
                  "children": [...]
                }
              ]
            }
        """
        style_guide = {
            "balanced": "广度和深度均衡，每层3-5个节点",
            "deep": "深度优先，每条分支尽量深入，节点少但层次多",
            "broad": "广度优先，第一层多分支（6-8个），层次浅",
            "structured": "高度结构化，类似目录层级",
        }.get(style, "广度和深度均衡")

        system = """你是一个专业的知识结构分析师，擅长将复杂主题分解为清晰的思维导图。
生成的思维导图要：
1. 层次清晰，逻辑严密
2. 覆盖主题的核心内容
3. 每个节点的importance(0-1)和difficulty(0-1)要合理
4. node_type从以下选择: concept, skill, example, practice, principle, fact
5. estimated_minutes表示学习该节点大约需要的分钟数
只返回JSON，不要有任何其他内容。"""

        schema = '''{
  "title": "主题名",
  "description": "主题简介",
  "nodes": [
    {
      "id": "1",
      "title": "子主题1",
      "description": "描述",
      "importance": 0.9,
      "difficulty": 0.4,
      "node_type": "concept",
      "estimated_minutes": 30,
      "children": [
        {
          "id": "1-1",
          "title": "具体知识点",
          "description": "描述",
          "importance": 0.7,
          "difficulty": 0.5,
          "node_type": "concept",
          "estimated_minutes": 20,
          "children": []
        }
      ]
    }
  ]
}'''

        prompt = f"""为以下主题生成{depth}层深度的思维导图：

主题：{topic}
深度要求：{depth}层
风格：{style_guide}
{f'补充背景：{context}' if context else ''}

请生成完整的思维导图JSON结构。"""

        result = self.generate_json(prompt, system=system, schema_hint=schema, max_tokens=4000)
        return result

    def generate_questions(self,
                           topic: str,
                           count: int = 5,
                           difficulty: str = "mixed",
                           node_type: str = "concept") -> List[Dict]:
        """
        生成学习问题

        Returns: List of {"question": "...", "type": "...", "difficulty": "easy/medium/hard", "hint": "..."}
        """
        difficulty_guide = {
            "easy": "基础理解题，考查定义和基本概念",
            "medium": "分析应用题，考查理解和实际应用",
            "hard": "综合创新题，考查深度理解和创造性思维",
            "mixed": "混合难度，包含简单、中等、困难各占1/3",
        }.get(difficulty, "混合难度")

        prompt = f"""为以下主题生成{count}道学习问题：

主题：{topic}
节点类型：{node_type}
难度要求：{difficulty_guide}

返回JSON数组，格式：
[
  {{
    "question": "问题内容",
    "type": "concept/application/analysis/creation",
    "difficulty": "easy/medium/hard",
    "hint": "提示（可选）",
    "answer_points": ["要点1", "要点2"]
  }}
]"""

        result = self.generate_json(prompt, max_tokens=2000)
        if isinstance(result, list):
            return result
        return []

    def evaluate_answer(self,
                        question: str,
                        answer: str,
                        topic: str = "") -> Dict:
        """评估学习回答质量"""
        prompt = f"""评估以下学习问答：

主题：{topic}
问题：{question}
学生回答：{answer}

请评估并返回JSON：
{{
  "score": 0.85,  // 0-1分
  "mastery_level": "understanding",  // exposure/familiarity/understanding/application/mastery
  "strengths": ["做得好的地方"],
  "gaps": ["知识盲点或不足"],
  "suggestions": ["改进建议"],
  "next_topics": ["建议继续学习的相关主题"]
}}"""

        result = self.generate_json(prompt, max_tokens=1000)
        if result:
            return result
        return {"score": 0.5, "mastery_level": "familiarity", "strengths": [], "gaps": [], "suggestions": []}

    def summarize_and_extract(self, text: str, max_length: int = 200) -> Dict:
        """从文本中提取知识点和摘要"""
        prompt = f"""分析以下文本，提取关键知识点：

文本：
{text[:3000]}

返回JSON：
{{
  "summary": "核心摘要（{max_length}字以内）",
  "key_concepts": ["概念1", "概念2"],
  "knowledge_points": [
    {{
      "title": "知识点名称",
      "content": "具体内容",
      "type": "concept/fact/principle/skill",
      "importance": 0.8
    }}
  ],
  "tags": ["标签1", "标签2"],
  "difficulty": 0.5
}}"""

        result = self.generate_json(prompt, max_tokens=2000)
        if result:
            return result
        return {"summary": text[:max_length], "key_concepts": [], "knowledge_points": [], "tags": []}

    def _call(self,
              messages: List[Dict],
              max_tokens: int = 2048,
              use_cache: bool = True) -> str:
        """底层调用，带缓存和重试"""

        # Mock模式
        if self.provider_name == "mock":
            return self._mock_response(messages)

        # 检查缓存
        if use_cache and self.cache:
            cached = self.cache.get(self.provider_name, self.model, messages)
            if cached:
                self._stats["cache_hits"] += 1
                return cached

        # 真实调用（带重试）
        last_error = None
        for attempt in range(3):
            try:
                response = self._http_call(messages, max_tokens)
                self._stats["calls"] += 1

                # 存缓存
                if use_cache and self.cache:
                    self.cache.set(self.provider_name, self.model, messages, response)

                return response

            except Exception as e:
                last_error = e
                self._stats["errors"] += 1
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 指数退避

        # 所有重试失败
        print(f"❌ LLM调用失败: {last_error}")
        return f"[LLM调用失败: {str(last_error)[:100]}]"

    def _http_call(self, messages: List[Dict], max_tokens: int) -> str:
        """实际HTTP请求"""
        api_key = self._get_api_key()

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}"
        }

        # OpenRouter额外headers
        if self.provider_name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/learning-system"
            headers["X-Title"] = "Autonomous Learning System"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        url = f"{self.config.base_url}/chat/completions"
        # 使用 Session 来确保正确的编码处理
        session = requests.Session()
        # 启用 SSL 证书验证（修复 SSL 警告）
        # 如果遇到 SSL 证书问题，可以考虑使用系统证书或下载证书
        try:
            r = session.post(url, headers=headers, json=payload, timeout=self.config.timeout, verify=True)
        except requests.exceptions.SSLError:
            # 如果 SSL 证书验证失败，记录警告并继续（保持向后兼容）
            import warnings
            from urllib3.exceptions import InsecureRequestWarning
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
            r = session.post(url, headers=headers, json=payload, timeout=self.config.timeout, verify=False)
        r.raise_for_status()

        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _http_call_with_tools(self, payload: Dict) -> Dict:
        """支持工具调用的 HTTP 请求"""
        import logging
        log = logging.getLogger(__name__)

        headers = {
            "Content-Type": "application/json",
        }

        # API Key
        if self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env)
            if api_key:
                if "volces" in self.config.base_url:  # 豆包
                    headers["Authorization"] = f"Bearer {api_key}"
                elif "xf-yun" in self.config.base_url:  # 讯飞
                    headers["Authorization"] = f"Bearer {api_key}"
                elif "siliconflow" in self.config.base_url:  # 硅基流动
                    headers["Authorization"] = f"Bearer {api_key}"
                else:  # OpenAI 兼容
                    headers["Authorization"] = f"Bearer {api_key}"

        # 讯飞星火特殊处理
        if "xf-yun" in self.config.base_url:
            # 讯飞 API 需要不同的认证方式
            password = os.getenv(self.config.api_key_env, "")
            headers["Authorization"] = password

        url = f"{self.config.base_url}/chat/completions"

        # 创建 session（复用连接）
        session = requests.Session()

        try:
            r = session.post(url, headers=headers, json=payload, timeout=self.config.timeout, verify=True)
        except requests.exceptions.SSLError:
            # 如果 SSL 证书验证失败，记录警告并继续（保持向后兼容）
            import warnings
            from urllib3.exceptions import InsecureRequestWarning
            warnings.filterwarnings("ignore", category=InsecureRequestWarning)
            r = session.post(url, headers=headers, json=payload, timeout=self.config.timeout, verify=False)
        r.raise_for_status()

        return r.json()

    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取JSON"""
        if not text:
            return None

        # 直接是JSON
        text = text.strip()
        if (text.startswith("{") and text.endswith("}")) or \
           (text.startswith("[") and text.endswith("]")):
            return text

        # 提取```json ... ``` 代码块
        import re
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",
            r"```\s*([\s\S]*?)\s*```",
            r"(\{[\s\S]*\})",
            r"(\[[\s\S]*\])",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                candidate = m.group(1).strip()
                try:
                    json.loads(candidate)
                    return candidate
                except Exception:
                    continue

        return None

    def _mock_response(self, messages: List[Dict]) -> str:
        """Mock响应，用于无LLM时的演示"""
        last_msg = messages[-1]["content"] if messages else ""

        if "思维导图" in last_msg or "mindmap" in last_msg.lower():
            return json.dumps({
                "title": "示例主题",
                "description": "这是一个演示思维导图（Mock模式）",
                "nodes": [
                    {
                        "id": "1", "title": "核心概念", "description": "基础理论",
                        "importance": 0.9, "difficulty": 0.4,
                        "node_type": "concept", "estimated_minutes": 30,
                        "children": [
                            {"id": "1-1", "title": "子概念A", "description": "详细内容",
                             "importance": 0.7, "difficulty": 0.3,
                             "node_type": "concept", "estimated_minutes": 20, "children": []}
                        ]
                    },
                    {
                        "id": "2", "title": "实践技能", "description": "动手操作",
                        "importance": 0.8, "difficulty": 0.6,
                        "node_type": "skill", "estimated_minutes": 60,
                        "children": []
                    }
                ]
            }, ensure_ascii=False)

        if "问题" in last_msg or "question" in last_msg.lower():
            return json.dumps([
                {"question": "这个概念的核心定义是什么？",
                 "type": "concept", "difficulty": "easy",
                 "hint": "思考最基本的定义", "answer_points": ["定义要点1", "定义要点2"]},
                {"question": "如何在实际场景中应用这个概念？",
                 "type": "application", "difficulty": "medium",
                 "hint": "结合具体案例", "answer_points": ["应用场景", "注意事项"]}
            ], ensure_ascii=False)

        return "这是Mock模式的演示回复。请配置真实的LLM以获得完整功能。\n\n建议：\n1. 安装Ollama（免费本地运行）\n2. 或配置Groq API Key（免费云端）"


    def stream(self,
               prompt: str,
               system: str = "",
               max_tokens: int = 2048):
        """
        流式响应（打字机效果）
        用法：
            for chunk in client.stream("你好"):
                print(chunk, end="", flush=True)
        """
        if self.provider_name == "mock":
            text = "这是演示流式响应。"
            for ch in text:
                import time; time.sleep(0.05)
                yield ch
            return

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        api_key = self._get_api_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        if self.provider_name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/learning-system"
            headers["X-Title"] = "VisionLearner"

        # 讯飞 model 名处理
        model = self.model
        if self.provider_name == "spark" and not model.startswith("spark-"):
            pass  # lite / generalv3 等直接用

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        url = f"{self.config.base_url}/chat/completions"
        try:
            with requests.post(url, headers=headers, json=payload,
                               stream=True, timeout=self.config.timeout, verify=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
        except Exception as e:
            yield f"[流式响应错误: {e}]"

    def call_llm(self, prompt: str, system_prompt: str = "",
                 max_tokens: int = 2048, temperature: float = None) -> str:
        """兼容原版 perception.py 的调用接口"""
        old_temp = self.temperature
        if temperature is not None:
            self.temperature = temperature
        result = self.chat(prompt, system=system_prompt, max_tokens=max_tokens)
        self.temperature = old_temp
        return result

    def get_stats(self) -> Dict:
        """获取调用统计"""
        return {**self._stats, "provider": self.provider_name, "model": self.model}

    def switch_provider(self, provider: str, model: Optional[str] = None) -> bool:
        """运行时切换provider"""
        if provider not in PROVIDERS:
            print(f"[X] 未知provider: {provider}")
            return False
        
        self.provider_name = provider
        self.config = PROVIDERS[provider]
        
        # 对于 Ollama，如果未指定模型，自动检测已安装的模型
        if provider == "ollama" and not model:
            try:
                r = requests.get("http://localhost:11434/api/tags", timeout=2)
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                    if models:
                        # 使用第一个已安装模型
                        model = models[0]
                        print(f"[INFO] 检测到 Ollama 模型: {model}")
            except Exception:
                pass
        
        self.model = model or self.config.default_model
        print(f"[OK] 已切换到 {self.config.name} / {self.model}")
        return True

    def list_available_providers(self) -> List[Dict]:
        """列出所有可用provider"""
        result = []
        for name, config in PROVIDERS.items():
            available = False
            if config.is_local:
                try:
                    r = requests.get("http://localhost:11434/api/tags", timeout=1)
                    available = r.status_code == 200
                except Exception:
                    pass
            else:
                available = bool(os.getenv(config.api_key_env, ""))

            result.append({
                "name": name,
                "display_name": config.name,
                "available": available,
                "is_free": config.is_local or name in ["groq", "gemini", "openrouter"],
                "default_model": config.default_model,
                "free_models": config.free_models,
                "notes": config.notes
            })
        return result


# ========== 便捷函数 ==========

_global_client: Optional[LLMClient] = None

def get_client(provider: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    """获取全局LLM客户端（单例）"""
    global _global_client
    if _global_client is None or provider is not None:
        _global_client = LLMClient(provider=provider, model=model)
    return _global_client


# ========== CLI测试 ==========

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("🤖 LLM客户端测试")
    print("=" * 60)

    # 列出可用provider
    client = LLMClient()
    providers = client.list_available_providers()

    print("\n📋 Provider状态：")
    for p in providers:
        status = "✅ 可用" if p["available"] else "❌ 未配置"
        free = "🆓 免费" if p["is_free"] else "💰 付费"
        print(f"  {status} {free} {p['display_name']} ({p['name']})")
        if not p["available"] and p["name"] != "mock":
            print(f"         → {p['notes']}")

    print("\n🧪 测试思维导图生成...")
    mindmap = client.generate_mindmap("Python数据分析", depth=2)
    if mindmap:
        print(f"✅ 生成成功: {mindmap.get('title')}")
        print(f"   顶层节点: {len(mindmap.get('nodes', []))}个")
    else:
        print("[!] Generation failed (check LLM config)")

    print("\n🧪 测试问题生成...")
    questions = client.generate_questions("机器学习基础", count=3)
    print(f"✅ 生成了{len(questions)}道问题")
    for q in questions:
        print(f"   [{q.get('difficulty', '?')}] {q.get('question', '')[:50]}")

    print("\n[%] Call Statistics:")
    stats = client.get_stats()
    print(f"  实际调用: {stats['calls']}次")
    print(f"  缓存命中: {stats['cache_hits']}次")
    print(f"  错误次数: {stats['errors']}次")
