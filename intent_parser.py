# intent_parser.py — LLM驱动的意图解析器
"""
替代 SentenceParser，用 LLM 做意图识别。

优点：
  - 不需要任何规则，自然语言直接理解
  - 代词消解、模糊匹配、多语言全部自动处理
  - 只需维护 prompt，不需要维护代码

token 消耗：
  - 每次提问额外消耗约 200-400 token（unit列表+问题）
  - 用 haiku/flash 等小模型可以极低成本
  - 如果命中缓存则 0 消耗
"""

import json
import hashlib
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ParsedIntent:
    """LLM解析出的意图结构"""
    unit: str           # 最相关的 unit 名（从知识库列表中选）
    focus: str          # 用户想了解的具体方面
    type: str           # query / quiz / progress / general
    lang: str           # zh / en / mixed
    raw: str            # 原始输入
    from_cache: bool = False  # 是否来自缓存


class LLMIntentParser:
    """
    用 LLM 做意图解析。
    
    用法：
        parser = LLMIntentParser(llm_client)
        parser.set_units(["认知革命：七万年前...", "农业革命：..."])
        intent = parser.parse("工业革命怎么发生的")
        # intent.unit = "工业革命：能源跃升如何改变人类时间感"
        # intent.focus = "工业革命的起因和发生过程"
    """

    def __init__(self, llm):
        self.llm = llm
        self._units: List[str] = []
        self._current_topic: Optional[str] = None  # 代词消解用
        self._cache: Dict[str, ParsedIntent] = {}  # 简单内存缓存

    def set_units(self, units: List[str]):
        """设置当前知识库的unit列表"""
        self._units = units
        self._cache.clear()  # unit列表变了，缓存失效

    def set_topic(self, topic: Optional[str]):
        """设置当前话题（用于代词消解）"""
        self._current_topic = topic

    def parse(self, user_input: str) -> ParsedIntent:
        """
        解析用户输入，返回结构化意图。
        先查缓存，没有再调LLM。
        """
        # ── 缓存查找 ──
        cache_key = self._make_cache_key(user_input)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.from_cache = True
            return cached

        # ── 先做规则快筛（0 token，处理明显的系统命令）──
        quick = self._quick_match(user_input)
        if quick:
            return quick

        # ── 调 LLM 做意图解析 ──
        result = self._llm_parse(user_input)
        self._cache[cache_key] = result
        return result

    # ───────────────────────────────────────────────────────────
    # 规则快筛（处理不需要LLM的明显意图）
    # ───────────────────────────────────────────────────────────

    def _quick_match(self, text: str) -> Optional[ParsedIntent]:
        """0 token 快速识别系统命令"""
        t = text.strip().lower()

        # 进度查询
        progress_kw = ['进度', '学了多少', '完成了多少', 'progress', '百分之']
        if any(k in t for k in progress_kw):
            return ParsedIntent(unit='', focus='', type='progress',
                                lang='zh', raw=text)

        # 测验请求
        quiz_kw = ['测试我', '考考我', '出题', 'quiz', '测验', '做题']
        if any(k in t for k in quiz_kw):
            return ParsedIntent(unit='', focus='', type='quiz',
                                lang='zh', raw=text)

        return None

    # ───────────────────────────────────────────────────────────
    # LLM 解析核心
    # ───────────────────────────────────────────────────────────

    def _llm_parse(self, user_input: str) -> ParsedIntent:
        """调LLM做意图解析，返回结构化结果"""

        units_text = "\n".join(f"- {u}" for u in self._units) if self._units else "（暂无）"

        # 代词消解提示
        topic_hint = ""
        if self._current_topic:
            topic_hint = f"\n注意：用户上一轮话题是「{self._current_topic}」，如果当前问题用了代词（它/这个/此），请解析为该话题。"

        prompt = f"""你是一个意图解析器。分析用户问题，返回JSON，不要说其他任何话。

当前知识库单元列表：
{units_text}
{topic_hint}

用户问题：「{user_input}」

返回格式（严格JSON，不加markdown）：
{{
  "unit": "从列表中选最相关的单元名（必须完整复制），如果没有相关单元则返回null",
  "focus": "用户想了解的具体方面，10字以内",
  "type": "query",
  "lang": "zh"
}}

type只能是：query（提问）/ general（闲聊）
lang只能是：zh / en / mixed"""

        system = "你是意图解析器，只返回JSON，绝对不说其他任何话。"

        try:
            raw = self.llm.chat(prompt, system=system)
            # 清理可能的markdown包裹
            raw = raw.strip()
            if raw.startswith('```'):
                raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]

            data = json.loads(raw)

            unit = data.get('unit') or ''
            # 验证unit是否真的在列表里（防止LLM瞎编）
            if unit and unit not in self._units:
                unit = self._fuzzy_match_unit(unit)

            return ParsedIntent(
                unit=unit,
                focus=data.get('focus', ''),
                type=data.get('type', 'query'),
                lang=data.get('lang', 'zh'),
                raw=user_input,
            )

        except Exception as e:
            log.warning(f"LLM意图解析失败: {e}, raw={raw if 'raw' in dir() else '?'}")
            # 降级：用原始输入做向量检索兜底
            return ParsedIntent(
                unit='', focus=user_input, type='query',
                lang='zh', raw=user_input,
            )

    def _fuzzy_match_unit(self, llm_unit: str) -> str:
        """
        LLM返回的unit不在列表里时，做模糊匹配。
        （防止LLM返回截断或改写的unit名）
        """
        llm_lower = llm_unit.lower()
        best, best_score = '', 0
        for u in self._units:
            u_lower = u.lower()
            # 包含关系
            if llm_lower in u_lower or u_lower in llm_lower:
                score = len(llm_lower) / max(len(u_lower), 1)
                if score > best_score:
                    best_score, best = score, u
        return best if best_score > 0.3 else ''

    def _make_cache_key(self, text: str) -> str:
        key = f"{text}|{self._current_topic}|{len(self._units)}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
