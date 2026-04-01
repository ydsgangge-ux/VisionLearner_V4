# conversation.py — v3.0  持久化上下文（治本版）
"""
治本方案：彻底解决上下文理解问题。

v2.0 的根本问题：
  1. dialog_log 是内存列表，重启清空，跨会话无记忆
  2. 对话历史只在层4（LLM兜底）才传给LLM，层2/层3完全无上下文
  3. 没有置信度、来源引用、后续问题推荐

v3.0 改进：
  1. 对话历史持久化到 SQLite（重启不丢失）
  2. get_context_for_llm() 供所有层统一注入上下文
  3. search_history() 跨会话 FTS 全文搜索历史对话
  4. DialogueTurn 增加 confidence / source / references / follow_up
  5. generate_follow_up() 每次回答后推荐后续问题

对外接口变化（完全向下兼容）：
  ctx.process(user_input)           → 不变
  ctx.update(user_input, resp, ent) → 不变，内部新增持久化
  ctx.set_goal(goal_id, goal_type)  → 不变
  ctx.get_dialog_context_text(n)    → 不变

新增接口：
  ctx.get_context_for_llm(n)        → 标准 messages 列表（所有层用这个）
  ctx.search_history(query)         → 跨会话 FTS 搜索历史对话
  ctx.get_follow_up_questions()     → 获取推荐后续问题

main.py 需要两处改动：
  1. __init__：ConversationContext() → ConversationContext(db=self.db)
  2. 每个 ctx.update() 加上 source / confidence / follow_up 参数
     （不加也没问题，有默认值，完全兼容）
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

from sentence_parser import SentenceParser, ParseResult

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class DialogueTurn:
    """一轮完整对话记录"""
    user_input:  str
    response:    str
    entity:      str = ""
    query_type:  str = "general"
    confidence:  float = 1.0        # 回答置信度（0~1）
    source:      str = ""           # local / vector / llm / skill
    references:  List[str] = field(default_factory=list)  # 引用的知识节点
    follow_up:   List[str] = field(default_factory=list)  # 推荐后续问题
    timestamp:   str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


@dataclass
class Intent:
    """对话意图（与旧版完全兼容）"""
    # 旧版字段
    type:          str
    subtype:       str = ""
    entity:        str = ""
    raw:           str = ""
    # 新增字段
    search_query:  str = ""
    dimension:     str = ""
    dimension_raw: str = ""
    lang:          str = "zh"
    parse_result:  Any = None


# ══════════════════════════════════════════════════════════════
# 对话上下文（治本版）
# ══════════════════════════════════════════════════════════════

class ConversationContext:
    """
    对话上下文管理器 v3.0

    关键改变：
      - 传入 db=DataManager 后自动持久化对话到 SQLite
      - 不传 db 则降级为纯内存模式（与旧版行为完全一致）
    """

    MAX_MEMORY_TURNS = 20   # 内存最多保留的轮数

    def __init__(self, db=None):
        """
        db: DataManager 实例（可选）
            传入后开启持久化，不传则纯内存模式。
        """
        self._db = db
        self._session_id = f"sess_{int(datetime.now().timestamp())}"

        # 当前状态
        self.current_topic:     Optional[str] = None
        self.current_goal_id:   Optional[str] = None
        self.current_goal_type: str = "general"
        self.recent_entities:   List[str] = []

        # 内存对话历史
        self._turns: List[DialogueTurn] = []

        # 最近一轮推荐的后续问题
        self._last_follow_up: List[str] = []

        # SentenceParser（代词消解）
        self._parser = SentenceParser()

        # 重启后恢复最近历史
        if self._db:
            self._restore_recent()

    # ──────────────────────────────────────────────────────────
    # 核心接口（与旧版完全兼容）
    # ──────────────────────────────────────────────────────────

    def process(self, user_input: str) -> Intent:
        """解析用户输入，返回 Intent（接口不变）"""
        self._parser.set_topic(self.current_topic)
        pr: ParseResult = self._parser.parse(user_input)
        return Intent(
            type=pr.intent_type,
            subtype=pr.dimension,
            entity=pr.object,
            raw=user_input,
            search_query=pr.search_query,
            dimension=pr.dimension,
            dimension_raw=pr.dimension_raw,
            lang=pr.lang,
            parse_result=pr,
        )

    def update(self, user_input: str, response: str,
               entity: str = "",
               query_type: str = "general",
               confidence: float = 1.0,
               source: str = "",
               references: List[str] = None,
               follow_up: List[str] = None):
        """
        对话结束后更新上下文状态。

        旧版调用方式完全兼容：
            ctx.update(user_input, response, entity)

        新增可选参数（不传有默认值）：
            source      — 回答来源（local/vector/llm/skill）
            confidence  — 置信度（0~1）
            references  — 引用的知识节点列表
            follow_up   — 推荐后续问题列表
        """
        follow_up  = follow_up  or []
        references = references or []

        # 更新话题栈
        if entity:
            self.current_topic = entity
            if entity in self.recent_entities:
                self.recent_entities.remove(entity)
            self.recent_entities.insert(0, entity)
            self.recent_entities = self.recent_entities[:10]

        # 记录到内存
        turn = DialogueTurn(
            user_input=user_input,
            response=response[:500],
            entity=entity,
            query_type=query_type,
            confidence=confidence,
            source=source,
            references=references,
            follow_up=follow_up,
        )
        self._turns.append(turn)
        if len(self._turns) > self.MAX_MEMORY_TURNS:
            self._turns = self._turns[-self.MAX_MEMORY_TURNS:]

        self._last_follow_up = follow_up

        # 持久化到 SQLite
        if self._db:
            try:
                self._db.storage.save_dialog(
                    session_id=self._session_id,
                    goal_id=self.current_goal_id or "",
                    role="user",
                    content=user_input,
                    entity=entity,
                )
                self._db.storage.save_dialog(
                    session_id=self._session_id,
                    goal_id=self.current_goal_id or "",
                    role="assistant",
                    content=response[:500],
                    entity=entity,
                )
            except Exception as e:
                log.warning(f"对话持久化失败（不影响功能）：{e}")

    def set_goal(self, goal_id: str, goal_type: str):
        """切换学习目标，重置话题（接口不变）"""
        self.current_goal_id   = goal_id
        self.current_goal_type = goal_type
        self.current_topic     = None

    # ──────────────────────────────────────────────────────────
    # 上下文获取（新增 + 旧版兼容）
    # ──────────────────────────────────────────────────────────

    def get_context_for_llm(self, n: int = 5) -> List[Dict]:
        """
        【新增——治本核心】
        返回标准 messages 列表，供所有层统一注入上下文。

        用法（在 _answer_auto 的层2/层3/层4 都加上）：
            history = self.ctx.get_context_for_llm(n=3)
            messages = history + [{"role": "user", "content": prompt}]
            ans = self.llm.chat_messages(messages, system=system)

        这样层2/层3命中知识库后，LLM 在组织回答时也能看到上下文，
        代词（它/这个/上面说的）自然就能正确理解了。
        """
        recent = self._turns[-n:] if self._turns else []
        messages = []
        for t in recent:
            messages.append({"role": "user",      "content": t.user_input})
            messages.append({"role": "assistant",  "content": t.response})
        return messages

    def get_dialog_context_text(self, n: int = 3) -> str:
        """返回近期对话纯文本（旧版接口，保持兼容）"""
        if not self._turns:
            return ""
        lines = []
        for t in self._turns[-n:]:
            lines.append(f"用户：{t.user_input}")
            lines.append(f"助手：{t.response[:150]}")
        return "\n".join(lines)

    def get_recent_dialog(self, n: int = 5) -> List[Dict]:
        """返回最近n轮 messages 格式（旧版接口兼容）"""
        return self.get_context_for_llm(n)

    def get_context_summary(self) -> str:
        """返回上下文摘要（旧版接口兼容）"""
        return (
            f"话题:{self.current_topic or '无'} | "
            f"近期:{self.recent_entities[:3]} | "
            f"目标:{self.current_goal_id or '无'}"
        )

    # ──────────────────────────────────────────────────────────
    # 跨会话搜索（新增）
    # ──────────────────────────────────────────────────────────

    def search_history(self, query: str,
                       limit: int = 5) -> List[Dict]:
        """
        【新增】跨会话 FTS5 全文搜索历史对话。

        例子：
          "我上次问过RAG是什么吗"  → 找到历史记录
          "之前学的认知革命内容"   → 找到相关对话

        返回：[{"role", "content", "entity", "created_at", "session_id"}]
        """
        if not self._db:
            return []
        try:
            return self._db.storage.search_dialog(
                query=query,
                goal_id=self.current_goal_id or "",
                limit=limit,
            )
        except Exception as e:
            log.warning(f"历史搜索失败：{e}")
            return []

    def get_follow_up_questions(self) -> List[str]:
        """【新增】获取上一轮推荐的后续问题"""
        return self._last_follow_up

    def get_turn_count(self) -> int:
        """当前会话已对话轮数"""
        return len(self._turns)

    # ──────────────────────────────────────────────────────────
    # 内部：重启后恢复
    # ──────────────────────────────────────────────────────────

    def _restore_recent(self):
        """从 SQLite 恢复最近对话到内存（重启后调用）"""
        try:
            rows = self._db.storage.get_recent_dialog(
                session_id=self._session_id,
                limit=self.MAX_MEMORY_TURNS * 2,
            )
            i = 0
            while i < len(rows) - 1:
                u = rows[i]
                a = rows[i + 1]
                if (u.get("role") == "user" and
                        a.get("role") == "assistant"):
                    self._turns.append(DialogueTurn(
                        user_input=u["content"],
                        response=a["content"],
                        entity=u.get("entity", ""),
                    ))
                    i += 2
                else:
                    i += 1
        except Exception:
            pass   # 静默失败，降级为空历史


# ══════════════════════════════════════════════════════════════
# 工具函数（保持与旧版完全兼容）
# ══════════════════════════════════════════════════════════════

def format_content(content: Any) -> str:
    """把知识节点的 content 格式化为可读字符串"""
    if content is None:
        return "（暂无）"
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "、".join(str(c) for c in content) if content else "（暂无）"
    if isinstance(content, dict):
        return "\n".join(
            f"{k}：{format_content(v)}" for k, v in content.items()
        )
    return str(content)


def compose_answer(intent: Intent, node_title: str,
                   content: Any, unit: str, goal_type: str) -> str:
    """把知识库内容组装成回答字符串（旧版接口兼容）"""
    fmt = format_content(content)
    if goal_type == "characters":
        labels = {
            "reading": f"「{unit}」读作：{fmt}",
            "strokes": f"「{unit}」{fmt}",
            "meaning": f"「{unit}」的意思：{fmt}",
            "usage":   f"「{unit}」{node_title}：{fmt}",
            "memory":  f"「{unit}」记忆方法：{fmt}",
        }
        return labels.get(intent.subtype,
                          f"「{unit}」的{node_title}：{fmt}")
    return (
        f"【{unit}】{node_title}：{fmt}"
        if intent.subtype else
        f"【{unit}】\n{fmt}"
    )


# ══════════════════════════════════════════════════════════════
# 后续问题生成（从上传文件借鉴，改造集成）
# ══════════════════════════════════════════════════════════════

def generate_follow_up(entity: str, source: str = "",
                       goal_type: str = "general") -> List[str]:
    """
    根据当前回答生成推荐后续问题。
    在 main.py 的 ctx.update() 时传入 follow_up 参数。

    用法：
        follow_up = generate_follow_up(unit, source="local")
        self.ctx.update(user_input, answer, unit,
                        source="local", follow_up=follow_up)
    """
    if not entity:
        return []

    if goal_type == "characters":
        return [
            f"「{entity}」怎么用在句子里？",
            f"「{entity}」有哪些近义词？",
            f"「{entity}」的起源是什么？",
        ]

    if source == "local":
        return [
            f"{entity}的具体例子有哪些？",
            f"{entity}和哪些概念相关？",
            f"可以测试我对{entity}的理解吗？",
        ]

    if source == "llm":
        return [
            f"{entity}的核心原理是什么？",
            f"能举一个{entity}的实际案例吗？",
            f"把{entity}存入我的知识库",
        ]

    return [
        f"{entity}还有哪些方面值得了解？",
        f"可以测试我对{entity}的掌握程度吗？",
    ]
