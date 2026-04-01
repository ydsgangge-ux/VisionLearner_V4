# tool_registry.py — 工具注册与 Agent 执行系统 v1.0
"""
让 LLM 通过 Function Calling 决定调用哪个工具，完成复合任务。

核心组件：
  Tool          — 单个工具定义
  ToolRegistry  — 工具注册表
  AgentLoop     — ReAct 执行循环（思考→选工具→执行→观察→再思考）

接入 main.py 只需三步：

  步骤1：在 __init__ 末尾加：
    from tool_registry import build_registry, AgentLoop
    self.registry = build_registry(self)
    self.agent = AgentLoop(self.llm, self.registry)

  步骤2：在 answer() 里加一个 agent 分支：
    if mode == "agent":
        return self.agent.run(user_input, self.ctx.current_goal_id, stream)

  步骤3：完成，其他代码不动。

用法示例：
  system.answer("认知革命是什么")                    # 原有逻辑，不变
  system.answer("学完认知革命然后测试我", mode="agent")  # Agent模式
  system.answer("我有哪些薄弱点", mode="agent")         # Agent模式
"""

import json
import logging
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class ToolParam:
    name: str
    type: str           # string / integer / boolean
    description: str
    required: bool = True
    enum: List[str] = field(default_factory=list)


@dataclass
class Tool:
    name: str
    description: str
    params: List[ToolParam]
    fn: Callable

    def to_schema(self) -> Dict:
        """转为 OpenAI Function Calling 格式"""
        props = {}
        required = []
        for p in self.params:
            prop: Dict = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            props[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }

    def execute(self, **kwargs) -> Any:
        return self.fn(**kwargs)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    result: Any
    error: str = ""

    def to_text(self) -> str:
        if not self.success:
            return f"执行失败：{self.error}"
        if isinstance(self.result, (dict, list)):
            return json.dumps(self.result, ensure_ascii=False, indent=2)
        return str(self.result) if self.result is not None else "（已执行，无返回内容）"


# ══════════════════════════════════════════════════════════════
# 工具注册表
# ══════════════════════════════════════════════════════════════

class ToolRegistry:

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        log.info(f"[ToolRegistry] 注册工具：{tool.name}")
        return self

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all_schemas(self) -> List[Dict]:
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool_name=name, success=False, result=None,
                error=f"工具 '{name}' 不存在，可用：{list(self._tools.keys())}"
            )
        try:
            result = tool.execute(**kwargs)
            return ToolResult(tool_name=name, success=True, result=result)
        except Exception as e:
            log.error(f"[Tool:{name}] 执行失败：{e}\n{traceback.format_exc()}")
            return ToolResult(tool_name=name, success=False,
                              result=None, error=str(e))

    def summary(self) -> str:
        lines = [f"已注册 {len(self._tools)} 个工具："]
        for t in self._tools.values():
            lines.append(f"  • {t.name}：{t.description[:60]}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 工具构建：把 LearningSystem 现有能力包装成工具
# ══════════════════════════════════════════════════════════════

def build_registry(system) -> ToolRegistry:
    """
    把 LearningSystem 的现有功能包装成工具。
    system = LearningSystem 实例（即 self）
    """
    reg = ToolRegistry()

    # ── 工具1：搜索本地知识库 ──
    def search_knowledge(query: str, goal_id: str = "",
                         top_k: int = 5) -> List[Dict]:
        gid = goal_id or (system.ctx.current_goal_id or "")
        results = system.vector.search(query=query, goal_id=gid, top_k=top_k)
        if not results:
            return [{"message": "本地知识库未找到相关内容"}]
        return [{"unit": r["unit"],
                 "content": r["content"][:300],
                 "score": r["score"]} for r in results]

    reg.register(Tool(
        name="search_knowledge",
        description="从本地知识库语义检索相关知识，适合回答「某概念是什么/为什么/怎么做」",
        params=[
            ToolParam("query", "string", "检索关键词或完整问题"),
            ToolParam("goal_id", "string", "限定搜索范围的目标ID，不填则搜全库",
                      required=False),
            ToolParam("top_k", "integer", "返回结果数量，默认5", required=False),
        ],
        fn=search_knowledge,
    ))

    # ── 工具2：查看学习进度 ──
    def get_progress(goal_id: str = "") -> Dict:
        gid = goal_id or (system.ctx.current_goal_id or "")
        if not gid:
            goals = system.db.load_all_goals()
            return {
                "all_goals": [
                    {"id": g["id"],
                     "description": g.get("description", ""),
                     "status": g.get("status", "")}
                    for g in goals[:10]
                ]
            }
        units = system.col.load_goal_units(gid)
        if not units:
            return {"error": f"目标 {gid} 未找到单元列表"}
        report = system.col.get_completion_report(gid, units)
        return {
            "goal_id": gid,
            "total_units": len(units),
            "completion_rate": report.get("completion_rate", 0),
            "completed_units": report.get("completed_units", [])[:20],
            "pending_units": report.get("pending_units", units)[:20],
        }

    reg.register(Tool(
        name="get_progress",
        description="查看学习进度，包括完成率、已完成单元、待学单元",
        params=[
            ToolParam("goal_id", "string",
                      "学习目标ID，不填则返回所有目标概览", required=False),
        ],
        fn=get_progress,
    ))

    # ── 工具3：主动学习一个单元 ──
    def learn_unit(unit: str, goal_id: str = "",
                   focus: str = "") -> str:
        gid = goal_id or (system.ctx.current_goal_id or "")
        if not gid:
            return "错误：未指定学习目标"
        goal_type = system._goal_type.get(gid, "general")
        tree = system._get_tree(gid, unit, goal_type)
        if not tree:
            return f"未找到单元「{unit}」的知识树，请先建立该单元"
        node_title = focus or unit
        node, content = system.col.collect_on_demand(
            node_title, tree, goal_type, unit, goal_id=gid)
        system.col.save_tree(gid, unit, tree)
        system._trees.setdefault(gid, {})[unit] = tree
        if node and node.collected and content:
            system._sync_node_to_vector(gid, unit, node_title, content)
            from conversation import format_content
            return (f"已学习「{unit}」的{node_title}：\n"
                    f"{format_content(content)[:500]}")
        return f"「{unit}」{node_title} 已触发学习任务"

    reg.register(Tool(
        name="learn_unit",
        description="主动学习一个知识单元，触发LLM采集内容并存入知识库",
        params=[
            ToolParam("unit", "string", "要学习的单元名称，如「认知革命」"),
            ToolParam("goal_id", "string", "学习目标ID", required=False),
            ToolParam("focus", "string",
                      "具体关注点，如「原因」「影响」「时间线」", required=False),
        ],
        fn=learn_unit,
    ))

    # ── 工具4：生成测验 ──
    def run_quiz(count: int = 3, goal_id: str = "") -> Dict:
        gid = goal_id or (system.ctx.current_goal_id or "")
        if not gid:
            return {"error": "未指定学习目标，无法生成测验"}
        try:
            results = system.quiz(count=count)
            return {"quiz_results": results}
        except Exception as e:
            return {"error": str(e)}

    reg.register(Tool(
        name="run_quiz",
        description="对当前学习目标生成测验题目，测试掌握程度",
        params=[
            ToolParam("count", "integer", "题目数量，默认3", required=False),
            ToolParam("goal_id", "string", "学习目标ID", required=False),
        ],
        fn=run_quiz,
    ))

    # ── 工具5：列出单元列表 ──
    def list_units(goal_id: str = "") -> Dict:
        gid = goal_id or (system.ctx.current_goal_id or "")
        if not gid:
            return {"error": "未指定学习目标"}
        units = system.col.load_goal_units(gid)
        return {"goal_id": gid, "units": units, "total": len(units)}

    reg.register(Tool(
        name="list_units",
        description="列出学习目标下的所有知识单元名称",
        params=[
            ToolParam("goal_id", "string",
                      "学习目标ID，不填则用当前目标", required=False),
        ],
        fn=list_units,
    ))

    # ── 工具6：保存笔记 ──
    def save_note(content: str, unit: str,
                  goal_id: str = "",
                  node_title: str = "笔记") -> str:
        gid = goal_id or (system.ctx.current_goal_id or "")
        if not gid:
            return "错误：未指定学习目标"
        system.vector.add_unit_knowledge(
            goal_id=gid,
            unit=unit,
            content_text=content,
            node_title=node_title,
        )
        return f"已保存笔记到「{unit}」/{node_title}"

    reg.register(Tool(
        name="save_note",
        description="把重要内容、心得、总结保存到知识库供日后检索",
        params=[
            ToolParam("content", "string", "要保存的内容"),
            ToolParam("unit", "string", "归属的知识单元名称"),
            ToolParam("goal_id", "string", "学习目标ID", required=False),
            ToolParam("node_title", "string",
                      "节点标题，默认「笔记」", required=False),
        ],
        fn=save_note,
    ))

    # ── 工具7：代码分析（有 code_analyzer.py 时自动注册）──
    try:
        from code_analyzer import CodeAnalyzer

        def analyze_code(path: str = "./") -> Dict:
            analyzer = CodeAnalyzer(path)
            report = analyzer.analyze()
            result = analyzer.to_dict()
            analyzer.sync_to_vector(system.vector, goal_id="project_arch")
            return {
                "files": result["stats"]["files"],
                "classes": result["stats"]["classes"],
                "functions": result["stats"]["functions"],
                "issues_count": result["stats"]["issues"],
                "top_issues": result["issues"][:5],
            }

        reg.register(Tool(
            name="analyze_code",
            description="静态分析Python项目代码结构，发现架构问题（方法不存在、循环依赖、函数过长等）",
            params=[
                ToolParam("path", "string",
                          "项目路径，默认当前目录", required=False),
            ],
            fn=analyze_code,
        ))
    except ImportError:
        pass

    log.info(f"[ToolRegistry] {reg.summary()}")
    return reg


# ══════════════════════════════════════════════════════════════
# Agent 执行循环（ReAct）
# ══════════════════════════════════════════════════════════════

class AgentLoop:
    """
    ReAct 循环：思考 → 选工具 → 执行 → 观察 → 再思考

    支持两种 LLM 工具调用方式：
      1. 原生 Function Calling（OpenAI / 兼容接口）
      2. JSON Prompt 降级（Ollama 等本地模型）
    """

    SYSTEM_PROMPT = """你是 VisionLearner 的智能助理，可以调用工具完成学习任务。

工作方式：
1. 分析用户需求，决定需要哪些工具
2. 调用工具，观察结果
3. 根据结果继续思考或完成任务
4. 给出简洁清晰的最终回答

原则：
- 优先用 search_knowledge 检索本地知识库
- 需要了解进度时先调用 get_progress
- 复合任务（如"学完X然后测试我"）拆分成多步执行
- 最终回答聚焦用户问题，不要照搬工具原始 JSON 输出
"""

    def __init__(self, llm, registry: ToolRegistry, max_steps: int = 6):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps

    def run(self, user_input: str, goal_id: str = "",
            stream: bool = False):
        """执行 Agent 循环，返回最终回答"""
        if stream:
            return self._run_stream(user_input, goal_id)
        return self._run_sync(user_input, goal_id)

    # ── 同步执行 ──

    def _run_sync(self, user_input: str, goal_id: str) -> str:
        system = self.SYSTEM_PROMPT
        if goal_id:
            system += f"\n当前学习目标 ID：{goal_id}"

        messages = [{"role": "user", "content": user_input}]
        schemas = self.registry.all_schemas()

        for step in range(self.max_steps):
            response = self._call_llm(messages, schemas, system)
            tool_calls = response.get("tool_calls", [])
            text = response.get("content", "")

            # 没有工具调用 → LLM 认为任务完成
            if not tool_calls:
                return text or "任务已完成。"

            # 把 LLM 决策加入消息历史
            messages.append({
                "role": "assistant",
                "content": text or "",
                "tool_calls": tool_calls,
            })

            # 执行每个工具调用
            for tc in tool_calls:
                fn_info = tc.get("function", {})
                tool_name = fn_info.get("name", "")
                try:
                    args = json.loads(fn_info.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                # 自动注入 goal_id（工具接受但用户没传时）
                if goal_id:
                    tool = self.registry.get(tool_name)
                    if tool:
                        param_names = [p.name for p in tool.params]
                        if "goal_id" in param_names and "goal_id" not in args:
                            args["goal_id"] = goal_id

                result = self.registry.execute(tool_name, **args)
                result_text = result.to_text()
                log.info(f"[Agent step {step+1}] {tool_name}({args}) "
                         f"→ {result_text[:80]}")

                # 把工具结果作为观察值传回给 LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{step}_{tool_name}"),
                    "content": result_text,
                })

        # 超出最大步数，强制总结
        messages.append({
            "role": "user",
            "content": "请根据以上工具执行结果，给出最终回答。"
        })
        final = self._call_llm(messages, [], system)
        return final.get("content", "任务执行完毕。")

    def _run_stream(self, user_input: str, goal_id: str):
        """流式输出：先同步跑完 Agent，再逐字符输出"""
        result = self._run_sync(user_input, goal_id)
        for char in result:
            yield char

    # ── LLM 调用 ──

    def _call_llm(self, messages: List[Dict],
                  schemas: List[Dict], system: str) -> Dict:
        """
        调用 LLM。优先原生 Function Calling，降级到 JSON Prompt。
        """
        # 有工具且 LLM 支持原生 Function Calling
        if schemas and hasattr(self.llm, "chat_with_tools"):
            try:
                return self.llm.chat_with_tools(
                    messages=messages, tools=schemas, system=system
                )
            except Exception as e:
                log.warning(f"原生 Function Calling 失败，降级：{e}")

        # 无工具 → 纯文本生成（最终总结阶段）
        if not schemas:
            last_user = next(
                (m["content"] for m in reversed(messages)
                 if m.get("role") == "user"), ""
            )
            ans = self.llm.chat(last_user, system=system)
            return {"content": ans, "tool_calls": []}

        # 降级：JSON Prompt（适用于 Ollama 本地模型）
        return self._json_prompt_call(messages, schemas, system)

    def _json_prompt_call(self, messages: List[Dict],
                          schemas: List[Dict], system: str) -> Dict:
        """
        降级方案：工具列表写进 prompt，让 LLM 返回 JSON 决策。
        """
        # 工具描述
        tools_lines = []
        for s in schemas:
            fn = s["function"]
            params_desc = "、".join(
                f"{k}（{v.get('description', '')}）"
                for k, v in fn["parameters"]["properties"].items()
            )
            tools_lines.append(
                f"- {fn['name']}: {fn['description']}\n  参数：{params_desc}"
            )
        tools_text = "\n".join(tools_lines)

        # 已完成的观察
        observations = []
        for m in messages:
            role = m.get("role", "")
            if role == "tool":
                observations.append(f"工具结果：{m['content'][:300]}")
            elif role == "assistant" and m.get("content"):
                observations.append(f"思考：{m['content'][:100]}")
        obs_text = "\n".join(observations) if observations else "（尚未执行任何工具）"

        user_question = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )

        prompt = f"""用户任务：{user_question}

已完成的步骤：
{obs_text}

可用工具：
{tools_text}

请决定下一步，只返回以下两种 JSON 格式之一：

需要调用工具：
{{"action": "tool", "tool": "工具名", "args": {{"参数名": "参数值"}}}}

任务已完成：
{{"action": "done", "answer": "最终回答内容"}}

只返回 JSON，不要其他内容。"""

        raw = self.llm.chat(prompt, system=system)

        try:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not json_match:
                return {"content": raw, "tool_calls": []}

            data = json.loads(json_match.group())

            if data.get("action") == "done":
                return {"content": data.get("answer", raw),
                        "tool_calls": []}

            if data.get("action") == "tool":
                tool_name = data.get("tool", "")
                args = data.get("args", {})
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": f"call_{tool_name}",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(
                                args, ensure_ascii=False
                            ),
                        },
                    }],
                }
        except (json.JSONDecodeError, AttributeError) as e:
            log.warning(f"JSON解析失败：{e}，原始：{raw[:200]}")

        return {"content": raw, "tool_calls": []}
