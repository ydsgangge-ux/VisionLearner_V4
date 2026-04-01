# main.py — VisionLearner v4.0
# 全量整合：原项目 + v3.2 + 方向A/B/C
"""
三层回答架构（token消耗从低到高）：
  ① SkillManager      → 技能匹配（0 token）
  ② NodeCollector     → 知识库命中（0 token）
  ③ LLM              → 按需学习（少量token，仅此一次）

完整功能：
  - 知识树按需收集（collector.py）
  - 对话自治：代词解析+意图识别（conversation.py）
  - 技能系统：LLM生成可扩展技能（skill_manager.py）
  - 自动驾驶模式（原版迁移）
  - 交互式思维导图探索（原版迁移）
  - QA上下文感知问答（原版迁移）
  - 自适应排课+进度监控（planner.py）
  - 文明愿景核心（vision_core.py）
  - 流式响应支持
  - REST API / Web UI 友好接口
"""

import sys
import os
import re
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime

# Windows GBK 终端安全输出
def safe_print(*args, **kwargs):
    if sys.platform == 'win32':
        _MAP = {
            '✅':'[OK]','❌':'[X]','⚠️':'[!]','⬜':'[ ]','📊':'[%]',
            '📋':'[=]','🌳':'[T]','🤖':'[A]','📚':'[B]','📝':'[Q]',
            '🌌':'[V]','🔧':'[S]','🆕':'[N]','📍':'[P]','📂':'[D]',
            '👋':'[~]','🗺':'[M]','⚖️':'[J]','🧠':'[brain]',
            '║':'|','╔':'+','╗':'+','╠':'+','╣':'+','╚':'+','╝':'+','═':'-',
            '█':'#','░':'.',
        }
        safe_args = []
        for a in args:
            s = str(a)
            for k, v in _MAP.items():
                s = s.replace(k, v)
            s = s.encode('gbk', errors='replace').decode('gbk')
            safe_args.append(s)
        print(*safe_args, **kwargs)
    else:
        print(*args, **kwargs)


# 核心模块
from foundation import (
    LearningGoal, MindMapNode, KnowledgeNode, generate_id,
    FoundationManager, GoalScale, GoalStatus
)
from llm_client import LLMClient, get_client
from storage import DataManager
from collector import NodeCollector, GOAL_TYPE_CONFIGS
from conversation import ConversationContext, Intent, format_content, generate_follow_up
from skill_manager import SkillManager
from storage import VectorStore

# 意图解析器
from intent_parser import LLMIntentParser

# 算法引擎
try:
    from planner import MindMapDrivenPlanner, AdaptiveScheduler, ProgressMonitor
    from explorer import ExplorerManager, MindMapVisualizer
    from perception import PerceptionManager
    from vision_core import CivilizationalVisionCore as VisionCore
    HAS_ENGINES = True
except ImportError as e:
    HAS_ENGINES = False
    logging.warning(f"部分算法引擎未加载: {e}")

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("VisionLearner")


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def detect_goal_type(description: str) -> str:
    desc = description.lower()
    if any(k in desc for k in ["汉字","字","文字","生字","识字"]): return "characters"
    if any(k in desc for k in ["单词","词汇","词语","英语","vocabulary"]): return "vocabulary"
    if any(k in desc for k in ["编程","代码","python","java","算法","函数","programming"]): return "programming"
    return "general"

def extract_count(description: str) -> Optional[int]:
    m = re.search(r'(\d+)', description)
    return int(m.group(1)) if m else None

def generate_unit_list(desc: str, goal_type: str,
                       count: Optional[int], llm: LLMClient) -> List[str]:
    if goal_type == "characters":
        top = list("的一是在不了有和人这中大为上个国我以要他时来用们生到作地"
                   "于出就分对成会可主发年动同工也能下过子说产种面而方后多定"
                   "行学法所民得经十三之进着等部度家电力里如水化高自二理起小"
                   "物现实加量都两体制机当使点从业本去把性好应开它合还因由其"
                   "些然前外天政四日那社义事平形相全表间样与关各重新线内数正"
                   "心反你明看先存而已通如被比于知质量再都特提到进后来安定经")
        seen, unique = set(), []
        for c in top:
            if c not in seen and '\u4e00' <= c <= '\u9fff':
                seen.add(c); unique.append(c)
        n = count or 100
        if n > len(unique):
            extra = llm.generate_json(
                f"列出{n-len(unique)}个常用汉字（不包括：{''.join(unique[:30])}），只返回JSON数组",
                system="只返回JSON数组。")
            if isinstance(extra, list):
                for c in extra:
                    if isinstance(c,str) and len(c)==1 and '\u4e00'<=c<='\u9fff' and c not in seen:
                        seen.add(c); unique.append(c)
        return unique[:n]
    result = llm.generate_json(
        f'对于学习目标"{desc}"，列出{"约"+str(count)+"个" if count else "合理数量的"}知识单元，只返回JSON数组',
        system="只返回JSON数组。")
    if isinstance(result, list):
        return [str(x) for x in result if x]
    return [desc]


# ═══════════════════════════════════════════════════════════════
# 主系统
# ═══════════════════════════════════════════════════════════════

class LearningSystem:
    """
    VisionLearner v4.0 核心系统

    集成：
    - 三层回答（技能→知识库→LLM）
    - 原版自动驾驶 / 交互式探索 / QA上下文
    - 自适应排课 + 进度监控
    - 文明愿景核心
    - Web API 友好接口（供 web_server.py 调用）
    """

    def __init__(self,
                 data_dir: str = "./learning_data",
                 skills_dir: str = "./skills",
                 provider: str = None,
                 model: str = None):
        log.info("VisionLearner v4.0 initializing...")

        # 基础设施
        self.llm      = get_client(provider=provider, model=model)
        self.db       = DataManager(data_dir)
        self.vector   = VectorStore(data_dir)          # RAG 向量检索层
        self.col      = NodeCollector(self.db, self.llm, self.vector)  # 传入vector
        self.ctx      = ConversationContext(db=self.db)  # 传入 db，启用持久化
        self.skills   = SkillManager(skills_dir, self.db, self.llm)
        self.foundation = FoundationManager()
        self.intent_parser = LLMIntentParser(self.llm)  # LLM意图解析器

        # 算法引擎（按需初始化）
        self.planner   = MindMapDrivenPlanner(self.llm) if HAS_ENGINES else None
        self.scheduler = AdaptiveScheduler() if HAS_ENGINES else None
        self.monitor   = ProgressMonitor() if HAS_ENGINES else None
        self.explorer  = ExplorerManager() if HAS_ENGINES else None
        self.perception= PerceptionManager(self.llm) if HAS_ENGINES else None
        self.vision    = VisionCore() if HAS_ENGINES else None

        # 运行时状态
        self._trees: Dict[str, Dict[str, MindMapNode]] = {}
        self._goal_type: Dict[str, str] = {}
        self.current_goal: Optional[LearningGoal] = None
        self.current_plan: Optional[Dict] = None
        self.current_schedule: Optional[Dict] = None
        self._start_time = datetime.now()

        # ── 工具注册表与 Agent ──────────────────────────────
        try:
            from tool_registry import build_registry, AgentLoop
            self.registry = build_registry(self)
            self.agent    = AgentLoop(self.llm, self.registry)
            log.info("OK ToolRegistry ready")
        except ImportError:
            self.registry = None
            self.agent    = None
            log.info("tool_registry.py 未找到，Agent模式不可用")

        self._restore()
        log.info("OK Ready\n")

    # ───────────────────────────────────────────────────────────
    # 目标管理
    # ───────────────────────────────────────────────────────────

    def create_goal(self, description: str, depth: int = 3) -> LearningGoal:
        goal = self.foundation.create_learning_goal(description)
        goal.status = GoalStatus.ACTIVE
        self.db.save_goal(goal)
        self.current_goal = goal

        goal_type = detect_goal_type(description)
        count     = extract_count(description)
        self._goal_type[goal.id] = goal_type
        self.ctx.set_goal(goal.id, goal_type)

        safe_print(f"\nOK Goal created: {goal.description}")
        safe_print(f"   Type: {goal_type}  Expected units: {count or 'TBD'}")

        safe_print("Generating knowledge unit list...")
        units = generate_unit_list(description, goal_type, count, self.llm)
        self.col.save_goal_units(goal.id, units)
        safe_print(f"   Total {len(units)} units")

        safe_print("Creating knowledge tree structure...")
        self._trees[goal.id] = {}
        for unit in units[:min(len(units), 30)]:
            tree = self.col.build_tree_from_template(unit, goal_type, max_depth=depth)
            self._trees[goal.id][unit] = tree
            self.col.save_tree(goal.id, unit, tree)

        # 文明愿景评估
        if self.vision:
            try:
                assessment = self.vision.assess_goal(description)
                if assessment:
                    safe_print(f"Vision fit: {assessment.get('score',0):.0%}")
            except Exception:
                pass

        safe_print(f"OK Ready! Ask questions directly, knowledge will be auto-filled\n")
        return goal

    def generate_mindmap_for_goal(self, goal: LearningGoal = None) -> Optional[MindMapNode]:
        g = goal or self.current_goal
        if not g: return None
        if self.perception:
            try:
                return self.perception.generate_mindmap_for_goal(g)
            except Exception as e:
                log.warning(f"MindMap生成失败: {e}")
        return None

    def create_learning_plan(self, goal: LearningGoal = None) -> Optional[Dict]:
        g = goal or self.current_goal
        if not g or not self.planner: return None
        try:
            units = self.col.load_goal_units(g.id)
            plan = self.planner.create_learning_plan(g, unit_list=units[:50])
            if plan:
                self.current_plan = plan
                self.db.save_learning_plan(g.id, plan)
                safe_print(f"OK Learning plan created: {len(plan.get('milestones',[]))} milestones")
            return plan
        except Exception as e:
            log.warning(f"计划创建失败: {e}")
            return None

    def schedule_sessions(self, goal: LearningGoal = None) -> Optional[Dict]:
        g = goal or self.current_goal
        if not g or not self.scheduler: return None
        try:
            schedule = self.scheduler.schedule_learning_sessions(
                goal=g, plan=self.current_plan, strategy="adaptive_schedule")
            if schedule:
                self.current_schedule = schedule
                self.db.save_schedule(g.id, schedule)
                n = len(schedule.get("scheduled_sessions", []))
                safe_print(f"OK Learning sessions scheduled: {n} sessions")
            return schedule
        except Exception as e:
            log.warning(f"排课失败: {e}")
            return None

    def execute_session(self, session_id: str = None) -> Dict:
        if not self.current_schedule:
            return {"success": False, "error": "没有激活的调度，请先运行 schedule"}
        sessions = self.current_schedule.get("scheduled_sessions", [])
        if not sessions:
            return {"success": False, "error": "没有可执行的学习会话"}
        session = next((s for s in sessions if s.get("id") == session_id), sessions[0])
        try:
            result = {"session_id": session.get("id"), "success": True,
                      "nodes_learned": 0, "duration_minutes": 0}
            topic = session.get("topic", "")
            if topic and self.current_goal:
                goal_type = self._goal_type.get(self.current_goal.id, "general")
                tree = self._get_tree(self.current_goal.id, topic, goal_type)
                if tree:
                    learned = []; start = time.time()
                    self.col.collect_tree(tree, topic, goal_type,
                                         on_progress=lambda c,t,title: learned.append(title))
                    self.col.save_tree(self.current_goal.id, topic, tree)
                    result["nodes_learned"] = len(learned)
                    result["duration_minutes"] = (time.time()-start)/60
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def monitor_progress(self) -> Dict:
        g = self.current_goal
        if not g: return {}
        units = self.col.load_goal_units(g.id)
        report = self.col.get_completion_report(g.id, units)
        pct = report['overall_completion']
        bar = "#"*int(pct*30) + "-"*(30-int(pct*30))
        safe_print(f"\n[Progress] Learning progress report")
        safe_print(f"   Goal: {g.description[:40]}")
        safe_print(f"   Units: {report['total_units']}  Learned: {report['learned_units']}")
        safe_print(f"   Nodes: {report['collected_nodes']}/{report['total_nodes']}")
        safe_print(f"   [{bar}] {pct:.0%}")
        if self.monitor and HAS_ENGINES:
            try:
                self.monitor.monitor_goal_progress(g, report)
            except Exception:
                pass
        progress_data = {**report, "timestamp": datetime.now().isoformat()}
        self.db.save_progress_data(g.id, progress_data)
        return report

    def run_auto_pilot(self, goal_description: str = None) -> Dict:
        """自动驾驶：一键完成完整学习流程"""
        safe_print("[AutoPilot] Starting auto-pilot mode...")
        results = {"steps": [], "success": False, "error": None}
        try:
            # 1. 目标
            if goal_description:
                goal = self.create_goal(goal_description)
            elif self.current_goal:
                goal = self.current_goal
            else:
                results["error"] = "没有指定学习目标"; return results
            results["steps"].append({"step": "goal", "success": True, "id": goal.id})

            # 2. 思维导图
            mm = self.generate_mindmap_for_goal(goal)
            results["steps"].append({"step": "mindmap", "success": bool(mm)})

            # 3. 计划
            plan = self.create_learning_plan(goal)
            results["steps"].append({"step": "plan", "success": bool(plan)})
            if not plan:
                results["error"] = "计划创建失败"; return results

            # 4. 排课
            schedule = self.schedule_sessions(goal)
            results["steps"].append({"step": "schedule", "success": bool(schedule)})

            # 5. 执行3个会话
            for i in range(3):
                r = self.execute_session()
                results["steps"].append({"step": f"session_{i+1}", "success": r.get("success"),
                                          "nodes": r.get("nodes_learned", 0)})

            # 6. 监控
            report = self.monitor_progress()
            results["steps"].append({"step": "monitor", "success": True})

            self.save_state()
            results["success"] = True
            results["completion"] = report.get("overall_completion", 0)
            safe_print(f"OK Auto-pilot completed: {len(results['steps'])} steps, completion {results['completion']:.0%}")
        except Exception as e:
            results["error"] = str(e)
            log.error(f"自动驾驶失败: {e}", exc_info=True)
        return results

    # ───────────────────────────────────────────────────────────
    # 三层回答（核心）
    # ───────────────────────────────────────────────────────────

    def answer(self, user_input: str, stream: bool = False, mode: str = "auto"):
        """
        回答入口，支持三种模式。

        参数：
            user_input: 用户问题
            stream: 是否流式输出
            mode: 模式选择
                - "auto": 自动模式（原有逻辑，技能→知识库→LLM）
                - "agent": Agent模式（LLM自主调用工具）
                - "local": 本地知识模式（只从本地知识库检索，不调用LLM）
                - "llm": 大模型学习模式（直接调用LLM，自动保存到知识库）

        stream=True 时返回生成器，False 返回完整字符串。
        """
        if mode == "agent":
            if self.agent:
                return self.agent.run(user_input, self.ctx.current_goal_id or "", stream)
            log.warning("Agent模式不可用，降级到auto")
        if mode == "local":
            return self._answer_local(user_input, stream)
        elif mode == "llm":
            return self._answer_llm(user_input, stream)
        else:
            return self._answer_auto(user_input, stream)

    # ───────────────────────────────────────────────────────────
    # 自动模式（原有逻辑）
    # ───────────────────────────────────────────────────────────

    def _answer_auto(self, user_input: str, stream: bool = False):
        """
        自动模式：技能→知识库→LLM

        流程：
          层1：技能系统（0 token）
          层2：SentenceParser 解析 → 精确匹配 unit → 直接返回知识库内容
          层3：向量语义检索 → 找到相关知识 → LLM 基于知识回答
          层4：纯 LLM 兜底（知识库完全没有时）
        """
        goal_id   = self.ctx.current_goal_id
        goal_type = self._goal_type.get(goal_id, "general") if goal_id else "general"

        # ── 层1：技能系统 ──
        progress_ctx = None
        if goal_id:
            units = self.col.load_goal_units(goal_id)
            progress_ctx = self.col.get_completion_report(goal_id, units[:100])
        skill_result = self.skills.handle(user_input, {"progress": progress_ctx})
        if skill_result:
            self.ctx.update(user_input, skill_result)
            return self._maybe_stream(f"[技能系统]\n{skill_result}", stream)

        # ── LLM 意图解析（替代 SentenceParser）──
        units = self.col.load_goal_units(goal_id) if goal_id else []
        self.intent_parser.set_units(units)
        self.intent_parser.set_topic(self.ctx.current_topic)
        intent = self.intent_parser.parse(user_input)

        # 处理特殊类型
        if intent.type == 'progress':
            report = self.monitor_progress()
            return self._maybe_stream(f"[学习进度]\n{json.dumps(report, ensure_ascii=False, indent=2)}", stream)
        if intent.type == 'quiz':
            results = self.quiz(count=3)
            return self._maybe_stream(f"[测验结果]\n{json.dumps(results, ensure_ascii=False, indent=2)}", stream)

        unit = intent.unit  # 直接就是正确的 unit 名
        focus = intent.focus

        # ── 层2：精确匹配 unit ──
        # 优先级：(1)向量库检索已学内容 > (2)知识树精确查找 > (3)LLM学习

        if unit and goal_id:
            # 先尝试从向量库检索相关内容（避免重复调用LLM）
            vector_results = self.vector.search(
                query=f"{unit} {focus}",
                goal_id=goal_id,
                top_k=3
            )
            # 如果向量库有匹配且unit匹配，直接使用
            if vector_results:
                top_result = vector_results[0]
                # TF-IDF分值天然偏低，有结果就用
                if top_result.get("unit") == unit:
                    from conversation import format_content
                    answer_text = format_content(top_result.get("content", ""))
                    if answer_text and answer_text != "（暂无）":
                        full = f"[本地知识]\n{answer_text}"
                        self.ctx.update(user_input, answer_text, unit)
                        return self._maybe_stream(full, stream)

            # 向量库没有高质量匹配，再去知识树精确查找
            tree = self._get_tree(goal_id, unit, goal_type)
            if tree:
                # 使用 LLM 解析的 focus 作为查询节点标题
                node_title = focus or unit
                node, content = self.col.collect_on_demand(
                    node_title, tree, goal_type, unit, goal_id=goal_id)
                self.col.save_tree(goal_id, unit, tree)
                self._trees.setdefault(goal_id, {})[unit] = tree

                # 同步写入向量库（增量更新）
                if node and node.collected and content:
                    self._sync_node_to_vector(goal_id, unit, node_title, content)

                actual_title = node.title if node else node_title
                answer_text  = format_content(content) if content else "（暂无）"
                source = "[NEW - Just learned]" if (node and "llm" in getattr(node, "collected_by", "")) else "[本地知识]"
                full   = f"{source}\n{answer_text}"

                self.ctx.update(user_input, answer_text, unit)
                return self._maybe_stream(full, stream)

        # ── 层3：向量语义检索 ──
        # [Bug 4 修复] 没有精确匹配，或对象为空，用原始输入去向量库找相关知识（不用 search_query）
        rag_results = []
        if goal_id:
            # [Bug 3 修复] top_k 3 → 5，阈值 0.3
            # [Bug 4 修复] 使用原始输入而非 search_query
            rag_results = self.vector.search(
                query=user_input, goal_id=goal_id, top_k=5)

        if rag_results:
            knowledge_text = self._format_rag_results(rag_results)
            context_text   = self.ctx.get_dialog_context_text(n=3)
            prompt = self._build_rag_prompt(
                user_input, knowledge_text, context_text)
            system = "你是一个学习助手，请根据提供的知识库内容准确回答问题。知识库没有的内容请如实说明。"

            top_unit = rag_results[0]["unit"] if rag_results else unit
            if stream:
                def _stream_and_update():
                    full_ans = ""
                    for chunk in self.llm.stream(prompt, system=system):
                        full_ans += chunk
                        yield chunk
                    self.ctx.update(user_input, full_ans, top_unit)
                return _stream_and_update()

            ans = self.llm.chat(prompt, system=system)
            self.ctx.update(user_input, ans, top_unit)
            return f"[本地知识（向量检索）]\n{ans}"

        # ── 层4：纯 LLM 兜底 ──
        context_text = self.ctx.get_dialog_context_text(n=3)
        prompt = user_input
        if context_text:
            prompt = f"对话上下文：\n{context_text}\n\n当前问题：{user_input}"
        system = "你是一个博学的学习助手，请准确回答问题。"

        if stream:
            def _stream_fallback():
                full_ans = ""
                for chunk in self.llm.stream(prompt, system=system):
                    full_ans += chunk
                    yield chunk
                self.ctx.update(user_input, full_ans, unit or "")
            return _stream_fallback()

        ans = self.llm.chat(prompt, system=system)
        self.ctx.update(user_input, ans, unit or "")
        return f"[LLM回答]\n{ans}"

    # ───────────────────────────────────────────────────────────
    # 本地知识模式
    # ───────────────────────────────────────────────────────────

    def _answer_local(self, user_input: str, stream: bool = False):
        """
        本地知识模式：只从本地知识库检索，不调用LLM。

        优先级：
          1. 技能系统
          2. 向量库检索
          3. 知识树精确查找
          4. 没有找到：提示"知识库中没有此内容"
        """
        goal_id   = self.ctx.current_goal_id
        goal_type = self._goal_type.get(goal_id, "general") if goal_id else "general"

        # ── 层1：技能系统 ──
        progress_ctx = None
        if goal_id:
            units = self.col.load_goal_units(goal_id)
            progress_ctx = self.col.get_completion_report(goal_id, units[:100])
        skill_result = self.skills.handle(user_input, {"progress": progress_ctx})
        if skill_result:
            self.ctx.update(user_input, skill_result)
            return self._maybe_stream(f"[技能系统]\n{skill_result}", stream)

        # ── LLM 意图解析（替代 SentenceParser）──
        units = self.col.load_goal_units(goal_id) if goal_id else []
        self.intent_parser.set_units(units)
        self.intent_parser.set_topic(self.ctx.current_topic)
        intent = self.intent_parser.parse(user_input)

        # 处理特殊类型
        if intent.type == 'progress':
            report = self.monitor_progress()
            return self._maybe_stream(f"[学习进度]\n{json.dumps(report, ensure_ascii=False, indent=2)}", stream)
        if intent.type == 'quiz':
            results = self.quiz(count=3)
            return self._maybe_stream(f"[测验结果]\n{json.dumps(results, ensure_ascii=False, indent=2)}", stream)

        unit = intent.unit
        focus = intent.focus

        # ── 层2：向量库检索 ──
        if goal_id:
            vector_results = self.vector.search(
                query=f"{unit} {focus}" if unit else user_input,
                goal_id=goal_id,
                top_k=5
            )
            if vector_results:
                # TF-IDF分值天然偏低，只要有结果就用，不设阈值
                top_result = vector_results[0]
                from conversation import format_content
                answer_text = format_content(top_result.get("content", ""))
                if answer_text and answer_text != "（暂无）":
                    full = f"[本地知识]\n{answer_text}"
                    self.ctx.update(user_input, answer_text, top_result.get("unit", unit))
                    return self._maybe_stream(full, stream)

        # ── 层3：知识树精确查找（遍历所有已收集节点）──
        if unit and goal_id:
            tree = self._get_tree(goal_id, unit, goal_type)
            if tree:
                # 不依赖 _intent_to_node，直接找树里第一个已收集的节点
                all_collected = [n for n in tree._all_nodes()
                                 if n.collected and n.content and n.depth > 0]
                if all_collected:
                    node = all_collected[0]
                    from conversation import format_content
                    answer_text = format_content(node.content)
                    full = f"[本地知识]\n{answer_text}"
                    self.ctx.update(user_input, answer_text, unit)
                    return self._maybe_stream(full, stream)

        # ── 层4：没有找到 ──
        answer_text = "本地知识库中没有找到相关内容。请切换到「大模型学习模式」来学习新知识。"
        return self._maybe_stream(answer_text, stream)

    # ───────────────────────────────────────────────────────────
    # 大模型学习模式
    # ───────────────────────────────────────────────────────────

    def _answer_llm(self, user_input: str, stream: bool = False):
        """
        大模型学习模式：直接调用LLM，自动保存到知识库。

        特点：
          - 不检查本地知识库
          - 直接调用LLM生成详细答案
          - 将答案保存到知识库
          - 适合学习新内容
        """
        # 检查是否有当前目标
        goal_id   = self.ctx.current_goal_id
        goal_type = self._goal_type.get(goal_id, "general") if goal_id else "general"

        # LLM 意图解析
        units = self.col.load_goal_units(goal_id) if goal_id else []
        self.intent_parser.set_units(units)
        self.intent_parser.set_topic(self.ctx.current_topic)
        intent = self.intent_parser.parse(user_input)

        unit = intent.unit
        focus = intent.focus

        # 构建详细的prompt
        prompt = user_input
        if unit and focus:
            prompt = f"请详细介绍「{unit}」的「{focus}」：{user_input}"

        system = "你是一个专业的学习助手，请用4-8句话详细回答，给出具体内容、例子和应用场景。"

        if stream:
            def _stream_and_save():
                full_ans = ""
                for chunk in self.llm.stream(prompt, system=system):
                    full_ans += chunk
                    yield chunk

                # 保存到知识库（如果有goal和unit）
                if goal_id and unit:
                    try:
                        tree = self._get_tree(goal_id, unit, goal_type)
                        if not tree:
                            # 如果没有知识树，先创建一个
                            tree = self.col.build_tree_from_template(unit, goal_type)

                        # 找到或创建节点
                        node_title = focus or unit
                        if node_title:
                            node = self.col._find_best_match(node_title, tree)
                            if not node:
                                # 创建新节点
                                from foundation import MindMapNode
                                import time
                                node = MindMapNode(
                                    id=f"node_{int(time.time()*1000)}_{hash(node_title) % 10000}",
                                    title=node_title,
                                    content=full_ans,
                                    collected=True,
                                    collected_at=datetime.now().isoformat(),
                                    collected_by=f"llm:{self.llm.model}"
                                )
                                tree.children.append(node)
                            else:
                                # 更新已有节点
                                node.content = full_ans
                                node.collected = True
                                node.collected_at = datetime.now().isoformat()
                                node.collected_by = f"llm:{self.llm.model}"

                        # 保存树
                        self.col.save_tree(goal_id, unit, tree)
                        self._trees.setdefault(goal_id, {})[unit] = tree

                        # 同步到向量库
                        self._sync_node_to_vector(goal_id, unit, node_title, full_ans)

                    except Exception as e:
                        log.warning(f"保存到知识库失败：{e}")

                self.ctx.update(user_input, full_ans, unit)
            return _stream_and_save()

        ans = self.llm.chat(prompt, system=system)

        # 保存到知识库（如果有goal和unit）
        if goal_id and unit:
            try:
                tree = self._get_tree(goal_id, unit, goal_type)
                if not tree:
                    tree = self.col.build_tree_from_template(unit, goal_type)

                node_title = focus or unit
                if node_title:
                    node = self.col._find_best_match(node_title, tree)
                    if not node:
                        from foundation import MindMapNode
                        import time
                        node = MindMapNode(
                            id=f"node_{int(time.time()*1000)}_{hash(node_title) % 10000}",
                            title=node_title,
                            content=ans,
                            collected=True,
                            collected_at=datetime.now().isoformat(),
                            collected_by=f"llm:{self.llm.model}"
                        )
                        tree.children.append(node)
                    else:
                        node.content = ans
                        node.collected = True
                        node.collected_at = datetime.now().isoformat()
                        node.collected_by = f"llm:{self.llm.model}"

                self.col.save_tree(goal_id, unit, tree)
                self._trees.setdefault(goal_id, {})[unit] = tree
                self._sync_node_to_vector(goal_id, unit, node_title, ans)

            except Exception as e:
                log.warning(f"保存到知识库失败：{e}")

        self.ctx.update(user_input, ans, unit)
        return f"[大模型]\n{ans}"

    # ───────────────────────────────────────────────────────────
    # RAG 辅助方法
    # ───────────────────────────────────────────────────────────

    def _sync_node_to_vector(self, goal_id: str, unit: str,
                              node_title: str, content: Any) -> None:
        """
        把刚学到的节点内容同步写入向量库。
        collector 收集完节点后由 answer() 调用。
        细粒度：单节点内容
        粗粒度：整个 unit 的内容合并（触发全量同步）
        """
        try:
            content_str = format_content(content)
            if not content_str or content_str == "（暂无）":
                return
            # 细粒度：这个节点
            self.vector.add_unit_knowledge(
                goal_id=goal_id,
                unit=unit,
                content_text=content_str,
                node_title=node_title,
            )
            # 粗粒度：把该 unit 所有已学内容合并，更新整体向量
            tree = self._trees.get(goal_id, {}).get(unit)
            if tree:
                self._sync_unit_coarse(goal_id, unit, tree)
        except Exception as e:
            log.warning(f"向量同步失败：{e}")

    def _sync_unit_coarse(self, goal_id: str, unit: str,
                           tree: "MindMapNode") -> None:
        """把整个 unit 的所有已学节点内容合并，写入粗粒度向量"""
        parts = []
        for node in tree._all_nodes():
            if node.collected and node.content and node.depth > 0:
                c = format_content(node.content)
                if c and c != "（暂无）":
                    parts.append(f"{node.title}：{c}")
        if parts:
            coarse_text = " ".join(parts)
            self.vector.add_unit_knowledge(
                goal_id=goal_id,
                unit=unit,
                content_text=coarse_text,
                node_title="",  # 空 = 粗粒度标记
            )

    def _format_rag_results(self, results: List[Dict]) -> str:
        """把向量检索结果格式化为 prompt 可用的知识文本"""
        lines = []
        seen_units = set()
        for r in results:
            unit      = r.get("unit", "")
            node      = r.get("node_title", "")
            content   = r.get("content", "")
            is_coarse = r.get("is_coarse", False)

            # 去掉 "unit node：" 前缀（存储时加的，显示时不需要）
            if content.startswith(f"{unit} {node}："):
                content = content[len(f"{unit} {node}："):]
            elif content.startswith(f"{unit} ："):
                content = content[len(f"{unit} ："):]

            label = f"【{unit}】" if is_coarse else f"【{unit} · {node}】"
            if label not in seen_units:
                lines.append(f"{label} {content}")
                seen_units.add(label)

        return "\n".join(lines)

    def _build_rag_prompt(self, user_input: str,
                           knowledge_text: str,
                           context_text: str = "") -> str:
        """构建 RAG prompt"""
        parts = []
        if context_text:
            parts.append(f"【对话上下文】\n{context_text}")
        parts.append(f"【知识库内容】\n{knowledge_text}")
        parts.append(f"【用户问题】\n{user_input}")
        parts.append("请根据知识库内容回答问题。如知识库中没有相关内容，请如实说明。")
        return "\n\n".join(parts)

    def _search_units(self, user_input: str, goal_id: str,
                       goal_type: str) -> tuple:
        """
        [Bug 2 修复] 三级匹配模糊查找 unit：
        1. 精确包含匹配
        2. 按":"和空格拆分，任意一段命中
        3. 向量库辅助兜底
        """
        units = self.col.load_goal_units(goal_id)
        if not units:
            return None, None

        user_lower = user_input.lower()

        # 级别1：精确包含匹配
        for unit_name in units:
            if unit_name.lower() in user_lower or user_lower in unit_name.lower():
                tree = self._get_tree(goal_id, unit_name, goal_type)
                if tree:
                    return unit_name, tree

        # 级别2：按 ":" 和空格拆分，任意一段命中
        for unit_name in units:
            # 拆分 unit 名称，例如 "贝叶斯定理：用证据更新信念" -> ["贝叶斯定理", "用证据更新信念"]
            parts = re.split(r'[：:\s]+', unit_name)
            for part in parts:
                if part.lower() in user_lower or user_lower in part.lower():
                    tree = self._get_tree(goal_id, unit_name, goal_type)
                    if tree:
                        return unit_name, tree

        # 级别3：向量库辅助兜底
        try:
            vector_results = self.vector.search(query=user_input, goal_id=goal_id, top_k=5)
            if vector_results:
                # 找到分数最高的匹配 unit
                for result in vector_results:
                    matched_unit = result.get("unit", "")
                    if matched_unit in units:
                        tree = self._get_tree(goal_id, matched_unit, goal_type)
                        if tree:
                            return matched_unit, tree
        except Exception as e:
            log.warning(f"向量库辅助匹配失败: {e}")

        return None, None

    def _build_context_prompt(self, user_input: str,
                               unit: str, tree: "MindMapNode") -> str:
        """保留旧接口，供外部调用兼容"""
        def collect_content(node, max_depth=2, current_depth=0):
            if current_depth > max_depth:
                return []
            contents = []
            if node.title and node.collected and node.content:
                contents.append(f"{node.title}: {node.content}")
            for child in node.children or []:
                contents.extend(collect_content(child, max_depth, current_depth + 1))
            return contents
        knowledge_parts = collect_content(tree)
        knowledge_text = "\n".join(knowledge_parts[:20])
        return (f"问题：{user_input}\n\n"
                f"相关知识库内容（关于\"{unit}\"）：\n{knowledge_text}\n\n"
                f"请基于以上知识库内容回答问题。")

    def _maybe_stream(self, text: str, stream: bool):
        if not stream:
            return text
        def _gen():
            for ch in text:
                yield ch
                time.sleep(0.01)
        return _gen()

    def _intent_to_node(self, intent) -> str:
        """兼容旧接口，现在直接返回 focus"""
        return getattr(intent, 'focus', '') or getattr(intent, 'entity', '')

    def _get_tree(self, goal_id: str, unit: str, goal_type: str) -> Optional[MindMapNode]:
        if goal_id in self._trees and unit in self._trees[goal_id]:
            return self._trees[goal_id][unit]
        tree = self.col.load_tree(goal_id, unit)
        if tree:
            self._trees.setdefault(goal_id, {})[unit] = tree
            return tree
        tree = self.col.build_tree_from_template(unit, goal_type)
        self._trees.setdefault(goal_id, {})[unit] = tree
        return tree

    # ───────────────────────────────────────────────────────────
    # 批量填充
    # ───────────────────────────────────────────────────────────

    def populate(self, unit_limit: int = None, goal_id: str = None) -> Dict:
        gid = goal_id or (self.current_goal.id if self.current_goal else None)
        if not gid: return {"error": "无活跃目标"}
        units     = self.col.load_goal_units(gid)
        goal_type = self._goal_type.get(gid, "general")
        if unit_limit: units = units[:unit_limit]
        total_nodes = 0
        safe_print(f"\n[Batch] Populating: {len(units)} units")
        for i, unit in enumerate(units):
            tree = self._get_tree(gid, unit, goal_type)
            learned = []
            self.col.collect_tree(tree, unit, goal_type,
                on_progress=lambda c,t,title: (
                    safe_print(f"\r  [{i*100//len(units):3d}%] {unit}: {title[:12]:<12}", end="", flush=True),
                    learned.append(title)),
                goal_id=gid)
            self.col.save_tree(gid, unit, tree)
            self._trees.setdefault(gid, {})[unit] = tree
            total_nodes += len(learned)
            # 同步到向量库（粗粒度 + 细粒度）
            try:
                self._sync_unit_coarse(gid, unit, tree)
                for node in tree._all_nodes():
                    if node.collected and node.content and node.depth > 0:
                        self._sync_node_to_vector(gid, unit, node.title, node.content)
            except Exception as e:
                log.warning(f"Vector sync failed [{unit}]: {e}")
        safe_print(f"\nOK Completed, learned {total_nodes} nodes")
        return {"units": len(units), "nodes": total_nodes}

    # ───────────────────────────────────────────────────────────
    # 测验
    # ───────────────────────────────────────────────────────────

    def quiz(self, count: int = 5) -> List[Dict]:
        import random
        gid       = self.current_goal.id if self.current_goal else None
        goal_type = self._goal_type.get(gid, "general") if gid else "general"
        units     = self.col.load_goal_units(gid) if gid else []
        if not units: print("请先创建学习目标"); return []

        random.shuffle(units); tested = []; results = []
        safe_print(f"\n[Quiz] ({count} questions)\n" + "-"*40)

        for unit in units:
            if len(tested) >= count: break
            tree = self._get_tree(gid, unit, goal_type)
            if not tree: continue
            nodes = [n for n in tree._all_nodes() if n.collected and n.depth > 0]
            if not nodes: continue
            node = random.choice(nodes)
            ans_fmt = format_content(node.content)
            if not ans_fmt or ans_fmt == "（暂无）": continue

            q_map = {"读音":f"「{unit}」怎么读？","含义":f"「{unit}」是什么意思？",
                     "组词":f"用「{unit}」组一个词","笔画":f"「{unit}」共几画？"}
            question = q_map.get(node.title, f"「{unit}」的{node.title}？")
            safe_print(f"\n第{len(tested)+1}题：{question}")
            ua = input("你的回答: ").strip()
            safe_print(f"参考答案: {ans_fmt}")
            score = self._score_answer(ua, ans_fmt)
            correct = score >= 0.8
            safe_print("OK Correct" if correct else "WARN Partial" if score>=0.4 else "FAIL Wrong")
            results.append({"q": question, "a": ua, "ref": ans_fmt, "score": score})
            tested.append(unit)

        if results:
            avg = sum(r["score"] for r in results)/len(results)
            safe_print(f"\n{'─'*40}\n平均得分：{avg:.0%}  ({len(results)}题)")
        return results

    def _score_answer(self, user: str, correct: str) -> float:
        if not user: return 0.0
        u, c = user.lower().strip(), correct.lower().strip()
        if u == c: return 1.0
        if u in c or c in u: return 0.8
        return len(set(u) & set(c)) / max(len(c), 1) * 0.6

    # ───────────────────────────────────────────────────────────
    # 技能
    # ───────────────────────────────────────────────────────────

    def add_skill(self, description: str) -> Dict:
        return self.skills.create_skill(description)

    def list_skills(self) -> List[Dict]:
        return self.skills.list_skills()

    # ───────────────────────────────────────────────────────────
    # 进度 & 目标列表
    # ───────────────────────────────────────────────────────────

    def progress(self) -> Dict:
        return self.monitor_progress()

    def list_goals(self) -> List[Dict]:
        goals = self.db.load_all_goals()
        result = []
        for g in goals:
            units  = self.col.load_goal_units(g["id"])
            report = self.col.get_completion_report(g["id"], units[:50])
            result.append({**g, **report})
        return result

    def select_goal(self, goal_id: str) -> bool:
        goals = self.db.load_all_goals()
        for g in goals:
            if g["id"] == goal_id:
                gt = detect_goal_type(g.get("description",""))
                self._goal_type[goal_id] = gt
                self.ctx.set_goal(goal_id, gt)
                # reconstruct LearningGoal
                lg = self.foundation.create_learning_goal(g["description"])
                lg.id = goal_id
                self.current_goal = lg
                safe_print(f"OK Switched to: {g['description'][:40]}")
                return True
        return False

    def delete_goal(self, goal_id: str) -> bool:
        """删除目标及其所有相关数据"""
        import shutil

        # 如果删除的是当前目标，清空当前目标
        if self.current_goal and self.current_goal.id == goal_id:
            self.current_goal = None

        # 删除目标文件
        deleted = self.db.delete_goal(goal_id)

        # 删除相关数据
        try:
            # 删除思维导图
            mindmap_path = Path(self.db.data_dir) / "mindmap_trees" / f"{goal_id}.json"
            if mindmap_path.exists():
                mindmap_path.unlink()

            # 删除目标单元
            goal_units_path = Path(self.db.data_dir) / "goal_units" / f"{goal_id}.json"
            if goal_units_path.exists():
                goal_units_path.unlink()

            # 删除进度数据
            progress_path = Path(self.db.data_dir) / "progress" / f"{goal_id}.json"
            if progress_path.exists():
                progress_path.unlink()

            # 删除学习计划
            plan_path = Path(self.db.data_dir) / "plans" / f"{goal_id}.json"
            if plan_path.exists():
                plan_path.unlink()

            # 删除调度
            schedule_path = Path(self.db.data_dir) / "schedules" / f"{goal_id}.json"
            if schedule_path.exists():
                schedule_path.unlink()

            # 从内部状态中移除
            if goal_id in self._goal_type:
                del self._goal_type[goal_id]
            if goal_id in self._trees:
                del self._trees[goal_id]

            safe_print(f"OK Goal deleted: {goal_id}")
            return True
        except Exception as e:
            log.warning(f"Delete goal failed: {e}")
            return False

    # ───────────────────────────────────────────────────────────
    # 状态持久化
    # ───────────────────────────────────────────────────────────

    def save_state(self) -> None:
        state = {
            "current_goal_id": self.current_goal.id if self.current_goal else None,
            "goal_types": self._goal_type,
            "saved_at": datetime.now().isoformat(),
        }
        self.db.save_system_state(state)

    def _restore(self) -> None:
        state = self.db.load_system_state()
        goals = self.db.load_all_goals()
        if not goals: return
        active = [g for g in goals if g.get("status") == "active"]
        if not active: active = goals
        last_g = active[-1]
        gid = last_g["id"]
        gt  = detect_goal_type(last_g.get("description",""))
        self._goal_type[gid] = gt
        self.ctx.set_goal(gid, gt)
        lg = self.foundation.create_learning_goal(last_g["description"])
        lg.id = gid; self.current_goal = lg
        plan = self.db.load_learning_plan(gid)
        if plan: self.current_plan = plan
        sched = self.db.load_schedule(gid)
        if sched: self.current_schedule = sched
        safe_print(f"[Restore] Restored goal: {last_g['description'][:40]}")

    # ───────────────────────────────────────────────────────────
    # Web API 接口（供 web_server.py 调用）
    # ───────────────────────────────────────────────────────────

    def api_status(self) -> Dict:
        gid = self.current_goal.id if self.current_goal else None
        units = self.col.load_goal_units(gid) if gid else []
        report = self.col.get_completion_report(gid, units) if gid else {}
        uptime = str(datetime.now() - self._start_time).split(".")[0]
        stats  = self.llm.get_stats()
        return {
            "goal": {"id": gid, "description": self.current_goal.description
                     if self.current_goal else None},
            "progress": report,
            "llm": {"provider": stats["provider"], "model": stats["model"],
                    "calls": stats["calls"], "cache_hits": stats["cache_hits"]},
            "skills": [s["name"] for s in self.skills.list_skills()],
            "uptime": uptime,
            "storage": self.db.get_data_statistics(),
        }

    def api_mindmap(self, unit: str = None) -> Dict:
        """返回当前目标/指定单元的思维导图JSON"""
        gid = self.current_goal.id if self.current_goal else None
        if not gid: return {}
        goal_type = self._goal_type.get(gid, "general")
        if not unit:
            units = self.col.load_goal_units(gid)
            unit  = units[0] if units else None
        if not unit: return {}
        tree = self._get_tree(gid, unit, goal_type)
        if not tree: return {}
        return self._tree_to_dict(tree)

    def _tree_to_dict(self, node: MindMapNode) -> Dict:
        return {
            "id": node.id, "title": node.title,
            "collected": node.collected, "depth": node.depth,
            "content": format_content(node.content) if node.collected else "",
            "children": [self._tree_to_dict(c) for c in node.children]
        }


# ═══════════════════════════════════════════════════════════════
# 交互式思维导图探索器（原版迁移）
# ═══════════════════════════════════════════════════════════════

class MindMapExplorer:
    """在思维导图里交互式"走"的探索器"""

    def __init__(self, system: LearningSystem):
        self.sys = system
        self.current_node: Optional[MindMapNode] = None
        self.node_map: Dict[str, MindMapNode] = {}
        self.history: List[MindMapNode] = []

    def explore(self, unit: str = None) -> None:
        gid = self.sys.current_goal.id if self.sys.current_goal else None
        if not gid: print("请先创建学习目标"); return
        goal_type = self.sys._goal_type.get(gid, "general")
        units     = self.sys.col.load_goal_units(gid)
        if not units: print("目标尚无知识单元"); return
        target = unit or units[0]
        tree   = self.sys._get_tree(gid, target, goal_type)
        if not tree: print(f"找不到单元：{target}"); return
        self.node_map = {}
        self._collect_nodes(tree)
        self.current_node = tree
        self._loop()

    def _collect_nodes(self, node: MindMapNode):
        self.node_map[node.id] = node
        for c in node.children: self._collect_nodes(c)

    def _loop(self):
        safe_print("\n" + "═"*50)
        safe_print("== MindMap Explorer Mode  (help for commands)")
        safe_print("═"*50)
        while True:
            self._show_current()
            cmd = input("\n[探索] > ").strip().lower()
            if cmd in ("exit","q","退出"): print("退出探索"); break
            elif cmd == "help":
                safe_print("  ls / 子节点  |  cd <序号>  |  up / 返回  |  ask  |  learn  |  exit")
            elif cmd in ("ls","子节点","children"): self._show_children()
            elif cmd in ("up","..","返回","parent"): self._go_parent()
            elif cmd == "ask":  self._ask_node()
            elif cmd == "learn": self._learn_node()
            elif cmd.startswith("cd "):
                idx = cmd[3:].strip()
                if idx.isdigit(): self._go_child(int(idx)-1)
            else:
                # 尝试作为提问处理
                ans = self.sys.answer(cmd)
                safe_print(f"\n{ans}")

    def _show_current(self):
        n = self.current_node
        if not n: return
        pct = ("[OK]" if n.collected else "[ ]") + f" [depth {n.depth}]"
        safe_print(f"\n[Node] {n.title}  {pct}")
        if n.collected and n.content:
            from conversation import format_content
            safe_print(f"   {format_content(n.content)[:80]}")
        if n.children:
            safe_print(f"   └─ {len(n.children)} 个子节点: {', '.join(c.title for c in n.children[:5])}")

    def _show_children(self):
        if not self.current_node or not self.current_node.children:
            safe_print("   （无子节点）"); return
        for i,c in enumerate(self.current_node.children,1):
            status = "[OK]" if c.collected else "[ ]"
            safe_print(f"  {i}. {status} {c.title}")

    def _go_child(self, idx: int):
        if not self.current_node: return
        children = self.current_node.children
        if 0 <= idx < len(children):
            self.history.append(self.current_node)
            self.current_node = children[idx]
        else:
            safe_print("   序号超范围")

    def _go_parent(self):
        if self.history:
            self.current_node = self.history.pop()
        else:
            safe_print("   已在根节点")

    def _ask_node(self):
        if not self.current_node: return
        q = f"{self.current_node.title}详细说一下"
        safe_print(f"提问：{q}")
        safe_print(self.sys.answer(q))

    def _learn_node(self):
        if not self.current_node: return
        gid = self.sys.current_goal.id if self.sys.current_goal else None
        if not gid: return
        goal_type = self.sys._goal_type.get(gid,"general")
        units = self.sys.col.load_goal_units(gid)
        unit  = units[0] if units else ""
        _, content = self.sys.col.collect_on_demand(
            self.current_node.title,
            self.sys._get_tree(gid, unit, goal_type),
            goal_type, unit)
        safe_print(f"🆕 学习完成：{format_content(content)[:80]}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

HELP = """
╔══════════════════════════════════════════════════════════╗
║  VisionLearner v4.0  命令参考                            ║
╠══════════════════════════════════════════════════════════╣
║  目标管理                                                 ║
║    new <描述>        创建学习目标                         ║
║    goals             列出所有目标                         ║
║    select <ID>       切换目标                             ║
║    progress          查看进度                             ║
║                                                           ║
║  学习                                                     ║
║    learn [n]         批量填充n个单元                      ║
║    ask <问题>        提问（或直接输入）                   ║
║    quiz [n]          随机测验n题                          ║
║    explore [单元]    交互式思维导图探索                   ║
║                                                           ║
║  自动化                                                   ║
║    autopilot [目标]  自动驾驶模式                         ║
║    plan              生成学习计划                         ║
║    schedule          排课                                 ║
║    session           执行一个学习会话                     ║
║                                                           ║
║  技能系统                                                 ║
║    skills            查看所有技能                         ║
║    skill add <描述>  新增技能                             ║
║                                                           ║
║  系统                                                     ║
║    status            系统状态                             ║
║    ctx               对话上下文                           ║
║    backup            备份数据                             ║
║    help              帮助                                 ║
║    exit              退出                                 ║
╚══════════════════════════════════════════════════════════╝
"""

def run_cli(sys_: LearningSystem):
    safe_print("""
╔══════════════════════════════════════════════════════════╗
║  AI VisionLearner v4.0                                   ║
║  知识自治 · 技能可扩展 · LLM按需调用 · 文明愿景          ║
╚══════════════════════════════════════════════════════════╝""")
    safe_print(HELP)
    explorer = MindMapExplorer(sys_)

    while True:
        try:
            topic = sys_.ctx.current_topic or (
                sys_.current_goal.description[:12] if sys_.current_goal else "无目标")
            raw   = input(f"[{topic}] > ").strip()
            if not raw: continue
            parts = raw.split(None, 1)
            cmd, args = parts[0].lower(), (parts[1] if len(parts)>1 else "")

            if cmd == "new":
                if not args: print("用法: new <目标描述>")
                else: sys_.create_goal(args)

            elif cmd == "learn":
                n = int(args) if args.isdigit() else None
                sys_.populate(unit_limit=n)

            elif cmd == "ask":
                if args:
                    # 流式输出
                    for chunk in sys_.answer(args, stream=True):
                        safe_print(chunk, end="", flush=True)
                    safe_print()
                else:
                    safe_print("用法: ask <问题>")

            elif cmd == "quiz":
                sys_.quiz(int(args) if args.isdigit() else 5)

            elif cmd == "progress":
                sys_.monitor_progress()

            elif cmd == "goals":
                goals = sys_.list_goals()
                safe_print(f"\n📋 共 {len(goals)} 个目标：")
                for g in goals:
                    pct = g.get("overall_completion", 0)
                    safe_print(f"  [{pct:.0%}] {g.get('description','')[:40]}")
                    safe_print(f"        ID: {g['id']}")

            elif cmd == "select":
                if not sys_.select_goal(args):
                    safe_print(f"未找到目标: {args}")

            elif cmd == "explore":
                explorer.explore(args or None)

            elif cmd == "autopilot":
                sys_.run_auto_pilot(args or None)

            elif cmd == "plan":
                sys_.create_learning_plan()

            elif cmd == "schedule":
                sys_.schedule_sessions()

            elif cmd == "session":
                r = sys_.execute_session()
                safe_print(f"OK Session completed: learned {r.get('nodes_learned',0)} nodes")

            elif cmd == "skills":
                skills = sys_.list_skills()
                safe_print(f"\n[Skills] Loaded ({len(skills)} skills):")
                for s in skills:
                    safe_print(f"  [{s['name']}] {s['desc']}")
                    safe_print(f"   Triggers: {', '.join(s['triggers'][:4])}")

            elif cmd == "skill" and args.startswith("add "):
                sys_.add_skill(args[4:].strip())

            elif cmd == "status":
                st = sys_.api_status()
                safe_print(json.dumps(st, ensure_ascii=False, indent=2, default=str))

            elif cmd == "ctx":
                safe_print(f"[Context] {sys_.ctx.get_context_summary()}")

            elif cmd == "backup":
                path = sys_.db.backup_data()
                safe_print(f"OK Backup completed: {path}")

            elif cmd in ("help","?"): print(HELP)

            elif cmd in ("exit","quit","q"):
                sys_.save_state()
                safe_print("Bye!"); break

            else:
                # 直接提问，流式输出
                for chunk in sys_.answer(raw, stream=True):
                    safe_print(chunk, end="", flush=True)
                safe_print()

        except KeyboardInterrupt:
            safe_print("\n（输入 exit 退出）")
        except EOFError:
            sys_.save_state(); break
        except Exception as e:
            safe_print(f"[ERROR] {e}")
            log.debug("", exc_info=True)


def main():
    import argparse
    p = argparse.ArgumentParser(description="VisionLearner v4.0")
    p.add_argument("--provider",   default=None)
    p.add_argument("--model",      default=None)
    p.add_argument("--data-dir",   default="./learning_data")
    p.add_argument("--skills-dir", default="./skills")
    p.add_argument("--web",        action="store_true", help="启动Web服务器")
    p.add_argument("--port",       default=5000, type=int)
    args = p.parse_args()

    sys_ = LearningSystem(args.data_dir, args.skills_dir, args.provider, args.model)

    if args.web:
        from web_server import run_server
        # 启用 Debug 模式：显示请求日志、自动重载、错误堆栈
        run_server(sys_, port=args.port, debug=True)
    else:
        run_cli(sys_)


if __name__ == "__main__":
    main()
