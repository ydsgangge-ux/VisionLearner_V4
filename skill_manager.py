# skill_manager.py - 技能系统
"""
让系统可以自己扩展能力。

核心思想（来自 OpenClaw）：
- 每个"技能"是一个独立的 Python 文件
- 技能文件放在 skills/ 目录下
- 系统启动时自动加载所有技能
- 用户描述新能力 → LLM生成技能文件 → 自动注册 → 立即可用

技能文件格式：
  skills/
    weather.py       # 查天气
    calculator.py    # 计算器
    timer.py         # 定时提醒
    crawler.py       # 网页爬取
    ...

每个技能文件必须有：
  SKILL_NAME = "天气查询"
  SKILL_DESC = "查询指定城市的天气"
  TRIGGERS   = ["天气", "weather", "气温"]   ← 触发关键词
  def run(query: str, context: dict) -> str: ← 执行入口
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from llm_client import LLMClient, get_client
from storage import DataManager


# ========== 技能基类 ==========

class SkillBase:
    """所有技能的基类（也可以不继承，只要有规定的属性就行）"""
    SKILL_NAME: str = ""
    SKILL_DESC: str = ""
    TRIGGERS: List[str] = []

    def run(self, query: str, context: dict) -> str:
        raise NotImplementedError


# ========== 内置技能示例 ==========

BUILTIN_CALCULATOR = '''
# skills/calculator.py - 计算器技能
SKILL_NAME = "计算器"
SKILL_DESC = "计算数学表达式"
TRIGGERS   = ["计算", "算一下", "等于多少", "+", "-", "*", "/", "="]

def run(query: str, context: dict) -> str:
    import re
    # 提取数学表达式
    expr = re.sub(r"[^0-9+\\-*/().% ]", "", query).strip()
    if not expr:
        return "没有找到可计算的表达式"
    try:
        result = eval(expr, {"__builtins__": {}})
        return f"{expr} = {result}"
    except Exception as e:
        return f"计算失败: {e}"
'''

BUILTIN_TIMER = '''
# skills/timer.py - 时间技能
SKILL_NAME = "时间日期"
SKILL_DESC = "查询当前时间和日期"
TRIGGERS   = ["现在几点", "今天日期", "今天是", "几号", "星期几", "时间"]

def run(query: str, context: dict) -> str:
    from datetime import datetime
    now = datetime.now()
    weekdays = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
    wd = weekdays[now.weekday()]
    return f"现在是 {now.strftime('%Y年%m月%d日')} {wd} {now.strftime('%H:%M:%S')}"
'''

BUILTIN_PROGRESS = '''
# skills/progress.py - 进度查询技能
SKILL_NAME = "学习进度"
SKILL_DESC = "查询当前学习目标的进度"
TRIGGERS   = ["进度", "学了多少", "完成度", "还差多少", "学会了吗"]

def run(query: str, context: dict) -> str:
    # context 里有系统传入的进度数据
    progress = context.get("progress")
    if not progress:
        return "暂无进度数据，请先创建学习目标"
    total = progress.get("total_units", 0)
    learned = progress.get("learned_units", 0)
    pct = progress.get("overall_completion", 0)
    bar = "█" * int(pct*20) + "░" * (20-int(pct*20))
    return (f"学习进度：\\n"
            f"[{bar}] {pct:.0%}\\n"
            f"总单元：{total} | 已学会：{learned}")
'''


# ========== 技能管理器 ==========

class SkillManager:
    """
    技能管理器

    职责：
    1. 启动时扫描 skills/ 目录，加载所有技能
    2. 用户提问时，判断是否有技能可以处理
    3. 用户描述新能力时，让LLM生成技能文件并注册
    """

    def __init__(self, skills_dir: str = "./skills",
                 db: Optional[DataManager] = None,
                 llm: Optional[LLMClient] = None):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(exist_ok=True)
        self.db  = db  or DataManager()
        self.llm = llm or get_client()

        # 已加载的技能: {skill_name: module}
        self._skills: Dict[str, Any] = {}
        # 触发词索引: {keyword: skill_name}
        self._trigger_index: Dict[str, str] = {}

        self._install_builtins()
        self._load_all()

    # ===== 初始化 =====

    def _install_builtins(self):
        """安装内置技能（如果还没有）"""
        builtins = {
            "calculator.py": BUILTIN_CALCULATOR,
            "timer.py":      BUILTIN_TIMER,
            "progress.py":   BUILTIN_PROGRESS,
        }
        for fname, code in builtins.items():
            fpath = self.skills_dir / fname
            if not fpath.exists():
                fpath.write_text(code.strip(), encoding="utf-8")

    def _load_all(self):
        """扫描并加载 skills/ 目录下所有技能"""
        loaded = 0
        for fpath in sorted(self.skills_dir.glob("*.py")):
            if fpath.name.startswith("_"):
                continue
            if self._load_skill_file(fpath):
                loaded += 1
        if loaded:
            print(f"   OK Loaded {loaded} skills")

    def _load_skill_file(self, fpath: Path) -> bool:
        """加载单个技能文件"""
        try:
            spec = importlib.util.spec_from_file_location(
                fpath.stem, fpath
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            name     = getattr(mod, "SKILL_NAME", fpath.stem)
            triggers = getattr(mod, "TRIGGERS", [])
            run_fn   = getattr(mod, "run", None)

            if not callable(run_fn):
                return False

            self._skills[name] = mod
            for kw in triggers:
                self._trigger_index[kw.lower()] = name

            return True
        except Exception as e:
            print(f"   [!]  Skill load failed {fpath.name}: {e}")
            return False

    # ===== 核心：判断+执行 =====

    def can_handle(self, query: str) -> Optional[str]:
        """
        判断是否有技能能处理这个查询。
        返回技能名，或 None。
        不调用LLM，纯关键词匹配。
        """
        q_lower = query.lower()
        for kw, skill_name in self._trigger_index.items():
            if kw in q_lower:
                return skill_name
        return None

    def execute(self, query: str, skill_name: str,
                context: dict = None) -> str:
        """执行指定技能"""
        mod = self._skills.get(skill_name)
        if not mod:
            return f"技能「{skill_name}」未找到"
        try:
            result = mod.run(query, context or {})
            return str(result)
        except Exception as e:
            return f"技能执行出错: {e}"

    def handle(self, query: str, context: dict = None) -> Optional[str]:
        """
        尝试用技能处理查询。
        有技能能处理就返回结果，否则返回 None（交给知识库/LLM处理）。
        """
        skill_name = self.can_handle(query)
        if skill_name:
            return self.execute(query, skill_name, context)
        return None

    # ===== 技能生成（LLM）=====

    def create_skill(self, description: str) -> Dict:
        """
        用户描述新能力 → LLM生成技能文件 → 注册

        例如：
        "我想让系统能查询股票价格"
        "帮我加一个翻译功能"
        "让系统能执行shell命令"
        """
        print(f"🔧 正在生成技能：{description}")

        prompt = f'''为以下功能生成一个 Python 技能文件：
"{description}"

文件必须包含：
1. SKILL_NAME = "技能名称"（中文，简短）
2. SKILL_DESC = "功能描述"（一句话）
3. TRIGGERS = ["触发词1", "触发词2", ...]（用户输入包含这些词时触发，3-8个）
4. def run(query: str, context: dict) -> str:
   - query 是用户的原始输入
   - context 是系统上下文字典
   - 返回字符串结果

要求：
- 只用 Python 标准库（不用第三方库）
- 代码简洁可用，有基本错误处理
- 如果需要外部API，在函数里说明如何配置

只返回 Python 代码，不要任何解释和 markdown 标记。'''

        code = self.llm.chat(prompt, system="只返回Python代码，不要```标记。")

        # 清理代码
        code = code.strip()
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:])
        if code.endswith("```"):
            code = "\n".join(code.split("\n")[:-1])
        code = code.strip()

        # 从代码提取技能名
        import re
        name_match = re.search(r'SKILL_NAME\s*=\s*["\'](.+?)["\']', code)
        skill_name = name_match.group(1) if name_match else "新技能"
        safe_name  = re.sub(r'[^\w]', '_', skill_name)

        # 保存文件
        fpath = self.skills_dir / f"{safe_name}.py"
        fpath.write_text(code, encoding="utf-8")

        # 加载注册
        success = self._load_skill_file(fpath)

        result = {
            "success": success,
            "skill_name": skill_name,
            "file": str(fpath),
            "triggers": [],
        }

        if success and skill_name in self._skills:
            mod = self._skills[skill_name]
            result["triggers"] = getattr(mod, "TRIGGERS", [])
            print(f"✅ 技能「{skill_name}」已创建并注册")
            print(f"   触发词: {result['triggers']}")
        else:
            print(f"[!] Skill file saved but load failed, check: {fpath}")

        return result

    # ===== 技能管理 =====

    def list_skills(self) -> List[Dict]:
        """列出所有已加载的技能"""
        result = []
        for name, mod in self._skills.items():
            result.append({
                "name": name,
                "desc": getattr(mod, "SKILL_DESC", ""),
                "triggers": getattr(mod, "TRIGGERS", [])[:3],
                "file": getattr(mod, "__file__", ""),
            })
        return result

    def reload_skill(self, skill_name: str) -> bool:
        """重新加载某个技能（修改后用）"""
        for name, mod in self._skills.items():
            if name == skill_name:
                fpath = Path(getattr(mod, "__file__", ""))
                if fpath.exists():
                    # 先移除旧的触发词
                    old_triggers = getattr(mod, "TRIGGERS", [])
                    for kw in old_triggers:
                        self._trigger_index.pop(kw.lower(), None)
                    self._skills.pop(name, None)
                    return self._load_skill_file(fpath)
        return False

    def delete_skill(self, skill_name: str) -> bool:
        """删除技能"""
        mod = self._skills.get(skill_name)
        if not mod:
            return False
        fpath = Path(getattr(mod, "__file__", ""))
        triggers = getattr(mod, "TRIGGERS", [])
        for kw in triggers:
            self._trigger_index.pop(kw.lower(), None)
        self._skills.pop(skill_name, None)
        if fpath.exists():
            fpath.unlink()
        return True


# ========== 测试 ==========

if __name__ == "__main__":
    import shutil
    print("🧪 测试 skill_manager.py\n")

    sm = SkillManager("./test_skills")

    # 1. 列出内置技能
    skills = sm.list_skills()
    print(f"Loaded skills: {len(skills)}")
    for s in skills:
        print(f"  [{s['name']}] {s['desc']} 触发词:{s['triggers']}")

    # 2. 测试技能触发
    print("\n触发测试：")
    tests = [
        ("1+1等于多少", True),
        ("现在几点了", True),
        ("蠢字怎么读", False),
    ]
    for q, expect in tests:
        skill = sm.can_handle(q)
        ok = "✅" if (skill is not None) == expect else "❌"
        print(f"  {ok} '{q}' → {skill or '无技能'}")

    # 3. 测试执行
    print("\n执行测试：")
    print(f"  计算: {sm.handle('100+200等于多少')}")
    print(f"  时间: {sm.handle('现在几点')}")

    shutil.rmtree("./test_skills", ignore_errors=True)
    print("\n✅ 测试完成")
