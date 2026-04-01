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
    return (f"学习进度：\n"
            f"[{bar}] {pct:.0%}\n"
            f"总单元：{total} | 已学会：{learned}")