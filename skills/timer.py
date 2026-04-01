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