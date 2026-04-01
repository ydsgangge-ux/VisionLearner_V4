# skills/calculator.py - 计算器技能
SKILL_NAME = "计算器"
SKILL_DESC = "计算数学表达式"
TRIGGERS   = ["计算", "算一下", "等于多少", "+", "-", "*", "/", "="]

def run(query: str, context: dict) -> str:
    import re
    # 提取数学表达式
    expr = re.sub(r"[^0-9+\-*/().% ]", "", query).strip()
    if not expr:
        return "没有找到可计算的表达式"
    try:
        result = eval(expr, {"__builtins__": {}})
        return f"{expr} = {result}"
    except Exception as e:
        return f"计算失败: {e}"