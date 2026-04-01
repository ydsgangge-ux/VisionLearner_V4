# collector.py - 节点收集器
"""
系统"学习"的核心执行者。

核心逻辑：
- 思维导图每个节点有 collected=True/False
- collected=False 的节点 = 系统还不知道这个知识
- 遇到空节点就调LLM学习，存入，标记为已收集
- 整棵树收集完 = 这个目标学会了

对外只需要两个方法：
  collect_on_demand(query, root)   ← 用户提问时触发，按需收集
  collect_tree(root)               ← 主动批量收集整棵树
"""

import time
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from llm_client import LLMClient, get_client
from storage import DataManager, VectorStore
from foundation import MindMapNode as FoundationMindMapNode


# ========== 目标类型配置 ==========

GOAL_TYPE_CONFIGS = {
    "characters": {
        "tree_template": {
            "title": "{unit}",
            "children": [
                {"title": "读音", "importance": 1.0, "node_type": "fact"},
                {"title": "字形", "importance": 0.9, "children": [
                    {"title": "笔画", "importance": 0.8, "node_type": "fact"},
                    {"title": "部首", "importance": 0.8, "node_type": "fact"},
                    {"title": "结构", "importance": 0.6, "node_type": "fact"},
                ]},
                {"title": "含义", "importance": 1.0, "node_type": "concept"},
                {"title": "用法", "importance": 0.9, "children": [
                    {"title": "组词", "importance": 0.9, "node_type": "example"},
                    {"title": "例句", "importance": 0.7, "node_type": "example"},
                ]},
                {"title": "记忆方法", "importance": 0.7, "node_type": "skill"},
                {"title": "关联", "importance": 0.5, "children": [
                    {"title": "形近字", "importance": 0.6, "node_type": "concept"},
                    {"title": "同音字", "importance": 0.5, "node_type": "concept"},
                ]},
            ]
        },
        "prompts": {
            "读音": '"{unit}"字的拼音是什么？注意：答案必须是拼音字母（如 chǔn），绝对不能是汉字本身。只返回JSON：{{"content": "拼音（声调）"}}',
            "笔画": '"{unit}"字共几画？请给出准确答案。只返回JSON：{{"content": "答案"}}',
            "部首": '"{unit}"字的部首是什么？请给出准确答案。只返回JSON：{{"content": "答案"}}',
            "结构": '"{unit}"字是什么结构（左右结构、上下结构等）？请给出准确答案。只返回JSON：{{"content": "答案"}}',
            "含义": '"{unit}"字有哪些主要含义？请详细分析并列举。只返回JSON：{{"content": "含义列表及说明"}}',
            "组词": '用"{unit}"字组常用词语，并详细解释每个词语的含义和用法。只返回JSON：{{"content": "词语1：含义和用法；词语2：含义和用法"}}',
            "例句": '用"{unit}"字造常用例句，要求句子有实际意义。只返回JSON：{{"content": ["例句1", "例句2", "例句3"]}}',
            "记忆方法": '"{unit}"字有什么好的记忆方法或口诀？请详细分析并给出实用方法。只返回JSON：{{"content": "记忆方法"}}',
            "形近字": '"{unit}"字有哪些形近字？请详细列举并说明如何区分。只返回JSON：{{"content": "形近字及区分方法"}}',
            "同音字": '"{unit}"字有哪些常见的同音字？请详细列举并标注常用程度。只返回JSON：{{"content": "同音字列表及说明"}}',
        },
        "default_prompt": '请详细分析汉字"{unit}"的{node_title}，给出真实准确的具体内容。只返回JSON格式：{{"content": "你的详细回答"}}',
    },
    "vocabulary": {
        "tree_template": {
            "title": "{unit}",
            "children": [
                {"title": "发音", "importance": 1.0, "node_type": "fact"},
                {"title": "释义", "importance": 1.0, "node_type": "concept"},
                {"title": "例句", "importance": 0.9, "node_type": "example"},
                {"title": "搭配", "importance": 0.8, "node_type": "skill"},
                {"title": "同义词", "importance": 0.6, "node_type": "concept"},
                {"title": "记忆方法", "importance": 0.7, "node_type": "skill"},
            ]
        },
        "prompts": {
            "发音": '"{unit}"的发音是什么？请详细分析，包括国际音标和英式/美式发音差异。只返回JSON：{{"content": "发音详情"}}',
            "释义": '"{unit}"的释义有哪些？请详细分析并列举。只返回JSON：{{"content": "含义列表及说明"}}',
            "例句": '"{unit}"的例句？要求句子有实际意义，体现词汇用法。只返回JSON：{{"content": ["例句1", "例句2", "例句3"]}}',
            "搭配": '"{unit}"的常用搭配有哪些？请详细列举并说明适用场景。只返回JSON：{{"content": "搭配1：场景；搭配2：场景；搭配3：场景"}}',
            "同义词": '"{unit}"的同义词有哪些？请详细列举并说明细微差异。只返回JSON：{{"content": "同义词及差异说明"}}',
            "记忆方法": '"{unit}"如何有效记忆？请详细分析并给出实用方法。只返回JSON：{{"content": "记忆方法"}}',
        },
        "default_prompt": '请详细分析词汇"{unit}"的{node_title}，给出真实准确的具体内容和例证。只返回JSON格式：{{"content": "你的详细回答"}}',
    },
    "programming": {
        "tree_template": {
            "title": "{unit}",
            "children": [
                {"title": "定义", "importance": 1.0, "node_type": "concept"},
                {"title": "语法", "importance": 1.0, "node_type": "skill"},
                {"title": "代码示例", "importance": 0.9, "node_type": "example"},
                {"title": "使用场景", "importance": 0.8, "node_type": "concept"},
                {"title": "常见错误", "importance": 0.8, "node_type": "fact"},
                {"title": "相关概念", "importance": 0.6, "node_type": "concept"},
            ]
        },
        "prompts": {
            "定义": '"{unit}"的定义是什么？请详细分析其核心概念、作用原理和特点。只返回JSON：{{"content": "详细定义"}}',
            "语法": '"{unit}"的语法格式是什么？请详细分析，给出具体语法规则、参数说明和使用注意事项。只返回JSON：{{"content": "语法详情"}}',
            "代码示例": '"{unit}"的代码示例？请详细分析，给出实际可运行的代码示例并解释每行代码的作用。只返回JSON：{{"content": "示例代码及详细注释"}}',
            "使用场景": '"{unit}"在什么场景下使用？请详细分析典型应用场景并说明为何选择该方案。只返回JSON：{{"content": "场景及说明列表"}}',
            "常见错误": '使用"{unit}"时容易犯哪些错误？请详细分析常见问题，说明错误原因和解决方法。只返回JSON：{{"content": "错误列表及解决方法"}}',
            "相关概念": '与"{unit}"相关的编程概念有哪些？请详细列举并说明关联关系。只返回JSON：{{"content": "相关概念及关联说明"}}',
        },
        "default_prompt": '请详细分析编程概念"{unit}"的{node_title}，给出真实准确的具体内容、代码或例子。只返回JSON格式：{{"content": "你的详细回答"}}',
    },
    "general": {
        "tree_template": {
            "title": "{unit}",
            "children": [
                {"title": "定义", "importance": 1.0, "node_type": "concept"},
                {"title": "核心要点", "importance": 0.9, "node_type": "concept"},
                {"title": "具体例子", "importance": 0.8, "node_type": "example"},
                {"title": "应用场景", "importance": 0.7, "node_type": "skill"},
                {"title": "记忆方法", "importance": 0.6, "node_type": "skill"},
            ]
        },
        "prompts": {
            "定义":   '请详细分析"{unit}"的定义，包括其核心概念和本质特征。只返回JSON：{{"content": "详细定义内容"}}',
            "核心要点": '"{unit}"的核心要点有哪些？请详细分析并列举。只返回JSON：{{"content": "要点列表及说明"}}',
            "具体例子": '"{unit}"的具体例子有哪些？请详细分析并举例。只返回JSON：{{"content": "例子列表及说明"}}',
            "应用场景": '"{unit}"在现实生活或工作中有哪些具体应用场景？请详细分析并列举。只返回JSON：{{"content": "应用场景列表及说明"}}',
            "记忆方法": '如何深入理解和记忆"{unit}"？请详细分析并给出实用方法。只返回JSON：{{"content": "记忆方法及说明"}}',
        },
        "default_prompt": '请详细分析"{unit}"的{node_title}，给出真实准确的具体内容、细节和例子，从多个维度展开（如定义、机制、影响、应用等），避免空泛的模板语言。只返回JSON格式：{{"content": "你的详细回答"}}',
    },
}


# ========== 节点收集器 ==========

class NodeCollector:
    """
    节点收集器 - 系统学习的执行者

    两种模式：
    1. collect_on_demand  用户提问时按需收集
    2. collect_tree       主动批量收集整棵树
    """

    def __init__(self, db: Optional[DataManager] = None,
                 llm: Optional[LLMClient] = None,
                 vector: Optional['VectorStore'] = None):
        self.db = db or DataManager()
        self.llm = llm or get_client()
        self.vector = vector                # VectorStore 引用，None时跳过向量写入
        self._counter = 0
        self._current_goal_id: str = ""     # 当前正在收集的 goal_id
        self._current_unit: str = ""        # 当前正在收集的 unit

    def _gen_id(self) -> str:
        self._counter += 1
        return f"node_{int(time.time()*1000)}_{self._counter}"

    # ===== 树的创建 =====

    def build_tree_from_template(self, unit: str, goal_type: str, max_depth: int = 3) -> FoundationMindMapNode:
        """根据目标类型为知识单元创建空树（所有节点 collected=False）"""
        config = GOAL_TYPE_CONFIGS.get(goal_type, GOAL_TYPE_CONFIGS["general"])
        return self._tmpl_to_tree(config["tree_template"], unit, depth=0, max_depth=max_depth)

    def _tmpl_to_tree(self, tmpl: Dict, unit: str,
                      depth: int, parent_id: str = "", max_depth: int = 3) -> FoundationMindMapNode:
        title = tmpl["title"].replace("{unit}", unit)
        node = FoundationMindMapNode(
            id=self._gen_id(),
            title=title,
            depth=depth,
            importance=tmpl.get("importance", 0.5),
            node_type=tmpl.get("node_type", "concept"),
            parent_id=parent_id,
            collected=False,
        )
        if depth < max_depth:
            for child_tmpl in tmpl.get("children", []):
                child = self._tmpl_to_tree(child_tmpl, unit, depth + 1, node.id, max_depth)
                node.children.append(child)
        return node

    def build_tree_from_llm(self, unit: str,
                            goal_description: str, depth: int = 3) -> FoundationMindMapNode:
        """让LLM为知识单元设计思维导图结构（用于非标准类型）"""
        result = self.llm.generate_json(
            prompt=f"""为学习主题"{unit}"（来自：{goal_description}）
设计{depth}层思维导图结构。返回JSON：
{{
  "title": "{unit}",
  "children": [
    {{"title": "子主题", "importance": 0.9, "node_type": "concept",
      "children": [{{"title": "知识点", "importance": 0.8,
                    "node_type": "fact", "children": []}}]}}
  ]
}}
importance 0~1，node_type 从 concept/skill/example/fact 选。""",
            system="只返回JSON。"
        )
        if result:
            return self._tmpl_to_tree(result, unit, depth=0)
        return self.build_tree_from_template(unit, "general")

    # ===== 按需收集 =====

    def collect_on_demand(self, query: str, root: FoundationMindMapNode,
                          goal_type: str, unit: str = "",
                          goal_id: str = "") -> Tuple[Optional[FoundationMindMapNode], Any]:
        """
        用户提问时触发。
        找到对应节点 → 已收集直接返回 → 未收集就学 → 节点不存在就创建再学
        goal_id：传入后自动同步向量库
        """
        # 记录当前上下文，供 _collect_single 写向量库用
        self._current_goal_id = goal_id
        self._current_unit    = unit or root.title

        node = self._find_best_match(query, root)

        if node and node.collected:
            return node, node.content

        if node and not node.collected:
            content = self._collect_single(node, unit or root.title, goal_type)
            return node, content

        # 节点不存在，创建并学习
        new_node = FoundationMindMapNode(
            id=self._gen_id(), title=query,
            depth=1, importance=0.5,
            node_type="concept", parent_id=root.id,
        )
        root.children.append(new_node)
        content = self._collect_single(new_node, unit or root.title, goal_type)
        return new_node, content

    def _find_best_match(self, query: str,
                         root: FoundationMindMapNode) -> Optional[FoundationMindMapNode]:
        query_lower = query.lower()
        best, best_score = None, 0.0
        for node in root._all_nodes():
            if node.id == root.id:
                continue
            node_lower = node.title.lower()
            if query_lower == node_lower:
                return node
            if query_lower in node_lower or node_lower in query_lower:
                score = len(node_lower) / max(len(query_lower), 1)
                if score > best_score:
                    best_score, best = score, node
        return best if best_score > 0.3 else None

    # ===== 单节点收集（唯一调用LLM的地方） =====

    def _collect_single(self, node: FoundationMindMapNode,
                        unit: str, goal_type: str) -> Any:
        prompt = self._build_prompt(node.title, unit, goal_type)
        result = self.llm.generate_json(prompt, max_tokens=3600)  # 增加到3600，让内容更详细

        if result and "content" in result:
            content = result["content"]
        elif result:
            content = result
        else:
            content = self.llm.chat(
                f'关于"{unit}"的"{node.title}"，请详细回答，给出具体内容和例子。',
                system="请用3-5句话详细回答，包含具体细节和例子。"
            )

        # ── 内容校验：防止 LLM 返回占位符 ──
        BAD = {"内容", "答案", "content", "回答", "你的实际回答",
               "你的回答", "内容列表", "含义列表", "形近字及区分方法"}

        def _is_bad(c):
            if c is None or c == "":
                return True
            if isinstance(c, str) and c.strip() in BAD:
                return True
            if isinstance(c, list) and all(
                isinstance(x, str) and x.strip() in BAD for x in c
            ):
                return True
            return False

        if _is_bad(content):
            # 重试一次，用更明确的 prompt
            retry_prompt = (
                f'请用4-6句话详细介绍"{unit}"的{node.title}，'
                f'给出真实具体的内容和实际案例，不要模板语言。'
            )
            content = self.llm.chat(retry_prompt, system="请直接给出详细准确的答案。")

            if _is_bad(content):
                # 重试仍然是废数据，不污染知识库
                node.collected = False
                return "（暂无，请重新提问）"

        node.content = content
        node.collected = True
        node.collected_at = datetime.now().isoformat()
        node.collected_by = f"llm:{self.llm.model}"

        # ── 同步写入向量库（如果有 VectorStore 且知道 goal_id）──
        if self.vector and self._current_goal_id and self._current_unit:
            try:
                from conversation import format_content
                content_str = format_content(content)
                if content_str and content_str != "（暂无）":
                    # 细粒度：单节点
                    self.vector.add_unit_knowledge(
                        goal_id   = self._current_goal_id,
                        unit      = self._current_unit,
                        content_text = content_str,
                        node_title   = node.title,
                    )
            except Exception as e:
                import logging
                logging.getLogger("NodeCollector").warning(f"向量写入失败：{e}")

        return content

    def _build_prompt(self, node_title: str, unit: str, goal_type: str) -> str:
        config = GOAL_TYPE_CONFIGS.get(goal_type, GOAL_TYPE_CONFIGS["general"])
        prompts = config.get("prompts", {})
        if node_title in prompts:
            return prompts[node_title].replace("{unit}", unit)
        for key, tmpl in prompts.items():
            if key in node_title or node_title in key:
                return tmpl.replace("{unit}", unit)
        default = config.get("default_prompt",
                             '关于"{unit}"的"{node_title}"给出内容。只返回JSON：{{"content": "内容"}}')
        return default.replace("{unit}", unit).replace("{node_title}", node_title)

    # ===== 批量收集 =====

    def collect_tree(self, root: FoundationMindMapNode, unit: str,
                     goal_type: str, on_progress=None,
                     goal_id: str = "") -> Dict:
        """主动收集整棵树，按重要性顺序"""
        # 设置上下文，让 _collect_single 知道写哪个向量库
        self._current_goal_id = goal_id
        self._current_unit    = unit

        uncollected = root.uncollected_nodes()
        total, done, failed = len(uncollected), 0, 0

        for node in uncollected:
            try:
                self._collect_single(node, unit, goal_type)
                done += 1
            except Exception as e:
                failed += 1
                print(f"   [!]  {node.title}: {e}")
            if on_progress:
                on_progress(done + failed, total, node.title)
            time.sleep(0.3)

        return {"total": total, "done": done, "failed": failed,
                "completion": root.completion_rate()}

    # ===== 持久化 =====

    def save_tree(self, goal_id: str, unit: str, root: FoundationMindMapNode):
        # 使用to_dict_with_children保存完整的树结构（向后兼容）
        tree_dict = root.to_dict_with_children()
        self.db.storage.save("mindmap_trees", f"{goal_id}_{_safe_key(unit)}", {
            "goal_id": goal_id, "unit": unit,
            "tree": tree_dict,
            "saved_at": datetime.now().isoformat(),
            "completion_rate": root.completion_rate(),
        })

    def load_tree(self, goal_id: str, unit: str) -> Optional[FoundationMindMapNode]:
        data = self.db.storage.load("mindmap_trees", f"{goal_id}_{_safe_key(unit)}")
        if data and "tree" in data:
            tree_dict = data["tree"]

            # 兼容旧数据：如果包含"children"数组，转换为children_ids
            def convert_tree_format(tree_data):
                """递归转换树格式：children -> children_ids"""
                if "children" in tree_data and isinstance(tree_data["children"], list):
                    # 保存children_ids
                    children_ids = []
                    for child in tree_data["children"]:
                        children_ids.append(child["id"])
                        # 递归处理子节点
                        convert_tree_format(child)
                    # 如果没有children_ids，则创建
                    if "children_ids" not in tree_data:
                        tree_data["children_ids"] = children_ids

            convert_tree_format(tree_dict)

            tree = FoundationMindMapNode.from_dict(tree_dict)
            # 构建children对象列表
            node_map: Dict[str, FoundationMindMapNode] = {}

            def build_children(node):
                node_map[node.id] = node
                children_list = []
                for child_id in node.children_ids:
                    # 如果这个节点已经处理过，直接从 node_map 获取
                    if child_id in node_map:
                        children_list.append(node_map[child_id])
                        continue
                    # 否则从 tree_data 中递归查找并构建
                    child = _find_child_in_tree(tree_dict, child_id)
                    if child:
                        children_list.append(child)
                        build_children(child)
                # 使用set_children方法而不是直接append
                node.set_children(children_list)

            def _find_child_in_tree(tree_data, node_id):
                """递归在字典树中查找节点"""
                if tree_data.get("id") == node_id:
                    return FoundationMindMapNode.from_dict(tree_data)
                for child_dict in tree_data.get("children", []):
                    found = _find_child_in_tree(child_dict, node_id)
                    if found:
                        return found
                return None

            build_children(tree)
            return tree
        return None

    def save_goal_units(self, goal_id: str, units: List[str]):
        self.db.storage.save("goal_units", goal_id, {
            "goal_id": goal_id, "units": units,
            "saved_at": datetime.now().isoformat(),
        })

    def load_goal_units(self, goal_id: str) -> List[str]:
        data = self.db.storage.load("goal_units", goal_id)
        return data.get("units", []) if data else []

    def get_completion_report(self, goal_id: str, units: List[str]) -> Dict:
        total_nodes = collected_nodes = 0
        learned = 0
        learned_content = []  # 存储已学习的内容
        for unit in units:
            tree = self.load_tree(goal_id, unit)
            if tree:
                all_n = tree._all_nodes()
                # 过滤掉根节点（depth=0），只计算知识节点
                all_knowledge_nodes = [n for n in all_n if n.depth > 0]
                c = sum(1 for n in all_knowledge_nodes if n.collected)
                total_nodes += len(all_knowledge_nodes)
                collected_nodes += c
                if tree.is_learned():
                    learned += 1

                # 收集已学习的内容
                for node in all_knowledge_nodes:
                    if node.collected and node.content and node.content != "（暂无，请重新提问）":
                        from conversation import format_content
                        learned_content.append({
                            "unit": unit,
                            "node": node.title,
                            "content": format_content(node.content),
                            "collected_at": node.collected_at
                        })
        return {
            "total_units": len(units),
            "learned_units": learned,
            "total_nodes": total_nodes,
            "collected_nodes": collected_nodes,
            "overall_completion": collected_nodes / total_nodes if total_nodes else 0,
            "learned_content": learned_content  # 新增：已学习的内容列表
        }


def _safe_key(text: str) -> str:
    import hashlib
    prefix = re.sub(r'[^\w]', '', text[:4])
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


# ========== 测试 ==========

if __name__ == "__main__":
    import shutil
    print("Test collector.py\n")

    db = DataManager("./test_collector")
    collector = NodeCollector(db)

    # 构建树
    tree = collector.build_tree_from_template("蠢", "characters")
    all_n = tree._all_nodes()
    print(f"树构建完成：{len(all_n)}个节点，完成度 {tree.completion_rate():.0%}")

    def show(node, indent=0):
        print(f"{'  '*indent}{'OK' if node.collected else 'NO'} {node.title}")
        for c in node.children:
            show(c, indent+1)
    show(tree)

    # 按需收集
    print("\n按需收集「读音」...")
    node, content = collector.collect_on_demand("读音", tree, "characters", "蠢")
    print(f"  内容: {content}")

    print("\n再次查询「读音」（应直接返回）...")
    node2, content2 = collector.collect_on_demand("读音", tree, "characters", "蠢")
    print(f"  内容: {content2}  已收集: {node2.collected}")

    # 持久化
    collector.save_tree("test_goal", "蠢", tree)
    loaded = collector.load_tree("test_goal", "蠢")
    reading = loaded.find_by_title("读音") if loaded else None
    print(f"\nPersistence test: after loading, reading node collected={reading.collected if reading else '?'}")

    shutil.rmtree("./test_collector", ignore_errors=True)
    print("\nOK - Test completed")
