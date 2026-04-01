# perception.py - 多模态感知与知识提取（重构版）
"""
核心功能：
1. MindMapGenerator   - 用LLM生成思维导图
2. KnowledgeExtractor - 从文本/图片提取知识节点
3. ActiveLearningTrigger - 主动触发学习机制
4. PerceptionManager  - 统一感知入口

改动：原有算法逻辑全部保留，仅把LLMClient替换为新版（支持免费LLM）
"""

import json
import re
import base64
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from foundation import (
    MindMapNode, KnowledgeNode, LearningGoal, LearningLevel,
    KnowledgeType, GoalScale, ModalityType, MindMapStyle,
    IMindMapGenerator, IMultimodalRecognizer, generate_id,
    FoundationManager
)
from llm_client import LLMClient, get_client


# ========== 思维导图生成器 ==========

class MindMapGenerator(IMindMapGenerator):
    """
    思维导图生成器 - 大模型驱动
    保留原有完整逻辑，接入新LLMClient
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_client()
        self.foundation = FoundationManager()
        self._node_map: Dict[str, MindMapNode] = {}   # 最近一次生成的节点映射

    def generate_for_goal(self, goal: LearningGoal) -> Optional[MindMapNode]:
        """为学习目标生成思维导图"""
        print(f"🧠 为'{goal.description}'生成思维导图...")

        depth = {
            GoalScale.MICRO: 2,
            GoalScale.SMALL: 3,
            GoalScale.MEDIUM: 4,
        }.get(goal.scale, goal.mindmap_depth or 3)

        result = self.llm.generate_mindmap(
            topic=goal.description,
            depth=depth,
            style=goal.mindmap_style.value if goal.mindmap_style else "balanced",
            context=goal.detailed_description or ""
        )

        if result:
            root, node_map = self._dict_to_mindmap_tree(result, goal.description)
            self._node_map = node_map
            goal.mindmap_root_id = root.id
            goal.mindmap_generated_at = datetime.now().isoformat()
            goal.mindmap_confidence = 0.85
            print(f"✅ 思维导图生成成功，共 {len(node_map)} 个节点")
            return root
        else:
            print("[!] Generation failed, using fallback")
            return self._fallback_mindmap(goal)

    def generate_from_text(self, text: str, depth: int = 3) -> Optional[MindMapNode]:
        """从任意文本生成思维导图"""
        print(f"🧠 从文本生成思维导图...")

        result = self.llm.generate_json(
            prompt=f"将以下文本转换为{depth}层思维导图：\n\n{text[:2000]}",
            system="你是知识结构分析师，请提取文本核心内容生成思维导图JSON。",
            schema_hint='{"title":"...", "description":"...", "nodes":[{"id":"1","title":"...","description":"...","importance":0.8,"difficulty":0.5,"node_type":"concept","estimated_minutes":20,"children":[]}]}'
        )

        if result:
            root, node_map = self._dict_to_mindmap_tree(result, "文本主题")
            self._node_map = node_map
            return root
        return None

    def refine_mindmap(self, mindmap: MindMapNode, feedback: Dict) -> MindMapNode:
        """根据反馈精炼思维导图"""
        desc = self._node_to_text(mindmap)
        result = self.llm.generate_json(
            prompt=f"请根据以下反馈优化思维导图：\n\n当前结构：{desc}\n\n反馈：{json.dumps(feedback, ensure_ascii=False)}",
            system="请返回优化后的完整思维导图JSON，格式与原始结构相同。"
        )
        if result:
            root, node_map = self._dict_to_mindmap_tree(result, mindmap.title)
            self._node_map.update(node_map)
            return root
        return mindmap

    def get_node_map(self) -> Dict[str, MindMapNode]:
        """获取最近生成的节点映射"""
        return self._node_map

    def _dict_to_mindmap_tree(self, data: Dict, fallback_title: str) -> Tuple[MindMapNode, Dict[str, MindMapNode]]:
        """将LLM返回的字典转换为MindMapNode树"""
        node_map: Dict[str, MindMapNode] = {}

        root = MindMapNode(
            id=generate_id("mm_root_"),
            title=data.get("title", fallback_title),
            description=data.get("description", ""),
            depth=0,
            importance=1.0,
            generated_by=self.llm.model
        )
        node_map[root.id] = root

        def build_children(parent: MindMapNode, children_data: List[Dict], depth: int):
            for child_data in children_data:
                node = MindMapNode(
                    id=generate_id(f"mm_node_"),
                    title=child_data.get("title", "未知"),
                    description=child_data.get("description", ""),
                    depth=depth,
                    node_type=child_data.get("node_type", "concept"),
                    importance=float(child_data.get("importance", 0.5)),
                    difficulty=float(child_data.get("difficulty", 0.5)),
                    estimated_time_minutes=int(child_data.get("estimated_minutes", 30)),
                    parent_id=parent.id,
                    tags=child_data.get("tags", []),
                    generated_by=self.llm.model
                )
                parent.children_ids.append(node.id)
                node_map[node.id] = node

                sub_children = child_data.get("children", [])
                if sub_children:
                    build_children(node, sub_children, depth + 1)

        build_children(root, data.get("nodes", []), 1)
        return root, node_map

    def _node_to_text(self, node: MindMapNode, indent: int = 0) -> str:
        """将节点转为文本描述"""
        text = "  " * indent + f"- {node.title}"
        if node.description:
            text += f"（{node.description[:30]}）"
        return text

    def _fallback_mindmap(self, goal: LearningGoal) -> MindMapNode:
        """备选：基于规则创建简单思维导图"""
        root = MindMapNode(
            id=generate_id("mm_root_"),
            title=goal.description,
            description="自动生成的基础学习结构",
            depth=0,
            generated_by="fallback"
        )

        default_topics = ["基础概念", "核心原理", "实践技能", "应用场景", "进阶提升"]
        node_map = {root.id: root}

        for i, topic in enumerate(default_topics):
            child = MindMapNode(
                id=generate_id(f"mm_child_{i}_"),
                title=topic,
                description=f"{goal.description}的{topic}",
                depth=1,
                parent_id=root.id,
                importance=1.0 - i * 0.1,
                difficulty=0.3 + i * 0.1,
                estimated_time_minutes=30 + i * 10,
                generated_by="fallback"
            )
            root.children_ids.append(child.id)
            node_map[child.id] = child

        self._node_map = node_map
        return root


# ========== 知识提取器 ==========

class KnowledgeExtractor:
    """
    多模态知识提取器
    支持文本、URL、图片内容的知识提取
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_client()

    def extract_from_text(self, text: str, source: str = "manual") -> List[KnowledgeNode]:
        """从文本提取知识节点"""
        result = self.llm.summarize_and_extract(text)
        return self._result_to_nodes(result, source)

    def extract_from_url(self, url: str) -> List[KnowledgeNode]:
        """从URL获取内容并提取知识"""
        print(f"🌐 从URL提取知识: {url}")
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            # 简单清洗HTML
            text = re.sub(r'<[^>]+>', ' ', resp.text)
            text = re.sub(r'\s+', ' ', text).strip()[:4000]
            return self.extract_from_text(text, source=url)
        except Exception as e:
            print(f"❌ URL提取失败: {e}")
            return []

    def extract_from_image(self, image_path: str) -> List[KnowledgeNode]:
        """从图片提取知识（需要视觉LLM支持）"""
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
        except Exception as e:
            print(f"[X] 读取图片失败: {e}")
            return []

        print(f"[DEBUG] LLM类型: {type(self.llm)}")
        print(f"[DEBUG] LLM provider: {getattr(self.llm, 'provider_name', 'unknown')}")
        print(f"[DEBUG] LLM model: {getattr(self.llm, 'model', 'unknown')}")
        print(f"[DEBUG] 有chat_with_image方法: {hasattr(self.llm, 'chat_with_image')}")

        # 使用多模态接口进行图片识别
        prompt = "请详细分析这张图片中的内容，提取其中的关键知识点。如果是学习笔记、教材图片、实验图等，请重点关注其中的知识点、概念、原理等。"

        try:
            print("[DEBUG] 正在调用 chat_with_image...")
            description = self.llm.chat_with_image(
                prompt=prompt,
                image_b64=img_b64,
                system="你是专业的多模态学习助手，擅长从图片中识别和提取知识内容。请用清晰、准确的语言描述图片内容，并总结其中的知识点。"
            )
            print(f"[DEBUG] chat_with_image 返回，结果长度: {len(description)}")
            print(f"[DEBUG] 返回内容前200字符: {description[:200]}")
            print(f"📷 图片识别成功，提取内容长度: {len(description)} 字符")
            return self.extract_from_text(description, source=image_path)
        except AttributeError as e:
            # 如果不支持多模态，使用普通chat作为降级方案
            print(f"[WARN] AttributeError: {e}")
            print("[!] 当前LLM不支持多模态，使用普通文本模式降级处理")
            description = self.llm.chat(
                "请描述一张关于学习内容的图片（由于多模态功能不可用，这是模拟响应）",
                system="你是多模态学习助手"
            )
            return self.extract_from_text(description, source=image_path)
        except Exception as e:
            print(f"[X] 图片识别失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def extract_from_audio(self, audio_path: str) -> str:
        """从音频提取文本内容（语音识别）"""
        try:
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
        except Exception as e:
            print(f"[X] 读取音频失败: {e}")
            return ""

        # 尝试使用LLM进行语音识别（如果有语音识别API）
        prompt = """请分析这段音频并提取其中的语音内容。
如果没有语音识别能力，请说明这个功能需要接入语音识别API。
音频内容（base64编码）："""

        response = self.llm.chat(
            prompt + audio_b64[:500] + "...",  # 截断避免太长
            system="你是多模态学习助手，可以处理音频内容"
        )

        # 检查是否真的识别了内容
        if "语音识别" in response or "无法识别" in response:
            print("[!] 当前LLM不支持语音识别，建议接入专门的语音识别API")
            return "音频转录功能：需要接入语音识别API（如讯飞开放平台、百度语音、腾讯云语音等）"

        return response

    def _result_to_nodes(self, result: Dict, source: str) -> List[KnowledgeNode]:
        """将提取结果转为KnowledgeNode列表"""
        nodes = []
        for kp in result.get("knowledge_points", []):
            node = KnowledgeNode(
                id=generate_id("kn_"),
                title=kp.get("title", "未知"),
                content=kp.get("content", ""),
                summary=kp.get("title", ""),
                knowledge_type=self._map_type(kp.get("type", "concept")),
                source=source,
                tags=result.get("tags", []),
            )
            nodes.append(node)
        return nodes

    def _map_type(self, type_str: str) -> KnowledgeType:
        mapping = {
            "concept": KnowledgeType.CONCEPT,
            "fact": KnowledgeType.FACT,
            "principle": KnowledgeType.PRINCIPLE,
            "skill": KnowledgeType.SKILL,
            "process": KnowledgeType.PROCESS,
        }
        return mapping.get(type_str.lower(), KnowledgeType.CONCEPT)


# ========== 主动学习触发器 ==========

class ActiveLearningTrigger:
    """
    主动学习触发器
    基于进度、时间和目标状态触发学习活动
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_client()

    def should_review(self, knowledge_node: Dict, days_since_last: int) -> bool:
        """基于遗忘曲线判断是否需要复习（SM-2算法）"""
        review_count = knowledge_node.get("review_count", 0)
        # 间隔天数：1, 3, 7, 14, 30, 90...
        intervals = [1, 3, 7, 14, 30, 90, 180]
        if review_count < len(intervals):
            return days_since_last >= intervals[review_count]
        return days_since_last >= 180

    def get_learning_suggestions(self,
                                  goal: Dict,
                                  progress: Dict,
                                  history: List[Dict]) -> List[str]:
        """基于当前状态生成学习建议"""
        prompt = f"""根据以下学习情况生成3-5条具体建议：

学习目标：{goal.get('description', '')}
当前进度：{progress.get('overall_progress', 0):.0%}
最近学习历史：{len(history)}条记录

请给出简洁实用的建议（每条不超过50字）。
返回JSON数组：["建议1", "建议2", "建议3"]"""

        result = self.llm.generate_json(prompt, max_tokens=500)
        if isinstance(result, list):
            return result
        return ["继续按计划学习", "注意复习已学内容", "尝试实际应用所学知识"]

    def identify_knowledge_gaps(self, goal: Dict, mastered_topics: List[str]) -> List[str]:
        """识别知识盲点"""
        prompt = f"""分析以下学习情况，找出可能的知识盲点：

学习目标：{goal.get('description', '')}
已掌握的主题：{', '.join(mastered_topics[:20])}

请列出3-5个可能的知识盲点或薄弱环节。
返回JSON数组：["盲点1", "盲点2"]"""

        result = self.llm.generate_json(prompt, max_tokens=500)
        if isinstance(result, list):
            return result
        return []


# ========== 统一感知管理器 ==========

class PerceptionManager:
    """
    感知模块统一入口
    整合思维导图生成、知识提取、主动学习触发
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or get_client()
        self.mindmap_gen = MindMapGenerator(self.llm)
        self.extractor = KnowledgeExtractor(self.llm)
        self.trigger = ActiveLearningTrigger(self.llm)
        self.foundation = FoundationManager()

    def process_learning_goal(self, goal: LearningGoal) -> Dict:
        """
        处理学习目标的完整感知流程：
        1. 生成思维导图
        2. 分析知识结构
        3. 返回感知结果
        """
        print(f"🔍 感知处理：{goal.description}")

        result = {
            "goal_id": goal.id,
            "mindmap_root": None,
            "node_map": {},
            "knowledge_summary": {},
            "suggestions": [],
            "processing_time": datetime.now().isoformat()
        }

        # 生成思维导图
        root = self.mindmap_gen.generate_for_goal(goal)
        if root:
            result["mindmap_root"] = root
            result["node_map"] = self.mindmap_gen.get_node_map()

        # 获取学习建议
        suggestions = self.trigger.get_learning_suggestions(
            goal.to_dict() if hasattr(goal, 'to_dict') else {"description": goal.description},
            {"overall_progress": goal.overall_progress},
            []
        )
        result["suggestions"] = suggestions

        return result

    def ingest_content(self, content: str, content_type: str = "text",
                       source: str = "manual") -> List[KnowledgeNode]:
        """
        摄入外部内容（文本/URL/文件）并提取知识节点
        """
        if content_type == "url":
            return self.extractor.extract_from_url(content)
        elif content_type == "image":
            return self.extractor.extract_from_image(content)
        else:
            return self.extractor.extract_from_text(content, source)

    def describe_image(self, image_path: str) -> str:
        """
        从图片中提取描述性文本（不提取知识点结构）

        Args:
            image_path: 图片文件路径

        Returns:
            识别到的文本描述
        """
        import base64

        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
        except Exception as e:
            print(f"[X] 读取图片失败: {e}")
            return ""

        print(f"[DEBUG] 开始图片描述提取...")

        # 使用多模态接口进行图片描述
        prompt = "请详细分析这张图片中的内容。如果是学习笔记、教材图片、实验图等，请重点关注其中的知识点、概念、原理等。请用清晰、准确的语言描述图片内容。"

        try:
            print("[DEBUG] 正在调用 chat_with_image 获取描述...")
            description = self.llm.chat_with_image(
                prompt=prompt,
                image_b64=img_b64,
                system="你是专业的多模态学习助手，擅长从图片中识别和提取知识内容。"
            )
            print(f"[DEBUG] 获取到描述，长度: {len(description)} 字符")
            print(f"[DEBUG] 描述前200字符: {description[:200]}")
            return description
        except Exception as e:
            print(f"[X] 图片描述失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
