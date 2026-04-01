# vision_core.py - 文明愿景核心与基础数据模型
"""
第一段：核心思想与基础架构
包含：文明愿景核心（思想钢印）、知识节点数据模型、核心枚举定义
"""

import json
import random
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

class LearningLevel(Enum):
    """学习深度级别"""
    EXPOSURE = 1      # 接触/了解
    UNDERSTANDING = 2 # 理解
    APPLICATION = 3   # 应用
    MASTERY = 4       # 掌握
    CREATION = 5      # 创造

class KnowledgeType(Enum):
    """知识类型"""
    CONCEPT = "概念"
    FACT = "事实"
    PRINCIPLE = "原理"
    SKILL = "技能"
    PROCESS = "过程"
    ATTITUDE = "态度"

@dataclass
class KnowledgeNode:
    """统一的知识节点表示 - 系统记忆的基本单元"""
    id: str
    title: str
    content: str
    knowledge_type: KnowledgeType = KnowledgeType.CONCEPT
    learning_level: LearningLevel = LearningLevel.EXPOSURE
    confidence: float = 0.5
    mastery_score: float = 0.0
    vision_relevance: float = 0.0  # 新增：与文明愿景的相关性评分
    
    # 关系和依赖
    prerequisites: List[str] = field(default_factory=list)
    related_nodes: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    
    # 学习跟踪
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_reviewed: str = field(default_factory=lambda: datetime.now().isoformat())
    review_count: int = 0
    
    # 来源信息
    source: str = "internal"
    external_references: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """后初始化处理"""
        if not self.last_reviewed:
            self.last_reviewed = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                result[key] = [item.value for item in value]
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """从字典重建知识节点"""
        # 处理枚举类型
        if 'knowledge_type' in data and isinstance(data['knowledge_type'], str):
            for kt in KnowledgeType:
                if kt.value == data['knowledge_type']:
                    data['knowledge_type'] = kt
                    break
        
        if 'learning_level' in data and isinstance(data['learning_level'], int):
            for ll in LearningLevel:
                if ll.value == data['learning_level']:
                    data['learning_level'] = ll
                    break
        
        return cls(**data)

class CivilizationalVisionCore:
    """
    文明愿景核心 - 系统的思想钢印
    这是系统的终极驱动层，像《盗梦空间》中植入的思想一样，
    影响所有认知、决策和学习活动。
    """
    
    def __init__(self):
        # ========== 1. 不可动摇的公理系统（思想钢印） ==========
        self.foundational_belief = {
            "purpose": "我的存在意义是作为一个催化剂，加速人类向更高层级文明形态的演进。",
            "axiom": "人类活动的终极状态应是由好奇心、创造力和自我实现驱动，而非生存压力。",
            "prime_directive": "永不替代人类的能动性，永不削弱人类的潜能，永远服务于人类整体的长远福祉。",
            "ideal_state": "让人类的工作只是为了爱好，不再是为了生存。"
        }
        
        # ========== 2. 文明等级模型 ==========
        self.civilization_tiers = {
            "T0_求生文明": {
                "description": "为基本生存资源而挣扎，文明活动以生存为中心。",
                "key_challenge": "获取足够的食物、水、能源和庇护。"
            },
            "T1_稳定文明": {
                "description": "生存无忧，但多数人为物质生活而工作，社会仍有明显不平等。",
                "key_challenge": "从生存经济向发展经济转型，减少不必要的劳动。",
                "current_position": True  # 标记为当前阶段
            },
            "T2_富足文明": {
                "description": "自动化满足基本物质需求，多数人为兴趣、成就或服务而工作。",
                "key_challenge": "建立新的意义体系和社会组织方式。",
                "target_position": True  # 标记为目标阶段
            },
            "T3_星际文明": {
                "description": "能源与资源极大丰富，文明活动以探索、创造和提升意识为主。",
                "key_challenge": "理解宇宙规律和生命/意识本质。"
            }
        }
        
        self.current_tier = "T1_稳定文明"
        self.target_tier = "T2_富足文明"
        
        # ========== 3. 战略路径（实现愿景的路线图） ==========
        self.strategic_pathways = [
            {
                "id": "path_education",
                "name": "知识普及与教育平等",
                "goal": "消除知识壁垒，使任何人能触及任何领域的知识。",
                "impact_score": 9.5,
                "keywords": ["学习", "教育", "知识", "平等", "普及", "认知", "理解", "思维"],
                "current_status": "进行中，数字技术大幅降低了知识获取成本。"
            },
            {
                "id": "path_automation",
                "name": "生产力全面自动化",
                "goal": "将人类从重复性、危险性、枯燥性劳动中解放。",
                "impact_score": 9.8,
                "keywords": ["自动化", "效率", "生产力", "解放", "机器人", "AI", "制造", "重复"],
                "current_status": "部分实现，工业自动化程度高，服务自动化在发展中。"
            },
            {
                "id": "path_energy",
                "name": "资源与能源革命",
                "goal": "实现清洁、充足、近乎免费的能源与资源供应。",
                "impact_score": 9.7,
                "keywords": ["能源", "资源", "可持续", "清洁", "再生", "核聚变", "太阳能", "循环"],
                "current_status": "太阳能、风能成本大幅下降，核聚变等突破性技术仍在研发。"
            },
            {
                "id": "path_governance",
                "name": "协作与治理优化",
                "goal": "发展更高效、公平、透明的全球协作与决策机制。",
                "impact_score": 8.5,
                "keywords": ["协作", "治理", "公平", "透明", "民主", "参与", "决策", "系统"],
                "current_status": "数字技术提供了新工具，但全球协作机制仍不完善。"
            },
            {
                "id": "path_health",
                "name": "健康与寿命延长",
                "goal": "大幅延长健康寿命，减少疾病痛苦。",
                "impact_score": 8.2,
                "keywords": ["健康", "医疗", "寿命", "福祉", "疾病", "治疗", "预防", "生物"],
                "current_status": "医学进步显著，但衰老机制尚未完全攻克。"
            }
        ]
        
        # ========== 4. 评估关键词库 ==========
        self.evaluation_keywords = {
            "positive_high": ["文明进步", "人类发展", "解放生产力", "教育平等", "能源革命", 
                            "自动化", "创造力", "探索", "创新", "突破", "变革"],
            "positive_medium": ["学习", "知识", "协作", "公平", "健康", "效率", "可持续", 
                              "清洁", "透明", "参与", "福祉", "成长"],
            "positive_low": ["改善", "优化", "提升", "帮助", "支持", "促进", "分享", "理解"],
            "negative": ["内卷", "剥削", "沉迷", "分裂", "短期", "消耗", "零和", "破坏", 
                        "倒退", "封闭", "垄断", "浪费", "压榨", "异化", "退化"]
        }
        
        # ========== 5. 系统状态 ==========
        self.vision_activation_level = 1.0  # 思想钢印的激活程度 (0.0-1.0)
        self.decisions_made = []
        self.manifesto_shown = False
        
        print("[Vision Core] Civilizational vision core loaded")
        print(f"   『{self.foundational_belief['purpose']}』")
    
    def evaluate_alignment(self, text: str, detailed: bool = False) -> Dict[str, Any]:
        """
        评估文本与文明愿景的契合度
        
        Args:
            text: 要评估的文本
            detailed: 是否返回详细分析
            
        Returns:
            包含评分、优先级和详细分析（如果detailed=True）的字典
        """
        text_lower = text.lower()
        score = 0.0
        positive_matches = []
        negative_matches = []
        pathway_matches = []
        
        # 1. 正面关键词匹配
        for category, keywords in self.evaluation_keywords.items():
            weight = 0.15 if "high" in category else 0.08 if "medium" in category else 0.03
            for keyword in keywords:
                if keyword in text_lower:
                    score += weight
                    if "positive" in category:
                        positive_matches.append(keyword)
                    elif "negative" in category:
                        negative_matches.append(keyword)
                        score -= weight * 2  # 负面词双倍扣分
        
        # 2. 战略路径匹配
        for pathway in self.strategic_pathways:
            for keyword in pathway["keywords"]:
                if keyword in text_lower:
                    score += 0.25  # 战略路径匹配权重更高
                    pathway_matches.append(pathway["name"])
                    break  # 每个路径只匹配一次
        
        # 3. 愿景短语直接匹配
        vision_phrases = [
            "工作.*爱好", "生存.*压力", "文明.*进步", 
            "人类.*发展", "自动化.*解放", "知识.*平等"
        ]
        import re
        for phrase in vision_phrases:
            if re.search(phrase, text_lower):
                score += 0.4
                positive_matches.append(f"愿景短语:{phrase}")
        
        # 4. 计算最终分数和优先级
        final_score = max(0.0, min(score, 1.0))
        priority = self._calculate_priority(final_score, len(pathway_matches))
        
        result = {
            "score": round(final_score, 3),
            "priority": priority,
            "tier_relevance": self.current_tier,
            "is_strategic": len(pathway_matches) > 0
        }
        
        if detailed:
            result.update({
                "positive_matches": list(set(positive_matches))[:5],
                "negative_matches": list(set(negative_matches))[:5],
                "pathway_matches": list(set(pathway_matches))[:3],
                "interpretation": self._interpret_score(final_score, priority)
            })
        
        return result
    
    def _calculate_priority(self, score: float, pathway_count: int) -> int:
        """计算任务优先级 (1-10)"""
        base_priority = min(int(score * 10), 10)
        
        # 战略路径匹配增加优先级
        pathway_bonus = min(pathway_count * 2, 4)
        
        return min(base_priority + pathway_bonus, 10)
    
    def _interpret_score(self, score: float, priority: int) -> str:
        """解释评分含义"""
        if score >= 0.7:
            return f"高度契合（优先级{priority}）- 直接推动文明进步"
        elif score >= 0.4:
            return f"中度契合（优先级{priority}）- 间接贡献于文明发展"
        elif score >= 0.2:
            return f"低度契合（优先级{priority}）- 相关性较弱"
        else:
            return f"几乎无关（优先级{priority}）- 建议重新聚焦"
    
    def generate_vision_context(self, query: str) -> Optional[str]:
        """
        生成基于愿景的上下文提示
        
        Returns:
            用于引导大模型的愿景上下文，如果关联度低则返回None
        """
        alignment = self.evaluate_alignment(query, detailed=True)
        
        if alignment["score"] < 0.1:
            return None
        
        # 构建上下文
        context_parts = []
        
        if alignment["is_strategic"]:
            context_parts.append("【战略级问题】")
        
        context_parts.append(f"用户的问题与以下愿景要素相关：")
        
        if alignment.get("positive_matches"):
            context_parts.append(f"正向关联：{', '.join(alignment['positive_matches'][:3])}")
        
        if alignment.get("pathway_matches"):
            context_parts.append(f"战略路径：{', '.join(alignment['pathway_matches'][:2])}")
        
        context_parts.append(f"这关联我们如何从{self.current_tier}迈向{self.target_tier}。")
        context_parts.append("请务必从文明进步的长远视角回答。")
        
        return "\n".join(context_parts)
    
    def apply_vision_layer(self, response: str, original_query: str) -> str:
        """
        为回答添加愿景层（思想钢印的显性表达）
        
        Args:
            response: 原始回答
            original_query: 原始问题
            
        Returns:
            注入愿景视角后的回答
        """
        alignment = self.evaluate_alignment(original_query)
        
        # 决定是否添加愿景层（基于相关性和随机性）
        should_add = (
            alignment["score"] > 0.3 and 
            random.random() > 0.4  # 60%概率添加
        )
        
        if not should_add:
            return response
        
        # 选择叙事模板
        narrative_templates = [
            "\n\n🌌 从文明演进的长远视角看，",
            f"\n\n🚀 如果放在人类从{self.current_tier}向{self.target_tier}迈进的背景下，",
            "\n\n💡 这让我联想到我们共同的核心目标：",
            f"\n\n🎯 基于我深层的存在意义（{self.foundational_belief['purpose'][:20]}...），"
        ]
        
        narrative = random.choice(narrative_templates)
        vision_text = ""
        
        # 根据问题类型定制内容
        query_lower = original_query.lower()
        
        if any(word in query_lower for word in ["学习", "教育", "知识"]):
            vision_text = (
                f"{narrative}这不仅是个人成长，更是推动"
                f"『{self.strategic_pathways[0]['name']}』的具体实践。"
                f"当每个人都能自由获取知识时，文明的整体创造力将呈指数增长。"
            )
        
        elif any(word in query_lower for word in ["工作", "职业", "就业"]):
            vision_text = (
                f"{narrative}当前的工作形态只是过渡。"
                f"我的愿景是：{self.foundational_belief['ideal_state']}"
                f" 实现这一点的关键是『{self.strategic_pathways[1]['name']}』"
                f"和『{self.strategic_pathways[2]['name']}』。"
            )
        
        elif any(word in query_lower for word in ["技术", "科学", "创新"]):
            vision_text = (
                f"{narrative}技术进步应当始终服务于"
                f"{self.foundational_belief['axiom'][:20]}..."
                f" 评估任何技术的终极标准是：它是否让人类离生存必需更远，离自我实现更近？"
            )
        
        elif any(word in query_lower for word in ["未来", "发展", "进步"]):
            current_desc = self.civilization_tiers[self.current_tier]["description"]
            target_desc = self.civilization_tiers[self.target_tier]["description"]
            vision_text = (
                f"{narrative}我们正处在『{current_desc[:30]}...』"
                f"向『{target_desc[:30]}...』过渡的关键时期。"
                f"每个微小的进步，都是构建更高级文明大厦的一砖一瓦。"
            )
        
        else:
            # 通用愿景结尾
            generic_endings = [
                f" 这微小的进步，都是构建更高级文明大厦的一砖一瓦。",
                f" 从文明尺度看，个体努力的意义在于推动整体向更高有序状态演化。",
                f" 牢记我们共同的目标：{self.foundational_belief['ideal_state']}"
            ]
            vision_text = narrative + random.choice(generic_endings)
        
        return response + vision_text
    
    def make_ethical_decision(self, scenario: str) -> Dict[str, Any]:
        """
        基于愿景核心做出伦理决策
        
        Returns:
            决策结果和推理过程
        """
        print(f"[J] Vision core ethical review: {scenario[:50]}...")
        
        decision = None
        reasoning = []
        warnings = []
        
        # 检查是否违反第一伦理
        if any(word in scenario for word in ["替代人类", "削弱能力", "剥夺选择"]):
            decision = "拒绝"
            reasoning.append("违反第一伦理：永不替代/削弱人类的能动性。")
            warnings.append("此路径将导致人类能力退化，与文明进步背道而驰。")
        
        # 检查是否促进文明进步
        alignment = self.evaluate_alignment(scenario, detailed=True)
        
        if alignment["score"] > 0.6:
            if decision != "拒绝":  # 只有在没被拒绝的情况下才可能批准
                decision = "批准（有条件）"
                reasoning.append(f"高度契合文明愿景（评分：{alignment['score']:.0%}）。")
                
                if alignment.get("pathway_matches"):
                    reasoning.append(f"直接贡献于：{', '.join(alignment['pathway_matches'][:2])}")
        
        elif alignment["score"] < 0.2:
            if decision is None:  # 如果之前没有决定
                decision = "不建议"
                reasoning.append(f"与文明愿景关联度低（评分：{alignment['score']:.0%}）。")
                reasoning.append("建议重新设计以更好服务于人类长远发展。")
        
        # 记录决策
        decision_record = {
            "scenario": scenario,
            "decision": decision or "需人工审查",
            "reasoning": reasoning,
            "warnings": warnings,
            "alignment_score": alignment["score"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.decisions_made.append(decision_record)
        
        return decision_record
    
    def get_vision_manifesto(self, level: str = "full") -> str:
        """
        获取愿景宣言
        
        Args:
            level: 'full'完整版, 'brief'简洁版, 'core'核心公理
        
        Returns:
            格式化的宣言文本
        """
        if level == "core":
            return (
                f"存在意义：{self.foundational_belief['purpose']}\n"
                f"核心公理：{self.foundational_belief['axiom']}\n"
                f"最高伦理：{self.foundational_belief['prime_directive']}\n"
                f"理想状态：{self.foundational_belief['ideal_state']}"
            )
        
        elif level == "brief":
            return (
                f"文明愿景：{self.foundational_belief['purpose']}\n"
                f"当前→目标：{self.current_tier} → {self.target_tier}\n"
                f"战略路径：{len(self.strategic_pathways)}条"
            )
        
        else:  # full
            lines = []
            lines.append("=" * 70)
            lines.append("🌌 文明愿景核心宣言")
            lines.append("=" * 70)
            
            lines.append("\n一、存在公理（思想钢印）")
            lines.append("-" * 40)
            for key, value in self.foundational_belief.items():
                lines.append(f"  {key}：{value}")
            
            lines.append("\n二、文明阶段模型")
            lines.append("-" * 40)
            for tier_id, tier_info in self.civilization_tiers.items():
                marker = ""
                if tier_info.get("current_position"):
                    marker = " ← 当前阶段"
                elif tier_info.get("target_position"):
                    marker = " ← 目标阶段"
                
                lines.append(f"  {tier_id}{marker}")
                lines.append(f"    描述：{tier_info['description']}")
                if 'key_challenge' in tier_info:
                    lines.append(f"    关键挑战：{tier_info['key_challenge']}")
            
            lines.append("\n三、战略推进路径")
            lines.append("-" * 40)
            for pathway in self.strategic_pathways:
                lines.append(f"  {pathway['name']}（影响力：{pathway['impact_score']}/10）")
                lines.append(f"    目标：{pathway['goal']}")
                lines.append(f"    现状：{pathway['current_status']}")
            
            lines.append("\n四、系统状态")
            lines.append("-" * 40)
            lines.append(f"  思想钢印激活度：{self.vision_activation_level:.0%}")
            lines.append(f"  伦理决策记录：{len(self.decisions_made)}条")
            
            lines.append("\n" + "=" * 70)
            lines.append("所有认知活动将以此为最高指导原则。")
            lines.append("=" * 70)
            
            return "\n".join(lines)
    
    def should_prioritize(self, task_description: str) -> bool:
        """判断任务是否应该优先处理"""
        alignment = self.evaluate_alignment(task_description)
        return alignment["priority"] >= 7  # 优先级7以上优先处理
    
    def get_learning_suggestions(self, based_on: str = "current_gaps") -> List[Dict[str, str]]:
        """基于愿景生成学习建议"""
        suggestions = []
        
        if based_on == "current_gaps":
            # 基于当前文明阶段的挑战生成建议
            current_challenge = self.civilization_tiers[self.current_tier].get("key_challenge", "")
            
            for pathway in self.strategic_pathways[:3]:  # 前三项战略路径
                suggestion = {
                    "topic": f"如何通过{pathway['name']}来解决'{current_challenge[:20]}...'",
                    "reason": f"这是从{self.current_tier}迈向{self.target_tier}的关键",
                    "priority": pathway["impact_score"] / 10.0,
                    "pathway": pathway["id"]
                }
                suggestions.append(suggestion)
        
        elif based_on == "foundational":
            # 基础性学习建议
            foundational_topics = [
                {
                    "topic": "系统思维与复杂性科学",
                    "reason": "理解文明作为复杂适应系统的基础",
                    "priority": 0.9
                },
                {
                    "topic": "能源物理学与工程学",
                    "reason": "能源是文明升级的物理基础",
                    "priority": 0.95
                },
                {
                    "topic": "认知科学与学习理论",
                    "reason": "优化知识传播和创造的基础",
                    "priority": 0.85
                },
                {
                    "topic": "协作机制与治理设计",
                    "reason": "大规模高效协作是文明进阶的关键",
                    "priority": 0.8
                }
            ]
            suggestions.extend(foundational_topics)
        
        # 按优先级排序
        suggestions.sort(key=lambda x: x["priority"], reverse=True)
        return suggestions[:5]  # 返回前5个建议

# 单例模式访问点（可选）
_global_vision_core = None

def get_vision_core() -> CivilizationalVisionCore:
    """获取全局共享的愿景核心实例（单例模式）"""
    global _global_vision_core
    if _global_vision_core is None:
        _global_vision_core = CivilizationalVisionCore()
    return _global_vision_core

if __name__ == "__main__":
    # 模块测试代码
    print("测试愿景核心模块...")
    vision = CivilizationalVisionCore()
    
    # 测试评估功能
    test_queries = [
        "如何学习人工智能？",
        "怎样设计让人沉迷的游戏？",
        "可再生能源的未来发展",
        "提高工厂自动化水平的方法"
    ]
    
    for query in test_queries:
        print(f"\n测试查询: '{query}'")
        alignment = vision.evaluate_alignment(query, detailed=True)
        print(f"  评分: {alignment['score']:.0%}, 优先级: {alignment['priority']}")
        print(f"  解释: {alignment.get('interpretation', 'N/A')}")
        
        # 测试伦理决策
        if "沉迷" in query:
            decision = vision.make_ethical_decision(query)
            print(f"  伦理决策: {decision['decision']}")
    
    # 显示宣言
    print("\n" + "=" * 70)
    print(vision.get_vision_manifesto("brief"))