# explorer.py - 深度知识探索与网络构建（重构版）
"""
第3段：深度知识探索与网络构建
功能：智能提问、思维导图可视化、知识网络构建、学习路径生成
特点：基于思维导图的深度探索、知识网络分析、可视化呈现
创新：思维导图驱动的深度提问，知识网络构建与可视化
"""

import json
import re
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import time
import os
from pathlib import Path

# 导入前两段的基础模块
from foundation import (
    MindMapNode, KnowledgeNode, LearningGoal, LearningLevel, 
    KnowledgeType, GoalScale, ModalityType, MindMapStyle,
    ProgressGranularity, generate_id, FoundationManager
)

from perception import (
    MindMapGenerator, KnowledgeExtractor,
    ActiveLearningTrigger, PerceptionManager
)
from llm_client import LLMClient, get_client

# ========== 智能提问引擎 ==========

class IntelligentQuestionEngine:
    """
    智能提问引擎 - 基于思维导图的深度提问系统
    通过系统性提问促进深度知识探索
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        
        # 提问模板库
        self.question_templates = {
            "concept": [
                "什么是{concept}？",
                "{concept}的核心特征是什么？",
                "{concept}与其他相关概念有什么区别？",
                "{concept}在现实中有哪些应用？"
            ],
            "principle": [
                "{principle}的基本原理是什么？",
                "{principle}是如何运作的？",
                "{principle}背后的理论基础是什么？",
                "如何验证{principle}的正确性？"
            ],
            "skill": [
                "如何掌握{skill}？",
                "{skill}的关键步骤是什么？",
                "实践{skill}时需要注意什么？",
                "如何评估{skill}的掌握程度？"
            ],
            "process": [
                "{process}的主要阶段是什么？",
                "如何优化{process}的效率？",
                "{process}中的关键决策点是什么？",
                "如何评估{process}的效果？"
            ],
            "system": [
                "{system}的主要组成部分是什么？",
                "{system}各组件如何相互作用？",
                "如何设计一个有效的{system}？",
                "{system}的评估标准是什么？"
            ],
            "example": [
                "这个例子说明了什么原理？",
                "这个例子中的关键点是什么？",
                "如何将这个例子应用到其他场景？",
                "这个例子有哪些局限性？"
            ],
            "general": [
                "关于{concept}，你最想了解什么？",
                "学习{concept}最大的困难是什么？",
                "{concept}未来可能如何发展？",
                "如何将{concept}与其他知识结合？"
            ]
        }
        
        # 提问深度级别
        self.question_depths = {
            "surface": {
                "name": "表面理解",
                "questions": ["是什么", "有什么", "什么时候", "谁", "哪里"],
                "thinking_time": 30
            },
            "understanding": {
                "name": "理解掌握", 
                "questions": ["为什么", "如何", "怎么样", "解释", "说明"],
                "thinking_time": 60
            },
            "application": {
                "name": "应用实践",
                "questions": ["如何应用", "如何解决", "如何改进", "如何使用"],
                "thinking_time": 90
            },
            "analysis": {
                "name": "分析评估",
                "questions": ["有什么联系", "有什么差异", "有什么影响", "分析", "比较"],
                "thinking_time": 120
            },
            "evaluation": {
                "name": "评价创造",
                "questions": ["有什么价值", "有什么局限", "如何评价", "批判", "判断"],
                "thinking_time": 150
            },
            "creation": {
                "name": "创新整合",
                "questions": ["如何创新", "如何设计", "如何整合", "创建", "发明"],
                "thinking_time": 180
            }
        }
        
        # 问题难度级别
        self.difficulty_levels = {
            "easy": {
                "description": "基础理解",
                "keywords": ["定义", "是什么", "举例", "描述"],
                "thinking_time": 30
            },
            "medium": {
                "description": "分析应用", 
                "keywords": ["为什么", "如何", "比较", "应用"],
                "thinking_time": 60
            },
            "hard": {
                "description": "综合创新",
                "keywords": ["设计", "评估", "创新", "批判"],
                "thinking_time": 120
            }
        }
    
    def generate_questions_for_node(self, 
                                  node: Union[MindMapNode, KnowledgeNode],
                                  depth_level: str = "understanding",
                                  count: int = 5) -> List[Dict[str, Any]]:
        """
        为节点生成问题
        
        Args:
            node: 思维导图节点或知识节点
            depth_level: 问题深度级别
            count: 问题数量
            
        Returns:
            问题列表，每个问题包含文本、类型、难度等信息
        """
        print(f"🤔 为节点生成问题: {node.title}")
        
        questions = []
        
        # 确定节点类型
        node_type = self._determine_node_type(node)
        
        # 使用模板生成基础问题
        template_key = node_type if node_type in self.question_templates else "general"
        templates = self.question_templates.get(template_key, self.question_templates["general"])
        
        # 根据深度级别过滤模板
        depth_info = self.question_depths.get(depth_level, self.question_depths["understanding"])
        depth_questions = depth_info["questions"]
        
        # 选择适合深度的模板
        selected_templates = []
        for template in templates:
            for depth_q in depth_questions:
                if depth_q in template:
                    selected_templates.append(template)
                    break
        
        # 如果没有匹配的模板，使用所有模板
        if not selected_templates:
            selected_templates = templates
        
        # 生成问题
        for i in range(min(count, len(selected_templates))):
            template = selected_templates[i % len(selected_templates)]
            
            # 替换占位符
            question_text = template.format(
                concept=node.title,
                principle=node.title,
                skill=node.title,
                process=node.title,
                system=node.title,
                example=node.title
            )
            
            # 确定难度
            difficulty = self._determine_difficulty(question_text, depth_level)
            
            # 创建问题字典
            question = {
                "id": generate_id(f"question_{i}_"),
                "text": question_text,
                "node_id": node.id,
                "node_title": node.title,
                "node_type": node_type,
                "depth_level": depth_level,
                "depth_name": depth_info["name"],
                "difficulty": difficulty,
                "difficulty_description": self.difficulty_levels[difficulty]["description"],
                "estimated_thinking_time": self.difficulty_levels[difficulty]["thinking_time"],
                "generated_at": datetime.now().isoformat(),
                "tags": [node_type, depth_level, difficulty]
            }
            
            questions.append(question)
        
        # 如果生成的问题不足，使用大模型补充
        if len(questions) < count:
            additional_questions = self._generate_questions_with_llm(
                node, depth_level, count - len(questions)
            )
            questions.extend(additional_questions)
        
        print(f"✅ 生成了 {len(questions)} 个问题")
        return questions
    
    def generate_questions_for_mindmap(self, 
                                     mindmap_root: MindMapNode,
                                     node_map: Dict[str, MindMapNode],
                                     depth_level: str = "understanding",
                                     questions_per_node: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """
        为整个思维导图生成问题
        
        Args:
            mindmap_root: 思维导图根节点
            node_map: 节点ID到节点的映射
            depth_level: 问题深度级别
            questions_per_node: 每个节点的问题数量
            
        Returns:
            按节点ID分组的问题字典
        """
        print(f"🧠 为思维导图生成问题 (深度: {depth_level})")
        
        all_questions = defaultdict(list)
        
        # 遍历所有节点
        for node_id, node in node_map.items():
            questions = self.generate_questions_for_node(
                node, depth_level, questions_per_node
            )
            
            if questions:
                all_questions[node_id] = questions
        
        return dict(all_questions)
    
    def generate_deep_questions_chain(self, 
                                    start_node: Union[MindMapNode, KnowledgeNode],
                                    node_map: Optional[Dict[str, Any]] = None,
                                    chain_length: int = 5) -> List[Dict[str, Any]]:
        """
        生成深度问题链 - 基于一个问题引发后续问题
        
        Args:
            start_node: 起始节点
            node_map: 节点映射（用于查找相关节点）
            chain_length: 问题链长度
            
        Returns:
            问题链列表
        """
        print(f"🔗 生成深度问题链 (起始: {start_node.title})")
        
        question_chain = []
        
        # 生成第一个问题（深度问题）
        initial_questions = self.generate_questions_for_node(
            start_node, "analysis", 1
        )
        
        if not initial_questions:
            return question_chain
        
        current_question = initial_questions[0]
        question_chain.append(current_question)
        
        # 使用大模型生成后续问题链
        chain_prompt = f"""基于以下问题生成一个深度问题链：

初始问题：{current_question['text']}
上下文：关于{start_node.title}

请生成一个包含{chain_length-1}个后续问题的问题链，要求：
1. 每个问题都基于前一个问题的答案
2. 问题逐渐深入，从理解到应用到分析到创新
3. 问题之间要有逻辑联系
4. 每个问题应该是开放式的，促进深度思考

请以JSON数组格式返回问题链，每个问题包含：
- text: 问题文本
- depth_level: 问题深度级别
- reasoning: 为什么提出这个问题
"""
        
        response = self.llm_client.call_llm(
            prompt=chain_prompt,
            system_prompt="你是问题设计专家，擅长创建连贯的深度问题链。",
            max_tokens=1500,
            temperature=0.7
        )
        
        if response:
            try:
                # 尝试解析JSON
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    chain_questions = json.loads(json_str)
                    
                    for i, q_data in enumerate(chain_questions):
                        if len(question_chain) >= chain_length:
                            break
                        
                        question = {
                            "id": generate_id(f"chain_{i}_"),
                            "text": q_data.get("text", f"问题 {i+1}"),
                            "node_id": start_node.id,
                            "node_title": start_node.title,
                            "depth_level": q_data.get("depth_level", "analysis"),
                            "depth_name": self.question_depths.get(
                                q_data.get("depth_level", "analysis"), {}
                            ).get("name", "分析评估"),
                            "difficulty": "hard",
                            "difficulty_description": "综合创新",
                            "estimated_thinking_time": 120,
                            "is_chained": True,
                            "chain_index": i + 1,
                            "reasoning": q_data.get("reasoning", ""),
                            "generated_at": datetime.now().isoformat()
                        }
                        
                        question_chain.append(question)
            except Exception as e:
                print(f"❌ 问题链解析失败: {str(e)}")
        
        # 如果问题链不足，生成补充问题
        if len(question_chain) < chain_length:
            remaining = chain_length - len(question_chain)
            additional_questions = self.generate_questions_for_node(
                start_node, "evaluation", remaining
            )
            
            for i, q in enumerate(additional_questions):
                if len(question_chain) >= chain_length:
                    break
                q["is_chained"] = True
                q["chain_index"] = len(question_chain) + 1
                question_chain.append(q)
        
        print(f"✅ 生成了 {len(question_chain)} 个问题的深度问题链")
        return question_chain
    
    def generate_comparison_questions(self, 
                                    node1: Union[MindMapNode, KnowledgeNode],
                                    node2: Union[MindMapNode, KnowledgeNode]) -> List[Dict[str, Any]]:
        """
        生成比较性问题 - 对比两个相关概念
        
        Args:
            node1: 第一个节点
            node2: 第二个节点
            
        Returns:
            比较性问题列表
        """
        print(f"[J] Generating comparative question: {node1.title} vs {node2.title}")
        
        comparison_questions = []
        
        # 比较问题模板
        comparison_templates = [
            "{concept1}和{concept2}有什么相同点？",
            "{concept1}和{concept2}有什么不同点？",
            "在什么情况下应该使用{concept1}而不是{concept2}？",
            "{concept1}和{concept2}如何相互影响？",
            "学习{concept1}对理解{concept2}有什么帮助？"
        ]
        
        for i, template in enumerate(comparison_templates):
            question_text = template.format(
                concept1=node1.title,
                concept2=node2.title
            )
            
            question = {
                "id": generate_id(f"comparison_{i}_"),
                "text": question_text,
                "node_ids": [node1.id, node2.id],
                "node_titles": [node1.title, node2.title],
                "question_type": "comparison",
                "depth_level": "analysis",
                "depth_name": "分析评估",
                "difficulty": "medium",
                "difficulty_description": "分析应用",
                "estimated_thinking_time": 90,
                "generated_at": datetime.now().isoformat(),
                "tags": ["比较", "分析", "关系"]
            }
            
            comparison_questions.append(question)
        
        # 使用大模型生成更多比较性问题
        comparison_prompt = f"""请生成一些深入比较{node1.title}和{node2.title}的问题：

要求：
1. 关注两者的本质区别和联系
2. 考虑实际应用场景
3. 包含高级分析性问题
4. 促进深度思考和理解

请生成3-5个比较性问题。"""
        
        response = self.llm_client.call_llm(
            prompt=comparison_prompt,
            max_tokens=800,
            temperature=0.6
        )
        
        if response:
            # 提取问题（假设每行一个问题）
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and '?' in line or '？' in line:
                    # 清理行号等前缀
                    clean_line = re.sub(r'^\d+[\.\)]?\s*', '', line)
                    
                    question = {
                        "id": generate_id("comparison_llm_"),
                        "text": clean_line,
                        "node_ids": [node1.id, node2.id],
                        "node_titles": [node1.title, node2.title],
                        "question_type": "comparison",
                        "depth_level": "analysis",
                        "depth_name": "分析评估",
                        "difficulty": "hard",
                        "difficulty_description": "综合创新",
                        "estimated_thinking_time": 120,
                        "generated_at": datetime.now().isoformat(),
                        "tags": ["比较", "深度分析", "大模型生成"]
                    }
                    
                    comparison_questions.append(question)
        
        print(f"✅ 生成了 {len(comparison_questions)} 个比较性问题")
        return comparison_questions
    
    def evaluate_answer_quality(self, 
                              question: Dict[str, Any],
                              answer: str,
                              reference_material: Optional[str] = None) -> Dict[str, Any]:
        """
        评估回答质量
        
        Args:
            question: 问题字典
            answer: 用户回答
            reference_material: 参考资料
            
        Returns:
            评估结果
        """
        print(f"📊 评估回答质量: {question['text'][:50]}...")
        
        evaluation = {
            "question_id": question["id"],
            "question_text": question["text"],
            "answer": answer,
            "evaluated_at": datetime.now().isoformat(),
            "scores": {},
            "feedback": "",
            "suggestions": []
        }
        
        # 构建评估提示
        prompt = f"""请评估以下回答的质量：

问题：{question['text']}
用户回答：{answer}

评估标准：
1. 准确性（0-10分）：回答是否准确反映了相关知识
2. 完整性（0-10分）：回答是否全面覆盖了问题的各个方面
3. 深度（0-10分）：回答是否展现了深入思考
4. 清晰度（0-10分）：回答是否表达清晰、条理分明

请提供：
1. 四个维度的分数
2. 总体反馈（指出优点和改进空间）
3. 具体的改进建议

{"参考资料：" + reference_material if reference_material else ""}
"""
        
        response = self.llm_client.call_llm(
            prompt=prompt,
            system_prompt="你是评估专家，擅长评估学习回答的质量。",
            max_tokens=1000,
            temperature=0.3
        )
        
        if response:
            # 尝试解析评分
            scores = {
                "accuracy": 5.0,
                "completeness": 5.0,
                "depth": 5.0,
                "clarity": 5.0
            }
            
            # 提取分数（简单正则匹配）
            score_patterns = {
                "accuracy": r"准确性[：:]?\s*(\d+(?:\.\d+)?)/10",
                "completeness": r"完整性[：:]?\s*(\d+(?:\.\d+)?)/10",
                "depth": r"深度[：:]?\s*(\d+(?:\.\d+)?)/10",
                "clarity": r"清晰度[：:]?\s*(\d+(?:\.\d+)?)/10"
            }
            
            for key, pattern in score_patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        scores[key] = float(match.group(1))
                    except:
                        pass
            
            evaluation["scores"] = scores
            
            # 计算平均分
            avg_score = sum(scores.values()) / len(scores)
            evaluation["average_score"] = avg_score
            
            # 提取反馈和建议
            lines = response.split('\n')
            feedback_lines = []
            suggestion_lines = []
            in_feedback = False
            in_suggestions = False
            
            for line in lines:
                line_lower = line.lower()
                if "反馈" in line_lower or "评价" in line_lower:
                    in_feedback = True
                    in_suggestions = False
                elif "建议" in line_lower:
                    in_feedback = False
                    in_suggestions = True
                elif "---" in line or "====" in line:
                    break
                
                if in_feedback and line.strip() and "反馈" not in line_lower:
                    feedback_lines.append(line.strip())
                elif in_suggestions and line.strip() and "建议" not in line_lower:
                    suggestion_lines.append(line.strip())
            
            evaluation["feedback"] = " ".join(feedback_lines) or "回答评估完成"
            evaluation["suggestions"] = suggestion_lines
            
            # 确定掌握程度
            if avg_score >= 8.0:
                evaluation["mastery_level"] = "精通"
                evaluation["mastery_description"] = "对该问题有深入理解和准确回答"
            elif avg_score >= 6.0:
                evaluation["mastery_level"] = "掌握"
                evaluation["mastery_description"] = "基本理解并正确回答了问题"
            elif avg_score >= 4.0:
                evaluation["mastery_level"] = "理解"
                evaluation["mastery_description"] = "部分理解但回答不够完整准确"
            else:
                evaluation["mastery_level"] = "初学"
                evaluation["mastery_description"] = "需要进一步学习和理解"
        
        else:
            # 备选评估方案
            evaluation["feedback"] = "自动评估完成，建议参考标准答案进一步学习。"
            evaluation["scores"] = {
                "accuracy": 5.0,
                "completeness": 5.0,
                "depth": 5.0,
                "clarity": 5.0
            }
            evaluation["average_score"] = 5.0
            evaluation["mastery_level"] = "评估中"
        
        return evaluation
    
    def _determine_node_type(self, node: Union[MindMapNode, KnowledgeNode]) -> str:
        """确定节点类型"""
        if isinstance(node, MindMapNode):
            return node.node_type if hasattr(node, 'node_type') else "concept"
        elif isinstance(node, KnowledgeNode):
            # 将KnowledgeType映射到问题类型
            type_mapping = {
                KnowledgeType.CONCEPT: "concept",
                KnowledgeType.FACT: "concept",
                KnowledgeType.PRINCIPLE: "principle",
                KnowledgeType.SKILL: "skill",
                KnowledgeType.PROCESS: "process",
                KnowledgeType.SYSTEM: "system",
                KnowledgeType.EXAMPLE: "example"
            }
            return type_mapping.get(node.knowledge_type, "concept")
        else:
            return "concept"
    
    def _determine_difficulty(self, question_text: str, depth_level: str) -> str:
        """确定问题难度"""
        # 基于深度级别
        if depth_level in ["evaluation", "creation"]:
            return "hard"
        elif depth_level in ["application", "analysis"]:
            return "medium"
        else:
            return "easy"
    
    def _generate_questions_with_llm(self, 
                                   node: Union[MindMapNode, KnowledgeNode],
                                   depth_level: str,
                                   count: int) -> List[Dict[str, Any]]:
        """使用大模型生成问题"""
        node_type = self._determine_node_type(node)
        depth_info = self.question_depths.get(depth_level, self.question_depths["understanding"])
        
        prompt = f"""请为以下{node_type}生成{count}个深度问题：

主题：{node.title}
描述：{getattr(node, 'description', getattr(node, 'content', ''))[:200]}
问题深度：{depth_info['name']}
问题类型：{depth_level}

要求：
1. 问题应该促进深度思考
2. 问题应该与主题密切相关
3. 问题应该是开放式的
4. 问题应该适合{node_type}类型

请返回JSON数组格式，每个问题包含：
- text: 问题文本
- reasoning: 为什么提出这个问题
"""
        
        response = self.llm_client.call_llm(
            prompt=prompt,
            max_tokens=800,
            temperature=0.7
        )
        
        questions = []
        
        if response:
            try:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    llm_questions = json.loads(json_str)
                    
                    for i, q_data in enumerate(llm_questions):
                        if i >= count:
                            break
                        
                        question = {
                            "id": generate_id(f"llm_question_{i}_"),
                            "text": q_data.get("text", f"关于{node.title}的问题"),
                            "node_id": node.id,
                            "node_title": node.title,
                            "node_type": node_type,
                            "depth_level": depth_level,
                            "depth_name": depth_info["name"],
                            "difficulty": self._determine_difficulty(q_data.get("text", ""), depth_level),
                            "difficulty_description": self.difficulty_levels[self._determine_difficulty(q_data.get("text", ""), depth_level)]["description"],
                            "estimated_thinking_time": self.difficulty_levels[self._determine_difficulty(q_data.get("text", ""), depth_level)]["thinking_time"],
                            "generated_by": "llm",
                            "reasoning": q_data.get("reasoning", ""),
                            "generated_at": datetime.now().isoformat()
                        }
                        
                        questions.append(question)
            except Exception as e:
                print(f"❌ LLM问题生成解析失败: {str(e)}")
        
        return questions

# ========== 思维导图可视化器 ==========

class MindMapVisualizer:
    """
    思维导图可视化器 - 将思维导图转换为可视化图表
    支持多种输出格式和可视化风格
    """
    
    def __init__(self):
        # 可视化配置
        self.visualization_config = {
            "node_size": {
                "root": 3000,
                "depth_1": 2000,
                "depth_2": 1500,
                "depth_3": 1000,
                "depth_4": 800,
                "default": 1000
            },
            "node_color": {
                "root": "#FF6B6B",     # 红色
                "concept": "#4ECDC4",   # 青色
                "skill": "#FFD166",     # 黄色
                "example": "#06D6A0",   # 绿色
                "practice": "#118AB2",  # 蓝色
                "principle": "#EF476F", # 粉色
                "fact": "#073B4C",      # 深蓝
                "default": "#999999"    # 灰色
            },
            "layout": {
                "spacing": 2.0,
                "layer_spacing": 1.5,
                "node_spacing": 1.0
            },
            "font": {
                "family": "SimHei, Microsoft YaHei, sans-serif",
                "size": {
                    "root": 14,
                    "depth_1": 12,
                    "depth_2": 11,
                    "depth_3": 10,
                    "default": 10
                }
            }
        }
        
        # 输出配置
        self.output_formats = ["png", "svg", "pdf", "jpg"]
        self.default_format = "png"
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def visualize_mindmap(self, 
                         mindmap_root: MindMapNode,
                         node_map: Dict[str, MindMapNode],
                         output_format: str = "png",
                         style: str = "balanced",
                         show_progress: bool = False) -> Optional[str]:
        """
        可视化思维导图
        
        Args:
            mindmap_root: 思维导图根节点
            node_map: 节点ID到节点的映射
            output_format: 输出格式
            style: 可视化风格
            show_progress: 是否显示学习进度
            
        Returns:
            输出文件路径，失败时返回None
        """
        print(f"🎨 可视化思维导图: {mindmap_root.title}")
        
        if output_format not in self.output_formats:
            print(f"❌ 不支持的输出格式: {output_format}")
            output_format = self.default_format
        
        try:
            # 创建NetworkX图
            G = nx.DiGraph()
            
            # 添加所有节点
            for node_id, node in node_map.items():
                G.add_node(node_id, **self._get_node_attributes(node, show_progress))
            
            # 添加边（父子关系）
            for node_id, node in node_map.items():
                if node.parent_id and node.parent_id in node_map:
                    G.add_edge(node.parent_id, node_id)
            
            # 创建图形
            plt.figure(figsize=(16, 12))
            
            # 选择布局
            pos = self._create_layout(G, mindmap_root.id, style)
            
            # 绘制节点
            node_colors = []
            node_sizes = []
            
            for node_id in G.nodes():
                node_data = G.nodes[node_id]
                node_colors.append(node_data.get('color', '#999999'))
                node_sizes.append(node_data.get('size', 1000))
            
            nx.draw_networkx_nodes(G, pos, 
                                 node_color=node_colors,
                                 node_size=node_sizes,
                                 alpha=0.9)
            
            # 绘制边
            nx.draw_networkx_edges(G, pos, 
                                 arrowstyle='-|>',
                                 arrowsize=20,
                                 edge_color='#666666',
                                 width=1.5,
                                 alpha=0.6)
            
            # 绘制标签
            labels = {}
            for node_id in G.nodes():
                node_data = G.nodes[node_id]
                labels[node_id] = node_data.get('label', node_id[:8])
            
            nx.draw_networkx_labels(G, pos, labels,
                                  font_size=10,
                                  font_family='sans-serif')
            
            # 添加标题
            plt.title(f"思维导图: {mindmap_root.title}", fontsize=16, pad=20)
            
            # 添加图例
            self._add_legend(plt, node_map)
            
            # 调整布局
            plt.axis('off')
            plt.tight_layout()
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mindmap_{mindmap_root.id[:8]}_{timestamp}.{output_format}"
            filepath = self.output_dir / filename
            
            # 保存图像
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 思维导图已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ 思维导图可视化失败: {str(e)}")
            plt.close()
            return None
    
    def visualize_knowledge_network(self,
                                  knowledge_nodes: List[KnowledgeNode],
                                  relations: Optional[List[Tuple[str, str, str]]] = None,
                                  output_format: str = "png") -> Optional[str]:
        """
        可视化知识网络
        
        Args:
            knowledge_nodes: 知识节点列表
            relations: 关系列表 (源节点ID, 目标节点ID, 关系类型)
            output_format: 输出格式
            
        Returns:
            输出文件路径
        """
        print(f"🌐 可视化知识网络 ({len(knowledge_nodes)}个节点)")
        
        if output_format not in self.output_formats:
            output_format = self.default_format
        
        try:
            # 创建NetworkX图
            G = nx.DiGraph()
            
            # 添加节点
            for node in knowledge_nodes:
                G.add_node(node.id, 
                         label=node.title[:15],
                         knowledge_type=node.knowledge_type.value,
                         color=self._get_knowledge_node_color(node),
                         size=self._get_knowledge_node_size(node))
            
            # 添加关系边
            if relations:
                for source, target, rel_type in relations:
                    if source in G and target in G:
                        G.add_edge(source, target, 
                                 relation=rel_type,
                                 label=rel_type[:10])
            
            # 如果没有显式关系，使用知识节点中的关系
            if not relations or len(relations) == 0:
                for node in knowledge_nodes:
                    # 添加先决条件关系
                    for prereq in node.prerequisites:
                        if prereq in G:
                            G.add_edge(prereq, node.id, 
                                     relation="先决条件",
                                     label="先决")
                    
                    # 添加相关节点关系
                    for related in node.related_nodes:
                        if related in G:
                            G.add_edge(node.id, related,
                                     relation="相关",
                                     label="相关")
            
            # 创建图形
            plt.figure(figsize=(14, 10))
            
            # 使用弹簧布局
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # 按知识类型分组
            node_types = {}
            for node_id, node_data in G.nodes(data=True):
                node_type = node_data.get('knowledge_type', '未知')
                if node_type not in node_types:
                    node_types[node_type] = []
                node_types[node_type].append(node_id)
            
            # 绘制每种类型的节点
            colors = plt.cm.tab20.colors
            for i, (node_type, node_ids) in enumerate(node_types.items()):
                color = colors[i % len(colors)]
                nx.draw_networkx_nodes(G, pos,
                                     nodelist=node_ids,
                                     node_color=[color],
                                     node_size=[G.nodes[nid].get('size', 800) for nid in node_ids],
                                     label=node_type,
                                     alpha=0.8)
            
            # 绘制边
            edge_colors = []
            edge_styles = []
            edge_widths = []
            
            for u, v, data in G.edges(data=True):
                rel_type = data.get('relation', '未知')
                if rel_type == "先决条件":
                    edge_colors.append('red')
                    edge_styles.append('solid')
                    edge_widths.append(2.0)
                elif rel_type == "相关":
                    edge_colors.append('blue')
                    edge_styles.append('dashed')
                    edge_widths.append(1.5)
                else:
                    edge_colors.append('gray')
                    edge_styles.append('dotted')
                    edge_widths.append(1.0)
            
            nx.draw_networkx_edges(G, pos,
                                 edge_color=edge_colors,
                                 style=edge_styles,
                                 width=edge_widths,
                                 alpha=0.6,
                                 arrowstyle='-|>',
                                 arrowsize=15)
            
            # 绘制节点标签
            labels = {node: G.nodes[node].get('label', node[:8]) for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels,
                                  font_size=9,
                                  font_family='sans-serif')
            
            # 绘制边标签
            edge_labels = {}
            for u, v, data in G.edges(data=True):
                if 'label' in data:
                    edge_labels[(u, v)] = data['label']
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                       font_size=8,
                                       label_pos=0.5)
            
            # 添加标题和图例
            plt.title(f"知识网络 ({len(knowledge_nodes)}个节点, {len(G.edges())}个关系)", 
                     fontsize=14, pad=20)
            plt.legend(title="知识类型", loc='upper left', bbox_to_anchor=(1, 1))
            plt.axis('off')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"knowledge_network_{timestamp}.{output_format}"
            filepath = self.output_dir / filename
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 知识网络已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ 知识网络可视化失败: {str(e)}")
            plt.close()
            return None
    
    def create_interactive_html(self,
                              mindmap_root: MindMapNode,
                              node_map: Dict[str, MindMapNode],
                              questions: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Optional[str]:
        """
        创建交互式HTML可视化
        
        Args:
            mindmap_root: 思维导图根节点
            node_map: 节点映射
            questions: 节点问题字典
            
        Returns:
            HTML文件路径
        """
        print(f"[PC] Creating interactive HTML visualization")
        
        try:
            # 创建HTML内容
            html_content = self._generate_html_content(mindmap_root, node_map, questions)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mindmap_interactive_{timestamp}.html"
            filepath = self.output_dir / filename
            
            # 保存HTML文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ 交互式HTML已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ 创建交互式HTML失败: {str(e)}")
            return None
    
    def _get_node_attributes(self, node: MindMapNode, show_progress: bool = False) -> Dict[str, Any]:
        """获取节点的可视化属性"""
        # 节点大小
        size_key = f"depth_{node.depth}" if node.depth <= 4 else "default"
        node_size = self.visualization_config["node_size"].get(
            size_key, self.visualization_config["node_size"]["default"]
        )
        
        # 节点颜色
        node_color = self.visualization_config["node_color"].get(
            node.node_type, self.visualization_config["node_color"]["default"]
        )
        
        # 如果是根节点
        if node.depth == 0:
            node_color = self.visualization_config["node_color"]["root"]
        
        # 标签文本
        if show_progress and hasattr(node, 'learning_status'):
            label = f"{node.title}\n({node.learning_status})"
        else:
            label = node.title
        
        # 如果标题太长，截断
        if len(label) > 20:
            label = label[:18] + "..."
        
        return {
            "label": label,
            "title": node.title,
            "depth": node.depth,
            "node_type": node.node_type,
            "color": node_color,
            "size": node_size,
            "description": node.description[:50] if node.description else ""
        }
    
    def _get_knowledge_node_color(self, node: KnowledgeNode) -> str:
        """获取知识节点的颜色"""
        color_map = {
            "概念": "#4ECDC4",    # 青色
            "事实": "#118AB2",    # 蓝色
            "原理": "#EF476F",    # 粉色
            "技能": "#FFD166",    # 黄色
            "过程": "#06D6A0",    # 绿色
            "系统": "#073B4C",    # 深蓝
            "示例": "#FF9A76",    # 橙色
            "练习": "#7209B7",    # 紫色
            "策略": "#F72585",    # 洋红
            "模式": "#3A86FF"     # 亮蓝
        }
        
        return color_map.get(node.knowledge_type.value, "#999999")
    
    def _get_knowledge_node_size(self, node: KnowledgeNode) -> int:
        """获取知识节点的大小"""
        # 基于重要性、难度等调整大小
        base_size = 800
        
        # 基于置信度调整
        confidence_factor = 0.5 + node.confidence  # 0.5-1.5
        
        # 基于掌握度调整
        mastery_factor = 0.5 + node.mastery_score  # 0.5-1.5
        
        # 最终大小
        size = int(base_size * confidence_factor * mastery_factor)
        
        return min(size, 2000)  # 限制最大大小
    
    def _create_layout(self, G: nx.DiGraph, root_id: str, style: str) -> Dict[str, Tuple[float, float]]:
        """创建布局"""
        if style == "hierarchical":
            # 层次化布局
            return self._hierarchical_layout(G, root_id)
        elif style == "radial":
            # 放射状布局
            return nx.shell_layout(G)
        elif style == "spring":
            # 弹簧布局
            return nx.spring_layout(G, k=2, iterations=50)
        else:
            # 默认：有向图分层布局
            return nx.multipartite_layout(G, subset_key="depth")
    
    def _hierarchical_layout(self, G: nx.DiGraph, root_id: str) -> Dict[str, Tuple[float, float]]:
        """创建层次化布局"""
        pos = {}
        
        # 按深度分组节点
        depth_groups = defaultdict(list)
        for node_id in G.nodes():
            depth = G.nodes[node_id].get('depth', 0)
            depth_groups[depth].append(node_id)
        
        # 确定每层的位置
        max_depth = max(depth_groups.keys()) if depth_groups else 0
        
        for depth, nodes in depth_groups.items():
            # 计算y坐标（深度越大，y越小）
            y = max_depth - depth
            
            # 均匀分布x坐标
            node_count = len(nodes)
            for i, node_id in enumerate(nodes):
                x = i - (node_count - 1) / 2
                pos[node_id] = (x, y)
        
        return pos
    
    def _add_legend(self, plt, node_map: Dict[str, MindMapNode]) -> None:
        """添加图例"""
        # 收集所有节点类型
        node_types = set()
        for node in node_map.values():
            node_types.add(node.node_type)
        
        # 创建图例句柄
        legend_elements = []
        for node_type in sorted(node_types):
            color = self.visualization_config["node_color"].get(
                node_type, self.visualization_config["node_color"]["default"]
            )
            
            # 创建图例句柄
            from matplotlib.patches import Patch
            legend_elements.append(
                Patch(facecolor=color, edgecolor='black', label=node_type)
            )
        
        # 添加图例
        if legend_elements:
            plt.legend(handles=legend_elements, 
                     title="节点类型",
                     loc='upper right',
                     bbox_to_anchor=(1.15, 1))
    
    def _generate_html_content(self,
                             mindmap_root: MindMapNode,
                             node_map: Dict[str, MindMapNode],
                             questions: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> str:
        """生成HTML内容"""
        
        # 创建节点数据
        nodes_data = []
        for node_id, node in node_map.items():
            node_info = {
                "id": node_id,
                "title": node.title,
                "description": node.description,
                "depth": node.depth,
                "node_type": node.node_type,
                "importance": node.importance,
                "difficulty": node.difficulty,
                "learning_status": node.learning_status,
                "estimated_time_minutes": node.estimated_time_minutes
            }
            
            # 添加问题（如果有）
            if questions and node_id in questions:
                node_info["questions"] = questions[node_id][:3]  # 最多3个问题
            
            nodes_data.append(node_info)
        
        # 创建边数据
        edges_data = []
        for node_id, node in node_map.items():
            if node.parent_id and node.parent_id in node_map:
                edges_data.append({
                    "from": node.parent_id,
                    "to": node_id,
                    "type": "parent-child"
                })
        
        # 生成HTML
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>思维导图: {mindmap_root.title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Microsoft YaHei', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }}
        
        .header h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        
        .header .description {{
            color: #666;
            font-size: 16px;
        }}
        
        .mindmap-container {{
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: auto;
            position: relative;
        }}
        
        .node {{
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .node:hover {{
            transform: scale(1.05);
        }}
        
        .node-text {{
            font-size: 12px;
            text-anchor: middle;
            pointer-events: none;
            fill: white;
            font-weight: bold;
        }}
        
        .controls {{
            margin-top: 20px;
            text-align: center;
        }}
        
        .btn {{
            background: #4ECDC4;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 5px;
            font-size: 14px;
        }}
        
        .btn:hover {{
            background: #3DBBB2;
        }}
        
        .details-panel {{
            margin-top: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 5px;
            display: none;
        }}
        
        .details-panel.active {{
            display: block;
        }}
        
        .node-info h3 {{
            color: #333;
            margin-top: 0;
        }}
        
        .questions-list {{
            margin-top: 20px;
        }}
        
        .question-item {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #4ECDC4;
        }}
        
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
            justify-content: center;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin-right: 15px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 {mindmap_root.title}</h1>
            <div class="description">{mindmap_root.description}</div>
            <div class="meta-info">
                <span>节点数量: {len(node_map)} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="resetView()">重置视图</button>
            <button class="btn" onclick="zoomIn()">放大</button>
            <button class="btn" onclick="zoomOut()">缩小</button>
            <button class="btn" onclick="exportPNG()">导出PNG</button>
        </div>
        
        <div class="mindmap-container" id="mindmap"></div>
        
        <div class="legend" id="legend">
            <!-- 图例将由JavaScript动态生成 -->
        </div>
        
        <div class="details-panel" id="detailsPanel">
            <div class="node-info" id="nodeInfo">
                <!-- 节点详情将动态加载 -->
            </div>
        </div>
    </div>
    
    <script>
        // 数据
        const nodesData = {json.dumps(nodes_data, ensure_ascii=False)};
        const edgesData = {json.dumps(edges_data, ensure_ascii=False)};
        
        // 节点类型颜色映射
        const nodeColors = {json.dumps(self.visualization_config["node_color"], ensure_ascii=False)};
        
        // 初始化变量
        let selectedNodeId = null;
        
        // 创建思维导图
        function createMindMap() {{
            const container = document.getElementById('mindmap');
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            // 创建SVG
            const svg = d3.select('#mindmap')
                .append('svg')
                .attr('width', width)
                .attr('height', height)
                .attr('id', 'mindmap-svg');
            
            // 添加缩放组
            const g = svg.append('g')
                .attr('id', 'mindmap-g');
            
            // 创建力导向图模拟
            const simulation = d3.forceSimulation(nodesData)
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('link', d3.forceLink(edgesData)
                    .id(d => d.id)
                    .distance(100))
                .force('collision', d3.forceCollide().radius(40));
            
            // 创建边
            const link = g.append('g')
                .attr('class', 'links')
                .selectAll('line')
                .data(edgesData)
                .enter()
                .append('line')
                .attr('stroke', '#999')
                .attr('stroke-width', 1.5)
                .attr('stroke-opacity', 0.6);
            
            // 创建节点
            const node = g.append('g')
                .attr('class', 'nodes')
                .selectAll('circle')
                .data(nodesData)
                .enter()
                .append('circle')
                .attr('class', 'node')
                .attr('r', d => {{
                    if (d.depth === 0) return 30;
                    if (d.depth === 1) return 25;
                    if (d.depth === 2) return 20;
                    return 15;
                }})
                .attr('fill', d => nodeColors[d.node_type] || nodeColors.default)
                .attr('stroke', '#fff')
                .attr('stroke-width', 2)
                .call(d3.drag()
                    .on('start', dragStarted)
                    .on('drag', dragged)
                    .on('end', dragEnded))
                .on('click', nodeClicked);
            
            // 添加节点标签
            const text = g.append('g')
                .attr('class', 'labels')
                .selectAll('text')
                .data(nodesData)
                .enter()
                .append('text')
                .attr('class', 'node-text')
                .text(d => d.title.length > 15 ? d.title.substring(0, 12) + '...' : d.title)
                .attr('font-size', d => {{
                    if (d.depth === 0) return '14px';
                    if (d.depth === 1) return '12px';
                    return '10px';
                }});
            
            // 更新位置
            simulation.on('tick', () => {{
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);
                
                text
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            }});
            
            // 添加缩放功能
            const zoom = d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => {{
                    g.attr('transform', event.transform);
                }});
            
            svg.call(zoom);
            
            // 创建图例
            createLegend();
        }}
        
        // 创建图例
        function createLegend() {{
            const legend = document.getElementById('legend');
            const uniqueTypes = [...new Set(nodesData.map(n => n.node_type))];
            
            uniqueTypes.forEach(type => {{
                const legendItem = document.createElement('div');
                legendItem.className = 'legend-item';
                
                const colorBox = document.createElement('div');
                colorBox.className = 'legend-color';
                colorBox.style.backgroundColor = nodeColors[type] || nodeColors.default;
                
                const label = document.createElement('span');
                label.textContent = type;
                
                legendItem.appendChild(colorBox);
                legendItem.appendChild(label);
                legend.appendChild(legendItem);
            }});
        }}
        
        // 节点点击事件
        function nodeClicked(event, d) {{
            // 更新选中状态
            d3.selectAll('.node')
                .attr('stroke', '#fff')
                .attr('stroke-width', 2);
            
            d3.select(event.currentTarget)
                .attr('stroke', '#FF6B6B')
                .attr('stroke-width', 4);
            
            // 显示节点详情
            selectedNodeId = d.id;
            showNodeDetails(d);
        }}
        
        // 显示节点详情
        function showNodeDetails(nodeData) {{
            const panel = document.getElementById('detailsPanel');
            const infoDiv = document.getElementById('nodeInfo');
            
            // 构建详情HTML
            let html = `
                <h3>📌 ${{nodeData.title}}</h3>
                <p><strong>描述:</strong> ${{nodeData.description || '无描述'}}</p>
                <p><strong>类型:</strong> ${{nodeData.node_type}}</p>
                <p><strong>深度:</strong> ${{nodeData.depth}}</p>
                <p><strong>重要性:</strong> ${{(nodeData.importance * 100).toFixed(0)}}%</p>
                <p><strong>难度:</strong> ${{(nodeData.difficulty * 100).toFixed(0)}}%</p>
                <p><strong>学习状态:</strong> ${{nodeData.learning_status}}</p>
                <p><strong>预估时间:</strong> ${{nodeData.estimated_time_minutes}}分钟</p>
            `;
            
            // 如果有问题，显示问题
            if (nodeData.questions && nodeData.questions.length > 0) {{
                html += `<div class="questions-list"><h4>💭 相关问题 (${{nodeData.questions.length}}个):</h4>`;
                
                nodeData.questions.forEach((q, i) => {{
                    html += `
                        <div class="question-item">
                            <p><strong>问题 ${{i+1}}:</strong> ${{q.text}}</p>
                            <p><small>难度: ${{q.difficulty_description}} | 预估思考时间: ${{q.estimated_thinking_time}}秒</small></p>
                        </div>
                    `;
                }});
                
                html += `</div>`;
            }}
            
            infoDiv.innerHTML = html;
            panel.classList.add('active');
        }}
        
        // 拖拽函数
        function dragStarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragEnded(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // 控制函数
        function resetView() {{
            const svg = d3.select('#mindmap-svg');
            svg.transition().duration(750).call(
                d3.zoom().transform,
                d3.zoomIdentity
            );
        }}
        
        function zoomIn() {{
            const svg = d3.select('#mindmap-svg');
            const currentTransform = d3.zoomTransform(svg.node());
            svg.transition().duration(750).call(
                d3.zoom().scaleBy,
                1.3
            );
        }}
        
        function zoomOut() {{
            const svg = d3.select('#mindmap-svg');
            svg.transition().duration(750).call(
                d3.zoom().scaleBy,
                0.7
            );
        }}
        
        function exportPNG() {{
            alert('导出PNG功能需要后端支持，请联系系统管理员。');
        }}
        
        // 初始化思维导图
        window.onload = function() {{
            createMindMap();
        }};
    </script>
</body>
</html>
"""
        
        return html_template

# ========== 知识网络构建器 ==========

class KnowledgeNetworkBuilder:
    """
    知识网络构建器 - 构建和分析知识网络
    识别知识节点之间的关系和模式
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        
        # 关系类型
        self.relation_types = {
            "prerequisite": {
                "name": "先决条件",
                "description": "学习A需要先掌握B",
                "strength": 0.9
            },
            "related": {
                "name": "相关",
                "description": "A和B有相关性",
                "strength": 0.6
            },
            "part_of": {
                "name": "组成部分",
                "description": "A是B的一部分",
                "strength": 0.8
            },
            "instance_of": {
                "name": "实例",
                "description": "A是B的一个实例",
                "strength": 0.7
            },
            "similar_to": {
                "name": "相似",
                "description": "A和B相似",
                "strength": 0.5
            },
            "contrast_with": {
                "name": "对比",
                "description": "A和B形成对比",
                "strength": 0.4
            },
            "leads_to": {
                "name": "导致",
                "description": "A导致或产生B",
                "strength": 0.7
            },
            "analogy": {
                "name": "类比",
                "description": "A与B类似",
                "strength": 0.5
            }
        }
        
        # 网络分析指标
        self.network_metrics = [
            "degree_centrality",  # 度中心性
            "betweenness_centrality",  # 介数中心性
            "closeness_centrality",  # 接近中心性
            "eigenvector_centrality",  # 特征向量中心性
            "pagerank",  # PageRank
            "clustering_coefficient",  # 聚类系数
            "community_detection"  # 社区检测
        ]
    
    def build_from_mindmap(self, 
                          mindmap_root: MindMapNode,
                          node_map: Dict[str, MindMapNode]) -> nx.DiGraph:
        """
        从思维导图构建知识网络
        
        Args:
            mindmap_root: 思维导图根节点
            node_map: 节点映射
            
        Returns:
            知识网络图
        """
        print(f"🔗 从思维导图构建知识网络")
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for node_id, node in node_map.items():
            G.add_node(node_id,
                     title=node.title,
                     description=node.description,
                     node_type=node.node_type,
                     depth=node.depth,
                     importance=node.importance,
                     difficulty=node.difficulty)
        
        # 添加边（父子关系）
        for node_id, node in node_map.items():
            if node.parent_id and node.parent_id in node_map:
                G.add_edge(node.parent_id, node_id,
                         relation_type="parent_child",
                         strength=0.9)
            
            # 添加同级关系
            for sibling_id in node.sibling_ids:
                if sibling_id in node_map:
                    G.add_edge(node_id, sibling_id,
                             relation_type="sibling",
                             strength=0.3)
        
        # 基于大模型识别额外关系
        extra_relations = self._identify_relations_with_llm(node_map)
        for source, target, rel_type, strength in extra_relations:
            if source in G and target in G:
                G.add_edge(source, target,
                         relation_type=rel_type,
                         strength=strength)
        
        print(f"✅ 知识网络构建完成: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
        return G
    
    def build_from_knowledge_nodes(self, 
                                  knowledge_nodes: List[KnowledgeNode]) -> nx.DiGraph:
        """
        从知识节点构建知识网络
        
        Args:
            knowledge_nodes: 知识节点列表
            
        Returns:
            知识网络图
        """
        print(f"🔗 从知识节点构建知识网络 ({len(knowledge_nodes)}个节点)")
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for node in knowledge_nodes:
            G.add_node(node.id,
                     title=node.title,
                     content=node.content[:100] if node.content else "",
                     knowledge_type=node.knowledge_type.value,
                     confidence=node.confidence,
                     mastery_score=node.mastery_score)
        
        # 添加先决条件关系
        for node in knowledge_nodes:
            for prereq_id in node.prerequisites:
                if prereq_id in G:
                    G.add_edge(prereq_id, node.id,
                             relation_type="prerequisite",
                             strength=0.9)
        
        # 添加相关关系
        for node in knowledge_nodes:
            for related_id in node.related_nodes:
                if related_id in G:
                    G.add_edge(node.id, related_id,
                             relation_type="related",
                             strength=0.6)
        
        # 基于内容相似性添加关系
        content_relations = self._identify_content_relations(knowledge_nodes)
        for source, target, strength in content_relations:
            if source in G and target in G:
                G.add_edge(source, target,
                         relation_type="content_similarity",
                         strength=strength)
        
        # 基于大模型识别语义关系
        semantic_relations = self._identify_semantic_relations(knowledge_nodes)
        for source, target, rel_type, strength in semantic_relations:
            if source in G and target in G:
                G.add_edge(source, target,
                         relation_type=rel_type,
                         strength=strength)
        
        print(f"✅ 知识网络构建完成: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
        return G
    
    def analyze_network(self, G: nx.Graph) -> Dict[str, Any]:
        """
        分析知识网络
        
        Args:
            G: 知识网络图
            
        Returns:
            网络分析结果
        """
        print(f"📊 分析知识网络")
        
        analysis = {
            "basic_stats": {},
            "centrality_measures": {},
            "community_structure": {},
            "key_nodes": {},
            "recommendations": []
        }
        
        # 基础统计
        analysis["basic_stats"] = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "density": nx.density(G),
            "is_connected": nx.is_weakly_connected(G) if isinstance(G, nx.DiGraph) else nx.is_connected(G),
            "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        }
        
        # 中心性分析
        if G.number_of_nodes() > 0:
            # 度中心性
            degree_centrality = nx.degree_centrality(G)
            analysis["centrality_measures"]["degree"] = self._get_top_nodes(degree_centrality, 5)
            
            # 介数中心性
            try:
                betweenness_centrality = nx.betweenness_centrality(G)
                analysis["centrality_measures"]["betweenness"] = self._get_top_nodes(betweenness_centrality, 5)
            except:
                analysis["centrality_measures"]["betweenness"] = []
            
            # 接近中心性
            try:
                closeness_centrality = nx.closeness_centrality(G)
                analysis["centrality_measures"]["closeness"] = self._get_top_nodes(closeness_centrality, 5)
            except:
                analysis["centrality_measures"]["closeness"] = []
            
            # PageRank
            try:
                pagerank = nx.pagerank(G)
                analysis["centrality_measures"]["pagerank"] = self._get_top_nodes(pagerank, 5)
            except:
                analysis["centrality_measures"]["pagerank"] = []
        
        # 社区检测
        try:
            if isinstance(G, nx.DiGraph):
                G_undirected = G.to_undirected()
            else:
                G_undirected = G
            
            # 使用Louvain算法检测社区
            import community as community_louvain
            partition = community_louvain.best_partition(G_undirected)
            
            # 统计社区信息
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            analysis["community_structure"] = {
                "community_count": len(communities),
                "communities": {cid: len(nodes) for cid, nodes in communities.items()},
                "largest_community": max(len(nodes) for nodes in communities.values()) if communities else 0
            }
        except Exception as e:
            analysis["community_structure"] = {"error": str(e)}
        
        # 关键节点识别
        analysis["key_nodes"] = self._identify_key_nodes(G, analysis["centrality_measures"])
        
        # 生成学习建议
        analysis["recommendations"] = self._generate_network_recommendations(G, analysis)
        
        return analysis
    
    def find_learning_path(self, 
                          G: nx.DiGraph,
                          start_node_id: str,
                          target_node_id: str) -> Optional[List[str]]:
        """
        查找学习路径
        
        Args:
            G: 知识网络图
            start_node_id: 起始节点ID
            target_node_id: 目标节点ID
            
        Returns:
            学习路径（节点ID列表）
        """
        print(f"[P] Finding learning path: {start_node_id} -> {target_node_id}")
        
        if start_node_id not in G or target_node_id not in G:
            print(f"❌ 节点不存在于网络中")
            return None
        
        # 使用Dijkstra算法查找最短路径
        try:
            # 创建带权重的图（权重 = 1 / 关系强度）
            weighted_G = nx.DiGraph()
            
            for u, v, data in G.edges(data=True):
                strength = data.get('strength', 0.5)
                weight = 1.0 / strength if strength > 0 else 100.0
                weighted_G.add_edge(u, v, weight=weight)
            
            # 查找最短路径
            path = nx.shortest_path(weighted_G, start_node_id, target_node_id, weight='weight')
            
            print(f"✅ 找到学习路径: {len(path)}个节点")
            return path
            
        except nx.NetworkXNoPath:
            print(f"❌ 未找到从 {start_node_id} 到 {target_node_id} 的路径")
            return None
        except Exception as e:
            print(f"❌ 查找学习路径失败: {str(e)}")
            return None
    
    def identify_knowledge_gaps(self, 
                               G: nx.DiGraph,
                               mastered_nodes: List[str]) -> Dict[str, Any]:
        """
        识别知识缺口
        
        Args:
            G: 知识网络图
            mastered_nodes: 已掌握的节点ID列表
            
        Returns:
            知识缺口分析
        """
        print(f"🔍 识别知识缺口 (已掌握: {len(mastered_nodes)}个节点)")
        
        gaps = {
            "missing_prerequisites": [],
            "isolated_clusters": [],
            "weak_connections": [],
            "recommended_nodes": []
        }
        
        # 识别缺失的先决条件
        for node_id in G.nodes():
            if node_id in mastered_nodes:
                continue
            
            # 检查该节点的所有先决条件是否都已掌握
            prerequisites = []
            for predecessor in G.predecessors(node_id):
                edge_data = G.get_edge_data(predecessor, node_id)
                if edge_data and edge_data.get('relation_type') == 'prerequisite':
                    prerequisites.append(predecessor)
            
            missing_prereqs = [p for p in prerequisites if p not in mastered_nodes]
            if missing_prereqs:
                gaps["missing_prerequisites"].append({
                    "node_id": node_id,
                    "node_title": G.nodes[node_id].get('title', node_id),
                    "missing_count": len(missing_prereqs),
                    "missing_nodes": missing_prereqs[:3]  # 只显示前3个
                })
        
        # 识别孤立的集群（如果网络未连通）
        if not nx.is_weakly_connected(G):
            components = list(nx.weakly_connected_components(G))
            mastered_component = None
            
            # 找到包含已掌握节点的组件
            for component in components:
                if any(node in component for node in mastered_nodes):
                    mastered_component = component
                    break
            
            # 其他组件为孤立集群
            if mastered_component:
                for component in components:
                    if component != mastered_component:
                        gaps["isolated_clusters"].append({
                            "component_size": len(component),
                            "component_nodes": list(component)[:5]  # 只显示前5个节点
                        })
        
        # 识别弱连接
        weak_threshold = 0.3
        for u, v, data in G.edges(data=True):
            strength = data.get('strength', 0.5)
            if strength < weak_threshold and u in mastered_nodes and v not in mastered_nodes:
                gaps["weak_connections"].append({
                    "from_node": u,
                    "to_node": v,
                    "strength": strength,
                    "relation_type": data.get('relation_type', 'unknown')
                })
        
        # 推荐学习节点
        gaps["recommended_nodes"] = self._recommend_learning_nodes(G, mastered_nodes)
        
        return gaps
    
    def _identify_relations_with_llm(self, node_map: Dict[str, MindMapNode]) -> List[Tuple[str, str, str, float]]:
        """使用大模型识别节点间关系"""
        relations = []
        
        # 如果节点太多，只分析部分节点
        nodes = list(node_map.values())
        if len(nodes) > 10:
            # 选择重要性高的节点
            nodes.sort(key=lambda x: x.importance, reverse=True)
            nodes = nodes[:10]
        
        # 准备节点信息
        node_info = []
        for node in nodes:
            node_info.append(f"{node.id}: {node.title} - {node.description[:50]}")
        
        prompt = f"""请分析以下知识节点之间的关系：

节点信息：
{chr(10).join(node_info)}

请识别节点之间可能存在的关系，包括：
1. 先决条件关系（学习A需要先掌握B）
2. 相关关系（A和B有相关性）
3. 组成部分关系（A是B的一部分）
4. 相似关系（A和B相似）

对于每个识别的关系，请提供：
- 源节点ID
- 目标节点ID
- 关系类型
- 关系强度（0.0-1.0）

请以JSON数组格式返回。"""
        
        response = self.llm_client.call_llm(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.4
        )
        
        if response:
            try:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    llm_relations = json.loads(json_str)
                    
                    for rel in llm_relations:
                        source = rel.get('source')
                        target = rel.get('target')
                        rel_type = rel.get('relation_type', 'related')
                        strength = float(rel.get('strength', 0.5))
                        
                        if source in node_map and target in node_map:
                            relations.append((source, target, rel_type, strength))
            except Exception as e:
                print(f"❌ LLM关系识别解析失败: {str(e)}")
        
        return relations
    
    def _identify_content_relations(self, knowledge_nodes: List[KnowledgeNode]) -> List[Tuple[str, str, float]]:
        """基于内容相似性识别关系"""
        relations = []
        
        # 简化的文本相似性计算（实际中可以使用TF-IDF或词向量）
        for i, node1 in enumerate(knowledge_nodes):
            for j, node2 in enumerate(knowledge_nodes):
                if i >= j:
                    continue
                
                # 基于标题和内容的简单相似性
                similarity = self._calculate_text_similarity(
                    f"{node1.title} {node1.content[:100]}",
                    f"{node2.title} {node2.content[:100]}"
                )
                
                # 如果相似性超过阈值，添加关系
                if similarity > 0.3:
                    relations.append((node1.id, node2.id, similarity))
        
        return relations
    
    def _identify_semantic_relations(self, knowledge_nodes: List[KnowledgeNode]) -> List[Tuple[str, str, str, float]]:
        """识别语义关系"""
        relations = []
        
        # 选择部分节点进行分析（避免太多API调用）
        if len(knowledge_nodes) > 8:
            selected_nodes = random.sample(knowledge_nodes, 8)
        else:
            selected_nodes = knowledge_nodes
        
        # 准备节点对
        node_pairs = []
        for i in range(len(selected_nodes)):
            for j in range(i + 1, len(selected_nodes)):
                node_pairs.append((selected_nodes[i], selected_nodes[j]))
        
        # 随机选择部分节点对进行分析
        if len(node_pairs) > 10:
            node_pairs = random.sample(node_pairs, 10)
        
        # 分析每个节点对
        for node1, node2 in node_pairs:
            prompt = f"""请分析以下两个知识概念之间的关系：

概念1: {node1.title}
描述: {node1.content[:100]}

概念2: {node2.title}
描述: {node2.content[:100]}

请分析它们之间的关系类型和强度：
1. 关系类型：先决条件、相关、组成部分、相似、对比等
2. 关系强度：0.0-1.0，表示关系的紧密程度
3. 关系描述：简要说明关系

请以JSON格式返回分析结果。"""
            
            response = self.llm_client.call_llm(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            if response:
                try:
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        analysis = json.loads(json_str)
                        
                        rel_type = analysis.get('relation_type', 'related')
                        strength = float(analysis.get('strength', 0.5))
                        
                        relations.append((node1.id, node2.id, rel_type, strength))
                except:
                    pass
        
        return relations
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似性（简化版）"""
        if not text1 or not text2:
            return 0.0
        
        # 转换为小写并分词
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # 计算Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _get_top_nodes(self, centrality_dict: Dict[str, float], top_n: int) -> List[Dict[str, Any]]:
        """获取中心性最高的节点"""
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
        top_nodes = []
        
        for node_id, centrality in sorted_nodes[:top_n]:
            top_nodes.append({
                "node_id": node_id,
                "centrality": centrality,
                "rank": len(top_nodes) + 1
            })
        
        return top_nodes
    
    def _identify_key_nodes(self, G: nx.Graph, centrality_measures: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """识别关键节点"""
        key_nodes = {
            "hubs": [],  # 连接度高的节点
            "bridges": [],  # 连接不同社区的节点
            "foundations": [],  # 先决条件多的节点
            "bottlenecks": []  # 关键路径上的节点
        }
        
        # 识别枢纽节点（度中心性高）
        if "degree" in centrality_measures:
            key_nodes["hubs"] = centrality_measures["degree"][:3]
        
        # 识别桥接节点（介数中心性高）
        if "betweenness" in centrality_measures:
            key_nodes["bridges"] = centrality_measures["betweenness"][:3]
        
        # 识别基础节点（入度高，很多节点的先决条件）
        if isinstance(G, nx.DiGraph):
            in_degrees = dict(G.in_degree())
            sorted_in_degree = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
            key_nodes["foundations"] = [
                {"node_id": node_id, "in_degree": degree, "rank": i+1}
                for i, (node_id, degree) in enumerate(sorted_in_degree[:3])
            ]
        
        # 识别瓶颈节点（出度低但重要的节点）
        if isinstance(G, nx.DiGraph):
            # 简单的瓶颈识别：入度高但出度低的节点
            bottlenecks = []
            for node_id in G.nodes():
                in_deg = G.in_degree(node_id)
                out_deg = G.out_degree(node_id)
                if in_deg > 3 and out_deg < 2:  # 阈值可以根据需要调整
                    bottlenecks.append((node_id, in_deg, out_deg))
            
            bottlenecks.sort(key=lambda x: x[1], reverse=True)
            key_nodes["bottlenecks"] = [
                {"node_id": node_id, "in_degree": in_deg, "out_degree": out_deg, "rank": i+1}
                for i, (node_id, in_deg, out_deg) in enumerate(bottlenecks[:3])
            ]
        
        return key_nodes
    
    def _generate_network_recommendations(self, G: nx.Graph, analysis: Dict[str, Any]) -> List[str]:
        """生成网络分析建议"""
        recommendations = []
        
        # 基于网络密度
        density = analysis["basic_stats"].get("density", 0)
        if density < 0.1:
            recommendations.append("网络连接稀疏，建议增加节点间的关联学习")
        elif density > 0.5:
            recommendations.append("网络连接紧密，可以考虑进行系统性的复习和整合")
        
        # 基于社区结构
        community_count = analysis["community_structure"].get("community_count", 1)
        if community_count > 3:
            recommendations.append(f"网络包含{community_count}个知识社区，建议按社区分组学习")
        
        # 基于关键节点
        if analysis["key_nodes"].get("bottlenecks"):
            bottleneck_count = len(analysis["key_nodes"]["bottlenecks"])
            recommendations.append(f"识别到{bottleneck_count}个瓶颈节点，建议优先学习这些节点以打通知识路径")
        
        # 基于连通性
        if not analysis["basic_stats"].get("is_connected", True):
            recommendations.append("网络未完全连通，可能存在孤立的知识点，建议建立连接")
        
        return recommendations
    
    def _recommend_learning_nodes(self, G: nx.DiGraph, mastered_nodes: List[str]) -> List[Dict[str, Any]]:
        """推荐学习节点"""
        recommendations = []
        
        # 找到已掌握节点的邻居
        frontier_nodes = set()
        for node_id in mastered_nodes:
            if node_id in G:
                # 获取该节点的所有后继节点
                for successor in G.successors(node_id):
                    if successor not in mastered_nodes:
                        frontier_nodes.add(successor)
        
        # 为每个边界节点计算优先级
        for node_id in frontier_nodes:
            # 计算该节点的先决条件掌握比例
            prerequisites = list(G.predecessors(node_id))
            mastered_prereqs = [p for p in prerequisites if p in mastered_nodes]
            prereq_ratio = len(mastered_prereqs) / len(prerequisites) if prerequisites else 1.0
            
            # 计算节点重要性（基于PageRank或度中心性）
            try:
                importance = nx.pagerank(G).get(node_id, 0.5)
            except:
                importance = G.out_degree(node_id) / (G.number_of_nodes() - 1) if G.number_of_nodes() > 1 else 0.5
            
            # 计算优先级
            priority = prereq_ratio * 0.6 + importance * 0.4
            
            recommendations.append({
                "node_id": node_id,
                "node_title": G.nodes[node_id].get('title', node_id),
                "prerequisite_ratio": prereq_ratio,
                "importance": importance,
                "priority": priority,
                "ready_to_learn": prereq_ratio >= 0.8  # 80%的先决条件已掌握
            })
        
        # 按优先级排序
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations[:10]  # 返回前10个推荐

# ========== 学习路径生成器 ==========

class LearningPathGenerator:
    """
    学习路径生成器 - 基于知识网络生成个性化学习路径
    """
    
    def __init__(self, 
                 network_builder: Optional[KnowledgeNetworkBuilder] = None,
                 llm_client: Optional[LLMClient] = None):
        self.network_builder = network_builder or KnowledgeNetworkBuilder()
        self.llm_client = llm_client or LLMClient()
        
        # 学习路径策略
        self.path_strategies = {
            "sequential": {
                "name": "顺序学习",
                "description": "按依赖关系顺序学习",
                "suitable_for": ["初学者", "系统性知识"]
            },
            "spiral": {
                "name": "螺旋式学习",
                "description": "多次循环，每次加深理解",
                "suitable_for": ["复杂概念", "技能学习"]
            },
            "modular": {
                "name": "模块化学习",
                "description": "按模块分组学习",
                "suitable_for": ["大规模知识", "并行学习"]
            },
            "priority": {
                "name": "优先级学习",
                "description": "按重要性优先级学习",
                "suitable_for": ["时间有限", "考试准备"]
            },
            "adaptive": {
                "name": "自适应学习",
                "description": "根据学习情况动态调整",
                "suitable_for": ["个性化学习", "持续学习"]
            }
        }
        
        # 学习阶段配置
        self.learning_stages = {
            "exploration": {
                "name": "探索阶段",
                "duration_ratio": 0.2,
                "focus": ["概览", "核心概念", "建立认知"]
            },
            "foundation": {
                "name": "基础阶段",
                "duration_ratio": 0.3,
                "focus": ["基本原理", "关键技能", "建立基础"]
            },
            "deepening": {
                "name": "深化阶段",
                "duration_ratio": 0.3,
                "focus": ["深度理解", "复杂应用", "建立联系"]
            },
            "integration": {
                "name": "整合阶段",
                "duration_ratio": 0.2,
                "focus": ["系统整合", "创新应用", "建立体系"]
            }
        }
    
    def generate_for_goal(self, 
                         goal: LearningGoal,
                         knowledge_network: nx.DiGraph,
                         current_knowledge: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        为目标生成学习路径
        
        Args:
            goal: 学习目标
            knowledge_network: 知识网络
            current_knowledge: 当前已掌握的知识节点ID列表
            
        Returns:
            学习路径计划
        """
        print(f"🛣️ 为学习目标生成学习路径: {goal.description}")
        
        learning_path = {
            "goal_id": goal.id,
            "goal_description": goal.description,
            "generated_at": datetime.now().isoformat(),
            "strategy": "adaptive",
            "stages": [],
            "total_nodes": 0,
            "estimated_time_hours": 0,
            "recommendations": []
        }
        
        # 确定学习策略
        strategy = self._determine_strategy(goal, knowledge_network)
        learning_path["strategy"] = strategy
        
        # 获取网络中的所有节点
        all_nodes = list(knowledge_network.nodes())
        
        # 确定当前知识状态
        if current_knowledge is None:
            current_knowledge = []
        
        # 识别需要学习的节点
        nodes_to_learn = [node_id for node_id in all_nodes if node_id not in current_knowledge]
        
        if not nodes_to_learn:
            learning_path["message"] = "所有知识节点都已掌握"
            return learning_path
        
        learning_path["total_nodes"] = len(nodes_to_learn)
        
        # 根据策略生成学习阶段
        if strategy == "sequential":
            stages = self._generate_sequential_path(nodes_to_learn, knowledge_network)
        elif strategy == "spiral":
            stages = self._generate_spiral_path(nodes_to_learn, knowledge_network, goal)
        elif strategy == "modular":
            stages = self._generate_modular_path(nodes_to_learn, knowledge_network)
        elif strategy == "priority":
            stages = self._generate_priority_path(nodes_to_learn, knowledge_network)
        else:  # adaptive
            stages = self._generate_adaptive_path(nodes_to_learn, knowledge_network, current_knowledge)
        
        # 计算每个阶段的时间估计
        total_time_hours = 0
        for stage in stages:
            stage_time = self._estimate_stage_time(stage, knowledge_network)
            stage["estimated_time_hours"] = stage_time
            total_time_hours += stage_time
        
        learning_path["stages"] = stages
        learning_path["estimated_time_hours"] = total_time_hours
        
        # 生成学习建议
        learning_path["recommendations"] = self._generate_path_recommendations(
            learning_path, knowledge_network, current_knowledge
        )
        
        print(f"✅ 学习路径生成完成: {len(stages)}个阶段, {total_time_hours:.1f}小时")
        return learning_path
    
    def generate_personalized_path(self,
                                 user_profile: Dict[str, Any],
                                 knowledge_network: nx.DiGraph,
                                 learning_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        生成个性化学习路径
        
        Args:
            user_profile: 用户画像
            knowledge_network: 知识网络
            learning_history: 学习历史
            
        Returns:
            个性化学习路径
        """
        print(f"👤 生成个性化学习路径")
        
        # 提取用户特征
        learning_style = user_profile.get("learning_style", "balanced")
        available_time = user_profile.get("available_time_hours_per_week", 10)
        prior_knowledge = user_profile.get("prior_knowledge", [])
        goals = user_profile.get("learning_goals", [])
        
        # 确定学习策略
        strategy = self._determine_personalized_strategy(user_profile)
        
        # 识别需要学习的节点
        all_nodes = list(knowledge_network.nodes())
        nodes_to_learn = [node_id for node_id in all_nodes if node_id not in prior_knowledge]
        
        # 根据学习风格调整路径
        if learning_style == "visual":
            # 视觉学习者：优先学习有图表、可视化的节点
            visual_nodes = self._identify_visual_nodes(knowledge_network)
            nodes_to_learn = self._prioritize_nodes(nodes_to_learn, visual_nodes)
        elif learning_style == "auditory":
            # 听觉学习者：优先学习有音频、讲解的节点
            auditory_nodes = self._identify_auditory_nodes(knowledge_network)
            nodes_to_learn = self._prioritize_nodes(nodes_to_learn, auditory_nodes)
        elif learning_style == "kinesthetic":
            # 动觉学习者：优先学习有实践、练习的节点
            practice_nodes = self._identify_practice_nodes(knowledge_network)
            nodes_to_learn = self._prioritize_nodes(nodes_to_learn, practice_nodes)
        
        # 根据可用时间调整学习强度
        intensity = self._determine_learning_intensity(available_time)
        
        # 生成路径
        path = {
            "user_id": user_profile.get("user_id", "unknown"),
            "learning_style": learning_style,
            "strategy": strategy,
            "intensity": intensity,
            "available_time_hours_per_week": available_time,
            "generated_at": datetime.now().isoformat(),
            "stages": [],
            "total_nodes": len(nodes_to_learn),
            "estimated_weeks": 0
        }
        
        # 生成学习阶段
        if strategy == "sequential":
            stages = self._generate_sequential_path(nodes_to_learn, knowledge_network)
        else:
            stages = self._generate_adaptive_path(nodes_to_learn, knowledge_network, prior_knowledge)
        
        # 根据强度调整阶段
        stages = self._adjust_stages_for_intensity(stages, intensity)
        
        # 计算总时间和周数
        total_hours = sum(stage.get("estimated_time_hours", 0) for stage in stages)
        estimated_weeks = math.ceil(total_hours / available_time) if available_time > 0 else 0
        
        path["stages"] = stages
        path["estimated_time_hours"] = total_hours
        path["estimated_weeks"] = estimated_weeks
        
        # 添加个性化建议
        path["personalized_recommendations"] = self._generate_personalized_recommendations(
            user_profile, path, knowledge_network
        )
        
        return path
    
    def adjust_path_based_on_progress(self,
                                    current_path: Dict[str, Any],
                                    progress_data: Dict[str, Any],
                                    knowledge_network: nx.DiGraph) -> Dict[str, Any]:
        """
        基于学习进度调整学习路径
        
        Args:
            current_path: 当前学习路径
            progress_data: 进度数据
            knowledge_network: 知识网络
            
        Returns:
            调整后的学习路径
        """
        print(f"🔄 基于进度调整学习路径")
        
        # 提取进度信息
        mastered_nodes = progress_data.get("mastered_nodes", [])
        struggling_nodes = progress_data.get("struggling_nodes", [])
        learning_speed = progress_data.get("learning_speed", 1.0)  # 1.0为正常速度
        
        # 复制当前路径
        adjusted_path = current_path.copy()
        adjusted_path["adjusted_at"] = datetime.now().isoformat()
        adjusted_path["adjustment_reason"] = []
        
        # 更新已掌握的节点
        original_total = adjusted_path.get("total_nodes", 0)
        remaining_nodes = []
        
        for stage in adjusted_path.get("stages", []):
            original_nodes = stage.get("node_ids", [])
            # 过滤掉已掌握的节点
            remaining_in_stage = [node_id for node_id in original_nodes if node_id not in mastered_nodes]
            
            if len(remaining_in_stage) < len(original_nodes):
                stage["node_ids"] = remaining_in_stage
                stage["node_count"] = len(remaining_in_stage)
                adjusted_path["adjustment_reason"].append(f"阶段'{stage['name']}'移除了{len(original_nodes) - len(remaining_in_stage)}个已掌握的节点")
            
            remaining_nodes.extend(remaining_in_stage)
        
        # 更新统计信息
        adjusted_path["total_nodes"] = len(remaining_nodes)
        adjusted_path["mastered_nodes"] = len(mastered_nodes)
        
        # 如果有学习困难的节点，重新安排或添加额外练习
        if struggling_nodes:
            # 为困难节点创建专门的复习阶段
            review_stage = {
                "name": "难点复习",
                "description": f"针对{len(struggling_nodes)}个学习困难的节点进行复习",
                "node_ids": struggling_nodes,
                "node_count": len(struggling_nodes),
                "focus": ["复习巩固", "克服难点", "额外练习"],
                "estimated_time_hours": len(struggling_nodes) * 0.5,  # 每个困难节点0.5小时
                "is_adjustment": True
            }
            
            # 将复习阶段插入到第二阶段（如果有的话）
            stages = adjusted_path.get("stages", [])
            if len(stages) > 1:
                stages.insert(1, review_stage)
            else:
                stages.append(review_stage)
            
            adjusted_path["stages"] = stages
            adjusted_path["adjustment_reason"].append(f"添加了难点复习阶段，包含{len(struggling_nodes)}个节点")
        
        # 根据学习速度调整时间估计
        if learning_speed != 1.0:
            original_hours = adjusted_path.get("estimated_time_hours", 0)
            adjusted_hours = original_hours / learning_speed if learning_speed > 0 else original_hours
            
            # 更新每个阶段的时间估计
            for stage in adjusted_path.get("stages", []):
                stage_hours = stage.get("estimated_time_hours", 0)
                stage["estimated_time_hours"] = stage_hours / learning_speed if learning_speed > 0 else stage_hours
            
            adjusted_path["estimated_time_hours"] = adjusted_hours
            
            if learning_speed < 0.8:
                adjusted_path["adjustment_reason"].append(f"检测到学习速度较慢({learning_speed:.1f}x)，已增加时间估计")
            elif learning_speed > 1.2:
                adjusted_path["adjustment_reason"].append(f"检测到学习速度较快({learning_speed:.1f}x)，已减少时间估计")
        
        # 重新计算总时间
        total_hours = sum(stage.get("estimated_time_hours", 0) for stage in adjusted_path.get("stages", []))
        adjusted_path["estimated_time_hours"] = total_hours
        
        print(f"✅ 学习路径调整完成: {len(adjusted_path.get('stages', []))}个阶段, {total_hours:.1f}小时")
        return adjusted_path
    
    def _determine_strategy(self, goal: LearningGoal, knowledge_network: nx.DiGraph) -> str:
        """确定学习策略"""
        # 基于目标规模
        if goal.scale in [GoalScale.MICRO, GoalScale.SMALL]:
            return "sequential"
        elif goal.scale == GoalScale.MEDIUM:
            return "spiral"
        elif goal.scale == GoalScale.LARGE:
            return "modular"
        elif goal.scale == GoalScale.MASSIVE:
            return "priority"
        else:
            return "adaptive"
    
    def _determine_personalized_strategy(self, user_profile: Dict[str, Any]) -> str:
        """确定个性化学习策略"""
        learning_style = user_profile.get("learning_style", "balanced")
        time_availability = user_profile.get("time_availability", "medium")
        experience_level = user_profile.get("experience_level", "intermediate")
        
        # 根据用户特征选择策略
        if time_availability == "low":
            return "priority"
        elif experience_level == "beginner":
            return "sequential"
        elif learning_style == "exploratory":
            return "spiral"
        elif time_availability == "high" and experience_level == "advanced":
            return "modular"
        else:
            return "adaptive"
    
    def _generate_sequential_path(self, nodes_to_learn: List[str], knowledge_network: nx.DiGraph) -> List[Dict[str, Any]]:
        """生成顺序学习路径"""
        # 基于拓扑排序（考虑依赖关系）
        try:
            # 创建子图
            subgraph = knowledge_network.subgraph(nodes_to_learn)
            
            # 进行拓扑排序
            sorted_nodes = list(nx.topological_sort(subgraph))
        except:
            # 如果存在环，使用简单的节点重要性排序
            node_importance = {}
            for node_id in nodes_to_learn:
                # 简单的重要性计算：入度 + 出度
                in_deg = knowledge_network.in_degree(node_id)
                out_deg = knowledge_network.out_degree(node_id)
                node_importance[node_id] = in_deg + out_deg
            
            sorted_nodes = sorted(nodes_to_learn, key=lambda x: node_importance.get(x, 0), reverse=True)
        
        # 将节点分组到阶段
        stage_count = min(4, max(1, len(sorted_nodes) // 5))
        nodes_per_stage = math.ceil(len(sorted_nodes) / stage_count)
        
        stages = []
        for i in range(stage_count):
            start_idx = i * nodes_per_stage
            end_idx = min((i + 1) * nodes_per_stage, len(sorted_nodes))
            
            stage_nodes = sorted_nodes[start_idx:end_idx]
            
            stage = {
                "name": f"阶段 {i+1}",
                "description": f"顺序学习第{i+1}部分",
                "node_ids": stage_nodes,
                "node_count": len(stage_nodes),
                "focus": ["顺序学习", "依赖关系", "逐步深入"],
                "order": "sequential"
            }
            
            stages.append(stage)
        
        return stages
    
    def _generate_spiral_path(self, nodes_to_learn: List[str], knowledge_network: nx.DiGraph, goal: LearningGoal) -> List[Dict[str, Any]]:
        """生成螺旋式学习路径"""
        # 识别核心节点
        core_nodes = self._identify_core_nodes(nodes_to_learn, knowledge_network)
        
        # 创建螺旋式阶段
        stages = []
        
        # 阶段1: 探索核心概念
        if core_nodes:
            stages.append({
                "name": "螺旋第1轮: 核心探索",
                "description": "探索核心概念，建立整体认知",
                "node_ids": core_nodes[:min(5, len(core_nodes))],
                "node_count": min(5, len(core_nodes)),
                "focus": ["核心概念", "整体认知", "初步了解"],
                "depth": "surface"
            })
        
        # 阶段2: 基础学习
        remaining_nodes = [node for node in nodes_to_learn if node not in core_nodes]
        if remaining_nodes:
            stages.append({
                "name": "螺旋第2轮: 基础建立",
                "description": "学习基础知识，建立理解框架",
                "node_ids": remaining_nodes[:min(8, len(remaining_nodes))],
                "node_count": min(8, len(remaining_nodes)),
                "focus": ["基础知识", "建立框架", "深入理解"],
                "depth": "understanding"
            })
        
        # 阶段3: 深化学习
        if nodes_to_learn:
            stages.append({
                "name": "螺旋第3轮: 深化理解",
                "description": "深化理解，建立联系",
                "node_ids": nodes_to_learn[:min(10, len(nodes_to_learn))],
                "node_count": min(10, len(nodes_to_learn)),
                "focus": ["深度理解", "建立联系", "分析应用"],
                "depth": "analysis"
            })
        
        # 阶段4: 整合应用
        if nodes_to_learn:
            stages.append({
                "name": "螺旋第4轮: 整合应用",
                "description": "整合知识，实践应用",
                "node_ids": nodes_to_learn[:min(8, len(nodes_to_learn))],
                "node_count": min(8, len(nodes_to_learn)),
                "focus": ["整合知识", "实践应用", "创新思考"],
                "depth": "application"
            })
        
        return stages
    
    def _generate_modular_path(self, nodes_to_learn: List[str], knowledge_network: nx.DiGraph) -> List[Dict[str, Any]]:
        """生成模块化学习路径"""
        # 检测社区（模块）
        try:
            if isinstance(knowledge_network, nx.DiGraph):
                G_undirected = knowledge_network.to_undirected()
            else:
                G_undirected = knowledge_network
            
            # 创建子图
            subgraph = G_undirected.subgraph(nodes_to_learn)
            
            # 使用Louvain算法检测社区
            import community as community_louvain
            partition = community_louvain.best_partition(subgraph)
            
            # 按社区分组
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            # 为每个社区创建阶段
            stages = []
            for i, (community_id, community_nodes) in enumerate(communities.items()):
                stage = {
                    "name": f"模块 {i+1}",
                    "description": f"学习知识模块{i+1}，包含{len(community_nodes)}个相关概念",
                    "node_ids": community_nodes,
                    "node_count": len(community_nodes),
                    "community_id": community_id,
                    "focus": ["模块学习", "内部关联", "并行掌握"]
                }
                stages.append(stage)
            
            # 按社区大小排序
            stages.sort(key=lambda x: x["node_count"], reverse=True)
            
            return stages
            
        except Exception as e:
            print(f"❌ 社区检测失败: {str(e)}，使用备选方案")
            # 备选方案：随机分组
            return self._generate_random_modular_path(nodes_to_learn)
    
    def _generate_priority_path(self, nodes_to_learn: List[str], knowledge_network: nx.DiGraph) -> List[Dict[str, Any]]:
        """生成优先级学习路径"""
        # 计算每个节点的优先级
        node_priority = {}
        
        for node_id in nodes_to_learn:
            # 基于中心性
            try:
                centrality = nx.degree_centrality(knowledge_network).get(node_id, 0)
            except:
                centrality = 0
            
            # 基于先决条件数量
            prereq_count = knowledge_network.in_degree(node_id)
            
            # 基于重要性（如果有的话）
            importance = knowledge_network.nodes[node_id].get('importance', 0.5)
            
            # 计算综合优先级
            priority = centrality * 0.4 + (prereq_count / 10) * 0.3 + importance * 0.3
            node_priority[node_id] = priority
        
        # 按优先级排序
        sorted_nodes = sorted(nodes_to_learn, key=lambda x: node_priority.get(x, 0), reverse=True)
        
        # 创建阶段：高优先级、中优先级、低优先级
        stages = []
        
        if sorted_nodes:
            # 阶段1: 高优先级（前30%）
            high_priority_count = max(1, int(len(sorted_nodes) * 0.3))
            stages.append({
                "name": "高优先级",
                "description": "学习最重要的核心概念",
                "node_ids": sorted_nodes[:high_priority_count],
                "node_count": high_priority_count,
                "priority": "high",
                "focus": ["核心概念", "关键知识", "高效学习"]
            })
            
            # 阶段2: 中优先级（中间40%）
            mid_priority_count = max(1, int(len(sorted_nodes) * 0.4))
            stages.append({
                "name": "中优先级",
                "description": "学习重要的支持性知识",
                "node_ids": sorted_nodes[high_priority_count:high_priority_count + mid_priority_count],
                "node_count": mid_priority_count,
                "priority": "medium",
                "focus": ["支持知识", "建立基础", "系统学习"]
            })
            
            # 阶段3: 低优先级（剩余30%）
            if high_priority_count + mid_priority_count < len(sorted_nodes):
                remaining_nodes = sorted_nodes[high_priority_count + mid_priority_count:]
                stages.append({
                    "name": "低优先级",
                    "description": "学习补充性知识",
                    "node_ids": remaining_nodes,
                    "node_count": len(remaining_nodes),
                    "priority": "low",
                    "focus": ["补充知识", "拓展学习", "完善体系"]
                })
        
        return stages
    
    def _generate_adaptive_path(self, 
                              nodes_to_learn: List[str], 
                              knowledge_network: nx.DiGraph,
                              current_knowledge: List[str]) -> List[Dict[str, Any]]:
        """生成自适应学习路径"""
        # 识别知识缺口
        gaps = self.network_builder.identify_knowledge_gaps(knowledge_network, current_knowledge)
        
        # 创建自适应阶段
        stages = []
        
        # 阶段1: 填补先决条件缺口
        missing_prereqs = []
        for gap in gaps.get("missing_prerequisites", []):
            missing_prereqs.extend(gap.get("missing_nodes", []))
        
        if missing_prereqs:
            # 去重
            missing_prereqs = list(set(missing_prereqs))
            
            stages.append({
                "name": "先决条件学习",
                "description": f"填补{len(missing_prereqs)}个先决知识缺口",
                "node_ids": missing_prereqs[:10],  # 最多10个
                "node_count": min(10, len(missing_prereqs)),
                "focus": ["先决知识", "基础准备", "打通路径"],
                "is_gap_filling": True
            })
        
        # 阶段2: 学习推荐节点
        recommended_nodes = gaps.get("recommended_nodes", [])
        if recommended_nodes:
            ready_nodes = [node["node_id"] for node in recommended_nodes if node.get("ready_to_learn", False)]
            
            if ready_nodes:
                stages.append({
                    "name": "推荐学习",
                    "description": f"学习{len(ready_nodes)}个推荐的知识节点",
                    "node_ids": ready_nodes[:8],  # 最多8个
                    "node_count": min(8, len(ready_nodes)),
                    "focus": ["推荐学习", "高效路径", "适时学习"]
                })
        
        # 阶段3: 扩展学习
        remaining_nodes = [node for node in nodes_to_learn 
                          if node not in missing_prereqs and 
                          node not in [r["node_id"] for r in recommended_nodes]]
        
        if remaining_nodes:
            # 按重要性排序
            node_importance = {}
            for node_id in remaining_nodes:
                importance = knowledge_network.nodes[node_id].get('importance', 0.5)
                node_importance[node_id] = importance
            
            sorted_remaining = sorted(remaining_nodes, key=lambda x: node_importance.get(x, 0), reverse=True)
            
            stages.append({
                "name": "扩展学习",
                "description": f"学习{len(sorted_remaining[:6])}个扩展知识节点",
                "node_ids": sorted_remaining[:6],  # 最多6个
                "node_count": min(6, len(sorted_remaining)),
                "focus": ["扩展知识", "完善体系", "深化理解"]
            })
        
        return stages
    
    def _generate_random_modular_path(self, nodes_to_learn: List[str]) -> List[Dict[str, Any]]:
        """生成随机模块化路径（备选方案）"""
        import random
        
        # 随机分组
        random.shuffle(nodes_to_learn)
        
        # 确定阶段数量
        stage_count = min(4, max(1, len(nodes_to_learn) // 3))
        nodes_per_stage = math.ceil(len(nodes_to_learn) / stage_count)
        
        stages = []
        for i in range(stage_count):
            start_idx = i * nodes_per_stage
            end_idx = min((i + 1) * nodes_per_stage, len(nodes_to_learn))
            
            stage_nodes = nodes_to_learn[start_idx:end_idx]
            
            stage = {
                "name": f"模块 {i+1}",
                "description": f"学习知识模块{i+1}",
                "node_ids": stage_nodes,
                "node_count": len(stage_nodes),
                "focus": ["模块学习", "知识分组", "系统掌握"]
            }
            
            stages.append(stage)
        
        return stages
    
    def _identify_core_nodes(self, nodes: List[str], knowledge_network: nx.DiGraph) -> List[str]:
        """识别核心节点"""
        core_nodes = []
        
        # 基于中心性
        try:
            centrality = nx.degree_centrality(knowledge_network)
            for node_id in nodes:
                if centrality.get(node_id, 0) > 0.1:  # 阈值
                    core_nodes.append(node_id)
        except:
            pass
        
        # 如果基于中心性的识别不够，使用其他方法
        if len(core_nodes) < 3:
            # 基于PageRank
            try:
                pagerank = nx.pagerank(knowledge_network)
                sorted_nodes = sorted(nodes, key=lambda x: pagerank.get(x, 0), reverse=True)
                core_nodes = sorted_nodes[:min(5, len(sorted_nodes))]
            except:
                # 最后手段：选择前几个节点
                core_nodes = nodes[:min(5, len(nodes))]
        
        return core_nodes
    
    def _identify_visual_nodes(self, knowledge_network: nx.DiGraph) -> List[str]:
        """识别视觉型学习节点"""
        visual_nodes = []
        
        for node_id, data in knowledge_network.nodes(data=True):
            # 这里可以根据实际数据判断
            # 例如：节点类型包含"图表"、"图像"、"可视化"等
            title = data.get('title', '').lower()
            if any(keyword in title for keyword in ['图', '表', '视觉', '可视化', '图表']):
                visual_nodes.append(node_id)
        
        return visual_nodes
    
    def _identify_auditory_nodes(self, knowledge_network: nx.DiGraph) -> List[str]:
        """识别听觉型学习节点"""
        auditory_nodes = []
        
        for node_id, data in knowledge_network.nodes(data=True):
            title = data.get('title', '').lower()
            if any(keyword in title for keyword in ['音频', '声音', '听力', '讲解', '讲座']):
                auditory_nodes.append(node_id)
        
        return auditory_nodes
    
    def _identify_practice_nodes(self, knowledge_network: nx.DiGraph) -> List[str]:
        """识别实践型学习节点"""
        practice_nodes = []
        
        for node_id, data in knowledge_network.nodes(data=True):
            title = data.get('title', '').lower()
            if any(keyword in title for keyword in ['练习', '实践', '操作', '实验', '项目']):
                practice_nodes.append(node_id)
        
        return practice_nodes
    
    def _prioritize_nodes(self, all_nodes: List[str], preferred_nodes: List[str]) -> List[str]:
        """优先排序节点"""
        # 将偏好的节点放在前面
        result = []
        
        # 添加偏好节点
        for node in preferred_nodes:
            if node in all_nodes and node not in result:
                result.append(node)
        
        # 添加其他节点
        for node in all_nodes:
            if node not in result:
                result.append(node)
        
        return result
    
    def _determine_learning_intensity(self, available_time: float) -> str:
        """确定学习强度"""
        if available_time < 5:
            return "low"
        elif available_time < 15:
            return "medium"
        else:
            return "high"
    
    def _adjust_stages_for_intensity(self, stages: List[Dict[str, Any]], intensity: str) -> List[Dict[str, Any]]:
        """根据学习强度调整阶段"""
        adjusted_stages = []
        
        if intensity == "low":
            # 低强度：减少每个阶段的节点数量
            for stage in stages:
                node_ids = stage.get("node_ids", [])
                if len(node_ids) > 4:
                    stage["node_ids"] = node_ids[:4]
                    stage["node_count"] = 4
                    stage["description"] += "（低强度调整）"
                adjusted_stages.append(stage)
        
        elif intensity == "high":
            # 高强度：合并阶段或增加节点数量
            if len(stages) > 3:
                # 合并前两个阶段
                merged_stage = {
                    "name": "高强度学习阶段1",
                    "description": "合并的高强度学习阶段",
                    "node_ids": [],
                    "node_count": 0,
                    "focus": ["高强度", "集中学习", "快速推进"]
                }
                
                for stage in stages[:2]:
                    merged_stage["node_ids"].extend(stage.get("node_ids", []))
                
                merged_stage["node_ids"] = list(set(merged_stage["node_ids"]))
                merged_stage["node_count"] = len(merged_stage["node_ids"])
                
                adjusted_stages.append(merged_stage)
                adjusted_stages.extend(stages[2:])
            else:
                adjusted_stages = stages
        
        else:
            # 中等强度：保持不变
            adjusted_stages = stages
        
        return adjusted_stages
    
    def _estimate_stage_time(self, stage: Dict[str, Any], knowledge_network: nx.DiGraph) -> float:
        """估计阶段学习时间（小时）"""
        node_count = stage.get("node_count", 0)
        
        # 基础时间：每个节点平均30分钟
        base_hours = node_count * 0.5
        
        # 根据阶段深度调整
        depth = stage.get("depth", "understanding")
        if depth == "surface":
            base_hours *= 0.7
        elif depth in ["analysis", "application"]:
            base_hours *= 1.3
        
        # 根据是否是难点调整
        if stage.get("is_gap_filling", False):
            base_hours *= 1.2
        
        return round(base_hours, 1)
    
    def _generate_path_recommendations(self, 
                                     learning_path: Dict[str, Any],
                                     knowledge_network: nx.DiGraph,
                                     current_knowledge: List[str]) -> List[str]:
        """生成路径学习建议"""
        recommendations = []
        
        strategy = learning_path.get("strategy", "adaptive")
        total_nodes = learning_path.get("total_nodes", 0)
        total_hours = learning_path.get("estimated_time_hours", 0)
        
        # 基于策略的建议
        if strategy == "sequential":
            recommendations.append("建议按顺序学习，不要跳过的前面的内容")
        elif strategy == "spiral":
            recommendations.append("建议多次循环学习，每次加深理解")
        elif strategy == "modular":
            recommendations.append("建议按模块学习，可以并行学习不同模块")
        elif strategy == "priority":
            recommendations.append("建议优先学习高优先级内容，时间有限时可以跳过低优先级内容")
        elif strategy == "adaptive":
            recommendations.append("建议根据学习情况动态调整学习计划")
        
        # 基于学习量的建议
        if total_nodes > 50:
            recommendations.append("学习内容较多，建议制定长期计划并坚持执行")
        elif total_nodes < 10:
            recommendations.append("学习内容较少，可以快速完成")
        
        # 基于时间的建议
        if total_hours > 40:
            recommendations.append(f"预计需要{total_hours:.0f}小时，建议分散在几周内完成")
        elif total_hours > 20:
            recommendations.append(f"预计需要{total_hours:.0f}小时，建议在一周内集中学习")
        else:
            recommendations.append(f"预计需要{total_hours:.0f}小时，可以在几天内完成")
        
        # 基于知识状态的建议
        if not current_knowledge:
            recommendations.append("从零开始学习，建议先建立整体认知")
        elif len(current_knowledge) > 10:
            recommendations.append("已有一定基础，可以快速推进学习")
        
        return recommendations
    
    def _generate_personalized_recommendations(self, 
                                             user_profile: Dict[str, Any],
                                             learning_path: Dict[str, Any],
                                             knowledge_network: nx.DiGraph) -> List[str]:
        """生成个性化建议"""
        recommendations = []
        
        learning_style = user_profile.get("learning_style", "balanced")
        available_time = user_profile.get("available_time_hours_per_week", 10)
        
        # 基于学习风格的建议
        if learning_style == "visual":
            recommendations.append("作为视觉型学习者，建议多使用图表、思维导图等可视化工具")
        elif learning_style == "auditory":
            recommendations.append("作为听觉型学习者，建议多听讲解、参与讨论")
        elif learning_style == "kinesthetic":
            recommendations.append("作为动觉型学习者，建议多动手实践、做练习")
        
        # 基于时间可用性的建议
        if available_time < 5:
            recommendations.append("每周学习时间有限，建议制定高效的学习计划")
        elif available_time > 20:
            recommendations.append("每周学习时间充足，可以按计划稳步推进")
        
        # 基于路径的建议
        estimated_weeks = learning_path.get("estimated_weeks", 0)
        if estimated_weeks > 8:
            recommendations.append(f"预计需要{estimated_weeks}周，建议设置阶段性目标保持动力")
        
        return recommendations

# ========== 探索管理器 ==========

class ExplorerManager:
    """探索管理器 - 整合深度知识探索功能"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.llm_client = LLMClient()
            self.question_engine = IntelligentQuestionEngine(self.llm_client)
            self.visualizer = MindMapVisualizer()
            self.network_builder = KnowledgeNetworkBuilder(self.llm_client)
            self.path_generator = LearningPathGenerator(self.network_builder, self.llm_client)
            self._initialized = True
    
    def explore_mindmap(self, 
                       mindmap_root: MindMapNode,
                       node_map: Dict[str, MindMapNode],
                       depth_level: str = "understanding") -> Dict[str, Any]:
        """
        探索思维导图
        
        Args:
            mindmap_root: 思维导图根节点
            node_map: 节点映射
            depth_level: 探索深度
            
        Returns:
            探索结果
        """
        print(f"🔍 探索思维导图: {mindmap_root.title}")
        
        exploration_result = {
            "mindmap_id": mindmap_root.id,
            "mindmap_title": mindmap_root.title,
            "explored_at": datetime.now().isoformat(),
            "depth_level": depth_level,
            "questions": {},
            "visualization": {},
            "network_analysis": {},
            "learning_path": {}
        }
        
        # 生成问题
        questions = self.question_engine.generate_questions_for_mindmap(
            mindmap_root, node_map, depth_level, 2
        )
        exploration_result["questions"] = questions
        
        # 可视化思维导图
        visualization_path = self.visualizer.visualize_mindmap(
            mindmap_root, node_map, "png", "balanced", True
        )
        if visualization_path:
            exploration_result["visualization"]["mindmap"] = visualization_path
        
        # 构建知识网络
        knowledge_network = self.network_builder.build_from_mindmap(mindmap_root, node_map)
        
        # 分析网络
        network_analysis = self.network_builder.analyze_network(knowledge_network)
        exploration_result["network_analysis"] = network_analysis
        
        # 可视化知识网络
        network_viz_path = self.visualizer.visualize_knowledge_network(
            list(node_map.values()), 
            list(knowledge_network.edges(data=True))
        )
        if network_viz_path:
            exploration_result["visualization"]["knowledge_network"] = network_viz_path
        
        # 创建交互式HTML
        html_path = self.visualizer.create_interactive_html(mindmap_root, node_map, questions)
        if html_path:
            exploration_result["visualization"]["interactive_html"] = html_path
        
        print(f"✅ 思维导图探索完成: {len(questions)}组问题, {len(network_analysis.get('key_nodes', {}))}个关键节点")
        return exploration_result
    
    def explore_knowledge_nodes(self, 
                               knowledge_nodes: List[KnowledgeNode],
                               current_mastery: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        探索知识节点
        
        Args:
            knowledge_nodes: 知识节点列表
            current_mastery: 当前已掌握的节点ID列表
            
        Returns:
            探索结果
        """
        print(f"🔍 探索知识节点 ({len(knowledge_nodes)}个)")
        
        exploration_result = {
            "knowledge_nodes_count": len(knowledge_nodes),
            "explored_at": datetime.now().isoformat(),
            "questions": [],
            "network_analysis": {},
            "learning_path": {},
            "knowledge_gaps": {}
        }
        
        # 生成深度问题链（为前3个重要节点）
        important_nodes = sorted(
            knowledge_nodes, 
            key=lambda x: x.confidence * x.mastery_score, 
            reverse=True
        )[:3]
        
        for node in important_nodes:
            question_chain = self.question_engine.generate_deep_questions_chain(node, chain_length=4)
            exploration_result["questions"].extend(question_chain)
        
        # 构建知识网络
        knowledge_network = self.network_builder.build_from_knowledge_nodes(knowledge_nodes)
        
        # 分析网络
        network_analysis = self.network_builder.analyze_network(knowledge_network)
        exploration_result["network_analysis"] = network_analysis
        
        # 可视化知识网络
        network_viz_path = self.visualizer.visualize_knowledge_network(knowledge_nodes)
        if network_viz_path:
            exploration_result["visualization"] = network_viz_path
        
        # 识别知识缺口
        if current_mastery:
            knowledge_gaps = self.network_builder.identify_knowledge_gaps(
                knowledge_network, current_mastery
            )
            exploration_result["knowledge_gaps"] = knowledge_gaps
        
        # 生成学习路径
        if current_mastery is None:
            current_mastery = []
        
        learning_path = self.path_generator.generate_for_goal(
            LearningGoal(
                id="exploration_goal",
                description=f"学习{len(knowledge_nodes)}个知识节点",
                target_knowledge_count=len(knowledge_nodes)
            ),
            knowledge_network,
            current_mastery
        )
        
        exploration_result["learning_path"] = learning_path
        
        print(f"✅ 知识节点探索完成: {len(exploration_result['questions'])}个问题, {len(learning_path.get('stages', []))}个学习阶段")
        return exploration_result
    
    def generate_personalized_learning_path(self,
                                          user_profile: Dict[str, Any],
                                          knowledge_nodes: List[KnowledgeNode]) -> Dict[str, Any]:
        """
        生成个性化学习路径
        
        Args:
            user_profile: 用户画像
            knowledge_nodes: 知识节点列表
            
        Returns:
            个性化学习路径
        """
        print(f"👤 生成个性化学习路径")
        
        # 构建知识网络
        knowledge_network = self.network_builder.build_from_knowledge_nodes(knowledge_nodes)
        
        # 生成个性化路径
        personalized_path = self.path_generator.generate_personalized_path(
            user_profile, knowledge_network
        )
        
        return personalized_path
    
    def adjust_learning_path(self,
                           current_path: Dict[str, Any],
                           progress_data: Dict[str, Any],
                           knowledge_nodes: List[KnowledgeNode]) -> Dict[str, Any]:
        """
        调整学习路径
        
        Args:
            current_path: 当前学习路径
            progress_data: 进度数据
            knowledge_nodes: 知识节点列表
            
        Returns:
            调整后的学习路径
        """
        print(f"🔄 调整学习路径")
        
        # 构建知识网络
        knowledge_network = self.network_builder.build_from_knowledge_nodes(knowledge_nodes)
        
        # 调整路径
        adjusted_path = self.path_generator.adjust_path_based_on_progress(
            current_path, progress_data, knowledge_network
        )
        
        return adjusted_path

# ========== 测试代码 ==========

if __name__ == "__main__":
    print("🧪 测试探索模块（深度知识探索与网络构建）...")
    print("=" * 70)
    
    # 初始化管理器
    manager = ExplorerManager()
    
    # 测试智能提问引擎
    print("\n🤔 测试智能提问引擎:")
    print("-" * 50)
    
    # 创建测试节点
    test_node = MindMapNode(
        id="test_concept_node",
        title="机器学习",
        description="使计算机能够从数据中学习并做出决策的技术",
        node_type="concept",
        importance=0.8,
        difficulty=0.6
    )
    
    # 生成问题
    questions = manager.question_engine.generate_questions_for_node(
        test_node, "understanding", 3
    )
    
    print(f"✅ 为'{test_node.title}'生成了 {len(questions)} 个问题:")
    for i, q in enumerate(questions):
        print(f"  {i+1}. {q['text']}")
        print(f"     难度: {q['difficulty_description']}, 预估时间: {q['estimated_thinking_time']}秒")
    
    # 测试深度问题链
    print("\n🔗 测试深度问题链:")
    print("-" * 50)
    
    chain = manager.question_engine.generate_deep_questions_chain(test_node, chain_length=4)
    
    print(f"✅ 生成了 {len(chain)} 个问题的深度问题链:")
    for i, q in enumerate(chain):
        print(f"  {i+1}. [{q['depth_name']}] {q['text']}")
    
    # 测试思维导图可视化
    print("\n🎨 测试思维导图可视化:")
    print("-" * 50)
    
    # 创建测试思维导图
    root_node = MindMapNode(
        id="test_root",
        title="人工智能学习",
        description="人工智能相关知识的思维导图",
        depth=0,
        node_type="concept"
    )
    
    # 创建子节点
    node_map = {root_node.id: root_node}
    
    sub_topics = ["机器学习", "深度学习", "自然语言处理", "计算机视觉"]
    for i, topic in enumerate(sub_topics):
        child = MindMapNode(
            id=f"test_child_{i}",
            title=topic,
            description=f"{topic}相关知识",
            depth=1,
            parent_id=root_node.id,
            node_type="concept",
            importance=0.7,
            difficulty=0.5
        )
        node_map[child.id] = child
        root_node.children_ids.append(child.id)
        
        # 为每个子主题创建孙节点
        for j in range(2):
            grandchild = MindMapNode(
                id=f"test_grandchild_{i}_{j}",
                title=f"{topic}子主题{j+1}",
                description=f"{topic}的详细知识点",
                depth=2,
                parent_id=child.id,
                node_type="concept",
                importance=0.5,
                difficulty=0.4
            )
            node_map[grandchild.id] = grandchild
            child.children_ids.append(grandchild.id)
    
    # 可视化思维导图
    viz_path = manager.visualizer.visualize_mindmap(root_node, node_map, "png", "balanced")
    
    if viz_path:
        print(f"✅ 思维导图可视化完成: {viz_path}")
    else:
        print("❌ 思维导图可视化失败")
    
    # 测试知识网络构建
    print("\n🔗 测试知识网络构建:")
    print("-" * 50)
    
    # 创建测试知识节点
    test_knowledge_nodes = []
    for i in range(8):
        node = KnowledgeNode(
            id=f"knowledge_node_{i}",
            title=f"知识概念{i+1}",
            content=f"这是知识概念{i+1}的详细内容",
            knowledge_type=KnowledgeType.CONCEPT if i % 3 == 0 else KnowledgeType.FACT,
            learning_level=LearningLevel.UNDERSTANDING,
            confidence=0.6 + i * 0.05,
            mastery_score=0.3 + i * 0.1
        )
        test_knowledge_nodes.append(node)
    
    # 构建知识网络
    network = manager.network_builder.build_from_knowledge_nodes(test_knowledge_nodes)
    
    print(f"✅ 知识网络构建完成: {network.number_of_nodes()}个节点, {network.number_of_edges()}条边")
    
    # 分析网络
    analysis = manager.network_builder.analyze_network(network)
    
    print(f"📊 网络分析:")
    print(f"  节点数量: {analysis['basic_stats'].get('node_count', 0)}")
    print(f"  边数量: {analysis['basic_stats'].get('edge_count', 0)}")
    print(f"  网络密度: {analysis['basic_stats'].get('density', 0):.3f}")
    
    if analysis['key_nodes'].get('hubs'):
        print(f"  枢纽节点: {len(analysis['key_nodes']['hubs'])}个")
    
    # 测试学习路径生成
    print("\n[P] Testing learning path generation:")
    print("-" * 50)
    
    test_goal = LearningGoal(
        id="test_goal",
        description="学习人工智能基础知识",
        target_knowledge_count=len(test_knowledge_nodes),
        scale=GoalScale.SMALL
    )
    
    learning_path = manager.path_generator.generate_for_goal(
        test_goal, network, []
    )
    
    print(f"✅ 学习路径生成完成:")
    print(f"  策略: {learning_path.get('strategy', '未知')}")
    print(f"  阶段数: {len(learning_path.get('stages', []))}")
    print(f"  总节点数: {learning_path.get('total_nodes', 0)}")
    print(f"  预估时间: {learning_path.get('estimated_time_hours', 0):.1f}小时")
    
    if learning_path.get('stages'):
        print(f"  阶段详情:")
        for i, stage in enumerate(learning_path['stages'][:2]):  # 只显示前两个阶段
            print(f"    阶段{i+1}: {stage['name']} ({stage['node_count']}个节点)")
    
    # 测试个性化学习路径
    print("\n👤 测试个性化学习路径:")
    print("-" * 50)
    
    user_profile = {
        "user_id": "test_user",
        "learning_style": "visual",
        "available_time_hours_per_week": 8,
        "experience_level": "beginner",
        "prior_knowledge": []
    }
    
    personalized_path = manager.generate_personalized_learning_path(
        user_profile, test_knowledge_nodes
    )
    
    print(f"✅ 个性化学习路径生成完成:")
    print(f"  学习风格: {personalized_path.get('learning_style', '未知')}")
    print(f"  策略: {personalized_path.get('strategy', '未知')}")
    print(f"  预估周数: {personalized_path.get('estimated_weeks', 0)}周")
    
    # 测试探索管理器
    print("\n🔍 测试探索管理器:")
    print("-" * 50)
    
    exploration_result = manager.explore_mindmap(root_node, node_map, "understanding")
    
    print(f"✅ 思维导图探索完成:")
    print(f"  问题数量: {len(exploration_result.get('questions', {}))}组")
    print(f"  可视化文件: {len(exploration_result.get('visualization', {}))}个")
    print(f"  网络分析: {len(exploration_result.get('network_analysis', {}))}项指标")
    
    print("\n✅ 探索模块测试完成")
    print("=" * 70)