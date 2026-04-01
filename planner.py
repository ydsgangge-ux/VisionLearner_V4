import json
import re
import math
import random
import heapq
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import statistics

# 导入前几段的基础模块
from foundation import (
    MindMapNode, KnowledgeNode, LearningGoal, LearningLevel, 
    KnowledgeType, GoalScale, LearningStrategy, MindMapStyle,
    ProgressGranularity, generate_id, FoundationManager,
    TimeEstimationModel
)

# explorer依赖在运行时按需导入，避免循环依赖

# ========== 层次化学习分配器 ==========

class HierarchicalLearningAllocator:
    """
    层次化学习分配器 - 基于思维导图进行层次化学习任务分配
    将学习目标分解为不同层次的学习任务
    """
    
    def __init__(self, time_model: Optional[TimeEstimationModel] = None):
        self.time_model = time_model or TimeEstimationModel()
        
        # 层次分配策略
        self.allocation_strategies = {
            "depth_first": {
                "name": "深度优先",
                "description": "按深度优先顺序分配学习任务",
                "priority": ["importance", "depth", "difficulty"]
            },
            "breadth_first": {
                "name": "广度优先", 
                "description": "按广度优先顺序分配学习任务",
                "priority": ["depth", "importance", "difficulty"]
            },
            "importance_first": {
                "name": "重要性优先",
                "description": "按重要性优先分配学习任务",
                "priority": ["importance", "difficulty", "depth"]
            },
            "balanced": {
                "name": "均衡分配",
                "description": "均衡考虑多个因素分配学习任务",
                "priority": ["composite_score", "depth", "prerequisites"]
            },
            "adaptive": {
                "name": "自适应分配",
                "description": "根据学习情况动态调整分配策略",
                "priority": ["dynamic_adjustment", "learning_history", "progress"]
            }
        }
        
        # 学习层次配置
        self.learning_levels_config = {
            "exploration": {
                "name": "探索层",
                "depth_range": [0, 1],
                "focus": ["整体认知", "核心概念", "建立框架"],
                "allocation_ratio": 0.2
            },
            "foundation": {
                "name": "基础层", 
                "depth_range": [2, 3],
                "focus": ["基础知识", "关键技能", "建立基础"],
                "allocation_ratio": 0.3
            },
            "deepening": {
                "name": "深化层",
                "depth_range": [4, 5],
                "focus": ["深度理解", "复杂应用", "建立联系"],
                "allocation_ratio": 0.3
            },
            "integration": {
                "name": "整合层",
                "depth_range": [6, float('inf')],
                "focus": ["系统整合", "创新应用", "建立体系"],
                "allocation_ratio": 0.2
            }
        }
        
        # 分配历史记录
        self.allocation_history = defaultdict(list)
    
    def allocate_by_mindmap(self,
                          goal: LearningGoal,
                          mindmap_root: MindMapNode,
                          node_map: Dict[str, MindMapNode],
                          strategy: str = "balanced",
                          available_time_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        基于思维导图进行层次化学习分配
        
        Args:
            goal: 学习目标
            mindmap_root: 思维导图根节点
            node_map: 节点映射
            strategy: 分配策略
            available_time_minutes: 可用学习时间（分钟）
            
        Returns:
            分配计划
        """
        print(f"📊 基于思维导图进行层次化学习分配 (策略: {strategy})")
        
        allocation_plan = {
            "goal_id": goal.id,
            "mindmap_id": mindmap_root.id,
            "strategy": strategy,
            "allocated_at": datetime.now().isoformat(),
            "total_nodes": len(node_map),
            "hierarchical_breakdown": {},
            "learning_sequences": [],
            "time_allocation": {},
            "recommendations": []
        }
        
        # 验证策略
        if strategy not in self.allocation_strategies:
            print(f"[!] Unknown strategy {strategy}, using balanced strategy")
            strategy = "balanced"
        
        # 分析思维导图结构
        structure_analysis = self._analyze_mindmap_structure(node_map)
        allocation_plan["structure_analysis"] = structure_analysis
        
        # 层次化分解
        hierarchical_nodes = self._hierarchical_decomposition(node_map)
        allocation_plan["hierarchical_breakdown"] = hierarchical_nodes
        
        # 计算节点优先级
        prioritized_nodes = self._calculate_node_priorities(
            node_map, strategy, goal
        )
        
        # 生成学习序列
        learning_sequences = self._generate_learning_sequences(
            prioritized_nodes, node_map, strategy, available_time_minutes
        )
        allocation_plan["learning_sequences"] = learning_sequences
        
        # 分配学习时间
        time_allocation = self._allocate_learning_time(
            learning_sequences, node_map, available_time_minutes
        )
        allocation_plan["time_allocation"] = time_allocation
        
        # 生成分配建议
        recommendations = self._generate_allocation_recommendations(
            allocation_plan, goal, node_map
        )
        allocation_plan["recommendations"] = recommendations
        
        # 记录分配历史
        self._record_allocation_history(goal.id, allocation_plan)
        
        print(f"✅ 层次化分配完成: {len(learning_sequences)}个学习序列，{time_allocation.get('total_minutes', 0)}分钟")
        return allocation_plan
    
    def allocate_by_knowledge_network(self,
                                    goal: LearningGoal,
                                    knowledge_network: Any,  # NetworkX图
                                    current_mastery: List[str] = None,
                                    strategy: str = "balanced") -> Dict[str, Any]:
        """
        基于知识网络进行学习分配
        
        Args:
            goal: 学习目标
            knowledge_network: 知识网络
            current_mastery: 当前已掌握节点
            strategy: 分配策略
            
        Returns:
            分配计划
        """
        print(f"🔗 基于知识网络进行学习分配 (策略: {strategy})")
        
        allocation_plan = {
            "goal_id": goal.id,
            "strategy": strategy,
            "allocated_at": datetime.now().isoformat(),
            "network_analysis": {},
            "learning_paths": [],
            "priority_nodes": [],
            "recommendations": []
        }
        
        try:
            import networkx as nx
            
            # 网络分析
            network_stats = {
                "node_count": knowledge_network.number_of_nodes(),
                "edge_count": knowledge_network.number_of_edges(),
                "density": nx.density(knowledge_network) if isinstance(knowledge_network, nx.Graph) else 0,
                "connected_components": nx.number_weakly_connected_components(knowledge_network) 
                if isinstance(knowledge_network, nx.DiGraph) 
                else nx.number_connected_components(knowledge_network)
            }
            allocation_plan["network_analysis"] = network_stats
            
            # 确定未掌握节点
            all_nodes = list(knowledge_network.nodes())
            if current_mastery is None:
                current_mastery = []
            
            nodes_to_learn = [node for node in all_nodes if node not in current_mastery]
            
            if not nodes_to_learn:
                allocation_plan["message"] = "所有节点已掌握"
                return allocation_plan
            
            # 根据策略计算节点优先级
            if strategy == "prerequisite_based":
                # 基于先决条件的拓扑排序
                try:
                    sorted_nodes = list(nx.topological_sort(knowledge_network))
                    # 过滤已掌握节点和排序
                    learning_order = [node for node in sorted_nodes if node in nodes_to_learn]
                except:
                    # 如果有环，使用简单排序
                    learning_order = nodes_to_learn
            elif strategy == "centrality_based":
                # 基于中心性排序
                try:
                    centrality = nx.degree_centrality(knowledge_network)
                    nodes_to_learn.sort(key=lambda x: centrality.get(x, 0), reverse=True)
                    learning_order = nodes_to_learn
                except:
                    learning_order = nodes_to_learn
            else:  # balanced
                # 均衡考虑多个因素
                learning_order = self._balance_multiple_factors(
                    nodes_to_learn, knowledge_network
                )
            
            # 创建学习路径
            learning_paths = self._create_network_learning_paths(
                learning_order, knowledge_network, goal
            )
            allocation_plan["learning_paths"] = learning_paths
            
            # 识别关键节点
            key_nodes = self._identify_key_network_nodes(
                knowledge_network, nodes_to_learn
            )
            allocation_plan["priority_nodes"] = key_nodes
            
            # 生成建议
            recommendations = self._generate_network_allocation_recommendations(
                allocation_plan, goal, knowledge_network
            )
            allocation_plan["recommendations"] = recommendations
            
        except Exception as e:
            print(f"❌ 知识网络分配失败: {str(e)}")
            allocation_plan["error"] = str(e)
        
        return allocation_plan
    
    def adjust_allocation(self,
                         original_plan: Dict[str, Any],
                         progress_data: Dict[str, Any],
                         node_map: Dict[str, MindMapNode]) -> Dict[str, Any]:
        """
        基于学习进度调整分配计划
        
        Args:
            original_plan: 原始分配计划
            progress_data: 进度数据
            node_map: 节点映射
            
        Returns:
            调整后的分配计划
        """
        print(f"🔄 基于进度调整分配计划")
        
        # 创建调整后的计划副本
        adjusted_plan = original_plan.copy()
        adjusted_plan["adjusted_at"] = datetime.now().isoformat()
        adjusted_plan["adjustment_reasons"] = []
        
        # 提取进度信息
        mastered_nodes = progress_data.get("mastered_nodes", [])
        struggling_nodes = progress_data.get("struggling_nodes", [])
        learning_speed = progress_data.get("learning_speed", 1.0)
        engagement_level = progress_data.get("engagement_level", 0.5)
        
        # 调整学习序列
        if "learning_sequences" in adjusted_plan:
            original_sequences = adjusted_plan["learning_sequences"]
            adjusted_sequences = []
            
            for seq in original_sequences:
                # 过滤已掌握的节点
                original_nodes = seq.get("node_ids", [])
                remaining_nodes = [n for n in original_nodes if n not in mastered_nodes]
                
                if not remaining_nodes:
                    # 如果序列中所有节点都已掌握，跳过该序列
                    adjusted_plan["adjustment_reasons"].append(
                        f"跳过序列 '{seq.get('name', '未知')}'，所有节点已掌握"
                    )
                    continue
                
                # 如果有学习困难的节点，添加额外支持
                struggling_in_seq = [n for n in struggling_nodes if n in original_nodes]
                if struggling_in_seq and len(remaining_nodes) > 0:
                    # 为困难节点添加标记
                    seq["has_struggling_nodes"] = True
                    seq["struggling_node_count"] = len(struggling_in_seq)
                    seq["recommended_support"] = "额外练习和复习"
                    
                    adjusted_plan["adjustment_reasons"].append(
                        f"序列 '{seq.get('name', '未知')}' 包含 {len(struggling_in_seq)} 个困难节点"
                    )
                
                # 更新序列
                seq["node_ids"] = remaining_nodes
                seq["node_count"] = len(remaining_nodes)
                adjusted_sequences.append(seq)
            
            adjusted_plan["learning_sequences"] = adjusted_sequences
            adjusted_plan["remaining_nodes"] = sum(seq["node_count"] for seq in adjusted_sequences)
        
        # 调整时间分配
        if "time_allocation" in adjusted_plan:
            original_time = adjusted_plan["time_allocation"]
            
            # 根据学习速度调整时间
            if learning_speed != 1.0:
                for key in ["estimated_minutes", "daily_minutes", "weekly_minutes"]:
                    if key in original_time:
                        original_time[key] = int(original_time[key] / learning_speed)
                
                if learning_speed < 0.8:
                    adjusted_plan["adjustment_reasons"].append(
                        f"学习速度较慢 ({learning_speed:.2f}x)，增加时间分配"
                    )
                elif learning_speed > 1.2:
                    adjusted_plan["adjustment_reasons"].append(
                        f"学习速度较快 ({learning_speed:.2f}x)，减少时间分配"
                    )
            
            # 根据参与度调整
            if engagement_level < 0.3:
                # 低参与度，减少每日学习量
                if "daily_minutes" in original_time:
                    original_time["daily_minutes"] = int(original_time["daily_minutes"] * 0.7)
                    adjusted_plan["adjustment_reasons"].append(
                        "检测到低参与度，减少每日学习量"
                    )
            
            adjusted_plan["time_allocation"] = original_time
        
        # 更新分配策略（如果需要）
        if len(struggling_nodes) > len(mastered_nodes) * 0.3:  # 超过30%的节点有困难
            adjusted_plan["strategy"] = "adaptive"
            adjusted_plan["adjustment_reasons"].append(
                "大量节点学习困难，切换到自适应策略"
            )
        
        # 生成新的建议
        adjusted_plan["recommendations"] = self._generate_adjustment_recommendations(
            adjusted_plan, progress_data, node_map
        )
        
        print(f"✅ 分配计划调整完成: {len(adjusted_plan.get('adjustment_reasons', []))}项调整")
        return adjusted_plan
    
    def _analyze_mindmap_structure(self, node_map: Dict[str, MindMapNode]) -> Dict[str, Any]:
        """分析思维导图结构"""
        analysis = {
            "total_nodes": len(node_map),
            "depth_distribution": defaultdict(int),
            "type_distribution": defaultdict(int),
            "importance_stats": {},
            "difficulty_stats": {},
            "connection_stats": {}
        }
        
        # 收集节点信息
        depths = []
        importances = []
        difficulties = []
        child_counts = []
        
        for node in node_map.values():
            # 深度分布
            analysis["depth_distribution"][node.depth] += 1
            depths.append(node.depth)
            
            # 类型分布
            analysis["type_distribution"][node.node_type] += 1
            
            # 重要性
            importances.append(node.importance)
            
            # 难度
            difficulties.append(node.difficulty)
            
            # 子节点数量
            child_counts.append(len(node.children_ids))
        
        # 计算统计信息
        if importances:
            analysis["importance_stats"] = {
                "mean": statistics.mean(importances),
                "median": statistics.median(importances),
                "min": min(importances),
                "max": max(importances)
            }
        
        if difficulties:
            analysis["difficulty_stats"] = {
                "mean": statistics.mean(difficulties),
                "median": statistics.median(difficulties),
                "min": min(difficulties),
                "max": max(difficulties)
            }
        
        if child_counts:
            analysis["connection_stats"] = {
                "total_children": sum(child_counts),
                "avg_children": statistics.mean(child_counts),
                "max_children": max(child_counts),
                "leaf_nodes": sum(1 for c in child_counts if c == 0)
            }
        
        # 最大深度
        if depths:
            analysis["max_depth"] = max(depths)
            analysis["avg_depth"] = statistics.mean(depths)
        
        return analysis
    
    def _hierarchical_decomposition(self, node_map: Dict[str, MindMapNode]) -> Dict[str, List[str]]:
        """层次化分解节点"""
        hierarchical_nodes = defaultdict(list)
        
        for node_id, node in node_map.items():
            # 根据深度确定层次
            if node.depth <= 1:
                level = "exploration"
            elif node.depth <= 3:
                level = "foundation"
            elif node.depth <= 5:
                level = "deepening"
            else:
                level = "integration"
            
            hierarchical_nodes[level].append(node_id)
        
        return dict(hierarchical_nodes)
    
    def _calculate_node_priorities(self, 
                                 node_map: Dict[str, MindMapNode],
                                 strategy: str,
                                 goal: LearningGoal) -> List[Tuple[str, float]]:
        """计算节点优先级"""
        priorities = []
        
        strategy_config = self.allocation_strategies.get(strategy, self.allocation_strategies["balanced"])
        priority_factors = strategy_config.get("priority", [])
        
        for node_id, node in node_map.items():
            # 计算每个因素的分数
            factor_scores = {}
            
            # 重要性分数
            factor_scores["importance"] = node.importance
            
            # 难度分数（难度越低，优先级越高）
            factor_scores["difficulty"] = 1.0 - node.difficulty
            
            # 深度分数（深度越浅，优先级越高）
            factor_scores["depth"] = 1.0 / (node.depth + 1)
            
            # 先决条件分数（先决条件越少，优先级越高）
            prereq_factor = 1.0
            if node.prerequisites:
                prereq_factor = 1.0 / (len(node.prerequisites) + 1)
            factor_scores["prerequisites"] = prereq_factor
            
            # 预估时间分数（时间越短，优先级越高）
            time_factor = 1.0
            if node.estimated_time_minutes > 0:
                time_factor = 30.0 / node.estimated_time_minutes  # 30分钟为基准
                time_factor = min(max(time_factor, 0.1), 2.0)  # 限制在0.1-2.0之间
            factor_scores["time"] = time_factor
            
            # 学习状态分数（未学习的优先级高）
            status_factor = 1.0
            if node.learning_status == "mastered":
                status_factor = 0.1
            elif node.learning_status == "learning":
                status_factor = 0.5
            elif node.learning_status == "reviewing":
                status_factor = 0.3
            factor_scores["status"] = status_factor
            
            # 计算综合分数
            composite_score = 0.0
            weight_sum = 0.0
            
            # 根据策略分配权重
            weights = {
                "importance": 0.3,
                "difficulty": 0.2,
                "depth": 0.15,
                "prerequisites": 0.15,
                "time": 0.1,
                "status": 0.1
            }
            
            # 调整策略权重
            if strategy == "depth_first":
                weights["depth"] = 0.4
                weights["importance"] = 0.2
            elif strategy == "breadth_first":
                weights["depth"] = 0.5
            elif strategy == "importance_first":
                weights["importance"] = 0.5
                weights["depth"] = 0.1
            
            for factor, weight in weights.items():
                composite_score += factor_scores.get(factor, 0.5) * weight
                weight_sum += weight
            
            if weight_sum > 0:
                composite_score /= weight_sum
            
            # 添加随机因子避免完全相同分数
            composite_score += random.uniform(-0.01, 0.01)
            
            priorities.append((node_id, composite_score))
        
        # 按优先级排序
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        return priorities
    
    def _generate_learning_sequences(self,
                                   prioritized_nodes: List[Tuple[str, float]],
                                   node_map: Dict[str, MindMapNode],
                                   strategy: str,
                                   available_time_minutes: Optional[int]) -> List[Dict[str, Any]]:
        """生成学习序列"""
        sequences = []
        
        # 确定序列数量（基于节点总数）
        total_nodes = len(prioritized_nodes)
        if total_nodes <= 5:
            sequence_count = 1
        elif total_nodes <= 15:
            sequence_count = 2
        elif total_nodes <= 30:
            sequence_count = 3
        else:
            sequence_count = max(3, total_nodes // 10)
        
        # 确定每个序列的节点数量
        nodes_per_sequence = math.ceil(total_nodes / sequence_count)
        
        # 根据策略调整序列生成
        if strategy == "depth_first":
            # 深度优先：每个序列包含一个分支的节点
            sequences = self._create_depth_first_sequences(prioritized_nodes, node_map)
        elif strategy == "breadth_first":
            # 广度优先：每个序列包含同一深度的节点
            sequences = self._create_breadth_first_sequences(prioritized_nodes, node_map)
        else:
            # 其他策略：按优先级分组
            for i in range(sequence_count):
                start_idx = i * nodes_per_sequence
                end_idx = min((i + 1) * nodes_per_sequence, total_nodes)
                
                sequence_nodes = prioritized_nodes[start_idx:end_idx]
                node_ids = [node_id for node_id, _ in sequence_nodes]
                
                sequence = {
                    "id": generate_id(f"sequence_{i}_"),
                    "name": f"学习序列 {i+1}",
                    "description": f"包含{len(node_ids)}个知识节点的学习序列",
                    "node_ids": node_ids,
                    "node_count": len(node_ids),
                    "avg_priority": statistics.mean([score for _, score in sequence_nodes]) if sequence_nodes else 0,
                    "estimated_time_minutes": sum(node_map[nid].estimated_time_minutes for nid in node_ids if nid in node_map)
                }
                
                sequences.append(sequence)
        
        # 如果提供了可用时间，调整序列
        if available_time_minutes:
            sequences = self._adjust_sequences_for_time(sequences, available_time_minutes)
        
        return sequences
    
    def _create_depth_first_sequences(self, 
                                    prioritized_nodes: List[Tuple[str, float]],
                                    node_map: Dict[str, MindMapNode]) -> List[Dict[str, Any]]:
        """创建深度优先学习序列"""
        sequences = []
        
        # 按深度分组
        depth_groups = defaultdict(list)
        for node_id, _ in prioritized_nodes:
            node = node_map.get(node_id)
            if node:
                depth_groups[node.depth].append(node_id)
        
        # 创建序列：每个序列专注于一个深度范围
        current_sequence = []
        current_depth = 0
        
        for depth in sorted(depth_groups.keys()):
            # 如果深度跳跃太大，开始新序列
            if depth - current_depth > 1 and current_sequence:
                sequence = {
                    "id": generate_id("depth_seq_"),
                    "name": f"深度{current_depth}学习序列",
                    "description": f"专注于深度{current_depth}的知识节点",
                    "node_ids": current_sequence,
                    "node_count": len(current_sequence),
                    "depth_range": [current_depth, current_depth],
                    "strategy": "depth_first"
                }
                sequences.append(sequence)
                current_sequence = []
            
            # 添加当前深度的节点
            current_sequence.extend(depth_groups[depth])
            current_depth = depth
        
        # 添加最后一个序列
        if current_sequence:
            sequence = {
                "id": generate_id("depth_seq_"),
                "name": f"深度{current_depth}学习序列",
                "description": f"专注于深度{current_depth}的知识节点",
                "node_ids": current_sequence,
                "node_count": len(current_sequence),
                "depth_range": [current_depth, current_depth],
                "strategy": "depth_first"
            }
            sequences.append(sequence)
        
        return sequences
    
    def _create_breadth_first_sequences(self,
                                       prioritized_nodes: List[Tuple[str, float]],
                                       node_map: Dict[str, MindMapNode]) -> List[Dict[str, Any]]:
        """创建广度优先学习序列"""
        sequences = []
        
        # 收集根节点和主要分支
        root_nodes = []
        for node_id, _ in prioritized_nodes:
            node = node_map.get(node_id)
            if node and node.depth == 0:
                root_nodes.append(node_id)
        
        # 如果没有明确的根节点，使用所有节点
        if not root_nodes:
            # 按优先级分组
            return self._generate_learning_sequences(prioritized_nodes, node_map, "balanced", None)
        
        # 为每个根节点创建序列
        for i, root_id in enumerate(root_nodes):
            # 收集该根节点的所有后代节点
            descendant_nodes = self._get_descendant_nodes(root_id, node_map)
            
            if descendant_nodes:
                sequence = {
                    "id": generate_id(f"breadth_seq_{i}_"),
                    "name": f"分支学习序列 {i+1}",
                    "description": f"学习以'{node_map[root_id].title}'为核心的知识分支",
                    "node_ids": descendant_nodes,
                    "node_count": len(descendant_nodes),
                    "root_node": root_id,
                    "strategy": "breadth_first"
                }
                sequences.append(sequence)
        
        # 如果序列太少，添加剩余节点
        if len(sequences) < 2 and prioritized_nodes:
            all_node_ids = [node_id for node_id, _ in prioritized_nodes]
            used_nodes = set()
            for seq in sequences:
                used_nodes.update(seq["node_ids"])
            
            remaining_nodes = [nid for nid in all_node_ids if nid not in used_nodes]
            if remaining_nodes:
                sequence = {
                    "id": generate_id("breadth_seq_remaining_"),
                    "name": "补充学习序列",
                    "description": "学习剩余的知识节点",
                    "node_ids": remaining_nodes,
                    "node_count": len(remaining_nodes),
                    "strategy": "breadth_first"
                }
                sequences.append(sequence)
        
        return sequences
    
    def _get_descendant_nodes(self, root_id: str, node_map: Dict[str, MindMapNode]) -> List[str]:
        """获取某个节点的所有后代节点"""
        descendants = []
        
        def collect_descendants(node_id: str):
            node = node_map.get(node_id)
            if not node:
                return
            
            for child_id in node.children_ids:
                if child_id not in descendants:
                    descendants.append(child_id)
                    collect_descendants(child_id)
        
        collect_descendants(root_id)
        return descendants
    
    def _adjust_sequences_for_time(self,
                                 sequences: List[Dict[str, Any]],
                                 available_time_minutes: int) -> List[Dict[str, Any]]:
        """根据可用时间调整学习序列"""
        if not sequences:
            return sequences
        
        # 计算总预估时间
        total_estimated = sum(seq.get("estimated_time_minutes", 0) for seq in sequences)
        
        if total_estimated <= available_time_minutes:
            # 时间充足，不需要调整
            return sequences
        
        # 时间不足，需要调整
        print(f"[!] Time shortage: estimated {total_estimated} mins, available {available_time_minutes} mins")
        
        # 计算调整比例
        adjustment_ratio = available_time_minutes / total_estimated
        
        adjusted_sequences = []
        for seq in sequences:
            original_nodes = seq.get("node_ids", [])
            original_time = seq.get("estimated_time_minutes", 0)
            
            # 调整节点数量
            if original_nodes:
                # 保留高优先级节点（假设节点按优先级排序）
                keep_count = max(1, int(len(original_nodes) * adjustment_ratio))
                adjusted_nodes = original_nodes[:keep_count]
                
                adjusted_seq = seq.copy()
                adjusted_seq["node_ids"] = adjusted_nodes
                adjusted_seq["node_count"] = len(adjusted_nodes)
                adjusted_seq["estimated_time_minutes"] = int(original_time * adjustment_ratio)
                adjusted_seq["time_adjusted"] = True
                adjusted_seq["original_node_count"] = len(original_nodes)
                
                adjusted_sequences.append(adjusted_seq)
        
        return adjusted_sequences
    
    def _allocate_learning_time(self,
                              sequences: List[Dict[str, Any]],
                              node_map: Dict[str, MindMapNode],
                              available_time_minutes: Optional[int]) -> Dict[str, Any]:
        """分配学习时间"""
        time_allocation = {
            "total_sequences": len(sequences),
            "sequence_allocation": []
        }
        
        # 计算总预估时间
        total_estimated = sum(seq.get("estimated_time_minutes", 0) for seq in sequences)
        time_allocation["total_estimated_minutes"] = total_estimated
        
        # 如果提供了可用时间，计算比例分配
        if available_time_minutes:
            time_allocation["available_minutes"] = available_time_minutes
            
            if total_estimated > 0:
                # 计算每个序列的时间分配比例
                for seq in sequences:
                    seq_time = seq.get("estimated_time_minutes", 0)
                    if total_estimated > 0:
                        time_ratio = seq_time / total_estimated
                        allocated_time = int(available_time_minutes * time_ratio)
                    else:
                        allocated_time = 0
                    
                    seq_allocation = {
                        "sequence_id": seq.get("id", ""),
                        "sequence_name": seq.get("name", ""),
                        "estimated_minutes": seq_time,
                        "allocated_minutes": allocated_time,
                        "time_ratio": time_ratio if total_estimated > 0 else 0
                    }
                    time_allocation["sequence_allocation"].append(seq_allocation)
                
                time_allocation["total_allocated_minutes"] = sum(
                    alloc["allocated_minutes"] for alloc in time_allocation["sequence_allocation"]
                )
            
            # 建议每日学习时间
            if available_time_minutes > 0:
                # 假设学习周期为2周（14天）
                daily_minutes = int(available_time_minutes / 14)
                weekly_minutes = daily_minutes * 7
                
                time_allocation["daily_recommendation"] = {
                    "minutes": daily_minutes,
                    "description": f"建议每日学习{daily_minutes}分钟"
                }
                time_allocation["weekly_recommendation"] = {
                    "minutes": weekly_minutes,
                    "description": f"建议每周学习{weekly_minutes}分钟"
                }
        
        # 如果没有提供可用时间，使用预估时间
        else:
            time_allocation["using_estimated_times"] = True
            time_allocation["recommendation"] = "使用节点预估时间进行分配"
            
            for seq in sequences:
                seq_time = seq.get("estimated_time_minutes", 0)
                seq_allocation = {
                    "sequence_id": seq.get("id", ""),
                    "sequence_name": seq.get("name", ""),
                    "allocated_minutes": seq_time,
                    "note": "使用节点预估时间"
                }
                time_allocation["sequence_allocation"].append(seq_allocation)
        
        return time_allocation
    
    def _balance_multiple_factors(self,
                                nodes: List[str],
                                knowledge_network: Any) -> List[str]:
        """均衡考虑多个因素排序节点"""
        try:
            import networkx as nx
            
            # 计算多个中心性指标
            centrality_measures = {}
            
            # 度中心性
            try:
                degree_centrality = nx.degree_centrality(knowledge_network)
                centrality_measures["degree"] = degree_centrality
            except:
                pass
            
            # PageRank
            try:
                pagerank = nx.pagerank(knowledge_network)
                centrality_measures["pagerank"] = pagerank
            except:
                pass
            
            # 综合分数
            composite_scores = {}
            for node in nodes:
                scores = []
                for measure_name, measure_dict in centrality_measures.items():
                    if node in measure_dict:
                        scores.append(measure_dict[node])
                
                if scores:
                    # 使用平均分
                    composite_scores[node] = statistics.mean(scores)
                else:
                    # 如果没有中心性分数，使用随机分数
                    composite_scores[node] = random.random()
            
            # 按综合分数排序
            sorted_nodes = sorted(nodes, key=lambda x: composite_scores.get(x, 0), reverse=True)
            return sorted_nodes
            
        except Exception as e:
            print(f"❌ 多因素平衡失败: {str(e)}")
            return nodes  # 返回原始顺序
    
    def _create_network_learning_paths(self,
                                      learning_order: List[str],
                                      knowledge_network: Any,
                                      goal: LearningGoal) -> List[Dict[str, Any]]:
        """创建网络学习路径"""
        paths = []
        
        # 将学习顺序分组为路径
        if len(learning_order) <= 10:
            # 节点少，一个路径
            paths.append({
                "name": "主要学习路径",
                "node_ids": learning_order,
                "node_count": len(learning_order),
                "description": "完整的学习路径"
            })
        else:
            # 节点多，分成多个路径
            path_count = min(4, max(2, len(learning_order) // 5))
            nodes_per_path = math.ceil(len(learning_order) / path_count)
            
            for i in range(path_count):
                start_idx = i * nodes_per_path
                end_idx = min((i + 1) * nodes_per_path, len(learning_order))
                
                path_nodes = learning_order[start_idx:end_idx]
                
                paths.append({
                    "name": f"学习路径 {i+1}",
                    "node_ids": path_nodes,
                    "node_count": len(path_nodes),
                    "description": f"学习路径第{i+1}部分"
                })
        
        return paths
    
    def _identify_key_network_nodes(self,
                                   knowledge_network: Any,
                                   nodes_to_learn: List[str]) -> List[Dict[str, Any]]:
        """识别关键网络节点"""
        key_nodes = []
        
        try:
            import networkx as nx
            
            # 计算介数中心性（识别桥接节点）
            try:
                betweenness = nx.betweenness_centrality(knowledge_network)
                for node in nodes_to_learn:
                    if node in betweenness and betweenness[node] > 0.1:  # 阈值
                        key_nodes.append({
                            "node_id": node,
                            "importance": "high",
                            "reason": "桥接节点（高介数中心性）",
                            "centrality": betweenness[node]
                        })
            except:
                pass
            
            # 如果没有找到桥接节点，使用度中心性
            if not key_nodes:
                try:
                    degree_centrality = nx.degree_centrality(knowledge_network)
                    top_nodes = sorted(
                        nodes_to_learn,
                        key=lambda x: degree_centrality.get(x, 0),
                        reverse=True
                    )[:3]  # 取前3个
                    
                    for node in top_nodes:
                        key_nodes.append({
                            "node_id": node,
                            "importance": "high",
                            "reason": "高连接度节点",
                            "centrality": degree_centrality.get(node, 0)
                        })
                except:
                    pass
            
        except Exception as e:
            print(f"❌ 关键节点识别失败: {str(e)}")
        
        # 如果没有找到关键节点，选择前几个节点
        if not key_nodes and nodes_to_learn:
            for i, node in enumerate(nodes_to_learn[:3]):
                key_nodes.append({
                    "node_id": node,
                    "importance": "medium",
                    "reason": "学习顺序靠前",
                    "order": i + 1
                })
        
        return key_nodes
    
    def _generate_allocation_recommendations(self,
                                           allocation_plan: Dict[str, Any],
                                           goal: LearningGoal,
                                           node_map: Dict[str, MindMapNode]) -> List[str]:
        """生成分配建议"""
        recommendations = []
        
        total_nodes = allocation_plan.get("total_nodes", 0)
        sequence_count = len(allocation_plan.get("learning_sequences", []))
        total_time = allocation_plan.get("time_allocation", {}).get("total_estimated_minutes", 0)
        
        # 基于节点数量的建议
        if total_nodes > 50:
            recommendations.append(f"学习内容较多（{total_nodes}个节点），建议制定长期计划")
        elif total_nodes < 10:
            recommendations.append(f"学习内容较少（{total_nodes}个节点），可以快速完成")
        
        # 基于序列数量的建议
        if sequence_count > 3:
            recommendations.append(f"分为{sequence_count}个学习序列，建议按顺序逐步完成")
        
        # 基于时间的建议
        if total_time > 0:
            total_hours = total_time / 60
            if total_hours > 20:
                recommendations.append(f"预计需要{total_hours:.1f}小时，建议分散在几周内学习")
            elif total_hours > 5:
                recommendations.append(f"预计需要{total_hours:.1f}小时，建议在一周内完成")
            else:
                recommendations.append(f"预计需要{total_hours:.1f}小时，可以在几天内完成")
        
        # 基于策略的建议
        strategy = allocation_plan.get("strategy", "balanced")
        if strategy == "depth_first":
            recommendations.append("使用深度优先策略，适合系统性深入学习")
        elif strategy == "breadth_first":
            recommendations.append("使用广度优先策略，适合建立整体认知框架")
        elif strategy == "importance_first":
            recommendations.append("使用重要性优先策略，适合时间有限的情况")
        
        # 基于目标规模的建议
        if goal.scale in [GoalScale.LARGE, GoalScale.MASSIVE]:
            recommendations.append("大规模学习目标，建议定期复习和进度检查")
        
        return recommendations
    
    def _generate_network_allocation_recommendations(self,
                                                   allocation_plan: Dict[str, Any],
                                                   goal: LearningGoal,
                                                   knowledge_network: Any) -> List[str]:
        """生成网络分配建议"""
        recommendations = []
        
        network_stats = allocation_plan.get("network_analysis", {})
        node_count = network_stats.get("node_count", 0)
        path_count = len(allocation_plan.get("learning_paths", []))
        
        if node_count > 30:
            recommendations.append(f"知识网络包含{node_count}个节点，建议按路径分阶段学习")
        
        if path_count > 1:
            recommendations.append(f"分为{path_count}条学习路径，可以并行或顺序学习")
        
        if allocation_plan.get("priority_nodes"):
            priority_count = len(allocation_plan["priority_nodes"])
            recommendations.append(f"识别出{priority_count}个关键节点，建议优先学习")
        
        return recommendations
    
    def _generate_adjustment_recommendations(self,
                                           adjusted_plan: Dict[str, Any],
                                           progress_data: Dict[str, Any],
                                           node_map: Dict[str, MindMapNode]) -> List[str]:
        """生成调整建议"""
        recommendations = []
        
        # 基于调整原因
        adjustment_reasons = adjusted_plan.get("adjustment_reasons", [])
        for reason in adjustment_reasons:
            if "学习速度较慢" in reason:
                recommendations.append("检测到学习速度较慢，建议增加学习时间或调整学习方法")
            elif "学习速度较快" in reason:
                recommendations.append("检测到学习速度较快，可以适当增加学习内容")
            elif "低参与度" in reason:
                recommendations.append("检测到低参与度，建议调整学习内容或增加互动")
            elif "学习困难" in reason:
                recommendations.append("检测到学习困难，建议增加练习和复习")
        
        # 基于剩余节点
        remaining_nodes = adjusted_plan.get("remaining_nodes", 0)
        if remaining_nodes > 0:
            mastered_nodes = progress_data.get("mastered_nodes", [])
            if mastered_nodes:
                progress_rate = len(mastered_nodes) / (len(mastered_nodes) + remaining_nodes)
                if progress_rate > 0.7:
                    recommendations.append(f"已完成{progress_rate:.0%}，继续保持当前学习节奏")
                elif progress_rate < 0.3:
                    recommendations.append(f"完成度较低({progress_rate:.0%})，建议加强学习")
        
        # 基于学习序列
        sequences = adjusted_plan.get("learning_sequences", [])
        if sequences:
            struggling_sequences = [s for s in sequences if s.get("has_struggling_nodes", False)]
            if struggling_sequences:
                recommendations.append(f"{len(struggling_sequences)}个学习序列包含困难节点，建议重点关注")
        
        return recommendations
    
    def _record_allocation_history(self, goal_id: str, allocation_plan: Dict[str, Any]) -> None:
        """记录分配历史"""
        history_entry = {
            "timestamp": allocation_plan.get("allocated_at", datetime.now().isoformat()),
            "strategy": allocation_plan.get("strategy"),
            "total_nodes": allocation_plan.get("total_nodes", 0),
            "sequence_count": len(allocation_plan.get("learning_sequences", [])),
            "estimated_minutes": allocation_plan.get("time_allocation", {}).get("total_estimated_minutes", 0)
        }
        
        self.allocation_history[goal_id].append(history_entry)
        
        # 限制历史记录长度
        if len(self.allocation_history[goal_id]) > 10:
            self.allocation_history[goal_id] = self.allocation_history[goal_id][-10:]

# ========== 思维导图驱动规划器 ==========

class MindMapDrivenPlanner:
    """
    思维导图驱动规划器 - 基于思维导图结构制定详细学习计划
    """
    
    def __init__(self, time_model: Optional[TimeEstimationModel] = None):
        self.time_model = time_model or TimeEstimationModel()
        self.allocation_history = {}
        
        # 规划模板
        self.planning_templates = {
            "micro_goal": {
                "name": "微目标计划",
                "description": "针对微小学习目标的详细计划",
                "components": ["daily_schedule", "learning_sessions", "review_plan"]
            },
            "small_goal": {
                "name": "小目标计划",
                "description": "针对小型学习目标的周计划",
                "components": ["weekly_schedule", "milestones", "progress_checkpoints"]
            },
            "medium_goal": {
                "name": "中目标计划", 
                "description": "针对中型学习目标的月度计划",
                "components": ["monthly_schedule", "phase_planning", "assessment_points"]
            },
            "large_goal": {
                "name": "大目标计划",
                "description": "针对大型学习目标的季度计划",
                "components": ["quarterly_schedule", "module_planning", "evaluation_stages"]
            },
            "massive_goal": {
                "name": "大规模目标计划",
                "description": "针对超大规模学习目标的年度计划",
                "components": ["annual_schedule", "project_planning", "comprehensive_reviews"]
            }
        }
        
        # 学习阶段配置
        self.learning_phases = {
            "exploration": {
                "name": "探索阶段",
                "duration_ratio": 0.1,
                "activities": ["概览学习", "建立认知", "识别重点"]
            },
            "acquisition": {
                "name": "获取阶段",
                "duration_ratio": 0.4,
                "activities": ["系统学习", "理解概念", "掌握技能"]
            },
            "practice": {
                "name": "实践阶段", 
                "duration_ratio": 0.3,
                "activities": ["应用练习", "解决问题", "项目实践"]
            },
            "review": {
                "name": "复习阶段",
                "duration_ratio": 0.1,
                "activities": ["巩固记忆", "查漏补缺", "系统回顾"]
            },
            "integration": {
                "name": "整合阶段",
                "duration_ratio": 0.1,
                "activities": ["知识整合", "创新应用", "体系构建"]
            }
        }
    
    def create_learning_plan(self,
                           goal: LearningGoal,
                           mindmap_root: Optional[MindMapNode] = None,
                           node_map: Optional[Dict[str, MindMapNode]] = None,
                           allocation_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建学习计划
        
        Args:
            goal: 学习目标
            mindmap_root: 思维导图根节点
            node_map: 节点映射
            allocation_plan: 分配计划
            
        Returns:
            学习计划
        """
        print(f"📋 创建学习计划: {goal.description}")
        
        learning_plan = {
            "goal_id": goal.id,
            "goal_description": goal.description,
            "created_at": datetime.now().isoformat(),
            "plan_type": "mindmap_driven",
            "plan_components": {},
            "timeline": {},
            "milestones": [],
            "schedules": {},
            "review_plan": {},
            "assessment_plan": {}
        }
        
        # 确定计划类型
        plan_type = self._determine_plan_type(goal)
        learning_plan["plan_type"] = plan_type
        
        # 获取模板
        template = self.planning_templates.get(plan_type, self.planning_templates["medium_goal"])
        learning_plan["template_used"] = template
        
        # 如果提供了分配计划，使用它
        if allocation_plan:
            learning_plan["allocation_plan"] = allocation_plan
            sequences = allocation_plan.get("learning_sequences", [])
            time_allocation = allocation_plan.get("time_allocation", {})
        else:
            # 创建基本分配
            sequences = self._create_basic_sequences(goal, node_map)
            time_allocation = {}
        
        # 创建时间线
        timeline = self._create_timeline(goal, sequences, time_allocation)
        learning_plan["timeline"] = timeline
        
        # 创建里程碑
        milestones = self._create_milestones(goal, sequences, timeline)
        learning_plan["milestones"] = milestones
        
        # 创建详细日程
        schedules = self._create_detailed_schedules(goal, sequences, timeline)
        learning_plan["schedules"] = schedules
        
        # 创建复习计划
        review_plan = self._create_review_plan(goal, sequences, timeline)
        learning_plan["review_plan"] = review_plan
        
        # 创建评估计划
        assessment_plan = self._create_assessment_plan(goal, milestones, timeline)
        learning_plan["assessment_plan"] = assessment_plan
        
        # 整合所有组件
        learning_plan["plan_components"] = self._integrate_plan_components(
            goal, timeline, milestones, schedules, review_plan, assessment_plan
        )
        
        # 生成计划摘要
        learning_plan["summary"] = self._generate_plan_summary(learning_plan)
        
        print(f"✅ 学习计划创建完成: {len(milestones)}个里程碑，{len(schedules.get('weekly_schedules', []))}周计划")
        return learning_plan
    
    def create_adaptive_plan(self,
                           goal: LearningGoal,
                           learning_history: List[Dict[str, Any]],
                           current_progress: Dict[str, Any],
                           available_time_per_week: int = 10) -> Dict[str, Any]:
        """
        创建自适应学习计划
        
        Args:
            goal: 学习目标
            learning_history: 学习历史
            current_progress: 当前进度
            available_time_per_week: 每周可用时间（小时）
            
        Returns:
            自适应学习计划
        """
        print(f"🔄 创建自适应学习计划")
        
        adaptive_plan = {
            "goal_id": goal.id,
            "created_at": datetime.now().isoformat(),
            "plan_type": "adaptive",
            "learning_profile": {},
            "adaptive_strategies": [],
            "flexible_schedule": {},
            "adjustment_rules": [],
            "contingency_plans": []
        }
        
        # 分析学习历史
        learning_profile = self._analyze_learning_profile(learning_history)
        adaptive_plan["learning_profile"] = learning_profile
        
        # 确定自适应策略
        strategies = self._determine_adaptive_strategies(learning_profile, current_progress)
        adaptive_plan["adaptive_strategies"] = strategies
        
        # 创建弹性时间表
        flexible_schedule = self._create_flexible_schedule(
            goal, available_time_per_week, learning_profile
        )
        adaptive_plan["flexible_schedule"] = flexible_schedule
        
        # 创建调整规则
        adjustment_rules = self._create_adjustment_rules(learning_profile, current_progress)
        adaptive_plan["adjustment_rules"] = adjustment_rules
        
        # 创建应急计划
        contingency_plans = self._create_contingency_plans(goal, learning_profile)
        adaptive_plan["contingency_plans"] = contingency_plans
        
        # 生成建议
        adaptive_plan["recommendations"] = self._generate_adaptive_recommendations(
            adaptive_plan, goal, current_progress
        )
        
        return adaptive_plan
    
    def adjust_plan_based_on_progress(self,
                                    original_plan: Dict[str, Any],
                                    progress_data: Dict[str, Any],
                                    learning_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        基于进度调整学习计划
        
        Args:
            original_plan: 原始计划
            progress_data: 进度数据
            learning_history: 学习历史
            
        Returns:
            调整后的计划
        """
        print(f"🔄 基于进度调整学习计划")
        
        adjusted_plan = original_plan.copy()
        adjusted_plan["last_adjusted_at"] = datetime.now().isoformat()
        adjusted_plan["adjustments_made"] = []
        
        # 提取进度信息
        current_progress = progress_data.get("overall_progress", 0)
        expected_progress = self._calculate_expected_progress(original_plan)
        progress_delta = current_progress - expected_progress
        
        mastered_nodes = progress_data.get("mastered_nodes", [])
        struggling_nodes = progress_data.get("struggling_nodes", [])
        learning_speed = progress_data.get("learning_speed", 1.0)
        
        # 调整时间线
        if "timeline" in adjusted_plan:
            timeline = adjusted_plan["timeline"]
            
            # 根据进度差异调整时间线
            if abs(progress_delta) > 0.1:  # 进度偏差超过10%
                if progress_delta > 0:
                    # 进度超前，可以缩短时间或增加内容
                    adjustment = self._adjust_for_ahead_schedule(timeline, progress_delta)
                    adjusted_plan["adjustments_made"].append(adjustment)
                else:
                    # 进度落后，需要延长时间或减少内容
                    adjustment = self._adjust_for_behind_schedule(timeline, abs(progress_delta))
                    adjusted_plan["adjustments_made"].append(adjustment)
            
            adjusted_plan["timeline"] = timeline
        
        # 调整里程碑
        if "milestones" in adjusted_plan:
            milestones = adjusted_plan["milestones"]
            adjusted_milestones = []
            
            for milestone in milestones:
                # 检查里程碑是否已达成
                if milestone.get("achieved", False):
                    adjusted_milestones.append(milestone)
                    continue
                
                # 调整未达成里程碑
                adjusted_milestone = self._adjust_milestone(
                    milestone, progress_data, learning_history
                )
                adjusted_milestones.append(adjusted_milestone)
            
            adjusted_plan["milestones"] = adjusted_milestones
        
        # 调整日程安排
        if "schedules" in adjusted_plan:
            schedules = adjusted_plan["schedules"]
            
            # 根据学习速度调整
            if learning_speed != 1.0:
                schedules = self._adjust_schedules_for_speed(schedules, learning_speed)
                adjusted_plan["adjustments_made"].append(
                    f"根据学习速度({learning_speed:.2f}x)调整日程"
                )
            
            # 如果有困难节点，调整日程
            if struggling_nodes:
                schedules = self._adjust_schedules_for_difficulty(schedules, struggling_nodes)
                adjusted_plan["adjustments_made"].append(
                    f"为{len(struggling_nodes)}个困难节点调整日程"
                )
            
            adjusted_plan["schedules"] = schedules
        
        # 生成调整摘要
        adjusted_plan["adjustment_summary"] = self._generate_adjustment_summary(
            adjusted_plan, progress_data
        )
        
        return adjusted_plan
    
    def _determine_plan_type(self, goal: LearningGoal) -> str:
        """确定计划类型"""
        if goal.scale == GoalScale.MICRO:
            return "micro_goal"
        elif goal.scale == GoalScale.SMALL:
            return "small_goal"
        elif goal.scale == GoalScale.MEDIUM:
            return "medium_goal"
        elif goal.scale == GoalScale.LARGE:
            return "large_goal"
        elif goal.scale == GoalScale.MASSIVE:
            return "massive_goal"
        else:
            return "medium_goal"
    
    def _create_basic_sequences(self,
                              goal: LearningGoal,
                              node_map: Optional[Dict[str, MindMapNode]]) -> List[Dict[str, Any]]:
        """创建基本学习序列"""
        if not node_map:
            # 如果没有节点映射，创建简单序列
            return [{
                "id": generate_id("basic_sequence_"),
                "name": "基础学习序列",
                "description": f"学习目标: {goal.description}",
                "node_ids": [],
                "node_count": goal.target_knowledge_count,
                "estimated_time_minutes": goal.target_knowledge_count * 30  # 每个知识点30分钟
            }]
        
        # 如果有节点映射，创建基于节点的序列
        total_nodes = len(node_map)
        if total_nodes <= 10:
            # 节点少，一个序列
            all_nodes = list(node_map.keys())
            return [{
                "id": generate_id("basic_sequence_"),
                "name": "完整学习序列",
                "description": "包含所有知识节点的学习序列",
                "node_ids": all_nodes,
                "node_count": total_nodes,
                "estimated_time_minutes": sum(
                    node.estimated_time_minutes for node in node_map.values()
                )
            }]
        else:
            # 节点多，分成多个序列
            sequence_count = min(4, total_nodes // 5)
            nodes_per_sequence = math.ceil(total_nodes / sequence_count)
            
            all_nodes = list(node_map.keys())
            sequences = []
            
            for i in range(sequence_count):
                start_idx = i * nodes_per_sequence
                end_idx = min((i + 1) * nodes_per_sequence, total_nodes)
                
                sequence_nodes = all_nodes[start_idx:end_idx]
                sequences.append({
                    "id": generate_id(f"basic_seq_{i}_"),
                    "name": f"学习序列 {i+1}",
                    "description": f"第{i+1}部分学习内容",
                    "node_ids": sequence_nodes,
                    "node_count": len(sequence_nodes),
                    "estimated_time_minutes": sum(
                        node_map[nid].estimated_time_minutes for nid in sequence_nodes
                    )
                })
            
            return sequences
    
    def _create_timeline(self,
                        goal: LearningGoal,
                        sequences: List[Dict[str, Any]],
                        time_allocation: Dict[str, Any]) -> Dict[str, Any]:
        """创建学习时间线"""
        timeline = {
            "total_estimated_hours": 0,
            "start_date": datetime.now().isoformat(),
            "end_date": None,
            "phases": [],
            "weekly_breakdown": []
        }
        
        # 计算总时间
        total_minutes = sum(seq.get("estimated_time_minutes", 0) for seq in sequences)
        if total_minutes == 0 and goal.target_knowledge_count > 0:
            # 使用时间模型估算
            total_minutes = self.time_model.estimate_for_goal(goal)
        
        total_hours = total_minutes / 60
        timeline["total_estimated_hours"] = total_hours
        
        # 计算学习周期
        if total_hours <= 10:
            # 10小时以内：1周
            timeline_days = 7
            timeline_weeks = 1
        elif total_hours <= 40:
            # 10-40小时：2周
            timeline_days = 14
            timeline_weeks = 2
        elif total_hours <= 100:
            # 40-100小时：1个月
            timeline_days = 30
            timeline_weeks = 4
        elif total_hours <= 300:
            # 100-300小时：2个月
            timeline_days = 60
            timeline_weeks = 8
        else:
            # 300小时以上：3个月
            timeline_days = 90
            timeline_weeks = 12
        
        timeline["timeline_days"] = timeline_days
        timeline["timeline_weeks"] = timeline_weeks
        
        # 计算结束日期
        start_date = datetime.now()
        end_date = start_date + timedelta(days=timeline_days)
        timeline["end_date"] = end_date.isoformat()
        
        # 创建学习阶段
        phases = self._create_learning_phases(timeline_weeks, sequences)
        timeline["phases"] = phases
        
        # 创建每周分解
        weekly_breakdown = self._create_weekly_breakdown(
            timeline_weeks, sequences, total_hours
        )
        timeline["weekly_breakdown"] = weekly_breakdown
        
        return timeline
    
    def _create_learning_phases(self,
                               total_weeks: int,
                               sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """创建学习阶段"""
        phases = []
        
        # 确定阶段数量
        if total_weeks <= 2:
            phase_count = 2
        elif total_weeks <= 4:
            phase_count = 3
        elif total_weeks <= 8:
            phase_count = 4
        else:
            phase_count = 5
        
        # 分配阶段
        weeks_per_phase = math.ceil(total_weeks / phase_count)
        
        for i in range(phase_count):
            phase_start_week = i * weeks_per_phase + 1
            phase_end_week = min((i + 1) * weeks_per_phase, total_weeks)
            
            # 选择阶段类型
            phase_types = list(self.learning_phases.keys())
            phase_type = phase_types[i % len(phase_types)]
            phase_config = self.learning_phases[phase_type]
            
            # 分配序列（如果有）
            sequences_for_phase = []
            if sequences:
                # 简单分配：每个阶段分配一些序列
                seq_per_phase = math.ceil(len(sequences) / phase_count)
                start_seq = i * seq_per_phase
                end_seq = min((i + 1) * seq_per_phase, len(sequences))
                sequences_for_phase = sequences[start_seq:end_seq]
            
            phase = {
                "id": generate_id(f"phase_{i}_"),
                "name": phase_config["name"],
                "type": phase_type,
                "description": phase_config["description"] if "description" in phase_config else "",
                "start_week": phase_start_week,
                "end_week": phase_end_week,
                "duration_weeks": phase_end_week - phase_start_week + 1,
                "activities": phase_config["activities"],
                "sequences": sequences_for_phase,
                "focus_areas": self._determine_phase_focus(phase_type, sequences_for_phase)
            }
            
            phases.append(phase)
        
        return phases
    
    def _determine_phase_focus(self,
                             phase_type: str,
                             sequences: List[Dict[str, Any]]) -> List[str]:
        """确定阶段重点"""
        focus_areas = []
        
        if phase_type == "exploration":
            focus_areas = ["整体认知", "建立框架", "识别重点"]
        elif phase_type == "acquisition":
            focus_areas = ["概念理解", "知识获取", "基础建立"]
        elif phase_type == "practice":
            focus_areas = ["应用练习", "技能掌握", "问题解决"]
        elif phase_type == "review":
            focus_areas = ["巩固记忆", "查漏补缺", "系统回顾"]
        elif phase_type == "integration":
            focus_areas = ["知识整合", "创新应用", "体系构建"]
        
        # 根据序列内容调整
        if sequences:
            if any("高级" in seq.get("name", "") for seq in sequences):
                focus_areas.append("深度理解")
            if any("实践" in seq.get("name", "") for seq in sequences):
                focus_areas.append("实践应用")
        
        return focus_areas
    
    def _create_weekly_breakdown(self,
                                total_weeks: int,
                                sequences: List[Dict[str, Any]],
                                total_hours: float) -> List[Dict[str, Any]]:
        """创建每周分解"""
        weekly_breakdown = []
        
        # 计算每周学习小时数
        hours_per_week = math.ceil(total_hours / total_weeks)
        
        for week in range(1, total_weeks + 1):
            # 分配序列（如果有）
            sequences_for_week = []
            if sequences:
                # 简单轮转分配
                seq_index = (week - 1) % len(sequences)
                sequences_for_week = [sequences[seq_index]]
            
            week_plan = {
                "week_number": week,
                "focus": self._determine_weekly_focus(week, total_weeks),
                "estimated_hours": hours_per_week,
                "daily_recommendation": f"每日{math.ceil(hours_per_week / 7)}小时",
                "sequences": sequences_for_week,
                "key_activities": self._get_weekly_activities(week, total_weeks),
                "milestones": []  # 将在后面填充
            }
            
            weekly_breakdown.append(week_plan)
        
        return weekly_breakdown
    
    def _determine_weekly_focus(self, week: int, total_weeks: int) -> str:
        """确定每周重点"""
        if week == 1:
            return "建立学习习惯，熟悉学习内容"
        elif week <= total_weeks // 3:
            return "系统学习，建立知识基础"
        elif week <= total_weeks * 2 // 3:
            return "深化理解，加强实践应用"
        elif week == total_weeks:
            return "总结回顾，整合知识体系"
        else:
            return "巩固提高，准备下一阶段"
    
    def _get_weekly_activities(self, week: int, total_weeks: int) -> List[str]:
        """获取每周活动"""
        activities = []
        
        # 每周基础活动
        base_activities = ["学习新知识", "完成练习", "复习巩固"]
        activities.extend(base_activities)
        
        # 特殊周活动
        if week == 1:
            activities.append("制定详细计划")
        elif week % 2 == 0:
            activities.append("进行小测验")
        elif week % 4 == 0:
            activities.append("进行阶段评估")
        
        if week == total_weeks:
            activities.append("进行最终总结")
        
        return activities
    
    def _create_milestones(self,
                          goal: LearningGoal,
                          sequences: List[Dict[str, Any]],
                          timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建里程碑"""
        milestones = []
        
        total_weeks = timeline.get("timeline_weeks", 4)
        phases = timeline.get("phases", [])
        
        # 阶段里程碑
        for phase in phases:
            milestone = {
                "id": generate_id(f"milestone_phase_{phase.get('name', '')}_"),
                "name": f"完成{phase.get('name', '')}",
                "type": "phase_completion",
                "description": f"完成{phase.get('name', '')}的学习",
                "target_week": phase.get("end_week", 0),
                "success_criteria": [
                    f"完成{phase.get('name', '')}的所有学习活动",
                    f"掌握{phase.get('name', '')}的核心概念"
                ],
                "reward": f"庆祝{phase.get('name', '')}完成，短暂休息"
            }
            milestones.append(milestone)
        
        # 序列里程碑（如果有序列）
        if sequences:
            for i, seq in enumerate(sequences):
                milestone_week = min((i + 1) * total_weeks // len(sequences), total_weeks)
                milestone = {
                    "id": generate_id(f"milestone_seq_{i}_"),
                    "name": f"完成{seq.get('name', f'序列{i+1}')}",
                    "type": "sequence_completion",
                    "description": f"完成{seq.get('name', f'第{i+1}个学习序列')}",
                    "target_week": milestone_week,
                    "success_criteria": [
                        f"完成{seq.get('name', f'序列{i+1}')}的所有学习内容",
                        f"通过{seq.get('name', f'序列{i+1}')}的测试"
                    ],
                    "reward": "继续下一阶段学习"
                }
                milestones.append(milestone)
        
        # 时间里程碑
        time_milestones = [
            (total_weeks // 4, "完成第一月学习", "检查学习进度，调整计划"),
            (total_weeks // 2, "完成一半学习", "进行中期评估，总结学习成果"),
            (total_weeks * 3 // 4, "完成第三月学习", "加强薄弱环节，准备最后冲刺"),
            (total_weeks, "完成全部学习", "进行最终评估，庆祝学习成果")
        ]
        
        for week, name, description in time_milestones:
            if week <= total_weeks:
                milestone = {
                    "id": generate_id(f"milestone_time_{week}_"),
                    "name": name,
                    "type": "time_based",
                    "description": description,
                    "target_week": week,
                    "success_criteria": [f"按计划完成第{week}周学习"],
                    "reward": "成就感奖励"
                }
                milestones.append(milestone)
        
        # 按目标周排序
        milestones.sort(key=lambda x: x.get("target_week", 0))
        
        return milestones
    
    def _create_detailed_schedules(self,
                                 goal: LearningGoal,
                                 sequences: List[Dict[str, Any]],
                                 timeline: Dict[str, Any]) -> Dict[str, Any]:
        """创建详细日程"""
        schedules = {
            "weekly_schedules": [],
            "daily_templates": [],
            "time_blocks": []
        }
        
        total_weeks = timeline.get("timeline_weeks", 4)
        weekly_breakdown = timeline.get("weekly_breakdown", [])
        
        # 创建每周日程
        for week_plan in weekly_breakdown:
            week_number = week_plan.get("week_number", 1)
            weekly_schedule = self._create_weekly_schedule(week_plan, week_number)
            schedules["weekly_schedules"].append(weekly_schedule)
        
        # 创建每日模板
        daily_templates = self._create_daily_templates(goal)
        schedules["daily_templates"] = daily_templates
        
        # 创建时间块
        time_blocks = self._create_time_blocks()
        schedules["time_blocks"] = time_blocks
        
        return schedules
    
    def _create_weekly_schedule(self, week_plan: Dict[str, Any], week_number: int) -> Dict[str, Any]:
        """创建周日程"""
        weekly_schedule = {
            "week_number": week_number,
            "focus": week_plan.get("focus", ""),
            "estimated_hours": week_plan.get("estimated_hours", 10),
            "daily_breakdown": []
        }
        
        # 每日学习时间分配（假设每周学习5天）
        daily_hours = math.ceil(week_plan.get("estimated_hours", 10) / 5)
        
        # 创建每日计划
        for day in range(1, 6):  # 周一至周五
            daily_plan = {
                "day": day,
                "day_name": ["周一", "周二", "周三", "周四", "周五"][day-1],
                "estimated_hours": daily_hours,
                "focus_areas": self._get_daily_focus_areas(day, week_number),
                "activities": self._get_daily_activities(day, week_number),
                "time_slots": self._create_daily_time_slots(daily_hours)
            }
            weekly_schedule["daily_breakdown"].append(daily_plan)
        
        # 周末计划
        weekend_plan = {
            "day": 6,
            "day_name": "周末",
            "estimated_hours": 2,  # 周末复习2小时
            "focus_areas": ["复习", "整理", "计划"],
            "activities": ["复习本周内容", "整理学习笔记", "制定下周计划"],
            "time_slots": [{"time": "灵活安排", "activity": "周末复习"}]
        }
        weekly_schedule["daily_breakdown"].append(weekend_plan)
        
        return weekly_schedule
    
    def _get_daily_focus_areas(self, day: int, week: int) -> List[str]:
        """获取每日重点领域"""
        focus_pattern = [
            ["新知识学习", "概念理解"],
            ["深化理解", "练习应用"],
            ["技能训练", "实践操作"],
            ["复习巩固", "查漏补缺"],
            ["整合应用", "创新思考"]
        ]
        
        pattern_index = (day - 1) % len(focus_pattern)
        return focus_pattern[pattern_index]
    
    def _get_daily_activities(self, day: int, week: int) -> List[str]:
        """获取每日活动"""
        base_activities = ["阅读学习材料", "完成练习", "复习笔记"]
        
        # 每周第一天添加计划活动
        if day == 1:
            base_activities.append("制定本周计划")
        
        # 每周最后一天添加总结活动
        if day == 5:
            base_activities.append("本周总结")
        
        # 特殊活动
        if week % 2 == 0 and day == 3:
            base_activities.append("进行小测验")
        
        return base_activities
    
    def _create_daily_time_slots(self, daily_hours: int) -> List[Dict[str, str]]:
        """创建每日时间块"""
        # 假设学习时间分布在多个时间段
        time_slots = []
        
        if daily_hours >= 3:
            # 长时间学习：分多个时间段
            slots = [
                {"time": "09:00-10:30", "activity": "上午学习", "duration": 90},
                {"time": "14:00-15:30", "activity": "下午学习", "duration": 90},
                {"time": "20:00-21:00", "activity": "晚间复习", "duration": 60}
            ]
        elif daily_hours >= 2:
            # 中等时间学习
            slots = [
                {"time": "19:00-20:30", "activity": "晚间学习", "duration": 90},
                {"time": "21:00-21:30", "activity": "晚间复习", "duration": 30}
            ]
        else:
            # 短时间学习
            slots = [
                {"time": "20:00-21:00", "activity": "集中学习", "duration": 60}
            ]
        
        return slots
    
    def _create_daily_templates(self, goal: LearningGoal) -> List[Dict[str, Any]]:
        """创建每日模板"""
        templates = []
        
        # 高强度学习日模板
        templates.append({
            "name": "高强度学习日",
            "description": "专注深度学习和复杂任务",
            "total_hours": 3,
            "time_blocks": [
                {"time": "09:00-10:30", "activity": "深度学习", "focus": "复杂概念"},
                {"time": "14:00-15:30", "activity": "实践练习", "focus": "技能应用"},
                {"time": "20:00-21:00", "activity": "复习总结", "focus": "知识巩固"}
            ],
            "suitable_for": ["重要概念学习", "技能训练", "项目实践"]
        })
        
        # 中等强度学习日模板
        templates.append({
            "name": "中等强度学习日",
            "description": "平衡学习和复习",
            "total_hours": 2,
            "time_blocks": [
                {"time": "19:00-20:30", "activity": "系统学习", "focus": "新知识获取"},
                {"time": "21:00-21:30", "activity": "快速复习", "focus": "记忆巩固"}
            ],
            "suitable_for": ["日常学习", "知识积累", "进度维持"]
        })
        
        # 低强度学习日模板
        templates.append({
            "name": "低强度学习日",
            "description": "轻量学习和复习",
            "total_hours": 1,
            "time_blocks": [
                {"time": "20:00-21:00", "activity": "集中学习", "focus": "重点复习"}
            ],
            "suitable_for": ["忙碌日子", "复习巩固", "保持学习习惯"]
        })
        
        return templates
    
    def _create_time_blocks(self) -> List[Dict[str, Any]]:
        """创建时间块"""
        time_blocks = [
            {
                "name": "清晨学习块",
                "time_range": "06:00-08:00",
                "duration_minutes": 120,
                "characteristics": ["头脑清醒", "记忆力好", "干扰少"],
                "suitable_activities": ["记忆性学习", "概念理解", "计划制定"]
            },
            {
                "name": "上午学习块",
                "time_range": "09:00-12:00",
                "duration_minutes": 180,
                "characteristics": ["精力充沛", "专注度高", "效率高"],
                "suitable_activities": ["深度学习", "复杂任务", "项目工作"]
            },
            {
                "name": "下午学习块",
                "time_range": "14:00-17:00",
                "duration_minutes": 180,
                "characteristics": ["稳定发挥", "适合实践", "互动性好"],
                "suitable_activities": ["实践练习", "小组学习", "技能训练"]
            },
            {
                "name": "晚间学习块",
                "time_range": "19:00-22:00",
                "duration_minutes": 180,
                "characteristics": ["安静环境", "适合复习", "总结整理"],
                "suitable_activities": ["复习巩固", "知识整理", "计划反思"]
            }
        ]
        
        return time_blocks
    
    def _create_review_plan(self,
                          goal: LearningGoal,
                          sequences: List[Dict[str, Any]],
                          timeline: Dict[str, Any]) -> Dict[str, Any]:
        """创建复习计划"""
        review_plan = {
            "review_strategy": "spaced_repetition",
            "review_schedule": [],
            "review_methods": [],
            "review_checkpoints": []
        }
        
        total_weeks = timeline.get("timeline_weeks", 4)
        
        # 复习时间表（基于间隔重复）
        review_intervals = [1, 3, 7, 14, 30]  # 学习后的第几天复习
        
        for interval in review_intervals:
            review_plan["review_schedule"].append({
                "interval_days": interval,
                "review_type": "spaced_repetition",
                "focus": "记忆巩固",
                "methods": ["快速回顾", "自我测试", "概念复述"]
            })
        
        # 复习方法
        review_methods = [
            {
                "name": "主动回忆",
                "description": "不看书本，尝试回忆学习内容",
                "effectiveness": "高",
                "time_required": "中"
            },
            {
                "name": "自我测试",
                "description": "通过测试题检查掌握程度",
                "effectiveness": "高",
                "time_required": "中"
            },
            {
                "name": "概念图",
                "description": "绘制概念关系图",
                "effectiveness": "中",
                "time_required": "高"
            },
            {
                "name": "费曼技巧",
                "description": "用简单语言解释复杂概念",
                "effectiveness": "高",
                "time_required": "中"
            }
        ]
        review_plan["review_methods"] = review_methods
        
        # 复习检查点
        if total_weeks >= 4:
            checkpoints = [
                {"week": 1, "type": "周复习", "focus": "第一周内容"},
                {"week": 2, "type": "双周复习", "focus": "前两周内容"},
                {"week": 4, "type": "月复习", "focus": "整月内容"}
            ]
            if total_weeks >= 8:
                checkpoints.append({"week": 8, "type": "中期复习", "focus": "前半段内容"})
            if total_weeks >= 12:
                checkpoints.append({"week": 12, "type": "最终复习", "focus": "全部内容"})
            
            review_plan["review_checkpoints"] = checkpoints
        
        return review_plan
    
    def _create_assessment_plan(self,
                              goal: LearningGoal,
                              milestones: List[Dict[str, Any]],
                              timeline: Dict[str, Any]) -> Dict[str, Any]:
        """创建评估计划"""
        assessment_plan = {
            "assessment_types": [],
            "assessment_schedule": [],
            "evaluation_criteria": [],
            "feedback_mechanisms": []
        }
        
        # 评估类型
        assessment_types = [
            {
                "name": "形成性评估",
                "purpose": "学习过程中的持续评估",
                "methods": ["自我测试", "小测验", "学习日记"],
                "frequency": "每周"
            },
            {
                "name": "总结性评估",
                "purpose": "阶段性的综合评估",
                "methods": ["阶段考试", "项目评估", "综合测试"],
                "frequency": "每月"
            },
            {
                "name": "诊断性评估",
                "purpose": "识别学习困难和需求",
                "methods": ["前测", "知识地图", "学习分析"],
                "frequency": "学习开始时和需要时"
            }
        ]
        assessment_plan["assessment_types"] = assessment_types
        
        # 评估时间表
        total_weeks = timeline.get("timeline_weeks", 4)
        
        for week in range(1, total_weeks + 1):
            assessments = []
            
            # 每周小测验
            if week % 2 == 0:  # 每两周一次
                assessments.append({
                    "type": "小测验",
                    "purpose": "检查周学习成果",
                    "estimated_time": "30分钟"
                })
            
            # 里程碑评估
            for milestone in milestones:
                if milestone.get("target_week") == week:
                    assessments.append({
                        "type": "里程碑评估",
                        "purpose": f"评估{milestone.get('name')}完成情况",
                        "estimated_time": "60分钟"
                    })
            
            if assessments:
                assessment_plan["assessment_schedule"].append({
                    "week": week,
                    "assessments": assessments
                })
        
        # 评估标准
        assessment_plan["evaluation_criteria"] = [
            {"criterion": "知识掌握", "weight": 0.4, "description": "对学习内容的掌握程度"},
            {"criterion": "技能应用", "weight": 0.3, "description": "将知识应用于实际问题的能力"},
            {"criterion": "学习进步", "weight": 0.2, "description": "相比之前的学习进步"},
            {"criterion": "学习参与", "weight": 0.1, "description": "学习过程中的参与和投入程度"}
        ]
        
        # 反馈机制
        assessment_plan["feedback_mechanisms"] = [
            {"mechanism": "自我反馈", "frequency": "每日", "format": "学习日记"},
            {"mechanism": "系统反馈", "frequency": "每次评估后", "format": "评估报告"},
            {"mechanism": "同伴反馈", "frequency": "每周", "format": "学习小组讨论"},
            {"mechanism": "专家反馈", "frequency": "每月", "format": "指导会议"}
        ]
        
        return assessment_plan
    
    def _integrate_plan_components(self,
                                 goal: LearningGoal,
                                 timeline: Dict[str, Any],
                                 milestones: List[Dict[str, Any]],
                                 schedules: Dict[str, Any],
                                 review_plan: Dict[str, Any],
                                 assessment_plan: Dict[str, Any]) -> Dict[str, Any]:
        """整合计划组件"""
        integrated_components = {
            "timeline_integration": {
                "total_weeks": timeline.get("timeline_weeks", 0),
                "start_date": timeline.get("start_date"),
                "end_date": timeline.get("end_date"),
                "milestone_count": len(milestones),
                "weekly_schedule_count": len(schedules.get("weekly_schedules", [])),
                "review_checkpoints": len(review_plan.get("review_checkpoints", [])),
                "assessment_schedule": len(assessment_plan.get("assessment_schedule", []))
            },
            "component_links": [],
            "coordination_points": []
        }
        
        # 创建组件链接
        for milestone in milestones:
            milestone_week = milestone.get("target_week", 0)
            
            # 链接到周计划
            for weekly_schedule in schedules.get("weekly_schedules", []):
                if weekly_schedule.get("week_number") == milestone_week:
                    integrated_components["component_links"].append({
                        "from": f"milestone_{milestone.get('name')}",
                        "to": f"weekly_schedule_week_{milestone_week}",
                        "relationship": "里程碑对应周计划"
                    })
            
            # 链接到复习检查点
            for checkpoint in review_plan.get("review_checkpoints", []):
                if checkpoint.get("week") == milestone_week:
                    integrated_components["component_links"].append({
                        "from": f"milestone_{milestone.get('name')}",
                        "to": f"review_checkpoint_week_{milestone_week}",
                        "relationship": "里程碑对应复习点"
                    })
        
        # 创建协调点
        coordination_points = []
        
        # 每周协调点
        for week in range(1, timeline.get("timeline_weeks", 0) + 1):
            coordination_points.append({
                "week": week,
                "activities": [
                    "检查周计划完成情况",
                    "调整下周计划",
                    "进行周复习",
                    "记录学习进展"
                ],
                "estimated_time": "60分钟"
            })
        
        # 里程碑协调点
        for milestone in milestones:
            coordination_points.append({
                "milestone": milestone.get("name"),
                "week": milestone.get("target_week"),
                "activities": [
                    f"评估{milestone.get('name')}完成情况",
                    "庆祝里程碑达成",
                    "调整后续计划",
                    "进行阶段性总结"
                ],
                "estimated_time": "90分钟"
            })
        
        integrated_components["coordination_points"] = coordination_points
        
        return integrated_components
    
    def _generate_plan_summary(self, learning_plan: Dict[str, Any]) -> Dict[str, Any]:
        """生成计划摘要"""
        timeline = learning_plan.get("timeline", {})
        milestones = learning_plan.get("milestones", [])
        schedules = learning_plan.get("schedules", {})
        
        summary = {
            "overview": {
                "total_weeks": timeline.get("timeline_weeks", 0),
                "total_milestones": len(milestones),
                "weekly_schedules": len(schedules.get("weekly_schedules", [])),
                "estimated_total_hours": timeline.get("total_estimated_hours", 0)
            },
            "key_dates": {
                "start_date": timeline.get("start_date"),
                "end_date": timeline.get("end_date"),
                "key_milestones": [
                    {
                        "name": milestone.get("name"),
                        "week": milestone.get("target_week"),
                        "type": milestone.get("type")
                    }
                    for milestone in milestones[:3]  # 只显示前3个重要里程碑
                ]
            },
            "weekly_commitment": {
                "average_hours_per_week": math.ceil(
                    timeline.get("total_estimated_hours", 0) / timeline.get("timeline_weeks", 1)
                ),
                "learning_days_per_week": 5,
                "daily_average_hours": math.ceil(
                    timeline.get("total_estimated_hours", 0) / (timeline.get("timeline_weeks", 1) * 5)
                )
            },
            "success_factors": [
                "坚持每日学习",
                "定期复习巩固",
                "积极参与评估",
                "及时调整计划"
            ]
        }
        
        return summary
    
    def _analyze_learning_profile(self, learning_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析学习画像"""
        if not learning_history:
            return {"message": "无学习历史数据"}
        
        profile = {
            "total_sessions": len(learning_history),
            "learning_patterns": {},
            "preferences": {},
            "strengths": [],
            "weaknesses": []
        }
        
        # 分析学习时间模式
        session_times = []
        session_durations = []
        
        for session in learning_history:
            # 记录会话时间（假设有timestamp字段）
            if "timestamp" in session:
                session_time = datetime.fromisoformat(session["timestamp"]).hour
                session_times.append(session_time)
            
            # 记录会话时长（假设有duration_minutes字段）
            if "duration_minutes" in session:
                session_durations.append(session["duration_minutes"])
        
        if session_times:
            # 分析最佳学习时间
            time_counts = {}
            for hour in session_times:
                time_counts[hour] = time_counts.get(hour, 0) + 1
            
            if time_counts:
                best_hour = max(time_counts.items(), key=lambda x: x[1])[0]
                profile["learning_patterns"]["preferred_time"] = f"{best_hour}:00-{best_hour+1}:00"
        
        if session_durations:
            avg_duration = statistics.mean(session_durations)
            profile["learning_patterns"]["average_session_duration"] = avg_duration
        
        # 分析学习偏好（从历史中提取）
        # 这里简化处理，实际中需要更复杂的分析
        profile["preferences"] = {
            "learning_style": "balanced",  # 可以从历史中分析
            "preferred_content_type": "mixed",
            "interaction_level": "medium"
        }
        
        return profile
    
    def _determine_adaptive_strategies(self,
                                     learning_profile: Dict[str, Any],
                                     current_progress: Dict[str, Any]) -> List[Dict[str, Any]]:
        """确定自适应策略"""
        strategies = []
        
        # 基于学习时间的策略
        if "preferred_time" in learning_profile.get("learning_patterns", {}):
            strategies.append({
                "strategy": "时间优化",
                "description": f"在{learning_profile['learning_patterns']['preferred_time']}进行主要学习",
                "implementation": "安排重要学习任务在最佳时间段"
            })
        
        # 基于学习时长的策略
        avg_duration = learning_profile.get("learning_patterns", {}).get("average_session_duration", 0)
        if avg_duration > 0:
            if avg_duration < 30:
                strategies.append({
                    "strategy": "短时高效",
                    "description": "学习会话较短，采用高效学习方法",
                    "implementation": "使用番茄工作法，25分钟专注学习"
                })
            elif avg_duration > 90:
                strategies.append({
                    "strategy": "深度专注",
                    "description": "能够长时间专注学习",
                    "implementation": "安排长时间深度学习任务"
                })
        
        # 基于进度的策略
        progress = current_progress.get("overall_progress", 0)
        if progress < 0.3:
            strategies.append({
                "strategy": "建立基础",
                "description": "学习初期，重点建立基础",
                "implementation": "放慢节奏，确保基础概念掌握"
            })
        elif progress > 0.7:
            strategies.append({
                "strategy": "加速推进",
                "description": "学习后期，可以加速推进",
                "implementation": "增加学习强度，快速完成剩余内容"
            })
        
        return strategies
    
    def _create_flexible_schedule(self,
                                goal: LearningGoal,
                                available_time_per_week: int,
                                learning_profile: Dict[str, Any]) -> Dict[str, Any]:
        """创建弹性时间表"""
        flexible_schedule = {
            "available_hours_per_week": available_time_per_week,
            "minimum_weekly_hours": max(2, available_time_per_week // 2),
            "maximum_weekly_hours": min(20, available_time_per_week * 2),
            "flexible_days": [],
            "backup_time_slots": []
        }
        
        # 确定灵活学习日
        days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        flexible_days = []
        
        # 基于可用时间确定灵活日
        if available_time_per_week >= 10:
            # 时间充足，工作日学习
            flexible_days = days[:5]
        else:
            # 时间有限，集中在少数几天
            flexible_days = days[:3] + [days[-1]]  # 周一到周三加周日
        
        flexible_schedule["flexible_days"] = flexible_days
        
        # 创建备用时间段
        backup_slots = [
            {"day": "周末", "time": "09:00-12:00", "purpose": "补学未完成内容"},
            {"day": "工作日", "time": "20:00-22:00", "purpose": "日常学习"}
        ]
        
        # 如果有偏好的学习时间，优先使用
        preferred_time = learning_profile.get("learning_patterns", {}).get("preferred_time", "")
        if preferred_time:
            backup_slots.insert(0, {
                "day": "最佳时间",
                "time": preferred_time,
                "purpose": "高效学习时间段"
            })
        
        flexible_schedule["backup_time_slots"] = backup_slots
        
        return flexible_schedule
    
    def _create_adjustment_rules(self,
                               learning_profile: Dict[str, Any],
                               current_progress: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建调整规则"""
        rules = []
        
        # 进度调整规则
        rules.append({
            "rule_id": "progress_adjustment",
            "condition": "进度落后计划10%以上",
            "action": "增加每周学习时间10%",
            "priority": "high"
        })
        
        rules.append({
            "rule_id": "progress_ahead",
            "condition": "进度超前计划20%以上",
            "action": "可以提前学习后续内容或增加难度",
            "priority": "medium"
        })
        
        # 时间调整规则
        rules.append({
            "rule_id": "time_constraint",
            "condition": "连续3天未达到每日学习目标",
            "action": "重新评估时间分配，调整学习计划",
            "priority": "high"
        })
        
        # 难度调整规则
        rules.append({
            "rule_id": "difficulty_adjustment",
            "condition": "连续3个学习会话遇到困难",
            "action": "降低学习难度，增加基础练习",
            "priority": "medium"
        })
        
        # 参与度调整规则
        rules.append({
            "rule_id": "engagement_adjustment",
            "condition": "学习参与度连续下降",
            "action": "改变学习方式，增加互动元素",
            "priority": "medium"
        })
        
        return rules
    
    def _create_contingency_plans(self,
                                goal: LearningGoal,
                                learning_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建应急计划"""
        contingency_plans = []
        
        # 时间不足应急计划
        contingency_plans.append({
            "scenario": "时间严重不足",
            "probability": "medium",
            "impact": "high",
            "response": [
                "聚焦核心概念，跳过次要内容",
                "采用高效学习方法（如费曼技巧）",
                "延长学习周期，降低每周强度"
            ],
            "trigger": "可用时间减少50%以上"
        })
        
        # 学习困难应急计划
        contingency_plans.append({
            "scenario": "遇到学习瓶颈",
            "probability": "high",
            "impact": "medium",
            "response": [
                "寻求外部帮助（导师、学习小组）",
                "改变学习策略",
                "暂时切换学习内容，避免倦怠"
            ],
            "trigger": "连续2周无明显进步"
        })
        
        # 动力不足应急计划
        contingency_plans.append({
            "scenario": "学习动力下降",
            "probability": "medium",
            "impact": "medium",
            "response": [
                "设置小奖励机制",
                "寻找学习伙伴",
                "回顾学习初衷和目标"
            ],
            "trigger": "连续3天缺乏学习动力"
        })
        
        return contingency_plans
    
    def _generate_adaptive_recommendations(self,
                                         adaptive_plan: Dict[str, Any],
                                         goal: LearningGoal,
                                         current_progress: Dict[str, Any]) -> List[str]:
        """生成自适应建议"""
        recommendations = []
        
        # 基于学习画像的建议
        learning_profile = adaptive_plan.get("learning_profile", {})
        if "preferred_time" in learning_profile.get("learning_patterns", {}):
            recommendations.append(
                f"根据历史数据，建议在{learning_profile['learning_patterns']['preferred_time']}进行主要学习"
            )
        
        # 基于可用时间的建议
        available_time = adaptive_plan.get("flexible_schedule", {}).get("available_hours_per_week", 0)
        if available_time < 5:
            recommendations.append("每周可用时间有限，建议采用高效学习方法")
        elif available_time > 15:
            recommendations.append("每周可用时间充足，可以安排深度学习和实践")
        
        # 基于进度的建议
        progress = current_progress.get("overall_progress", 0)
        if progress > 0 and progress < 0.3:
            recommendations.append("学习初期，建议放慢节奏打好基础")
        elif progress > 0.7:
            recommendations.append("学习后期，可以加速完成剩余内容")
        
        return recommendations
    
    def _calculate_expected_progress(self, plan: Dict[str, Any]) -> float:
        """计算预期进度"""
        timeline = plan.get("timeline", {})
        created_at = plan.get("created_at")
        
        if not created_at or "start_date" not in timeline:
            return 0.0
        
        try:
            start_date = datetime.fromisoformat(timeline["start_date"])
            current_date = datetime.now()
            
            # 计算已过时间比例
            if "end_date" in timeline:
                end_date = datetime.fromisoformat(timeline["end_date"])
                total_days = (end_date - start_date).days
                elapsed_days = (current_date - start_date).days
                
                if total_days > 0:
                    return min(max(elapsed_days / total_days, 0.0), 1.0)
            
            # 如果没有结束日期，使用周数估算
            total_weeks = timeline.get("timeline_weeks", 4)
            if total_weeks > 0:
                # 假设每周进度均匀
                elapsed_days = (current_date - start_date).days
                elapsed_weeks = elapsed_days / 7
                return min(max(elapsed_weeks / total_weeks, 0.0), 1.0)
            
        except Exception as e:
            print(f"❌ 计算预期进度失败: {str(e)}")
        
        return 0.0
    
    def _adjust_for_ahead_schedule(self,
                                 timeline: Dict[str, Any],
                                 progress_delta: float) -> Dict[str, Any]:
        """为进度超前调整时间线"""
        adjustment = {
            "type": "ahead_schedule",
            "progress_delta": progress_delta,
            "actions": []
        }
        
        # 如果进度超前超过20%，可以考虑提前结束
        if progress_delta > 0.2:
            adjustment["actions"].append("考虑提前完成学习目标")
            adjustment["actions"].append("可以增加学习内容深度")
        
        # 如果进度超前10-20%，可以保持节奏或增加内容
        elif progress_delta > 0.1:
            adjustment["actions"].append("保持当前学习节奏")
            adjustment["actions"].append("可以考虑增加扩展学习")
        
        return adjustment
    
    def _adjust_for_behind_schedule(self,
                                  timeline: Dict[str, Any],
                                  progress_delta: float) -> Dict[str, Any]:
        """为进度落后调整时间线"""
        adjustment = {
            "type": "behind_schedule",
            "progress_delta": progress_delta,
            "actions": []
        }
        
        # 如果进度落后超过20%，需要大幅调整
        if progress_delta > 0.2:
            adjustment["actions"].append("需要大幅增加学习时间")
            adjustment["actions"].append("考虑延长学习周期")
            adjustment["actions"].append("聚焦核心内容，跳过次要部分")
        
        # 如果进度落后10-20%，需要适当调整
        elif progress_delta > 0.1:
            adjustment["actions"].append("适当增加每周学习时间")
            adjustment["actions"].append("加强薄弱环节学习")
            adjustment["actions"].append("优化学习方法提高效率")
        
        return adjustment
    
    def _adjust_milestone(self,
                         milestone: Dict[str, Any],
                         progress_data: Dict[str, Any],
                         learning_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调整里程碑"""
        adjusted_milestone = milestone.copy()
        
        # 检查里程碑是否应该调整
        current_week = self._get_current_week_from_plan(progress_data)
        target_week = milestone.get("target_week", 0)
        
        if current_week > target_week and not milestone.get("achieved", False):
            # 里程碑已过期但未达成
            adjusted_milestone["status"] = "overdue"
            adjusted_milestone["adjustment_needed"] = True
            
            # 重新安排里程碑
            new_target_week = current_week + 1  # 安排到下一周
            adjusted_milestone["target_week"] = new_target_week
            adjusted_milestone["original_target_week"] = target_week
            adjusted_milestone["rescheduled_at"] = datetime.now().isoformat()
        
        return adjusted_milestone
    
    def _get_current_week_from_plan(self, progress_data: Dict[str, Any]) -> int:
        """从进度数据获取当前周"""
        # 这里简化处理，实际中需要更复杂的逻辑
        overall_progress = progress_data.get("overall_progress", 0)
        
        # 假设进度均匀，计算当前周
        if "learning_weeks" in progress_data:
            total_weeks = progress_data["learning_weeks"]
            return min(math.ceil(overall_progress * total_weeks), total_weeks)
        
        return math.ceil(overall_progress * 12)  # 默认12周
    
    def _adjust_schedules_for_speed(self,
                                  schedules: Dict[str, Any],
                                  learning_speed: float) -> Dict[str, Any]:
        """根据学习速度调整日程"""
        adjusted_schedules = schedules.copy()
        
        # 调整每周计划
        if "weekly_schedules" in adjusted_schedules:
            for weekly_schedule in adjusted_schedules["weekly_schedules"]:
                # 调整预估时间
                if "estimated_hours" in weekly_schedule:
                    original_hours = weekly_schedule["estimated_hours"]
                    adjusted_hours = original_hours / learning_speed
                    weekly_schedule["estimated_hours"] = max(1, math.ceil(adjusted_hours))
                
                # 调整每日计划
                if "daily_breakdown" in weekly_schedule:
                    for daily_plan in weekly_schedule["daily_breakdown"]:
                        if "estimated_hours" in daily_plan:
                            daily_hours = daily_plan["estimated_hours"]
                            adjusted_daily = daily_hours / learning_speed
                            daily_plan["estimated_hours"] = max(0.5, math.ceil(adjusted_daily * 2) / 2)  # 保留0.5小时精度
        
        return adjusted_schedules
    
    def _adjust_schedules_for_difficulty(self,
                                       schedules: Dict[str, Any],
                                       struggling_nodes: List[str]) -> Dict[str, Any]:
        """为困难节点调整日程"""
        adjusted_schedules = schedules.copy()
        
        # 为困难节点增加额外时间
        extra_time_per_node = 0.5  # 每个困难节点额外0.5小时
        
        if "weekly_schedules" in adjusted_schedules:
            for weekly_schedule in adjusted_schedules["weekly_schedules"]:
                # 检查这周是否有困难节点
                has_struggling_nodes = False
                
                # 这里简化处理，实际中需要更精确的匹配
                if random.random() < 0.3:  # 30%的概率这周有困难节点
                    has_struggling_nodes = True
                
                if has_struggling_nodes:
                    # 增加额外时间
                    if "estimated_hours" in weekly_schedule:
                        weekly_schedule["estimated_hours"] += extra_time_per_node * len(struggling_nodes)
                    
                    # 添加说明
                    weekly_schedule["note"] = f"包含{len(struggling_nodes)}个困难节点的额外练习时间"
        
        return adjusted_schedules
    
    def _generate_adjustment_summary(self,
                                   adjusted_plan: Dict[str, Any],
                                   progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成调整摘要"""
        summary = {
            "adjustment_count": len(adjusted_plan.get("adjustments_made", [])),
            "adjustment_types": [],
            "current_status": {},
            "next_steps": []
        }
        
        # 收集调整类型
        adjustments_made = adjusted_plan.get("adjustments_made", [])
        for adjustment in adjustments_made:
            if isinstance(adjustment, dict):
                summary["adjustment_types"].append(adjustment.get("type", "unknown"))
            elif isinstance(adjustment, str):
                summary["adjustment_types"].append(adjustment.split(" ")[0] if " " in adjustment else adjustment)
        
        # 当前状态
        progress = progress_data.get("overall_progress", 0)
        summary["current_status"] = {
            "progress_percentage": f"{progress*100:.1f}%",
            "adjustment_reason": ", ".join(set(summary["adjustment_types"])),
            "plan_health": "良好" if len(adjustments_made) < 3 else "需要关注"
        }
        
        # 下一步
        if progress < 0.3:
            summary["next_steps"] = ["继续按计划学习", "建立学习习惯", "定期检查进度"]
        elif progress < 0.7:
            summary["next_steps"] = ["保持学习节奏", "加强薄弱环节", "准备中期评估"]
        else:
            summary["next_steps"] = ["加速完成剩余内容", "进行综合复习", "准备最终评估"]
        
        return summary

# ========== 自适应调度器 ==========

class AdaptiveScheduler:
    """
    自适应调度器 - 根据学习情况动态调整学习调度
    """
    
    def __init__(self):
        self.scheduling_strategies = {
            "fixed_schedule": {
                "name": "固定日程",
                "description": "按固定时间表学习",
                "flexibility": "低",
                "suitable_for": ["规律生活", "初学者", "建立习惯"]
            },
            "flexible_schedule": {
                "name": "弹性日程",
                "description": "在时间窗口内灵活安排",
                "flexibility": "中",
                "suitable_for": ["工作繁忙", "时间不定", "中级学习者"]
            },
            "dynamic_schedule": {
                "name": "动态日程",
                "description": "根据状态和进度实时调整",
                "flexibility": "高",
                "suitable_for": ["高级学习者", "自适应学习", "个性化需求"]
            },
            "adaptive_schedule": {
                "name": "自适应日程",
                "description": "基于多因素智能调整",
                "flexibility": "最高",
                "suitable_for": ["复杂目标", "多任务学习", "优化学习体验"]
            }
        }
        
        # 调度因素权重
        self.scheduling_factors = {
            "time_availability": {"weight": 0.3, "description": "时间可用性"},
            "energy_level": {"weight": 0.2, "description": "精力水平"},
            "learning_progress": {"weight": 0.25, "description": "学习进度"},
            "task_difficulty": {"weight": 0.15, "description": "任务难度"},
            "personal_preference": {"weight": 0.1, "description": "个人偏好"}
        }
        
        # 调度历史
        self.scheduling_history = defaultdict(list)
    
    def schedule_learning_sessions(self,
                                  learning_plan: Dict[str, Any],
                                  current_context: Dict[str, Any],
                                  strategy: str = "adaptive_schedule") -> Dict[str, Any]:
        """
        调度学习会话
        
        Args:
            learning_plan: 学习计划
            current_context: 当前上下文（时间、精力等）
            strategy: 调度策略
            
        Returns:
            调度结果
        """
        print(f"⏰ 调度学习会话 (策略: {strategy})")
        
        schedule = {
            "scheduled_at": datetime.now().isoformat(),
            "strategy": strategy,
            "context_analysis": {},
            "scheduled_sessions": [],
            "recommendations": [],
            "flexibility_score": 0.0
        }
        
        # 分析当前上下文
        context_analysis = self._analyze_current_context(current_context)
        schedule["context_analysis"] = context_analysis
        
        # 根据策略调度
        if strategy == "fixed_schedule":
            sessions = self._create_fixed_schedule(learning_plan, context_analysis)
        elif strategy == "flexible_schedule":
            sessions = self._create_flexible_schedule(learning_plan, context_analysis)
        elif strategy == "dynamic_schedule":
            sessions = self._create_dynamic_schedule(learning_plan, context_analysis)
        else:  # adaptive_schedule
            sessions = self._create_adaptive_schedule(learning_plan, context_analysis)
        
        schedule["scheduled_sessions"] = sessions
        
        # 计算灵活性分数
        flexibility_score = self._calculate_flexibility_score(sessions, context_analysis)
        schedule["flexibility_score"] = flexibility_score
        
        # 生成调度建议
        recommendations = self._generate_scheduling_recommendations(
            schedule, learning_plan, context_analysis
        )
        schedule["recommendations"] = recommendations
        
        # 记录调度历史
        self._record_scheduling_history(learning_plan.get("goal_id", "unknown"), schedule)
        
        print(f"✅ 学习会话调度完成: {len(sessions)}个会话，灵活性分数: {flexibility_score:.2f}")
        return schedule
    
    def reschedule_based_on_feedback(self,
                                   original_schedule: Dict[str, Any],
                                   session_feedback: Dict[str, Any],
                                   current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于反馈重新调度
        
        Args:
            original_schedule: 原始调度
            session_feedback: 会话反馈
            current_context: 当前上下文
            
        Returns:
            重新调度结果
        """
        print(f"🔄 基于反馈重新调度")
        
        reschedule = original_schedule.copy()
        reschedule["rescheduled_at"] = datetime.now().isoformat()
        reschedule["original_schedule_id"] = original_schedule.get("scheduled_at", "")
        
        # 分析反馈
        feedback_analysis = self._analyze_session_feedback(session_feedback)
        reschedule["feedback_analysis"] = feedback_analysis
        
        # 提取需要调整的信息
        completed_sessions = session_feedback.get("completed_sessions", [])
        canceled_sessions = session_feedback.get("canceled_sessions", [])
        session_ratings = session_feedback.get("session_ratings", {})
        
        # 调整会话
        original_sessions = original_schedule.get("scheduled_sessions", [])
        adjusted_sessions = []
        
        for session in original_sessions:
            session_id = session.get("session_id", "")
            
            # 如果会话已完成或取消，跳过
            if session_id in completed_sessions or session_id in canceled_sessions:
                continue
            
            # 如果有评分，根据评分调整
            if session_id in session_ratings:
                rating = session_ratings[session_id]
                adjusted_session = self._adjust_session_based_on_rating(session, rating)
                adjusted_sessions.append(adjusted_session)
            else:
                # 没有评分，保持原样或轻微调整
                adjusted_sessions.append(session)
        
        # 添加新会话（如果需要）
        if feedback_analysis.get("need_more_sessions", False):
            additional_sessions = self._create_additional_sessions(
                feedback_analysis, current_context, len(adjusted_sessions)
            )
            adjusted_sessions.extend(additional_sessions)
        
        reschedule["scheduled_sessions"] = adjusted_sessions
        reschedule["adjustments_made"] = len(original_sessions) - len(adjusted_sessions) + len(additional_sessions)
        
        # 更新上下文分析
        reschedule["context_analysis"] = self._analyze_current_context(current_context)
        
        print(f"✅ 重新调度完成: {reschedule['adjustments_made']}项调整")
        return reschedule
    
    def optimize_schedule_for_goals(self,
                                  goals: List[LearningGoal],
                                  available_time: Dict[str, int],  # 每周可用时间（小时）
                                  priority_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        为多个目标优化调度
        
        Args:
            goals: 学习目标列表
            available_time: 可用时间
            priority_weights: 优先级权重
            
        Returns:
            优化调度
        """
        print(f"⚡ 为{len(goals)}个目标优化调度")
        
        optimization = {
            "optimized_at": datetime.now().isoformat(),
            "goals_count": len(goals),
            "available_time": available_time,
            "time_allocation": {},
            "conflict_resolution": [],
            "optimized_schedule": {}
        }
        
        if not goals:
            optimization["message"] = "没有需要优化的目标"
            return optimization
        
        # 如果没有提供优先级权重，自动计算
        if priority_weights is None:
            priority_weights = self._calculate_priority_weights(goals)
        
        optimization["priority_weights"] = priority_weights
        
        # 计算每个目标的时间分配
        time_allocation = self._allocate_time_to_goals(goals, available_time, priority_weights)
        optimization["time_allocation"] = time_allocation
        
        # 解决时间冲突
        conflicts = self._identify_scheduling_conflicts(goals, time_allocation)
        optimization["conflict_resolution"] = conflicts
        
        # 创建优化后的日程
        optimized_schedule = self._create_optimized_schedule(goals, time_allocation, conflicts)
        optimization["optimized_schedule"] = optimized_schedule
        
        # 计算优化效果
        optimization["optimization_metrics"] = self._calculate_optimization_metrics(
            goals, time_allocation, optimized_schedule
        )
        
        print(f"✅ 多目标优化完成: {len(time_allocation)}个时间分配，解决{len(conflicts)}个冲突")
        return optimization
    
    def _analyze_current_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析当前上下文"""
        analysis = {
            "time_analysis": {},
            "energy_analysis": {},
            "focus_analysis": {},
            "readiness_score": 0.0
        }
        
        # 时间分析
        current_hour = datetime.now().hour
        current_weekday = datetime.now().weekday()  # 0=周一, 6=周日
        
        time_analysis = {
            "current_hour": current_hour,
            "current_weekday": current_weekday,
            "time_category": self._categorize_time(current_hour, current_weekday),
            "available_minutes": context.get("available_minutes", 60)
        }
        analysis["time_analysis"] = time_analysis
        
        # 精力分析
        energy_level = context.get("energy_level", 0.5)
        energy_trend = context.get("energy_trend", "stable")
        
        energy_analysis = {
            "energy_level": energy_level,
            "energy_trend": energy_trend,
            "suggested_activity_intensity": self._suggest_activity_intensity(energy_level)
        }
        analysis["energy_analysis"] = energy_analysis
        
        # 专注度分析
        focus_level = context.get("focus_level", 0.5)
        distractions = context.get("distractions", [])
        
        focus_analysis = {
            "focus_level": focus_level,
            "distraction_count": len(distractions),
            "distraction_types": list(set(distractions)),
            "suggested_focus_duration": self._suggest_focus_duration(focus_level)
        }
        analysis["focus_analysis"] = focus_analysis
        
        # 准备度分数
        readiness_score = self._calculate_readiness_score(
            time_analysis, energy_analysis, focus_analysis
        )
        analysis["readiness_score"] = readiness_score
        
        return analysis
    
    def _categorize_time(self, hour: int, weekday: int) -> str:
        """分类时间"""
        if 6 <= hour < 9:
            return "清晨"
        elif 9 <= hour < 12:
            return "上午"
        elif 12 <= hour < 14:
            return "午间"
        elif 14 <= hour < 18:
            return "下午"
        elif 18 <= hour < 22:
            return "晚间"
        else:
            return "深夜"
    
    def _suggest_activity_intensity(self, energy_level: float) -> str:
        """建议活动强度"""
        if energy_level > 0.7:
            return "高强度"
        elif energy_level > 0.4:
            return "中等强度"
        else:
            return "低强度"
    
    def _suggest_focus_duration(self, focus_level: float) -> int:
        """建议专注时长（分钟）"""
        if focus_level > 0.7:
            return 60
        elif focus_level > 0.4:
            return 45
        else:
            return 25
    
    def _calculate_readiness_score(self,
                                 time_analysis: Dict[str, Any],
                                 energy_analysis: Dict[str, Any],
                                 focus_analysis: Dict[str, Any]) -> float:
        """计算准备度分数"""
        factors = []
        
        # 时间因子
        time_category = time_analysis.get("time_category", "")
        time_factors = {
            "清晨": 0.8,
            "上午": 0.9,
            "午间": 0.6,
            "下午": 0.7,
            "晚间": 0.8,
            "深夜": 0.4
        }
        factors.append(time_factors.get(time_category, 0.5))
        
        # 精力因子
        energy_level = energy_analysis.get("energy_level", 0.5)
        factors.append(energy_level)
        
        # 专注因子
        focus_level = focus_analysis.get("focus_level", 0.5)
        factors.append(focus_level)
        
        # 可用时间因子
        available_minutes = time_analysis.get("available_minutes", 60)
        time_factor = min(available_minutes / 120, 1.0)  # 120分钟为理想值
        factors.append(time_factor)
        
        # 计算平均分
        if factors:
            return sum(factors) / len(factors)
        else:
            return 0.5
    
    def _create_fixed_schedule(self,
                              learning_plan: Dict[str, Any],
                              context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建固定日程"""
        sessions = []
        
        # 固定日程：每天固定时间学习
        daily_templates = learning_plan.get("schedules", {}).get("daily_templates", [])
        
        if not daily_templates:
            # 如果没有模板，创建基本会话
            base_session = {
                "session_id": generate_id("fixed_session_"),
                "type": "fixed",
                "scheduled_time": "19:00-20:00",
                "duration_minutes": 60,
                "flexibility": "low",
                "recommended_activities": ["系统学习", "完成练习"],
                "priority": "medium"
            }
            sessions.append(base_session)
        else:
            # 使用模板创建会话
            for i, template in enumerate(daily_templates[:3]):  # 最多3个模板
                session = {
                    "session_id": generate_id(f"fixed_session_{i}_"),
                    "type": "fixed",
                    "template_name": template.get("name", ""),
                    "scheduled_time": template.get("time_blocks", [{}])[0].get("time", "19:00-20:00"),
                    "duration_minutes": template.get("total_hours", 1) * 60,
                    "flexibility": "low",
                    "recommended_activities": template.get("suitable_for", ["通用学习"]),
                    "priority": "medium"
                }
                sessions.append(session)
        
        return sessions
    
    def _create_flexible_schedule(self,
                                learning_plan: Dict[str, Any],
                                context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建弹性日程"""
        sessions = []
        
        # 弹性日程：在时间窗口内灵活安排
        time_windows = [
            {"window": "09:00-12:00", "duration_minutes": 90, "priority": "high"},
            {"window": "14:00-17:00", "duration_minutes": 90, "priority": "medium"},
            {"window": "19:00-22:00", "duration_minutes": 60, "priority": "medium"}
        ]
        
        for i, window in enumerate(time_windows):
            session = {
                "session_id": generate_id(f"flexible_session_{i}_"),
                "type": "flexible",
                "time_window": window["window"],
                "duration_minutes": window["duration_minutes"],
                "flexibility": "medium",
                "suggested_time": self._suggest_time_in_window(window["window"]),
                "priority": window["priority"],
                "conditions": [
                    "在时间窗口内完成",
                    "根据精力选择具体时间",
                    "可以调整时长±30分钟"
                ]
            }
            sessions.append(session)
        
        return sessions
    
    def _create_dynamic_schedule(self,
                               learning_plan: Dict[str, Any],
                               context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建动态日程"""
        sessions = []
        
        # 动态日程：根据当前状态实时调整
        readiness_score = context_analysis.get("readiness_score", 0.5)
        available_minutes = context_analysis.get("time_analysis", {}).get("available_minutes", 60)
        
        # 确定会话类型和参数
        if readiness_score > 0.7 and available_minutes >= 90:
            # 状态好，时间长：深度学习会话
            session_type = "deep_learning"
            duration = 90
            intensity = "high"
        elif readiness_score > 0.5 and available_minutes >= 60:
            # 状态中等，时间中等：常规学习会话
            session_type = "regular_learning"
            duration = 60
            intensity = "medium"
        elif readiness_score > 0.3 and available_minutes >= 30:
            # 状态一般，时间短：复习或轻量学习
            session_type = "light_review"
            duration = 30
            intensity = "low"
        else:
            # 状态差或时间少：微学习
            session_type = "micro_learning"
            duration = 15
            intensity = "very_low"
        
        # 创建会话
        session = {
            "session_id": generate_id("dynamic_session_"),
            "type": "dynamic",
            "session_type": session_type,
            "duration_minutes": duration,
            "intensity": intensity,
            "flexibility": "high",
            "suggested_time": "立即开始",
            "priority": "high" if session_type == "deep_learning" else "medium",
            "adaptation_rules": [
                f"根据准备度分数({readiness_score:.2f})调整",
                f"根据可用时间({available_minutes}分钟)调整",
                "可以随时中断和恢复"
            ]
        }
        
        sessions.append(session)
        
        # 如果状态和时间允许，安排第二个会话
        if readiness_score > 0.6 and available_minutes >= duration + 30:
            # 安排一个补充会话
            supplemental_session = {
                "session_id": generate_id("dynamic_supplemental_"),
                "type": "dynamic",
                "session_type": "supplemental_practice",
                "duration_minutes": 30,
                "intensity": "medium",
                "flexibility": "high",
                "suggested_time": f"{duration}分钟后",
                "priority": "medium",
                "conditions": ["完成主会话后执行", "根据剩余精力调整"]
            }
            sessions.append(supplemental_session)
        
        return sessions
    
    def _create_adaptive_schedule(self,
                                learning_plan: Dict[str, Any],
                                context_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建自适应日程"""
        sessions = []
        
        # 自适应日程：基于多因素智能调整
        readiness_score = context_analysis.get("readiness_score", 0.5)
        energy_level = context_analysis.get("energy_analysis", {}).get("energy_level", 0.5)
        focus_level = context_analysis.get("focus_analysis", {}).get("focus_level", 0.5)
        available_minutes = context_analysis.get("time_analysis", {}).get("available_minutes", 60)
        
        # 计算多因素综合分数
        factor_scores = [
            readiness_score * 0.4,
            energy_level * 0.3,
            focus_level * 0.2,
            min(available_minutes / 120, 1.0) * 0.1
        ]
        composite_score = sum(factor_scores)
        
        # 确定最佳学习类型
        learning_types = self._determine_optimal_learning_types(
            composite_score, energy_level, focus_level, available_minutes
        )
        
        # 创建自适应会话
        for i, learning_type in enumerate(learning_types[:2]):  # 最多2种类型
            # 为每种类型计算最佳参数
            session_params = self._calculate_session_parameters(
                learning_type, composite_score, available_minutes
            )
            
            session = {
                "session_id": generate_id(f"adaptive_session_{i}_"),
                "type": "adaptive",
                "learning_type": learning_type,
                "composite_score": composite_score,
                "duration_minutes": session_params["duration"],
                "intensity": session_params["intensity"],
                "flexibility": "highest",
                "suggested_time": session_params["suggested_time"],
                "priority": session_params["priority"],
                "adaptation_factors": {
                    "readiness_score": readiness_score,
                    "energy_level": energy_level,
                    "focus_level": focus_level,
                    "available_minutes": available_minutes
                },
                "adjustment_rules": [
                    "根据实时状态动态调整",
                    "可以随时切换学习类型",
                    "支持中断和继续"
                ]
            }
            sessions.append(session)
        
        return sessions
    
    def _determine_optimal_learning_types(self,
                                        composite_score: float,
                                        energy_level: float,
                                        focus_level: float,
                                        available_minutes: int) -> List[str]:
        """确定最佳学习类型"""
        learning_types = []
        
        # 基于综合分数
        if composite_score > 0.8:
            learning_types.extend(["深度学习", "复杂问题解决", "创新思考"])
        elif composite_score > 0.6:
            learning_types.extend(["系统学习", "实践练习", "技能训练"])
        elif composite_score > 0.4:
            learning_types.extend(["知识复习", "概念理解", "基础练习"])
        else:
            learning_types.extend(["微学习", "记忆巩固", "轻量阅读"])
        
        # 基于精力水平调整
        if energy_level < 0.4:
            # 精力低，优先轻量学习
            learning_types = [lt for lt in learning_types if lt not in ["深度学习", "复杂问题解决"]]
            learning_types.insert(0, "轻量学习")
        
        # 基于专注度调整
        if focus_level < 0.4:
            # 专注度低，优先简单任务
            learning_types = [lt for lt in learning_types if lt not in ["系统学习", "创新思考"]]
            learning_types.insert(0, "简单任务")
        
        # 基于可用时间调整
        if available_minutes < 30:
            # 时间短，优先高效学习
            learning_types = [lt for lt in learning_types if lt in ["微学习", "记忆巩固", "轻量阅读"]]
        
        return list(set(learning_types))  # 去重
    
    def _calculate_session_parameters(self,
                                    learning_type: str,
                                    composite_score: float,
                                    available_minutes: int) -> Dict[str, Any]:
        """计算会话参数"""
        # 默认参数
        params = {
            "duration": 45,
            "intensity": "medium",
            "suggested_time": "尽快开始",
            "priority": "medium"
        }
        
        # 根据学习类型调整
        type_configs = {
            "深度学习": {"duration": 90, "intensity": "high", "priority": "high"},
            "系统学习": {"duration": 60, "intensity": "medium", "priority": "high"},
            "实践练习": {"duration": 45, "intensity": "medium", "priority": "medium"},
            "知识复习": {"duration": 30, "intensity": "low", "priority": "medium"},
            "微学习": {"duration": 15, "intensity": "very_low", "priority": "low"}
        }
        
        for key, config in type_configs.items():
            if key in learning_type:
                params.update(config)
                break
        
        # 根据综合分数调整
        if composite_score > 0.8:
            params["duration"] = min(params["duration"] + 30, available_minutes)
        elif composite_score < 0.4:
            params["duration"] = max(15, params["duration"] - 15)
        
        # 确保不超过可用时间
        params["duration"] = min(params["duration"], available_minutes)
        
        # 建议时间
        if params["duration"] >= 60:
            params["suggested_time"] = "安排专门时间段"
        elif params["duration"] >= 30:
            params["suggested_time"] = "利用碎片时间"
        else:
            params["suggested_time"] = "随时可以开始"
        
        return params
    
    def _suggest_time_in_window(self, time_window: str) -> str:
        """在时间窗口内建议具体时间"""
        # 简单实现：建议窗口中间时间
        if "-" in time_window:
            start_str, end_str = time_window.split("-")
            
            # 转换为分钟
            try:
                start_hour, start_minute = map(int, start_str.split(":"))
                end_hour, end_minute = map(int, end_str.split(":"))
                
                start_total = start_hour * 60 + start_minute
                end_total = end_hour * 60 + end_minute
                
                # 计算中间时间
                middle_total = (start_total + end_total) // 2
                middle_hour = middle_total // 60
                middle_minute = middle_total % 60
                
                return f"{middle_hour:02d}:{middle_minute:02d}"
            except:
                pass
        
        return time_window.split("-")[0]  # 返回开始时间
    
    def _calculate_flexibility_score(self,
                                   sessions: List[Dict[str, Any]],
                                   context_analysis: Dict[str, Any]) -> float:
        """计算灵活性分数"""
        if not sessions:
            return 0.0
        
        flexibility_scores = []
        
        for session in sessions:
            flexibility = session.get("flexibility", "medium")
            
            if flexibility == "low":
                flexibility_scores.append(0.3)
            elif flexibility == "medium":
                flexibility_scores.append(0.6)
            elif flexibility == "high":
                flexibility_scores.append(0.8)
            elif flexibility == "highest":
                flexibility_scores.append(1.0)
            else:
                flexibility_scores.append(0.5)
        
        # 考虑上下文灵活性
        readiness_score = context_analysis.get("readiness_score", 0.5)
        context_flexibility = readiness_score * 0.3 + 0.7  # 准备度越高，灵活性越高
        
        # 综合分数
        if flexibility_scores:
            avg_session_flexibility = sum(flexibility_scores) / len(flexibility_scores)
            return (avg_session_flexibility * 0.7 + context_flexibility * 0.3)
        else:
            return context_flexibility
    
    def _generate_scheduling_recommendations(self,
                                           schedule: Dict[str, Any],
                                           learning_plan: Dict[str, Any],
                                           context_analysis: Dict[str, Any]) -> List[str]:
        """生成调度建议"""
        recommendations = []
        
        strategy = schedule.get("strategy", "")
        flexibility_score = schedule.get("flexibility_score", 0.5)
        session_count = len(schedule.get("scheduled_sessions", []))
        readiness_score = context_analysis.get("readiness_score", 0.5)
        
        # 基于策略的建议
        if strategy == "fixed_schedule":
            recommendations.append("固定日程适合建立学习习惯，请严格遵守时间")
        elif strategy == "flexible_schedule":
            recommendations.append("弹性日程提供了灵活性，请在时间窗口内完成学习")
        elif strategy == "dynamic_schedule":
            recommendations.append("动态日程根据状态调整，请关注自身状态变化")
        elif strategy == "adaptive_schedule":
            recommendations.append("自适应日程智能优化，系统会自动调整最佳学习安排")
        
        # 基于灵活性分数的建议
        if flexibility_score < 0.4:
            recommendations.append("当前日程灵活性较低，请确保按计划执行")
        elif flexibility_score > 0.7:
            recommendations.append("当前日程灵活性高，可以根据状态调整学习安排")
        
        # 基于会话数量的建议
        if session_count > 3:
            recommendations.append(f"安排了{session_count}个学习会话，建议合理分配精力")
        elif session_count == 1:
            recommendations.append("安排了一个主要学习会话，请专注于完成它")
        
        # 基于准备度的建议
        if readiness_score < 0.4:
            recommendations.append("当前准备度较低，建议从简单任务开始")
        elif readiness_score > 0.8:
            recommendations.append("当前准备度很高，适合进行深度学习和复杂任务")
        
        return recommendations
    
    def _analyze_session_feedback(self, session_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """分析会话反馈"""
        analysis = {
            "completion_rate": 0.0,
            "average_rating": 0.0,
            "common_issues": [],
            "success_factors": [],
            "need_more_sessions": False
        }
        
        completed_sessions = session_feedback.get("completed_sessions", [])
        total_sessions = session_feedback.get("total_sessions", 0)
        session_ratings = session_feedback.get("session_ratings", {})
        issues = session_feedback.get("issues", [])
        
        # 完成率
        if total_sessions > 0:
            analysis["completion_rate"] = len(completed_sessions) / total_sessions
        
        # 平均评分
        if session_ratings:
            ratings = list(session_ratings.values())
            if ratings:
                analysis["average_rating"] = statistics.mean(ratings)
        
        # 常见问题
        if issues:
            issue_counts = {}
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            analysis["common_issues"] = [issue for issue, count in common_issues]
        
        # 成功因素（从正面反馈中提取）
        positive_feedback = session_feedback.get("positive_feedback", [])
        if positive_feedback:
            analysis["success_factors"] = positive_feedback[:3]
        
        # 是否需要更多会话
        if analysis["completion_rate"] < 0.5 or analysis["average_rating"] < 0.6:
            analysis["need_more_sessions"] = True
        
        return analysis
    
    def _adjust_session_based_on_rating(self,
                                      session: Dict[str, Any],
                                      rating: float) -> Dict[str, Any]:
        """基于评分调整会话"""
        adjusted_session = session.copy()
        
        # 根据评分调整参数
        if rating < 0.4:
            # 评分低：简化会话
            original_duration = adjusted_session.get("duration_minutes", 60)
            adjusted_session["duration_minutes"] = max(15, original_duration * 0.5)
            adjusted_session["intensity"] = "low"
            adjusted_session["note"] = "根据低评分简化了会话"
        elif rating > 0.8:
            # 评分高：保持或加强
            original_duration = adjusted_session.get("duration_minutes", 60)
            adjusted_session["duration_minutes"] = min(120, original_duration * 1.2)
            adjusted_session["intensity"] = "high" if session.get("intensity") != "very_low" else "medium"
            adjusted_session["note"] = "根据高评分加强了会话"
        
        return adjusted_session
    
    def _create_additional_sessions(self,
                                  feedback_analysis: Dict[str, Any],
                                  current_context: Dict[str, Any],
                                  existing_session_count: int) -> List[Dict[str, Any]]:
        """创建额外会话"""
        additional_sessions = []
        
        # 根据反馈分析确定需要多少额外会话
        completion_rate = feedback_analysis.get("completion_rate", 0.0)
        need_more = feedback_analysis.get("need_more_sessions", False)
        
        if not need_more:
            return additional_sessions
        
        # 计算需要补充的会话数量
        if completion_rate < 0.3:
            additional_count = 3
        elif completion_rate < 0.6:
            additional_count = 2
        else:
            additional_count = 1
        
        # 限制总会话数量
        max_total_sessions = 5
        additional_count = min(additional_count, max_total_sessions - existing_session_count)
        
        if additional_count <= 0:
            return additional_sessions
        
        # 创建补充会话
        for i in range(additional_count):
            session = {
                "session_id": generate_id(f"additional_session_{i}_"),
                "type": "supplemental",
                "purpose": "补充学习",
                "duration_minutes": 30,
                "intensity": "medium",
                "flexibility": "high",
                "suggested_time": "利用碎片时间",
                "priority": "low",
                "conditions": ["在主要会话完成后进行", "根据时间灵活安排"]
            }
            additional_sessions.append(session)
        
        return additional_sessions
    
    def _calculate_priority_weights(self, goals: List[LearningGoal]) -> Dict[str, float]:
        """计算优先级权重"""
        if not goals:
            return {}
        
        # 基于目标属性计算权重
        weights = {}
        total_weight = 0.0
        
        for goal in goals:
            weight = 0.0
            
            # 基于规模
            scale_weights = {
                GoalScale.MICRO: 1.0,
                GoalScale.SMALL: 2.0,
                GoalScale.MEDIUM: 3.0,
                GoalScale.LARGE: 4.0,
                GoalScale.MASSIVE: 5.0
            }
            weight += scale_weights.get(goal.scale, 2.0)
            
            # 基于优先级
            weight += goal.priority / 2.0  # 优先级1-10，转换为0.5-5
            
            # 基于复杂度
            weight += goal.complexity * 2.0
            
            # 基于时间紧迫性
            if goal.deadline:
                try:
                    deadline_date = datetime.fromisoformat(goal.deadline)
                    days_until_deadline = (deadline_date - datetime.now()).days
                    if days_until_deadline > 0:
                        time_factor = 10.0 / days_until_deadline  # 越近权重越高
                        weight += min(time_factor, 5.0)
                except:
                    pass
            
            weights[goal.id] = weight
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            for goal_id in weights:
                weights[goal_id] = weights[goal_id] / total_weight
        
        return weights
    
    def _allocate_time_to_goals(self,
                              goals: List[LearningGoal],
                              available_time: Dict[str, int],
                              priority_weights: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """分配时间给各个目标"""
        time_allocation = {}
        
        # 总可用时间
        total_available = sum(available_time.values())
        
        if total_available <= 0:
            return time_allocation
        
        # 为每个目标分配时间
        for goal in goals:
            goal_id = goal.id
            weight = priority_weights.get(goal_id, 0.0)
            
            # 基于权重分配时间
            allocated_hours = total_available * weight
            
            # 考虑目标自身的时间需求
            estimated_time = self._estimate_goal_time(goal)
            if estimated_time > 0:
                # 如果预估时间小于分配时间，使用预估时间
                allocated_hours = min(allocated_hours, estimated_time)
            
            # 分配到具体时间（简化处理）
            weekly_allocation = {}
            days = list(available_time.keys())
            
            if days:
                # 平均分配到每天
                hours_per_day = allocated_hours / len(days)
                for day in days:
                    weekly_allocation[day] = hours_per_day
            
            time_allocation[goal_id] = {
                "goal_name": goal.description,
                "priority_weight": weight,
                "allocated_hours_per_week": allocated_hours,
                "weekly_allocation": weekly_allocation,
                "estimated_completion_weeks": self._estimate_completion_weeks(goal, allocated_hours)
            }
        
        return time_allocation
    
    def _estimate_goal_time(self, goal: LearningGoal) -> float:
        """估算目标所需时间（小时）"""
        # 使用时间模型估算
        try:
            time_model = TimeEstimationModel()
            minutes = time_model.estimate_for_goal(goal)
            return minutes / 60
        except:
            # 备用估算
            return goal.target_knowledge_count * 0.5  # 每个知识点0.5小时
    
    def _estimate_completion_weeks(self, goal: LearningGoal, weekly_hours: float) -> float:
        """估算完成周数"""
        if weekly_hours <= 0:
            return float('inf')
        
        total_hours = self._estimate_goal_time(goal)
        return math.ceil(total_hours / weekly_hours)
    
    def _identify_scheduling_conflicts(self,
                                     goals: List[LearningGoal],
                                     time_allocation: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别调度冲突"""
        conflicts = []
        
        # 简化冲突检测：检查总时间分配是否合理
        total_allocated = 0
        for allocation in time_allocation.values():
            total_allocated += allocation.get("allocated_hours_per_week", 0)
        
        # 假设每周最多学习40小时
        max_reasonable_hours = 40
        
        if total_allocated > max_reasonable_hours:
            conflicts.append({
                "type": "overtime",
                "description": f"总分配时间({total_allocated:.1f}小时)超过合理上限({max_reasonable_hours}小时)",
                "severity": "high",
                "solutions": [
                    "减少某些目标的时间分配",
                    "延长学习周期",
                    "调整目标优先级"
                ]
            })
        
        # 检查目标间的时间冲突（简化）
        if len(goals) > 3:
            # 如果目标太多，可能存在时间冲突
            avg_hours_per_goal = total_allocated / len(goals)
            if avg_hours_per_goal < 2:
                conflicts.append({
                    "type": "insufficient_time_per_goal",
                    "description": f"每个目标平均只有{avg_hours_per_goal:.1f}小时，可能不足",
                    "severity": "medium",
                    "solutions": [
                        "聚焦少数重要目标",
                        "增加总学习时间",
                        "提高学习效率"
                    ]
                })
        
        return conflicts
    
    def _create_optimized_schedule(self,
                                 goals: List[LearningGoal],
                                 time_allocation: Dict[str, Dict[str, Any]],
                                 conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建优化后的日程"""
        optimized_schedule = {
            "daily_schedule": {},
            "weekly_overview": {},
            "conflict_resolutions_applied": []
        }
        
        # 应用冲突解决方案
        for conflict in conflicts:
            if conflict.get("type") == "overtime":
                # 减少时间分配
                self._apply_overtime_solution(time_allocation)
                optimized_schedule["conflict_resolutions_applied"].append("减少总时间分配")
        
        # 创建每日日程
        days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        
        for day in days:
            daily_plan = {
                "total_hours": 0,
                "goal_sessions": [],
                "recommended_time_slots": []
            }
            
            # 为每个目标分配当天的学习时间
            for goal_id, allocation in time_allocation.items():
                daily_hours = allocation.get("weekly_allocation", {}).get(day, 0)
                if daily_hours > 0:
                    daily_plan["goal_sessions"].append({
                        "goal_id": goal_id,
                        "goal_name": allocation.get("goal_name", ""),
                        "allocated_hours": daily_hours,
                    })
                    daily_plan["total_hours"] += daily_hours

            daily_plans[day] = daily_plan

        optimized_schedule = {
            "daily_schedule": daily_plans,
            "conflict_resolutions_applied": [],
        }

        # 创建周度概览（复用下方已有逻辑）
        # 创建每日计划模板（第二次实现，补全断层）
        days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        daily_plans = {}
        
        for day in days:
            daily_plan = {
                "total_hours": 0,
                "goal_sessions": [],
                "recommended_time_slots": [],
                "flexibility": "medium"
            }
            
            # 为每个目标分配当天的学习时间
            for goal_id, allocation in time_allocation.items():
                daily_hours = allocation.get("weekly_allocation", {}).get(day, 0)
                if daily_hours > 0:
                    goal_session = {
                        "goal_id": goal_id,
                        "goal_name": allocation.get("goal_name", ""),
                        "allocated_hours": daily_hours,
                        "priority_weight": allocation.get("priority_weight", 0.0),
                        "suggested_time": self._suggest_goal_time(day, goal_id, allocation)
                    }
                    daily_plan["goal_sessions"].append(goal_session)
                    daily_plan["total_hours"] += daily_hours
            
            # 推荐时间段
            if daily_plan["total_hours"] > 0:
                if daily_plan["total_hours"] <= 2:
                    daily_plan["recommended_time_slots"] = [
                        {"time": "19:00-21:00", "intensity": "medium"}
                    ]
                elif daily_plan["total_hours"] <= 4:
                    daily_plan["recommended_time_slots"] = [
                        {"time": "09:00-11:00", "intensity": "high"},
                        {"time": "19:00-21:00", "intensity": "medium"}
                    ]
                else:
                    daily_plan["recommended_time_slots"] = [
                        {"time": "09:00-12:00", "intensity": "high"},
                        {"time": "14:00-16:00", "intensity": "medium"},
                        {"time": "19:00-21:00", "intensity": "low"}
                    ]
            
            daily_plans[day] = daily_plan
        
        optimized_schedule["daily_schedule"] = daily_plans
        
        # 创建周度概览
        weekly_overview = {
            "total_goals": len(goals),
            "total_weekly_hours": sum(alloc.get("allocated_hours_per_week", 0) for alloc in time_allocation.values()),
            "goal_distribution": {alloc.get("goal_name", ""): alloc.get("allocated_hours_per_week", 0) 
                                 for alloc in time_allocation.values()},
            "daily_breakdown": {day: daily_plans[day]["total_hours"] for day in days if day in daily_plans},
            "busiest_day": max(daily_plans.items(), key=lambda x: x[1]["total_hours"])[0] if daily_plans else "无"
        }
        optimized_schedule["weekly_overview"] = weekly_overview
        
        # 记录应用的冲突解决方案
        for conflict in conflicts:
            if conflict.get("solutions_applied"):
                optimized_schedule["conflict_resolutions_applied"].extend(conflict["solutions_applied"])
        
        return optimized_schedule
    
    def _suggest_goal_time(self, day: str, goal_id: str, allocation: Dict[str, Any]) -> str:
        """为目标推荐学习时间"""
        # 基于优先级和目标特性推荐时间
        priority = allocation.get("priority_weight", 0.0)
        
        if priority > 0.7:
            return "上午高效时段"
        elif priority > 0.4:
            return "下午专注时段"
        else:
            return "晚间灵活时段"
    
    def _apply_overtime_solution(self, time_allocation: Dict[str, Dict[str, Any]]) -> None:
        """应用超时解决方案"""
        # 减少所有目标的时间分配，优先保护高优先级目标
        total_allocated = sum(alloc.get("allocated_hours_per_week", 0) for alloc in time_allocation.values())
        max_hours = 40  # 每周最多40小时
        
        if total_allocated <= max_hours:
            return
        
        # 计算需要减少的比例
        reduction_factor = max_hours / total_allocated
        
        # 按优先级调整：低优先级目标减少更多
        for goal_id, allocation in time_allocation.items():
            original_hours = allocation.get("allocated_hours_per_week", 0)
            priority = allocation.get("priority_weight", 0.5)
            
            # 高优先级目标减少较少
            if priority > 0.7:
                adjustment_factor = 0.9  # 只减少10%
            elif priority > 0.4:
                adjustment_factor = 0.8  # 减少20%
            else:
                adjustment_factor = 0.6  # 减少40%
            
            # 应用调整
            new_hours = original_hours * reduction_factor * adjustment_factor
            allocation["allocated_hours_per_week"] = max(1, new_hours)  # 至少1小时
        
    def _calculate_optimization_metrics(self,
                                      goals: List[LearningGoal],
                                      time_allocation: Dict[str, Dict[str, Any]],
                                      optimized_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """计算优化指标"""
        metrics = {
            "efficiency_score": 0.0,
            "balance_score": 0.0,
            "feasibility_score": 0.0,
            "satisfaction_score": 0.0,
            "improvements": []
        }
        
        # 计算效率分数（时间分配与优先级匹配度）
        priority_alignment = 0.0
        for goal_id, allocation in time_allocation.items():
            priority = allocation.get("priority_weight", 0.0)
            time_ratio = allocation.get("allocated_hours_per_week", 0) / 40  # 假设40小时为上限
            
            # 理想情况：时间分配与优先级成正比
            alignment = 1.0 - abs(priority - time_ratio)
            priority_alignment += alignment * priority
        
        if time_allocation:
            priority_alignment /= len(time_allocation)
        
        metrics["efficiency_score"] = priority_alignment
        
        # 计算平衡分数（每日时间分布均衡）
        daily_schedule = optimized_schedule.get("daily_schedule", {})
        daily_hours = [plan.get("total_hours", 0) for plan in daily_schedule.values()]
        
        if daily_hours:
            avg_hours = sum(daily_hours) / len(daily_hours)
            variance = sum((h - avg_hours) ** 2 for h in daily_hours) / len(daily_hours)
            balance_score = 1.0 / (1.0 + variance)  # 方差越小，分数越高
            metrics["balance_score"] = min(balance_score, 1.0)
        
        # 计算可行性分数（每日不超过合理上限）
        feasible_days = 0
        for day, plan in daily_schedule.items():
            total_hours = plan.get("total_hours", 0)
            if total_hours <= 8:  # 假设每天最多8小时学习
                feasible_days += 1
        
        if daily_schedule:
            metrics["feasibility_score"] = feasible_days / len(daily_schedule)
        
        # 计算满意度分数（综合指标）
        metrics["satisfaction_score"] = (
            metrics["efficiency_score"] * 0.4 +
            metrics["balance_score"] * 0.3 +
            metrics["feasibility_score"] * 0.3
        )
        
        # 改进建议
        if metrics["efficiency_score"] < 0.7:
            metrics["improvements"].append("优化时间分配以更好地匹配目标优先级")
        
        if metrics["balance_score"] < 0.6:
            metrics["improvements"].append("调整日程以使每日学习时间更均衡")
        
        if metrics["feasibility_score"] < 0.8:
            metrics["improvements"].append("减少某些日的学习量以提高可行性")
        
        return metrics
    
    def _record_scheduling_history(self, goal_id: str, schedule: Dict[str, Any]) -> None:
        """记录调度历史"""
        history_entry = {
            "scheduled_at": schedule.get("scheduled_at", datetime.now().isoformat()),
            "strategy": schedule.get("strategy", ""),
            "session_count": len(schedule.get("scheduled_sessions", [])),
            "flexibility_score": schedule.get("flexibility_score", 0.0)
        }
        
        self.scheduling_history[goal_id].append(history_entry)
        
        # 限制历史记录长度
        if len(self.scheduling_history[goal_id]) > 20:
            self.scheduling_history[goal_id] = self.scheduling_history[goal_id][-20:]

# ========== 进度监控器 ==========

class ProgressMonitor:
    """
    进度监控器 - 实时监控学习进度并提供反馈
    """
    
    def __init__(self):
        self.monitoring_strategies = {
            "continuous": {
                "name": "持续监控",
                "description": "实时监控学习进度",
                "update_frequency": "real_time",
                "granularity": "fine"
            },
            "periodic": {
                "name": "定期监控",
                "description": "按固定时间间隔监控",
                "update_frequency": "daily",
                "granularity": "medium"
            },
            "milestone": {
                "name": "里程碑监控",
                "description": "在关键里程碑检查进度",
                "update_frequency": "milestone_based",
                "granularity": "coarse"
            },
            "adaptive": {
                "name": "自适应监控",
                "description": "根据学习情况调整监控频率",
                "update_frequency": "adaptive",
                "granularity": "variable"
            }
        }
        
        # 进度指标配置
        self.progress_metrics = {
            "completion_rate": {
                "name": "完成率",
                "description": "已完成项目的比例",
                "weight": 0.3,
                "target": 1.0
            },
            "learning_speed": {
                "name": "学习速度",
                "description": "单位时间内学习的项目数",
                "weight": 0.2,
                "target": "dynamic"
            },
            "mastery_level": {
                "name": "掌握程度",
                "description": "知识的掌握深度",
                "weight": 0.25,
                "target": 0.8
            },
            "consistency": {
                "name": "一致性",
                "description": "学习过程的稳定程度",
                "weight": 0.15,
                "target": 0.9
            },
            "engagement": {
                "name": "参与度",
                "description": "学习过程的投入程度",
                "weight": 0.1,
                "target": 0.8
            }
        }
        
        # 监控历史
        self.monitoring_history = defaultdict(list)
    
    def monitor_goal_progress(self,
                             goal: LearningGoal,
                             progress_data: Dict[str, Any],
                             monitoring_strategy: str = "adaptive") -> Dict[str, Any]:
        """
        监控目标进度
        
        Args:
            goal: 学习目标
            progress_data: 进度数据
            monitoring_strategy: 监控策略
            
        Returns:
            监控报告
        """
        print(f"[^] Monitoring goal progress: {goal.description}")
        
        monitoring_report = {
            "goal_id": goal.id,
            "monitored_at": datetime.now().isoformat(),
            "strategy": monitoring_strategy,
            "current_progress": {},
            "progress_metrics": {},
            "trend_analysis": {},
            "alerts": [],
            "recommendations": []
        }
        
        # 验证策略
        if monitoring_strategy not in self.monitoring_strategies:
            print(f"[!] Unknown monitoring strategy {monitoring_strategy}, using adaptive strategy")
            monitoring_strategy = "adaptive"
        
        monitoring_report["strategy"] = monitoring_strategy
        
        # 分析当前进度
        current_progress = self._analyze_current_progress(goal, progress_data)
        monitoring_report["current_progress"] = current_progress
        
        # 计算进度指标
        progress_metrics = self._calculate_progress_metrics(goal, progress_data)
        monitoring_report["progress_metrics"] = progress_metrics
        
        # 分析趋势
        trend_analysis = self._analyze_progress_trends(goal, progress_data, current_progress)
        monitoring_report["trend_analysis"] = trend_analysis
        
        # 检查预警
        alerts = self._check_progress_alerts(goal, current_progress, progress_metrics, trend_analysis)
        monitoring_report["alerts"] = alerts
        
        # 生成建议
        recommendations = self._generate_progress_recommendations(
            goal, current_progress, progress_metrics, trend_analysis, alerts
        )
        monitoring_report["recommendations"] = recommendations
        
        # 记录监控历史
        self._record_monitoring_history(goal.id, monitoring_report)
        
        print(f"[OK] Progress monitoring complete: {len(alerts)} alerts, {len(recommendations)} recommendations")
        return monitoring_report
    
    def monitor_mindmap_progress(self,
                                goal: LearningGoal,
                                mindmap_root: MindMapNode,
                                node_map: Dict[str, MindMapNode],
                                progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        监控思维导图进度
        
        Args:
            goal: 学习目标
            mindmap_root: 思维导图根节点
            node_map: 节点映射
            progress_data: 进度数据
            
        Returns:
            思维导图进度报告
        """
        print(f"[brain] Monitoring mindmap progress")
        
        mindmap_report = {
            "goal_id": goal.id,
            "mindmap_id": mindmap_root.id,
            "monitored_at": datetime.now().isoformat(),
            "node_progress": {},
            "layer_progress": {},
            "structural_analysis": {},
            "weak_areas": [],
            "strong_areas": []
        }
        
        # 分析节点进度
        node_progress = self._analyze_node_progress(node_map, progress_data)
        mindmap_report["node_progress"] = node_progress
        
        # 分析层级进度
        layer_progress = self._analyze_layer_progress(node_map, progress_data)
        mindmap_report["layer_progress"] = layer_progress
        
        # 结构分析
        structural_analysis = self._analyze_mindmap_structure_progress(node_map, progress_data)
        mindmap_report["structural_analysis"] = structural_analysis
        
        # 识别薄弱区域
        weak_areas = self._identify_weak_areas(node_map, progress_data)
        mindmap_report["weak_areas"] = weak_areas
        
        # 识别优势区域
        strong_areas = self._identify_strong_areas(node_map, progress_data)
        mindmap_report["strong_areas"] = strong_areas
        
        # 生成思维导图学习建议
        mindmap_report["mindmap_recommendations"] = self._generate_mindmap_recommendations(
            node_map, progress_data, weak_areas, strong_areas
        )
        
        return mindmap_report
    
    def generate_progress_visualization(self,
                                      goal: LearningGoal,
                                      progress_data: Dict[str, Any],
                                      mindmap_node_map: Optional[Dict[str, MindMapNode]] = None) -> Dict[str, Any]:
        """
        生成进度可视化数据
        
        Args:
            goal: 学习目标
            progress_data: 进度数据
            mindmap_node_map: 思维导图节点映射
            
        Returns:
            可视化数据
        """
        print(f"📊 生成进度可视化")
        
        visualization = {
            "goal_id": goal.id,
            "generated_at": datetime.now().isoformat(),
            "progress_charts": {},
            "trend_visualizations": {},
            "mindmap_visualizations": {}
        }
        
        # 进度图表数据
        progress_charts = self._create_progress_charts(goal, progress_data)
        visualization["progress_charts"] = progress_charts
        
        # 趋势可视化
        trend_visualizations = self._create_trend_visualizations(goal, progress_data)
        visualization["trend_visualizations"] = trend_visualizations
        
        # 思维导图可视化（如果有思维导图）
        if mindmap_node_map:
            mindmap_visualizations = self._create_mindmap_visualizations(
                goal, mindmap_node_map, progress_data
            )
            visualization["mindmap_visualizations"] = mindmap_visualizations
        
        return visualization
    
    def predict_completion_time(self,
                               goal: LearningGoal,
                               progress_data: Dict[str, Any],
                               learning_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        预测完成时间
        
        Args:
            goal: 学习目标
            progress_data: 进度数据
            learning_history: 学习历史
            
        Returns:
            完成时间预测
        """
        print(f"[T] Predicted completion time: {goal.description}")
        
        prediction = {
            "goal_id": goal.id,
            "predicted_at": datetime.now().isoformat(),
            "current_progress": progress_data.get("overall_progress", 0),
            "prediction_models": {},
            "confidence_scores": {},
            "recommended_actions": []
        }
        
        # 使用多种模型预测
        prediction_models = {}
        
        # 1. 线性外推模型
        linear_prediction = self._predict_with_linear_model(goal, progress_data, learning_history)
        prediction_models["linear"] = linear_prediction
        
        # 2. 学习曲线模型
        learning_curve_prediction = self._predict_with_learning_curve(goal, progress_data, learning_history)
        prediction_models["learning_curve"] = learning_curve_prediction
        
        # 3. 时间序列模型
        time_series_prediction = self._predict_with_time_series(goal, progress_data, learning_history)
        prediction_models["time_series"] = time_series_prediction
        
        # 4. 自适应模型
        adaptive_prediction = self._predict_with_adaptive_model(goal, progress_data, learning_history)
        prediction_models["adaptive"] = adaptive_prediction
        
        prediction["prediction_models"] = prediction_models
        
        # 计算置信度
        confidence_scores = self._calculate_prediction_confidence(prediction_models)
        prediction["confidence_scores"] = confidence_scores
        
        # 综合预测（加权平均）
        weighted_prediction = self._calculate_weighted_prediction(prediction_models, confidence_scores)
        prediction["weighted_prediction"] = weighted_prediction
        
        # 生成建议
        if weighted_prediction["on_track"] == "behind":
            prediction["recommended_actions"].append("增加每日学习时间")
            prediction["recommended_actions"].append("优化学习方法")
        elif weighted_prediction["on_track"] == "ahead":
            prediction["recommended_actions"].append("可以提前完成或增加学习深度")
        
        return prediction
    
    def _analyze_current_progress(self,
                                 goal: LearningGoal,
                                 progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析当前进度"""
        current_progress = {
            "overall_progress": goal.overall_progress,
            "progress_details": {},
            "milestone_status": {},
            "recent_activity": {}
        }
        
        # 详细进度
        if goal.batch_progress:
            current_progress["progress_details"]["batch_progress"] = goal.batch_progress
        
        if goal.item_progress:
            current_progress["progress_details"]["item_progress"] = goal.item_progress
        
        if goal.subgoal_progress:
            current_progress["progress_details"]["subgoal_progress"] = goal.subgoal_progress
        
        if goal.mindmap_layer_progress:
            current_progress["progress_details"]["mindmap_layer_progress"] = goal.mindmap_layer_progress
        
        # 里程碑状态
        if "milestones" in progress_data:
            for milestone in progress_data["milestones"]:
                milestone_id = milestone.get("id", "")
                status = milestone.get("status", "pending")
                current_progress["milestone_status"][milestone_id] = status
        
        # 最近活动
        recent_days = 7
        current_progress["recent_activity"] = {
            "learning_sessions": progress_data.get("recent_sessions", []),
            "daily_progress": progress_data.get("daily_progress", {}),
            "active_days": progress_data.get("active_days", 0)
        }
        
        return current_progress
    
    def _calculate_progress_metrics(self,
                                  goal: LearningGoal,
                                  progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算进度指标"""
        metrics = {}
        
        # 完成率
        completion_rate = goal.overall_progress
        metrics["completion_rate"] = {
            "value": completion_rate,
            "status": "good" if completion_rate >= 0.7 else 
                     "warning" if completion_rate >= 0.3 else "poor"
        }
        
        # 学习速度（项目/天）
        total_items = goal.target_knowledge_count
        completed_items = sum(1 for p in goal.item_progress.values() if p >= 0.8) if goal.item_progress else 0
        days_elapsed = self._calculate_days_elapsed(goal)
        
        if days_elapsed > 0:
            learning_speed = completed_items / days_elapsed
        else:
            learning_speed = 0
        
        metrics["learning_speed"] = {
            "value": learning_speed,
            "status": "good" if learning_speed >= 5 else 
                     "warning" if learning_speed >= 2 else "poor"
        }
        
        # 掌握程度（基于测试成绩）
        mastery_level = progress_data.get("mastery_level", 0.0)
        metrics["mastery_level"] = {
            "value": mastery_level,
            "status": "good" if mastery_level >= 0.8 else 
                     "warning" if mastery_level >= 0.6 else "poor"
        }
        
        # 一致性（学习天数比例）
        total_days = self._calculate_total_days(goal)
        if total_days > 0:
            consistency = progress_data.get("active_days", 0) / total_days
        else:
            consistency = 0
        
        metrics["consistency"] = {
            "value": consistency,
            "status": "good" if consistency >= 0.7 else 
                     "warning" if consistency >= 0.4 else "poor"
        }
        
        # 参与度（基于学习时长和专注度）
        engagement = progress_data.get("engagement_level", 0.5)
        metrics["engagement"] = {
            "value": engagement,
            "status": "good" if engagement >= 0.7 else 
                     "warning" if engagement >= 0.5 else "poor"
        }
        
        # 综合进度分数
        weights = {
            "completion_rate": 0.3,
            "learning_speed": 0.2,
            "mastery_level": 0.25,
            "consistency": 0.15,
            "engagement": 0.1
        }
        
        weighted_score = 0
        for metric_name, metric_data in metrics.items():
            weight = weights.get(metric_name, 0)
            score = metric_data["value"]
            weighted_score += score * weight
        
        metrics["overall_score"] = {
            "value": weighted_score,
            "status": "good" if weighted_score >= 0.7 else 
                     "warning" if weighted_score >= 0.5 else "poor"
        }
        
        return metrics
    
    def _calculate_days_elapsed(self, goal: LearningGoal) -> int:
        """计算已过天数"""
        try:
            if goal.started_at:
                start_date = datetime.fromisoformat(goal.started_at)
                current_date = datetime.now()
                days_elapsed = (current_date - start_date).days
                return max(days_elapsed, 0)
        except:
            pass
        
        return 0
    
    def _calculate_total_days(self, goal: LearningGoal) -> int:
        """计算总天数（从开始到预计完成）"""
        try:
            if goal.started_at and goal.estimated_completion:
                start_date = datetime.fromisoformat(goal.started_at)
                end_date = datetime.fromisoformat(goal.estimated_completion)
                total_days = (end_date - start_date).days
                return max(total_days, 1)
        except:
            pass
        
        # 如果没有预估完成时间，使用默认值
        return 30  # 默认30天
    
    def _analyze_progress_trends(self,
                                goal: LearningGoal,
                                progress_data: Dict[str, Any],
                                current_progress: Dict[str, Any]) -> Dict[str, Any]:
        """分析进度趋势"""
        trends = {
            "progress_trend": "stable",
            "velocity_trend": "stable",
            "consistency_trend": "stable",
            "predicted_completion": None,
            "risk_factors": []
        }
        
        # 分析历史进度趋势
        if "progress_history" in progress_data:
            history = progress_data["progress_history"]
            
            if len(history) >= 3:
                # 计算近期进度变化
                recent_changes = []
                for i in range(1, min(4, len(history))):
                    if i < len(history):
                        change = history[-i].get("progress", 0) - history[-i-1].get("progress", 0)
                        recent_changes.append(change)
                
                if recent_changes:
                    avg_change = sum(recent_changes) / len(recent_changes)
                    
                    if avg_change > 0.05:  # 每周进度增加超过5%
                        trends["progress_trend"] = "accelerating"
                    elif avg_change < -0.02:  # 每周进度减少超过2%
                        trends["progress_trend"] = "decelerating"
                        trends["risk_factors"].append("学习进度在下降")
                    else:
                        trends["progress_trend"] = "stable"
        
        # 分析学习速度趋势
        if "learning_velocity" in progress_data:
            velocity_history = progress_data["learning_velocity"]
            
            if len(velocity_history) >= 3:
                recent_velocity = velocity_history[-3:]
                avg_velocity = sum(recent_velocity) / len(recent_velocity)
                
                # 与早期速度比较
                if len(velocity_history) >= 6:
                    early_velocity = velocity_history[-6:-3]
                    avg_early_velocity = sum(early_velocity) / len(early_velocity) if early_velocity else 0
                    
                    if avg_velocity > avg_early_velocity * 1.2:
                        trends["velocity_trend"] = "increasing"
                    elif avg_velocity < avg_early_velocity * 0.8:
                        trends["velocity_trend"] = "decreasing"
                        trends["risk_factors"].append("学习速度在下降")
        
        # 预测完成时间
        days_elapsed = self._calculate_days_elapsed(goal)
        current_progress_value = current_progress.get("overall_progress", 0)
        
        if days_elapsed > 0 and current_progress_value > 0:
            # 线性预测
            if current_progress_value > 0:
                estimated_total_days = days_elapsed / current_progress_value
                days_remaining = estimated_total_days - days_elapsed
                
                try:
                    predicted_date = datetime.now() + timedelta(days=days_remaining)
                    trends["predicted_completion"] = predicted_date.isoformat()
                    
                    # 检查是否按时
                    if goal.estimated_completion:
                        estimated_date = datetime.fromisoformat(goal.estimated_completion)
                        days_until_deadline = (estimated_date - datetime.now()).days
                        
                        if days_remaining > days_until_deadline * 1.2:
                            trends["risk_factors"].append(f"预计将延期{int(days_remaining - days_until_deadline)}天")
                except:
                    pass
        
        return trends
    
    def _check_progress_alerts(self,
                              goal: LearningGoal,
                              current_progress: Dict[str, Any],
                              progress_metrics: Dict[str, Any],
                              trend_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查进度预警"""
        alerts = []
        
        # 进度过慢预警
        current_progress_value = current_progress.get("overall_progress", 0)
        days_elapsed = self._calculate_days_elapsed(goal)
        
        if days_elapsed > 7 and current_progress_value < 0.2:
            alerts.append({
                "type": "slow_progress",
                "severity": "high",
                "message": f"学习7天后进度仅{current_progress_value:.1%}，可能过慢",
                "suggested_action": "检查学习方法或增加学习时间"
            })
        
        # 学习速度下降预警
        if trend_analysis.get("velocity_trend") == "decreasing":
            alerts.append({
                "type": "decreasing_velocity",
                "severity": "medium",
                "message": "学习速度在下降",
                "suggested_action": "分析原因并调整学习策略"
            })
        
        # 一致性预警
        consistency = progress_metrics.get("consistency", {}).get("value", 0)
        if consistency < 0.3:
            alerts.append({
                "type": "low_consistency",
                "severity": "medium",
                "message": f"学习一致性较低({consistency:.1%})",
                "suggested_action": "建立更规律的学习习惯"
            })
        
        # 参与度预警
        engagement = progress_metrics.get("engagement", {}).get("value", 0)
        if engagement < 0.4:
            alerts.append({
                "type": "low_engagement",
                "severity": "medium",
                "message": f"学习参与度较低({engagement:.1%})",
                "suggested_action": "增加学习互动性或调整内容"
            })
        
        # 进度停滞预警
        if "progress_history" in goal.metadata:
            history = goal.metadata["progress_history"]
            if len(history) >= 3:
                recent_progress = [h.get("progress", 0) for h in history[-3:]]
                if max(recent_progress) - min(recent_progress) < 0.02:  # 几乎无变化
                    alerts.append({
                        "type": "progress_stagnation",
                        "severity": "high",
                        "message": "最近3次检查进度几乎无变化",
                        "suggested_action": "突破学习瓶颈，尝试新方法"
                    })
        
        # 思维导图进度不均衡预警
        if "mindmap_layer_progress" in current_progress.get("progress_details", {}):
            layer_progress = current_progress["progress_details"]["mindmap_layer_progress"]
            if layer_progress:
                progress_values = list(layer_progress.values())
                if len(progress_values) >= 2:
                    progress_range = max(progress_values) - min(progress_values)
                    if progress_range > 0.5:  # 不同层级进度差异过大
                        alerts.append({
                            "type": "unbalanced_mindmap_progress",
                            "severity": "medium",
                            "message": "思维导图不同层级学习进度不均衡",
                            "suggested_action": "调整学习重点，加强薄弱层级"
                        })
        
        return alerts
    
    def _generate_progress_recommendations(self,
                                         goal: LearningGoal,
                                         current_progress: Dict[str, Any],
                                         progress_metrics: Dict[str, Any],
                                         trend_analysis: Dict[str, Any],
                                         alerts: List[Dict[str, Any]]) -> List[str]:
        """生成进度建议"""
        recommendations = []
        
        # 基于进度状态
        current_progress_value = current_progress.get("overall_progress", 0)
        
        if current_progress_value < 0.3:
            recommendations.append("学习初期，建议打好基础，不要急于求成")
        elif current_progress_value < 0.7:
            recommendations.append("学习中期，建议加强练习和复习")
        else:
            recommendations.append("学习后期，建议进行综合应用和总结")
        
        # 基于学习速度
        learning_speed = progress_metrics.get("learning_speed", {}).get("value", 0)
        if learning_speed < 2:
            recommendations.append(f"当前学习速度较低({learning_speed:.1f}项目/天)，建议提高学习效率")
        elif learning_speed > 10:
            recommendations.append(f"学习速度很快({learning_speed:.1f}项目/天)，可以考虑增加学习深度")
        
        # 基于趋势
        progress_trend = trend_analysis.get("progress_trend", "stable")
        if progress_trend == "decelerating":
            recommendations.append("检测到学习进度在减慢，建议分析原因并调整")
        elif progress_trend == "accelerating":
            recommendations.append("学习进度在加速，继续保持当前节奏")
        
        # 基于风险因素
        if trend_analysis.get("risk_factors"):
            for risk in trend_analysis["risk_factors"]:
                if "延期" in risk:
                    recommendations.append("预计将延期完成，建议增加学习时间或调整目标")
        
        # 基于整体分数
        overall_score = progress_metrics.get("overall_score", {}).get("value", 0)
        if overall_score < 0.5:
            recommendations.append("整体学习状况有待改善，建议全面检查学习计划")
        elif overall_score > 0.8:
            recommendations.append("学习状况良好，可以继续保持或挑战更高目标")
        
        return recommendations
    
    def _analyze_node_progress(self,
                              node_map: Dict[str, MindMapNode],
                              progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析节点进度"""
        node_progress = {
            "total_nodes": len(node_map),
            "completed_nodes": 0,
            "learning_nodes": 0,
            "pending_nodes": 0,
            "node_status_distribution": {},
            "node_progress_by_type": {},
            "node_progress_by_depth": {}
        }
        
        # 统计节点状态
        for node_id, node in node_map.items():
            status = node.learning_status
            
            if status not in node_progress["node_status_distribution"]:
                node_progress["node_status_distribution"][status] = 0
            node_progress["node_status_distribution"][status] += 1
            
            if status == "mastered":
                node_progress["completed_nodes"] += 1
            elif status == "learning":
                node_progress["learning_nodes"] += 1
            else:
                node_progress["pending_nodes"] += 1
            
            # 按类型统计
            node_type = node.node_type
            if node_type not in node_progress["node_progress_by_type"]:
                node_progress["node_progress_by_type"][node_type] = {
                    "total": 0,
                    "completed": 0,
                    "progress": 0.0
                }
            
            node_progress["node_progress_by_type"][node_type]["total"] += 1
            if status == "mastered":
                node_progress["node_progress_by_type"][node_type]["completed"] += 1
            
            # 按深度统计
            depth = node.depth
            if depth not in node_progress["node_progress_by_depth"]:
                node_progress["node_progress_by_depth"][depth] = {
                    "total": 0,
                    "completed": 0,
                    "progress": 0.0
                }
            
            node_progress["node_progress_by_depth"][depth]["total"] += 1
            if status == "mastered":
                node_progress["node_progress_by_depth"][depth]["completed"] += 1
        
        # 计算进度百分比
        for node_type, stats in node_progress["node_progress_by_type"].items():
            if stats["total"] > 0:
                stats["progress"] = stats["completed"] / stats["total"]
        
        for depth, stats in node_progress["node_progress_by_depth"].items():
            if stats["total"] > 0:
                stats["progress"] = stats["completed"] / stats["total"]
        
        return node_progress
    
    def _analyze_layer_progress(self,
                               node_map: Dict[str, MindMapNode],
                               progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析层级进度"""
        layer_progress = {}
        
        # 按深度分组
        depth_groups = defaultdict(list)
        for node_id, node in node_map.items():
            depth_groups[node.depth].append(node)
        
        # 计算每层进度
        for depth, nodes in depth_groups.items():
            total_nodes = len(nodes)
            mastered_nodes = sum(1 for node in nodes if node.learning_status == "mastered")
            progress = mastered_nodes / total_nodes if total_nodes > 0 else 0.0
            
            layer_progress[depth] = {
                "total_nodes": total_nodes,
                "mastered_nodes": mastered_nodes,
                "progress": progress,
                "status": "completed" if progress >= 0.8 else
                          "in_progress" if progress >= 0.3 else "pending"
            }
        
        return layer_progress
    
    def _analyze_mindmap_structure_progress(self,
                                          node_map: Dict[str, MindMapNode],
                                          progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析思维导图结构进度"""
        structure_progress = {
            "branch_completion": {},
            "prerequisite_chains": {},
            "structural_integrity": 0.0
        }
        
        # 分析分支完成情况
        branch_roots = [node for node in node_map.values() if node.depth == 1]
        
        for branch_root in branch_roots:
            # 获取分支所有节点
            branch_nodes = self._get_branch_nodes(branch_root.id, node_map)
            total_nodes = len(branch_nodes)
            mastered_nodes = sum(1 for node_id in branch_nodes
                                if node_map.get(node_id, MindMapNode(id="fallback", title="", description="", learning_status="pending")).learning_status == "mastered")
            
            if total_nodes > 0:
                branch_progress = mastered_nodes / total_nodes
                structure_progress["branch_completion"][branch_root.title] = {
                    "total_nodes": total_nodes,
                    "mastered_nodes": mastered_nodes,
                    "progress": branch_progress
                }
        
        # 分析先决条件链
        prerequisite_chains = self._identify_prerequisite_chains(node_map)
        structure_progress["prerequisite_chains"] = prerequisite_chains
        
        # 计算结构完整性
        if node_map:
            # 结构完整性 = 已完成节点的重要性加权平均
            total_importance = 0
            completed_importance = 0
            
            for node in node_map.values():
                total_importance += node.importance
                if node.learning_status == "mastered":
                    completed_importance += node.importance
            
            if total_importance > 0:
                structure_progress["structural_integrity"] = completed_importance / total_importance
        
        return structure_progress
    
    def _get_branch_nodes(self, root_id: str, node_map: Dict[str, MindMapNode]) -> List[str]:
        """获取分支所有节点"""
        branch_nodes = []
        
        def collect_nodes(node_id: str):
            node = node_map.get(node_id)
            if not node:
                return
            
            branch_nodes.append(node_id)
            for child_id in node.children_ids:
                collect_nodes(child_id)
        
        collect_nodes(root_id)
        return branch_nodes
    
    def _identify_prerequisite_chains(self, node_map: Dict[str, MindMapNode]) -> Dict[str, Any]:
        """识别先决条件链"""
        chains = {
            "completed_chains": [],
            "incomplete_chains": [],
            "blocking_nodes": []
        }
        
        # 查找有先决条件的节点
        for node_id, node in node_map.items():
            if node.prerequisites:
                # 检查先决条件是否满足
                prerequisites_completed = all(
                    node_map.get(prereq_id, MindMapNode(id="fallback", title="", description="", learning_status="pending")).learning_status == "mastered"
                    for prereq_id in node.prerequisites
                )
                
                chain = {
                    "node_id": node_id,
                    "node_title": node.title,
                    "prerequisites": node.prerequisites,
                    "all_completed": prerequisites_completed,
                    "blocked": not prerequisites_completed and node.learning_status != "mastered"
                }
                
                if prerequisites_completed:
                    chains["completed_chains"].append(chain)
                else:
                    chains["incomplete_chains"].append(chain)
                    
                    # 识别阻塞节点
                    if chain["blocked"]:
                        chains["blocking_nodes"].append(node_id)
        
        return chains
    
    def _identify_weak_areas(self,
                            node_map: Dict[str, MindMapNode],
                            progress_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别薄弱区域"""
        weak_areas = []
        
        # 识别学习困难的节点
        for node_id, node in node_map.items():
            if node.learning_status == "learning":
                # 检查学习时间是否过长
                if node.actual_time_minutes > node.estimated_time_minutes * 2:
                    weak_areas.append({
                        "node_id": node_id,
                        "node_title": node.title,
                        "reason": "学习时间远超预估",
                        "actual_time": node.actual_time_minutes,
                        "estimated_time": node.estimated_time_minutes
                    })
        
        # 识别重要性高但未掌握的节点
        for node_id, node in node_map.items():
            if node.learning_status != "mastered" and node.importance > 0.7:
                weak_areas.append({
                    "node_id": node_id,
                    "node_title": node.title,
                    "reason": "高重要性节点尚未掌握",
                    "importance": node.importance,
                    "status": node.learning_status
                })
        
        # 识别先决条件未满足的阻塞节点
        for node_id, node in node_map.items():
            if node.prerequisites and node.learning_status != "mastered":
                incomplete_prereqs = []
                for prereq_id in node.prerequisites:
                    prereq_node = node_map.get(prereq_id)
                    if prereq_node and prereq_node.learning_status != "mastered":
                        incomplete_prereqs.append(prereq_id)
                
                if incomplete_prereqs:
                    weak_areas.append({
                        "node_id": node_id,
                        "node_title": node.title,
                        "reason": "先决条件未满足",
                        "incomplete_prerequisites": incomplete_prereqs
                    })
        
        return weak_areas
    
    def _identify_strong_areas(self,
                              node_map: Dict[str, MindMapNode],
                              progress_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别优势区域"""
        strong_areas = []
        
        # 识别快速掌握的节点
        for node_id, node in node_map.items():
            if node.learning_status == "mastered":
                # 检查是否快速掌握
                if node.actual_time_minutes > 0 and node.actual_time_minutes < node.estimated_time_minutes * 0.5:
                    strong_areas.append({
                        "node_id": node_id,
                        "node_title": node.title,
                        "reason": "快速掌握",
                        "actual_time": node.actual_time_minutes,
                        "estimated_time": node.estimated_time_minutes,
                        "efficiency_ratio": node.estimated_time_minutes / node.actual_time_minutes
                    })
        
        # 识别重要性高且已掌握的节点
        for node_id, node in node_map.items():
            if node.learning_status == "mastered" and node.importance > 0.8:
                strong_areas.append({
                    "node_id": node_id,
                    "node_title": node.title,
                    "reason": "高重要性节点已掌握",
                    "importance": node.importance,
                    "mastery_level": "excellent"
                })
        
        # 识别完整掌握的分支
        branch_roots = [node for node in node_map.values() if node.depth == 1]

        for branch_root in branch_roots:
            branch_nodes = self._get_branch_nodes(branch_root.id, node_map)
            mastered_count = sum(1 for node_id in branch_nodes
                               if node_map.get(node_id, MindMapNode(id="fallback", title="", description="", learning_status="pending")).learning_status == "mastered")
            
            if mastered_count == len(branch_nodes) and len(branch_nodes) > 3:
                strong_areas.append({
                    "branch_root_id": branch_root.id,
                    "branch_title": branch_root.title,
                    "reason": "完整分支掌握",
                    "total_nodes": len(branch_nodes),
                    "mastered_nodes": mastered_count
                })
        
        return strong_areas
    
    def _generate_mindmap_recommendations(self,
                                        node_map: Dict[str, MindMapNode],
                                        progress_data: Dict[str, Any],
                                        weak_areas: List[Dict[str, Any]],
                                        strong_areas: List[Dict[str, Any]]) -> List[str]:
        """生成思维导图学习建议"""
        recommendations = []
        
        # 针对薄弱区域的建议
        if weak_areas:
            weak_count = len(weak_areas)
            recommendations.append(f"发现{weak_count}个薄弱区域，建议优先加强")
            
            # 具体建议
            for weak_area in weak_areas[:3]:  # 最多显示3个
                reason = weak_area.get("reason", "")
                if "学习时间远超预估" in reason:
                    recommendations.append(f"节点'{weak_area['node_title']}'学习时间过长，建议简化学习内容或寻求帮助")
                elif "高重要性节点尚未掌握" in reason:
                    recommendations.append(f"高重要性节点'{weak_area['node_title']}'尚未掌握，建议优先学习")
                elif "先决条件未满足" in reason:
                    recommendations.append(f"节点'{weak_area['node_title']}'的先决条件未满足，建议先学习先决节点")
        
        # 利用优势区域的建议
        if strong_areas:
            strong_count = len(strong_areas)
            recommendations.append(f"识别出{strong_count}个优势区域，可以在此基础上深化学习")
        
        # 结构优化建议
        branch_completion = progress_data.get("branch_completion", {})
        if branch_completion:
            incomplete_branches = [branch for branch, stats in branch_completion.items() 
                                 if stats.get("progress", 0) < 0.5]
            
            if incomplete_branches:
                recommendations.append(f"发现{len(incomplete_branches)}个完成度较低的分支，建议集中学习")
        
        # 进度均衡建议
        layer_progress = progress_data.get("layer_progress", {})
        if layer_progress and len(layer_progress) >= 2:
            progress_values = [stats.get("progress", 0) for stats in layer_progress.values()]
            if max(progress_values) - min(progress_values) > 0.4:
                recommendations.append("不同层级学习进度不均衡，建议加强薄弱层级")
        
        return recommendations
    
    def _create_progress_charts(self,
                               goal: LearningGoal,
                               progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建进度图表数据"""
        charts = {
            "progress_over_time": {
                "labels": [],
                "datasets": [
                    {
                        "label": "整体进度",
                        "data": [],
                        "borderColor": "#4CAF50",
                        "backgroundColor": "rgba(76, 175, 80, 0.1)"
                    }
                ]
            },
            "learning_velocity": {
                "labels": [],
                "datasets": [
                    {
                        "label": "学习速度(项目/天)",
                        "data": [],
                        "borderColor": "#2196F3",
                        "backgroundColor": "rgba(33, 150, 243, 0.1)"
                    }
                ]
            },
            "progress_by_category": {
                "labels": [],
                "datasets": [
                    {
                        "label": "完成率",
                        "data": [],
                        "backgroundColor": ["#4CAF50", "#FFC107", "#F44336", "#9C27B0", "#03A9F4"]
                    }
                ]
            }
        }
        
        # 进度随时间变化
        if "progress_history" in progress_data:
            history = progress_data["progress_history"]
            for entry in history[-10:]:  # 最近10次记录
                if "date" in entry and "progress" in entry:
                    charts["progress_over_time"]["labels"].append(entry["date"])
                    charts["progress_over_time"]["datasets"][0]["data"].append(entry["progress"])
        
        # 学习速度
        if "learning_velocity_history" in progress_data:
            velocity_history = progress_data["learning_velocity_history"]
            for i, velocity in enumerate(velocity_history[-10:]):
                charts["learning_velocity"]["labels"].append(f"第{i+1}周")
                charts["learning_velocity"]["datasets"][0]["data"].append(velocity)
        
        # 按类别进度
        if "category_progress" in progress_data:
            category_progress = progress_data["category_progress"]
            for category, progress in category_progress.items():
                charts["progress_by_category"]["labels"].append(category)
                charts["progress_by_category"]["datasets"][0]["data"].append(progress)
        
        return charts
    
    def _create_trend_visualizations(self,
                                   goal: LearningGoal,
                                   progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建趋势可视化"""
        visualizations = {
            "trend_indicators": [],
            "comparison_charts": [],
            "forecast_visualization": {}
        }
        
        # 趋势指标
        current_progress = goal.overall_progress
        days_elapsed = self._calculate_days_elapsed(goal)
        
        if days_elapsed > 0:
            daily_progress_rate = current_progress / days_elapsed if current_progress > 0 else 0
            
            visualizations["trend_indicators"] = [
                {
                    "name": "当前进度",
                    "value": f"{current_progress:.1%}",
                    "trend": "up" if current_progress > 0 else "stable"
                },
                {
                    "name": "日均进度",
                    "value": f"{daily_progress_rate:.2%}",
                    "trend": "up" if daily_progress_rate > 0.01 else "stable"
                },
                {
                    "name": "预计完成天数",
                    "value": f"{int((1 - current_progress) / daily_progress_rate) if daily_progress_rate > 0 else '未知'}",
                    "trend": "down" if daily_progress_rate > 0.02 else "stable"
                }
            ]
        
        # 对比图表（目标 vs 实际）
        if "planned_progress" in progress_data and "actual_progress" in progress_data:
            planned = progress_data["planned_progress"]
            actual = progress_data["actual_progress"]
            
            if len(planned) == len(actual):
                comparison_data = {
                    "labels": [f"第{i+1}周" for i in range(len(planned))],
                    "datasets": [
                        {
                            "label": "计划进度",
                            "data": planned,
                            "borderColor": "#FF9800",
                            "backgroundColor": "transparent"
                        },
                        {
                            "label": "实际进度",
                            "data": actual,
                            "borderColor": "#4CAF50",
                            "backgroundColor": "transparent"
                        }
                    ]
                }
                visualizations["comparison_charts"].append(comparison_data)
        
        # 预测可视化
        prediction = self.predict_completion_time(goal, progress_data, [])
        if "weighted_prediction" in prediction:
            forecast = prediction["weighted_prediction"]
            visualizations["forecast_visualization"] = {
                "predicted_completion": forecast.get("predicted_date"),
                "confidence": forecast.get("confidence", 0),
                "on_track": forecast.get("on_track", "unknown")
            }
        
        return visualizations
    
    def _create_mindmap_visualizations(self,
                                     goal: LearningGoal,
                                     node_map: Dict[str, MindMapNode],
                                     progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建思维导图可视化"""
        visualizations = {
            "node_status_heatmap": {},
            "progress_by_depth": {},
            "learning_path_visualization": {}
        }
        
        # 节点状态热图数据
        node_status_data = []
        for node_id, node in node_map.items():
            status_color = {
                "mastered": "#4CAF50",
                "learning": "#FFC107",
                "reviewing": "#2196F3",
                "pending": "#9E9E9E"
            }.get(node.learning_status, "#9E9E9E")
            
            node_status_data.append({
                "id": node_id,
                "title": node.title,
                "depth": node.depth,
                "status": node.learning_status,
                "color": status_color,
                "importance": node.importance,
                "difficulty": node.difficulty
            })
        
        visualizations["node_status_heatmap"] = {
            "nodes": node_status_data,
            "color_scheme": {
                "mastered": "#4CAF50",
                "learning": "#FFC107",
                "reviewing": "#2196F3",
                "pending": "#9E9E9E"
            }
        }
        
        # 按深度进度数据
        depth_progress = {}
        depth_groups = defaultdict(list)
        
        for node in node_map.values():
            depth_groups[node.depth].append(node)
        
        for depth, nodes in depth_groups.items():
            total = len(nodes)
            mastered = sum(1 for node in nodes if node.learning_status == "mastered")
            progress = mastered / total if total > 0 else 0
            
            depth_progress[depth] = {
                "total": total,
                "mastered": mastered,
                "progress": progress
            }
        
        visualizations["progress_by_depth"] = depth_progress
        
        # 学习路径可视化
        learning_path = self._extract_learning_path(node_map, progress_data)
        visualizations["learning_path_visualization"] = {
            "path_nodes": learning_path["nodes"],
            "path_connections": learning_path["connections"],
            "current_position": learning_path["current_position"]
        }
        
        return visualizations
    
    def _extract_learning_path(self,
                              node_map: Dict[str, MindMapNode],
                              progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取学习路径"""
        path_nodes = []
        path_connections = []
        
        # 从根节点开始
        root_nodes = [node for node in node_map.values() if node.depth == 0]
        
        if root_nodes:
            root = root_nodes[0]
            path_nodes.append({
                "id": root.id,
                "title": root.title,
                "status": root.learning_status,
                "type": "root"
            })
        
        # 添加已掌握的节点
        mastered_nodes = [node for node in node_map.values() if node.learning_status == "mastered"]
        for node in mastered_nodes:
            if node.depth > 0:  # 排除根节点
                path_nodes.append({
                    "id": node.id,
                    "title": node.title,
                    "status": node.learning_status,
                    "type": "mastered",
                    "depth": node.depth
                })
        
        # 添加正在学习的节点
        learning_nodes = [node for node in node_map.values() if node.learning_status == "learning"]
        current_position = None
        
        for node in learning_nodes:
            path_nodes.append({
                "id": node.id,
                "title": node.title,
                "status": node.learning_status,
                "type": "current",
                "depth": node.depth
            })
            
            # 设置当前位置
            if not current_position or node.importance > 0.7:
                current_position = node.id
        
        # 创建连接关系
        for node in node_map.values():
            for child_id in node.children_ids:
                if (node.id in [n["id"] for n in path_nodes] and 
                    child_id in [n["id"] for n in path_nodes]):
                    path_connections.append({
                        "from": node.id,
                        "to": child_id,
                        "type": "parent_child"
                    })
        
        return {
            "nodes": path_nodes,
            "connections": path_connections,
            "current_position": current_position
        }
    
    def _predict_with_linear_model(self,
                                  goal: LearningGoal,
                                  progress_data: Dict[str, Any],
                                  learning_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用线性模型预测"""
        current_progress = goal.overall_progress
        days_elapsed = self._calculate_days_elapsed(goal)
        
        if days_elapsed <= 0 or current_progress <= 0:
            return {
                "method": "linear",
                "predicted_date": None,
                "days_remaining": None,
                "confidence": 0.0,
                "on_track": "unknown"
            }
        
        # 线性外推
        total_days_needed = days_elapsed / current_progress
        days_remaining = total_days_needed - days_elapsed
        
        try:
            predicted_date = datetime.now() + timedelta(days=days_remaining)
            
            # 检查是否按时
            on_track = "on_time"
            if goal.estimated_completion:
                estimated_date = datetime.fromisoformat(goal.estimated_completion)
                days_until_deadline = (estimated_date - datetime.now()).days
                
                if days_remaining > days_until_deadline * 1.1:
                    on_track = "behind"
                elif days_remaining < days_until_deadline * 0.9:
                    on_track = "ahead"
            
            # 置信度基于数据点数量
            confidence = min(days_elapsed / 7, 0.9)  # 每过一周增加置信度，最高0.9
            
            return {
                "method": "linear",
                "predicted_date": predicted_date.isoformat(),
                "days_remaining": max(0, int(days_remaining)),
                "confidence": confidence,
                "on_track": on_track
            }
        except:
            return {
                "method": "linear",
                "predicted_date": None,
                "days_remaining": None,
                "confidence": 0.0,
                "on_track": "unknown"
            }
    
    def _predict_with_learning_curve(self,
                                    goal: LearningGoal,
                                    progress_data: Dict[str, Any],
                                    learning_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用学习曲线模型预测"""
        # 学习曲线模型考虑了学习效率随时间的变化
        current_progress = goal.overall_progress
        days_elapsed = self._calculate_days_elapsed(goal)
        
        if days_elapsed <= 0 or current_progress <= 0:
            return {
                "method": "learning_curve",
                "predicted_date": None,
                "days_remaining": None,
                "confidence": 0.0,
                "on_track": "unknown"
            }
        
        # 获取历史学习速度
        if learning_history and len(learning_history) >= 3:
            # 计算平均学习速度
            daily_progress_rates = []
            for i in range(1, min(4, len(learning_history))):
                if i < len(learning_history):
                    session = learning_history[-i]
                    if "progress_gain" in session and "duration_days" in session:
                        rate = session["progress_gain"] / session["duration_days"]
                        daily_progress_rates.append(rate)
            
            if daily_progress_rates:
                avg_daily_rate = sum(daily_progress_rates) / len(daily_progress_rates)
                
                # 应用学习曲线效应：后期学习可能变慢或变快
                remaining_progress = 1.0 - current_progress
                
                # 简单模型：剩余进度除以平均速度
                days_remaining = remaining_progress / avg_daily_rate if avg_daily_rate > 0 else 0
                
                # 调整因子：基于学习曲线
                curve_factor = 1.0
                if current_progress > 0.7:
                    # 后期可能遇到困难，学习变慢
                    curve_factor = 1.2
                elif current_progress < 0.3:
                    # 初期可能较慢，但建立基础后可能加速
                    curve_factor = 0.9
                
                days_remaining *= curve_factor
                
                try:
                    predicted_date = datetime.now() + timedelta(days=days_remaining)
                    
                    return {
                        "method": "learning_curve",
                        "predicted_date": predicted_date.isoformat(),
                        "days_remaining": max(0, int(days_remaining)),
                        "confidence": 0.7,
                        "on_track": "on_time"  # 简化处理
                    }
                except:
                    pass
        
        # 回退到线性模型
        return self._predict_with_linear_model(goal, progress_data, learning_history)
    
    def _predict_with_time_series(self,
                                 goal: LearningGoal,
                                 progress_data: Dict[str, Any],
                                 learning_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用时间序列模型预测"""
        # 简化版时间序列预测
        if learning_history and len(learning_history) >= 5:
            # 提取时间序列数据
            dates = []
            progress_values = []
            
            for session in learning_history[-5:]:
                if "date" in session and "progress" in session:
                    dates.append(session["date"])
                    progress_values.append(session["progress"])
            
            if len(progress_values) >= 3:
                # 简单趋势分析
                recent_trend = progress_values[-1] - progress_values[-3]
                
                # 预测剩余时间
                current_progress = progress_values[-1] if progress_values else 0
                remaining_progress = 1.0 - current_progress
                
                if recent_trend > 0:
                    # 有进展趋势
                    estimated_daily_rate = recent_trend / 3  # 最近3天的平均日进展
                    if estimated_daily_rate > 0:
                        days_remaining = remaining_progress / estimated_daily_rate
                        
                        try:
                            predicted_date = datetime.now() + timedelta(days=days_remaining)
                            
                            return {
                                "method": "time_series",
                                "predicted_date": predicted_date.isoformat(),
                                "days_remaining": max(0, int(days_remaining)),
                                "confidence": 0.6,
                                "on_track": "on_time"
                            }
                        except:
                            pass
        
        # 回退到线性模型
        return self._predict_with_linear_model(goal, progress_data, learning_history)
    
    def _predict_with_adaptive_model(self,
                                    goal: LearningGoal,
                                    progress_data: Dict[str, Any],
                                    learning_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用自适应模型预测"""
        # 自适应模型考虑多个因素
        factors = []
        
        # 1. 当前进度因素
        current_progress = goal.overall_progress
        if current_progress > 0.8:
            factors.append(0.9)  # 接近完成，预测较准确
        elif current_progress > 0.5:
            factors.append(0.7)
        elif current_progress > 0.2:
            factors.append(0.5)
        else:
            factors.append(0.3)  # 初期不确定性高
        
        # 2. 历史数据量因素
        if learning_history:
            data_points = len(learning_history)
            data_factor = min(data_points / 10, 1.0)  # 10个数据点为充分
            factors.append(data_factor)
        else:
            factors.append(0.1)
        
        # 3. 进度稳定性因素
        if "progress_history" in progress_data:
            history = progress_data["progress_history"]
            if len(history) >= 3:
                recent_changes = []
                for i in range(1, min(4, len(history))):
                    if i < len(history):
                        change = abs(history[-i].get("progress", 0) - history[-i-1].get("progress", 0))
                        recent_changes.append(change)
                
                if recent_changes:
                    avg_change = sum(recent_changes) / len(recent_changes)
                    stability_factor = 1.0 - min(avg_change * 5, 0.8)  # 变化越小越稳定
                    factors.append(stability_factor)
        
        # 计算综合置信度
        if factors:
            confidence = sum(factors) / len(factors)
        else:
            confidence = 0.5
        
        # 使用线性模型作为基础
        linear_prediction = self._predict_with_linear_model(goal, progress_data, learning_history)
        
        # 调整置信度
        linear_prediction["confidence"] = confidence
        linear_prediction["method"] = "adaptive"
        
        return linear_prediction
    
    def _calculate_prediction_confidence(self,
                                       prediction_models: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """计算预测置信度"""
        confidences = {}
        
        for model_name, prediction in prediction_models.items():
            confidences[model_name] = prediction.get("confidence", 0.5)
        
        # 综合置信度（加权平均）
        if confidences:
            weights = {
                "linear": 0.3,
                "learning_curve": 0.3,
                "time_series": 0.2,
                "adaptive": 0.2
            }
            
            weighted_sum = 0
            weight_sum = 0
            
            for model_name, confidence in confidences.items():
                weight = weights.get(model_name, 0.1)
                weighted_sum += confidence * weight
                weight_sum += weight
            
            if weight_sum > 0:
                confidences["overall"] = weighted_sum / weight_sum
            else:
                confidences["overall"] = 0.5
        
        return confidences
    
    def _calculate_weighted_prediction(self,
                                     prediction_models: Dict[str, Dict[str, Any]],
                                     confidence_scores: Dict[str, float]) -> Dict[str, Any]:
        """计算加权预测"""
        # 收集所有预测日期
        predictions = []
        
        for model_name, prediction in prediction_models.items():
            predicted_date = prediction.get("predicted_date")
            confidence = confidence_scores.get(model_name, 0.5)
            
            if predicted_date:
                try:
                    date_obj = datetime.fromisoformat(predicted_date)
                    predictions.append({
                        "date": date_obj,
                        "confidence": confidence,
                        "model": model_name
                    })
                except:
                    continue
        
        if not predictions:
            return {
                "predicted_date": None,
                "confidence": 0.0,
                "on_track": "unknown"
            }
        
        # 加权平均日期
        total_weight = sum(p["confidence"] for p in predictions)
        weighted_date = datetime.now()
        
        if total_weight > 0:
            # 计算加权平均的天数偏移
            weighted_days = 0
            
            for pred in predictions:
                days_offset = (pred["date"] - datetime.now()).days
                weight = pred["confidence"] / total_weight
                weighted_days += days_offset * weight
            
            weighted_date = datetime.now() + timedelta(days=weighted_days)
        
        # 判断是否按时
        on_track = "on_time"
        
        # 检查不同模型的一致性
        if len(predictions) >= 2:
            dates = [p["date"] for p in predictions]
            min_date = min(dates)
            max_date = max(dates)
            date_range = (max_date - min_date).days
            
            if date_range > 14:  # 预测差异超过14天
                on_track = "uncertain"
        
        return {
            "predicted_date": weighted_date.isoformat(),
            "confidence": confidence_scores.get("overall", 0.5),
            "on_track": on_track
        }
    
    def _record_monitoring_history(self, goal_id: str, monitoring_report: Dict[str, Any]) -> None:
        """记录监控历史"""
        history_entry = {
            "monitored_at": monitoring_report.get("monitored_at", datetime.now().isoformat()),
            "strategy": monitoring_report.get("strategy", ""),
            "progress": monitoring_report.get("current_progress", {}).get("overall_progress", 0),
            "alerts_count": len(monitoring_report.get("alerts", [])),
            "metrics_score": monitoring_report.get("progress_metrics", {}).get("overall_score", {}).get("value", 0)
        }
        
        self.monitoring_history[goal_id].append(history_entry)
        
        # 限制历史记录长度
        if len(self.monitoring_history[goal_id]) > 50:
            self.monitoring_history[goal_id] = self.monitoring_history[goal_id][-50:]

# ========== 规划模块测试 ==========

if __name__ == "__main__":
    print("🧪 测试规划模块...")
    print("=" * 70)
    
    # 创建测试目标
    test_goal = LearningGoal(
        id="test_goal_001",
        description="学习Python编程基础",
        scale=GoalScale.MEDIUM,
        target_knowledge_count=50,
        overall_progress=0.4
    )
    
    # 创建测试思维导图
    test_mindmap = MindMapNode(
        id="test_mindmap_root",
        title="Python编程基础",
        description="Python编程基础知识体系",
        depth=0
    )
    
    # 创建一些子节点
    child_nodes = []
    for i in range(5):
        child = MindMapNode(
            id=f"test_child_{i}",
            title=f"Python概念{i+1}",
            description=f"Python编程概念{i+1}",
            depth=1,
            parent_id=test_mindmap.id,
            importance=random.uniform(0.3, 0.9),
            difficulty=random.uniform(0.2, 0.8),
            learning_status="mastered" if i < 2 else "learning"
        )
        child_nodes.append(child)
    
    # 构建节点映射
    node_map = {test_mindmap.id: test_mindmap}
    for child in child_nodes:
        node_map[child.id] = child
        test_mindmap.children_ids.append(child.id)
    
    # 测试层次化学习分配器
    print("\n📊 测试层次化学习分配器:")
    print("-" * 50)
    
    allocator = HierarchicalLearningAllocator()
    
    allocation_plan = allocator.allocate_by_mindmap(
        goal=test_goal,
        mindmap_root=test_mindmap,
        node_map=node_map,
        strategy="balanced",
        available_time_minutes=600  # 10小时
    )
    
    print(f"分配计划: {len(allocation_plan.get('learning_sequences', []))}个学习序列")
    print(f"时间分配: {allocation_plan.get('time_allocation', {}).get('total_estimated_minutes', 0)}分钟")
    
    # 测试思维导图驱动规划器
    print("\n\n📋 测试思维导图驱动规划器:")
    print("-" * 50)
    
    planner = MindMapDrivenPlanner()
    
    learning_plan = planner.create_learning_plan(
        goal=test_goal,
        mindmap_root=test_mindmap,
        node_map=node_map,
        allocation_plan=allocation_plan
    )
    
    print(f"学习计划: {len(learning_plan.get('milestones', []))}个里程碑")
    print(f"时间线: {learning_plan.get('timeline', {}).get('timeline_weeks', 0)}周")
    
    # 测试自适应调度器
    print("\n\n⏰ 测试自适应调度器:")
    print("-" * 50)
    
    scheduler = AdaptiveScheduler()
    
    current_context = {
        "available_minutes": 120,
        "energy_level": 0.7,
        "focus_level": 0.8,
        "distractions": []
    }
    
    schedule = scheduler.schedule_learning_sessions(
        learning_plan=learning_plan,
        current_context=current_context,
        strategy="adaptive_schedule"
    )
    
    print(f"调度结果: {len(schedule.get('scheduled_sessions', []))}个学习会话")
    print(f"灵活性分数: {schedule.get('flexibility_score', 0):.2f}")
    
    # 测试进度监控器
    print("\n\n[^] Testing progress monitor:")
    print("-" * 50)
    
    monitor = ProgressMonitor()
    
    progress_data = {
        "overall_progress": 0.4,
        "mastery_level": 0.6,
        "engagement_level": 0.7,
        "active_days": 10,
        "daily_progress": {"2024-01-01": 0.1, "2024-01-02": 0.15, "2024-01-03": 0.2}
    }
    
    monitoring_report = monitor.monitor_goal_progress(
        goal=test_goal,
        progress_data=progress_data,
        monitoring_strategy="adaptive"
    )
    
    print(f"监控报告: {len(monitoring_report.get('alerts', []))}个预警")
    print(f"进度指标: {monitoring_report.get('progress_metrics', {}).get('overall_score', {}).get('value', 0):.2f}")
    
    # 测试多目标优化
    print("\n\n⚡ 测试多目标优化:")
    print("-" * 50)
    
    # 创建多个测试目标
    goals = [
        LearningGoal(
            id=f"goal_{i}",
            description=f"学习目标{i+1}",
            scale=random.choice([GoalScale.SMALL, GoalScale.MEDIUM, GoalScale.LARGE]),
            target_knowledge_count=random.randint(20, 200),
            priority=random.randint(3, 9)
        )
        for i in range(3)
    ]
    
    available_time = {
        "周一": 2, "周二": 2, "周三": 2, "周四": 2, "周五": 2,
        "周六": 4, "周日": 4
    }
    
    optimization = scheduler.optimize_schedule_for_goals(
        goals=goals,
        available_time=available_time
    )
    
    print(f"优化结果: {len(optimization.get('time_allocation', {}))}个时间分配")
    print(f"优化指标: 效率{optimization.get('optimization_metrics', {}).get('efficiency_score', 0):.2f}")
    
    print("\n✅ 规划模块测试完成")
    print("=" * 70)
                   