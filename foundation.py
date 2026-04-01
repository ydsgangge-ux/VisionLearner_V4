# foundation.py - 数据模型与核心接口（重构版）
"""
第1段：数据模型与核心接口
功能：定义系统的基础数据结构、抽象接口、核心枚举
特点：支持思维导图驱动的学习、规模感知、抽象接口设计
创新：引入MindMapNode模型，支持大模型生成的层次化知识结构
"""

import json
import hashlib
import time
import random
import math
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from abc import ABC, abstractmethod

# ========== 核心枚举定义 ==========

class LearningLevel(Enum):
    """学习深度级别 - 支持多粒度"""
    EXPOSURE = 1      # 接触/了解（30秒/项目）
    FAMILIARITY = 2   # 熟悉（1分钟/项目）
    UNDERSTANDING = 3 # 理解（2分钟/项目）
    APPLICATION = 4   # 应用（5分钟/项目）
    MASTERY = 5       # 掌握（10分钟/项目）
    CREATION = 6      # 创造（30+分钟/项目）

class KnowledgeType(Enum):
    """知识类型 - 扩展支持分层"""
    CONCEPT = "概念"      # 基本概念
    FACT = "事实"         # 具体事实
    PRINCIPLE = "原理"    # 原理原则
    SKILL = "技能"        # 技能方法
    PROCESS = "过程"      # 流程过程
    SYSTEM = "系统"       # 系统性知识
    PATTERN = "模式"      # 模式规律
    STRATEGY = "策略"     # 策略方法
    EXAMPLE = "示例"      # 具体示例
    PRACTICE = "练习"     # 练习任务

class GoalScale(Enum):
    """目标规模 - 核心新增：规模感知"""
    MICRO = "微目标"      # 1-10个知识点
    SMALL = "小目标"      # 11-100个知识点
    MEDIUM = "中目标"     # 101-1000个知识点
    LARGE = "大目标"      # 1001-10000个知识点
    MASSIVE = "大规模目标" # 10000+知识点

class LearningStrategy(Enum):
    """学习策略 - 针对不同规模"""
    INDIVIDUAL = "逐个学习"      # 微目标：精细学习
    BATCH = "批次学习"           # 小/中目标：批量处理
    PIPELINE = "流水线学习"       # 大目标：并行处理
    HIERARCHICAL = "分层学习"    # 大规模目标：层次化分解
    ADAPTIVE = "自适应学习"      # 动态调整策略
    MINDMAP_DRIVEN = "思维导图驱动" # 基于思维导图的学习

class GoalStatus(Enum):
    """目标状态"""
    PENDING = "待开始"
    ACTIVE = "进行中"
    PAUSED = "已暂停"
    COMPLETED = "已完成"
    CANCELLED = "已取消"

class ModalityType(Enum):
    """模态类型 - 支持多模态输入"""
    TEXT = "文本"
    IMAGE = "图片"
    AUDIO = "音频"
    VIDEO = "视频"
    MULTIMODAL = "多模态混合"
    MINDMAP = "思维导图"  # 新增：思维导图模态

class ProgressGranularity(Enum):
    """进度粒度 - 多粒度进度跟踪"""
    OVERALL = "整体进度"
    BATCH = "批次进度"
    ITEM = "项目进度"
    SUBGOAL = "子目标进度"
    MINDMAP_LAYER = "思维导图层级进度"  # 新增

class MindMapStyle(Enum):
    """思维导图风格"""
    BALANCED = "平衡型"     # 广度深度均衡
    DEEP = "深度型"        # 深度优先，层次多
    BROAD = "广度型"       # 广度优先，分支多
    STRUCTURED = "结构化"   # 高度结构化
    CREATIVE = "创意型"    # 非结构化，创意发散

# ========== 核心数据模型 ==========

@dataclass
class MindMapNode:
    """
    思维导图节点 - 大模型生成的层次化知识结构
    核心创新：每个学习目标对应一个思维导图树
    """

    # 基础信息
    id: str
    title: str
    description: str = ""

    # 学习内容
    content: Any = None              # 节点的知识内容
    collected: bool = False            # 是否已收集/学习
    collected_at: str = ""            # 收集时间
    collected_by: str = ""            # 收集来源（llm模型名称）

    # 结构属性
    depth: int = 0                    # 在树中的深度（0=根节点）
    node_type: str = "concept"       # concept, skill, example, practice, principle, fact
    importance: float = 0.5          # 重要性 0.0-1.0
    difficulty: float = 0.5          # 难度 0.0-1.0
    prerequisite_score: float = 0.0  # 先决条件重要性
    
    # 学习属性
    learning_status: str = "pending"  # pending, learning, reviewing, mastered
    estimated_time_minutes: int = 30  # 预估学习时间
    actual_time_minutes: int = 0      # 实际学习时间
    
    # 结构关系
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    sibling_ids: List[str] = field(default_factory=list)  # 同级节点
    _children_cache: List['MindMapNode'] = field(default_factory=list, repr=False)  # 运行时缓存children对象

    @property
    def children(self) -> List['MindMapNode']:
        """获取子节点对象列表（从node_map获取）"""
        return self._children_cache

    def set_children(self, children: List['MindMapNode']) -> None:
        """设置子节点对象列表"""
        self._children_cache = children
    
    # 关联知识
    knowledge_node_ids: List[str] = field(default_factory=list)  # 关联的知识节点
    prerequisites: List[str] = field(default_factory=list)       # 先决节点
    related_nodes: List[str] = field(default_factory=list)       # 相关节点
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    notes: str = ""  # 学习笔记
    confidence: float = 0.8  # 大模型生成置信度
    
    # 生成信息
    generated_by: str = "unknown"  # 生成模型
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_prompt: str = ""  # 生成提示词
    
    def __post_init__(self):
        """后初始化处理"""
        if not self.description:
            self.description = f"{self.title}的相关知识"
    
    def is_root(self) -> bool:
        """是否是根节点"""
        return self.depth == 0
    
    def is_leaf(self) -> bool:
        """是否是叶子节点"""
        return len(self.children_ids) == 0
    
    def get_ancestor_ids(self, node_map: Dict[str, 'MindMapNode']) -> List[str]:
        """获取所有祖先节点IDs"""
        ancestors = []
        current = self
        
        while current.parent_id and current.parent_id in node_map:
            ancestors.append(current.parent_id)
            current = node_map[current.parent_id]
        
        return ancestors
    
    def get_descendant_ids(self, node_map: Dict[str, 'MindMapNode']) -> List[str]:
        """获取所有后代节点IDs（递归）"""
        descendants = []
        
        def collect_descendants(node_id: str):
            node = node_map.get(node_id)
            if not node:
                return
            
            for child_id in node.children_ids:
                descendants.append(child_id)
                collect_descendants(child_id)
        
        collect_descendants(self.id)
        return descendants
    
    def calculate_complexity(self) -> float:
        """计算节点复杂度"""
        # 基于深度、难度、重要性、子节点数量
        complexity = 0.0
        
        # 深度因子：越深越复杂
        depth_factor = min(self.depth / 5, 1.0)
        complexity += depth_factor * 0.2
        
        # 难度因子
        complexity += self.difficulty * 0.3
        
        # 重要性因子：越重要可能越复杂
        complexity += self.importance * 0.1
        
        # 子节点数量因子
        children_factor = min(len(self.children_ids) / 10, 1.0)
        complexity += children_factor * 0.2
        
        # 预估时间因子
        time_factor = min(self.estimated_time_minutes / 120, 1.0)
        complexity += time_factor * 0.2
        
        return min(complexity, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        result = asdict(self)
        # _children_cache 不需要序列化
        if '_children_cache' in result:
            del result['_children_cache']
        return result

    def to_dict_with_children(self) -> Dict[str, Any]:
        """转换为可序列化的字典，包含完整的children数组（用于向后兼容）"""
        result = asdict(self)
        # _children_cache 不需要序列化
        if '_children_cache' in result:
            del result['_children_cache']
        # 递归添加children数组
        result['children'] = [child.to_dict_with_children() for child in self.children]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MindMapNode':
        """从字典重建节点"""
        # 过滤掉不是 dataclass 字段的键，特别是 'children' 和 '_children_cache'
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    # 兼容旧版方法
    def is_learned(self) -> bool:
        """自己收集了，且所有子节点都收集了"""
        return self.collected

    def _all_nodes(self) -> List['MindMapNode']:
        """获取所有节点（递归）"""
        nodes = [self]
        for child in self.children:
            nodes.extend(child._all_nodes())
        return nodes

    def uncollected_nodes(self) -> List['MindMapNode']:
        """获取所有未收集的节点（递归）"""
        uncollected = []
        if not self.collected and self.depth > 0:
            uncollected.append(self)
        for child in self.children:
            uncollected.extend(child.uncollected_nodes())
        return uncollected

    def completion_rate(self) -> float:
        """计算完成率（已收集节点/总节点）"""
        all_nodes = self._all_nodes()
        if not all_nodes:
            return 0.0
        collected = sum(1 for node in all_nodes if node.collected and node.depth > 0)
        total = sum(1 for node in all_nodes if node.depth > 0)
        return collected / total if total > 0 else 0.0

    def find_by_title(self, title: str, node_map: Dict[str, 'MindMapNode']) -> Optional['MindMapNode']:
        """递归查找指定标题的节点"""
        if self.title == title:
            return self
        for child_id in self.children_ids:
            child = node_map.get(child_id)
            if child:
                found = child.find_by_title(title, node_map)
                if found:
                    return found
        return None

@dataclass
class KnowledgeNode:
    """
    知识节点 - 统一的知识表示单元
    支持层次化组织，形成知识树
    增强版：与思维导图节点关联
    """
    
    # 基础信息
    id: str
    title: str
    content: str
    summary: str = ""  # 摘要，用于快速预览
    
    # 分类信息
    knowledge_type: KnowledgeType = KnowledgeType.CONCEPT
    learning_level: LearningLevel = LearningLevel.EXPOSURE
    modality: ModalityType = ModalityType.TEXT
    
    # 掌握程度
    confidence: float = 0.5  # 置信度 0.0-1.0
    mastery_score: float = 0.0  # 掌握分数 0.0-1.0
    vision_relevance: float = 0.0  # 与文明愿景的相关性
    
    # 思维导图关联
    mindmap_node_id: Optional[str] = None  # 关联的思维导图节点
    mindmap_depth: int = 0  # 在思维导图中的深度
    
    # 层次关系（支持树状结构）
    parent_id: Optional[str] = None  # 父节点ID
    children_ids: List[str] = field(default_factory=list)  # 子节点IDs
    depth: int = 0  # 在树中的深度
    
    # 关联关系
    prerequisites: List[str] = field(default_factory=list)  # 先决条件节点IDs
    related_nodes: List[str] = field(default_factory=list)  # 相关节点IDs
    applications: List[str] = field(default_factory=list)   # 应用场景
    
    # 学习跟踪
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_reviewed: str = field(default_factory=lambda: datetime.now().isoformat())
    next_review: Optional[str] = None  # 下次复习时间
    review_count: int = 0
    learning_sessions: List[str] = field(default_factory=list)  # 关联的学习会话IDs
    
    # 学习统计
    learning_time_minutes: int = 0  # 总学习时间
    test_score_history: List[float] = field(default_factory=list)  # 测试成绩历史
    success_rate: float = 0.0  # 成功率
    
    # 多模态支持
    image_data: Optional[str] = None  # base64编码的图片数据或图片URL
    audio_data: Optional[str] = None  # base64编码的音频数据或音频URL
    
    # 元数据
    source: str = "internal"
    external_references: List[str] = field(default_factory=list)  # 外部引用
    tags: List[str] = field(default_factory=list)  # 标签，用于分类和搜索
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展元数据
    
    def __post_init__(self):
        """后初始化处理"""
        if not self.summary and self.content:
            # 自动生成摘要
            self.summary = self.content[:100] + "..." if len(self.content) > 100 else self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        result = asdict(self)
        
        # 处理枚举类型
        for key, value in result.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                result[key] = [item.value for item in value]
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """从字典重建知识节点"""
        # 处理枚举类型
        enum_fields = {
            'knowledge_type': KnowledgeType,
            'learning_level': LearningLevel,
            'modality': ModalityType
        }
        
        for field_name, enum_class in enum_fields.items():
            if field_name in data and isinstance(data[field_name], str):
                for member in enum_class:
                    if member.value == data[field_name]:
                        data[field_name] = member
                        break
        
        return cls(**data)
    
    def is_leaf(self) -> bool:
        """是否是叶子节点（没有子节点）"""
        return len(self.children_ids) == 0
    
    def calculate_complexity(self) -> float:
        """计算知识节点复杂度"""
        complexity = 0.0
        
        # 基于学习级别
        complexity += self.learning_level.value * 0.2
        
        # 基于内容长度
        content_complexity = min(len(self.content) / 500, 1.0)  # 500字为上限
        complexity += content_complexity * 0.3
        
        # 基于先决条件数量
        prereq_complexity = min(len(self.prerequisites) / 5, 1.0)  # 5个先决条件为上限
        complexity += prereq_complexity * 0.2
        
        # 基于关联节点数量
        related_complexity = min(len(self.related_nodes) / 10, 1.0)  # 10个关联为上限
        complexity += related_complexity * 0.1
        
        # 基于学习时间
        time_complexity = min(self.learning_time_minutes / 120, 1.0)  # 120分钟为上限
        complexity += time_complexity * 0.2
        
        return min(complexity, 1.0)

@dataclass
class LearningGoal:
    """
    学习目标 - 支持思维导图驱动的层次化学习
    核心：大模型生成思维导图，按层次分配学习任务
    """
    
    # 基础信息
    id: str
    description: str
    detailed_description: str = ""  # 详细描述
    
    # 目标属性
    scale: GoalScale = GoalScale.MICRO  # 目标规模
    strategy: LearningStrategy = LearningStrategy.INDIVIDUAL  # 学习策略
    priority: int = 5  # 优先级 1-10（10最高）
    complexity: float = 0.5  # 复杂度 0.0-1.0
    
    # 思维导图驱动学习（新增）
    mindmap_root_id: Optional[str] = None  # 思维导图根节点ID
    mindmap_generated_at: Optional[str] = None  # 思维导图生成时间
    mindmap_depth: int = 3  # 思维导图深度（层数）
    mindmap_style: MindMapStyle = MindMapStyle.BALANCED  # 思维导图风格
    mindmap_confidence: float = 0.0  # 思维导图质量置信度
    
    # 层次结构（用于大规模目标分解）
    parent_goal_id: Optional[str] = None  # 父目标ID
    subgoal_ids: List[str] = field(default_factory=list)  # 子目标IDs
    depth: int = 0  # 在目标树中的深度
    is_leaf: bool = True  # 是否是叶子目标
    
    # 知识关联
    target_knowledge_count: int = 1  # 目标知识点数量
    knowledge_node_ids: List[str] = field(default_factory=list)  # 关联的知识节点IDs
    knowledge_structure: Dict[str, Any] = field(default_factory=dict)  # 知识结构描述
    
    # 进度跟踪（多粒度）
    overall_progress: float = 0.0  # 整体进度 0.0-1.0
    batch_progress: Dict[str, float] = field(default_factory=dict)  # 批次进度 {batch_id: progress}
    item_progress: Dict[str, float] = field(default_factory=dict)  # 单个项目进度 {item_id: progress}
    subgoal_progress: Dict[str, float] = field(default_factory=dict)  # 子目标进度
    mindmap_layer_progress: Dict[int, float] = field(default_factory=dict)  # 思维导图层级进度
    
    # 时间管理
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_completion: Optional[str] = None
    deadline: Optional[str] = None
    
    # 学习参数
    batch_size: int = 10  # 批次大小
    daily_limit: int = 100  # 每日学习上限（知识点数）
    review_interval_days: int = 7  # 复习间隔天数
    
    # 思维导图学习参数（新增）
    learning_depth_strategy: str = "standard"  # quick, standard, deep, mastery
    max_learning_depth: int = 3  # 最大学习深度
    min_node_importance: float = 0.3  # 最小节点重要性阈值
    
    # 统计信息
    total_sessions: int = 0
    total_learning_time_minutes: int = 0
    avg_test_score: float = 0.0
    completion_rate: float = 0.0  # 完成率
    
    # 状态
    status: str = "pending"  # pending, active, paused, completed, failed, adjusted
    current_batch: int = 0  # 当前批次索引
    total_batches: int = 1  # 总批次数
    
    # 愿景关联
    vision_alignment: float = 0.0
    vision_priority: int = 5
    strategic_pathway: str = ""  # 战略路径
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化处理"""
        if not self.detailed_description:
            self.detailed_description = self.description
        
        # 初始化进度字典
        if not self.batch_progress:
            self.batch_progress = {}
        if not self.item_progress:
            self.item_progress = {}
        if not self.subgoal_progress:
            self.subgoal_progress = {}
        if not self.mindmap_layer_progress:
            self.mindmap_layer_progress = {}
    
    def update_progress(self, progress_data: Optional[Dict[str, Any]] = None) -> None:
        """
        更新进度信息
        
        Args:
            progress_data: 进度数据，可包含各种进度信息
        """
        if progress_data:
            # 更新各项进度
            if 'overall' in progress_data:
                self.overall_progress = progress_data['overall']
            
            if 'batch' in progress_data:
                self.batch_progress.update(progress_data['batch'])
            
            if 'item' in progress_data:
                self.item_progress.update(progress_data['item'])
            
            if 'subgoal' in progress_data:
                self.subgoal_progress.update(progress_data['subgoal'])
            
            if 'mindmap_layer' in progress_data:
                self.mindmap_layer_progress.update(progress_data['mindmap_layer'])
        
        # 重新计算整体进度（如果基于子目标）
        if self.subgoal_ids and self.subgoal_progress:
            total = 0.0
            for subgoal_id in self.subgoal_ids:
                total += self.subgoal_progress.get(subgoal_id, 0.0)
            self.overall_progress = total / len(self.subgoal_ids)
        elif self.item_progress:
            # 基于项目进度计算
            if self.item_progress:
                completed = sum(1 for p in self.item_progress.values() if p >= 0.8)
                self.overall_progress = completed / len(self.item_progress) if self.item_progress else 0.0
        
        # 检查是否完成
        if self.overall_progress >= 0.95 and self.status != "completed":
            self.status = "completed"
            self.completed_at = datetime.now().isoformat()
    
    def get_next_batch(self) -> List[str]:
        """获取下一批要学习的知识节点ID"""
        if not self.knowledge_node_ids:
            return []
        
        # 计算批次范围
        all_items = self.knowledge_node_ids
        start = self.current_batch * self.batch_size
        end = min(start + self.batch_size, len(all_items))
        
        if start >= len(all_items):
            return []
        
        return all_items[start:end]
    
    def mark_batch_completed(self, batch_items: List[str], progress: float = 1.0) -> None:
        """标记批次完成"""
        # 更新项目进度
        for item_id in batch_items:
            self.item_progress[item_id] = progress
        
        # 更新批次进度
        batch_key = f"batch_{self.current_batch}"
        self.batch_progress[batch_key] = progress
        
        # 移动到下一批次
        self.current_batch += 1
        
        # 重新计算整体进度
        self.update_progress()
    
    def add_subgoal(self, subgoal_id: str, weight: float = 1.0) -> None:
        """添加子目标"""
        if subgoal_id not in self.subgoal_ids:
            self.subgoal_ids.append(subgoal_id)
            self.subgoal_progress[subgoal_id] = 0.0
            self.is_leaf = False
    
    def set_mindmap_root(self, mindmap_node_id: str) -> None:
        """设置思维导图根节点"""
        self.mindmap_root_id = mindmap_node_id
        self.mindmap_generated_at = datetime.now().isoformat()
    
    def calculate_estimated_time(self, time_model: 'TimeEstimationModel') -> int:
        """计算预估学习时间（分钟）"""
        return time_model.estimate_for_goal(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        
        # 处理枚举类型
        for key, value in result.items():
            if isinstance(value, Enum):
                result[key] = value.value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningGoal':
        """从字典重建学习目标"""
        # 处理枚举类型
        enum_fields = {
            'scale': GoalScale,
            'strategy': LearningStrategy,
            'mindmap_style': MindMapStyle
        }
        
        for field_name, enum_class in enum_fields.items():
            if field_name in data and isinstance(data[field_name], str):
                for member in enum_class:
                    if member.value == data[field_name]:
                        data[field_name] = member
                        break
        
        return cls(**data)

# ========== 抽象接口定义 ==========

class IGoalAnalyzer(ABC):
    """目标分析器接口 - 分析学习目标的特性"""
    
    @abstractmethod
    def analyze(self, goal_description: str) -> Dict[str, Any]:
        """分析目标描述，返回特征字典"""
        pass
    
    @abstractmethod
    def extract_quantity(self, description: str) -> Optional[int]:
        """从描述中提取数量信息"""
        pass
    
    @abstractmethod
    def determine_scale(self, quantity: Optional[int]) -> GoalScale:
        """确定目标规模"""
        pass

class IGoalDecomposer(ABC):
    """目标分解器接口 - 将大目标分解为小目标"""
    
    @abstractmethod
    def decompose(self, goal: LearningGoal, context: Dict[str, Any]) -> List[LearningGoal]:
        """分解目标为子目标"""
        pass
    
    @abstractmethod
    def create_knowledge_structure(self, goal: LearningGoal) -> Dict[str, Any]:
        """为目标创建知识结构"""
        pass

class IMindMapGenerator(ABC):
    """思维导图生成器接口 - 大模型生成思维导图"""
    
    @abstractmethod
    def generate_for_goal(self, goal: LearningGoal) -> Optional[MindMapNode]:
        """为学习目标生成思维导图"""
        pass
    
    @abstractmethod
    def generate_from_text(self, text: str, depth: int = 3) -> Optional[MindMapNode]:
        """从文本生成思维导图"""
        pass
    
    @abstractmethod
    def refine_mindmap(self, mindmap: MindMapNode, feedback: Dict[str, Any]) -> MindMapNode:
        """根据反馈精炼思维导图"""
        pass

class ILearningStrategy(ABC):
    """学习策略接口 - 不同规模使用不同策略"""
    
    @abstractmethod
    def plan(self, goal: LearningGoal) -> Dict[str, Any]:
        """为学习目标制定计划"""
        pass
    
    @abstractmethod
    def execute_batch(self, goal: LearningGoal, batch_items: List[str]) -> Dict[str, Any]:
        """执行一个批次的学习"""
        pass
    
    @abstractmethod
    def adjust_strategy(self, goal: LearningGoal, feedback: Dict[str, Any]) -> None:
        """根据反馈调整策略"""
        pass

class IProgressTracker(ABC):
    """进度跟踪器接口 - 多粒度进度跟踪"""
    
    @abstractmethod
    def track(self, goal: LearningGoal, granularity: ProgressGranularity) -> Dict[str, Any]:
        """跟踪进度"""
        pass
    
    @abstractmethod
    def get_progress_summary(self, goal: LearningGoal) -> Dict[str, Any]:
        """获取进度摘要"""
        pass

class IBatchProcessor(ABC):
    """批处理器接口 - 批量处理学习任务"""
    
    @abstractmethod
    def process_batch(self, items: List[Any], goal: LearningGoal) -> List[KnowledgeNode]:
        """批量处理项目，生成知识节点"""
        pass
    
    @abstractmethod
    def optimize_batch_size(self, goal: LearningGoal) -> int:
        """优化批次大小"""
        pass

class IMultimodalRecognizer(ABC):
    """多模态识别器接口"""
    
    @abstractmethod
    def recognize_image(self, image_data: Any) -> Dict[str, Any]:
        """识别图片内容"""
        pass
    
    @abstractmethod
    def recognize_audio(self, audio_data: Any) -> Dict[str, Any]:
        """识别音频内容"""
        pass
    
    @abstractmethod
    def recognize_text(self, text: str) -> Dict[str, Any]:
        """深度分析文本内容"""
        pass
    
    @abstractmethod
    def extract_concepts(self, recognition_result: Dict[str, Any]) -> List[str]:
        """从识别结果中提取概念"""
        pass

# ========== 时间估算模型 ==========

class TimeEstimationModel:
    """
    现实时间估算模型
    基于批量处理效率和真实学习速度
    增强版：支持思维导图层次时间估算
    """
    
    def __init__(self):
        # 不同类型知识的基准学习时间（分钟/项目）
        self.base_times = {
            'characters': {  # 汉字
                LearningLevel.EXPOSURE: 0.5,      # 30秒
                LearningLevel.FAMILIARITY: 1.0,   # 1分钟
                LearningLevel.UNDERSTANDING: 2.0, # 2分钟
                LearningLevel.APPLICATION: 5.0,   # 5分钟
                LearningLevel.MASTERY: 10.0       # 10分钟
            },
            'words': {  # 单词
                LearningLevel.EXPOSURE: 0.5,
                LearningLevel.FAMILIARITY: 1.5,
                LearningLevel.UNDERSTANDING: 3.0,
                LearningLevel.APPLICATION: 8.0,
                LearningLevel.MASTERY: 15.0
            },
            'concepts': {  # 概念
                LearningLevel.EXPOSURE: 2.0,
                LearningLevel.FAMILIARITY: 5.0,
                LearningLevel.UNDERSTANDING: 15.0,
                LearningLevel.APPLICATION: 30.0,
                LearningLevel.MASTERY: 60.0
            },
            'skills': {  # 技能
                LearningLevel.EXPOSURE: 5.0,
                LearningLevel.FAMILIARITY: 15.0,
                LearningLevel.UNDERSTANDING: 45.0,
                LearningLevel.APPLICATION: 120.0,
                LearningLevel.MASTERY: 300.0
            }
        }
        
        # 批量效率因子（基于项目数量）
        self.batch_efficiency = {
            1: 1.00,    # 无效率增益
            10: 0.90,   # 10%效率增益
            50: 0.70,   # 30%效率增益
            100: 0.60,  # 40%效率增益
            500: 0.50,  # 50%效率增益
            1000: 0.40, # 60%效率增益
            5000: 0.30  # 70%效率增益
        }
        
        # 策略效率因子
        self.strategy_efficiency = {
            LearningStrategy.INDIVIDUAL: 1.0,    # 无额外效率
            LearningStrategy.BATCH: 0.8,         # 20%效率提升
            LearningStrategy.PIPELINE: 0.7,      # 30%效率提升
            LearningStrategy.HIERARCHICAL: 0.6,  # 40%效率提升
            LearningStrategy.ADAPTIVE: 0.75,     # 25%效率提升
            LearningStrategy.MINDMAP_DRIVEN: 0.65 # 35%效率提升（结构化学习更高效）
        }
        
        # 思维导图层级时间因子
        self.mindmap_depth_factor = {
            0: 1.0,   # 根节点：基准时间
            1: 0.8,   # 第一层：80%时间
            2: 0.6,   # 第二层：60%时间
            3: 0.4,   # 第三层：40%时间
            4: 0.3,   # 第四层：30%时间
            5: 0.2    # 第五层：20%时间
        }
    
    def estimate_for_goal(self, goal: LearningGoal) -> int:
        """为目标估算学习时间（分钟）"""
        # 确定知识类型
        knowledge_type = self._detect_knowledge_type(goal.description)
        
        # 确定学习深度
        learning_level = self._detect_learning_level(goal.description)
        
        # 获取基础时间
        if knowledge_type in self.base_times and learning_level in self.base_times[knowledge_type]:
            base_time = self.base_times[knowledge_type][learning_level]
        else:
            base_time = 5.0  # 默认5分钟
        
        # 计算基础总时间
        quantity = goal.target_knowledge_count
        base_total = base_time * quantity
        
        # 应用批量效率
        efficiency_factor = self._get_efficiency_factor(quantity)
        
        # 应用策略效率
        strategy_factor = self.strategy_efficiency.get(goal.strategy, 1.0)
        
        # 应用规模效率（大规模目标有额外效率）
        scale_factor = self._get_scale_factor(goal.scale)
        
        # 思维导图效率增益（如果有思维导图）
        mindmap_factor = 1.0
        if goal.mindmap_root_id and goal.strategy == LearningStrategy.MINDMAP_DRIVEN:
            mindmap_factor = 0.7  # 思维导图驱动学习效率提升30%
        
        # 最终时间估算
        estimated_minutes = base_total * efficiency_factor * strategy_factor * scale_factor * mindmap_factor
        
        return int(estimated_minutes)
    
    def estimate_for_mindmap_node(self, node: MindMapNode) -> int:
        """为思维导图节点估算学习时间"""
        # 基础时间基于节点属性
        base_time = 30  # 默认30分钟
        
        # 根据重要性调整
        importance_factor = 0.5 + node.importance  # 0.5-1.5
        
        # 根据难度调整
        difficulty_factor = 0.5 + node.difficulty  # 0.5-1.5
        
        # 根据深度调整
        depth_factor = self.mindmap_depth_factor.get(node.depth, 1.0)
        
        # 最终时间
        estimated_time = base_time * importance_factor * difficulty_factor * depth_factor
        
        # 如果有预估时间，使用它
        if node.estimated_time_minutes > 0:
            estimated_time = node.estimated_time_minutes
        
        return int(estimated_time)
    
    def _detect_knowledge_type(self, description: str) -> str:
        """从描述中检测知识类型"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["汉字", "文字", "字"]):
            return "characters"
        elif any(word in description_lower for word in ["单词", "词汇", "词语"]):
            return "words"
        elif any(word in description_lower for word in ["概念", "理论", "原理", "定义"]):
            return "concepts"
        elif any(word in description_lower for word in ["技能", "方法", "技巧", "操作"]):
            return "skills"
        else:
            return "concepts"  # 默认
    
    def _detect_learning_level(self, description: str) -> LearningLevel:
        """从描述中检测学习深度"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["了解", "接触", "知道", "认识"]):
            return LearningLevel.EXPOSURE
        elif any(word in description_lower for word in ["熟悉", "认得", "辨认"]):
            return LearningLevel.FAMILIARITY
        elif any(word in description_lower for word in ["理解", "明白", "懂得", "掌握"]):
            return LearningLevel.UNDERSTANDING
        elif any(word in description_lower for word in ["应用", "使用", "实践", "操作"]):
            return LearningLevel.APPLICATION
        elif any(word in description_lower for word in ["精通", "熟练", "专精", "专家"]):
            return LearningLevel.MASTERY
        else:
            return LearningLevel.UNDERSTANDING  # 默认
    
    def _get_efficiency_factor(self, quantity: int) -> float:
        """获取批量效率因子"""
        # 找到最接近的阈值
        thresholds = sorted(self.batch_efficiency.keys())
        
        for threshold in reversed(thresholds):
            if quantity >= threshold:
                return self.batch_efficiency[threshold]
        
        return 1.0
    
    def _get_scale_factor(self, scale: GoalScale) -> float:
        """获取规模效率因子"""
        scale_factors = {
            GoalScale.MICRO: 1.0,
            GoalScale.SMALL: 0.9,
            GoalScale.MEDIUM: 0.8,
            GoalScale.LARGE: 0.7,
            GoalScale.MASSIVE: 0.6
        }
        return scale_factors.get(scale, 1.0)
    
    def generate_schedule_options(self, total_minutes: int) -> Dict[str, Dict[str, Any]]:
        """生成多种学习时间安排选项"""
        total_hours = total_minutes / 60
        
        schedule_options = {
            "轻松模式": {
                "daily_hours": 1,
                "days_per_week": 5,
                "estimated_days": math.ceil(total_hours / (1 * 5)) * 7 / 5,
                "intensity": "低",
                "description": "每天1小时，适合轻松学习"
            },
            "常规模式": {
                "daily_hours": 2,
                "days_per_week": 5,
                "estimated_days": math.ceil(total_hours / (2 * 5)) * 7 / 5,
                "intensity": "中",
                "description": "每天2小时，均衡学习"
            },
            "强化模式": {
                "daily_hours": 4,
                "days_per_week": 5,
                "estimated_days": math.ceil(total_hours / (4 * 5)) * 7 / 5,
                "intensity": "高",
                "description": "每天4小时，快速进步"
            },
            "冲刺模式": {
                "daily_hours": 8,
                "days_per_week": 7,
                "estimated_days": math.ceil(total_hours / (8 * 7)),
                "intensity": "极高",
                "description": "每天8小时，集中突破"
            }
        }
        
        return schedule_options

# ========== 基础实现类 ==========

class SimpleGoalAnalyzer(IGoalAnalyzer):
    """简单目标分析器 - 基础实现"""
    
    def __init__(self):
        self.time_model = TimeEstimationModel()
    
    def analyze(self, goal_description: str) -> Dict[str, Any]:
        """分析目标特征"""
        import re
        
        features = {
            "description": goal_description,
            "has_quantity": False,
            "quantity": None,
            "scale": GoalScale.MICRO,
            "complexity": "medium",
            "structure_type": "unknown",
            "knowledge_type": "concepts",
            "learning_level": LearningLevel.UNDERSTANDING,
            "keywords": [],
            "estimated_items": 1,
            "estimated_time_minutes": 0,
            "suggested_mindmap_depth": 3,
            "suggested_mindmap_style": MindMapStyle.BALANCED.value
        }
        
        # 提取数量
        quantity = self.extract_quantity(goal_description)
        if quantity:
            features["has_quantity"] = True
            features["quantity"] = quantity
            features["estimated_items"] = quantity
            features["scale"] = self.determine_scale(quantity)
        
        # 分析复杂度
        complexity_keywords = {
            "掌握": "high",
            "精通": "high", 
            "深入": "high",
            "系统": "high",
            "全面": "high",
            "基础": "low",
            "入门": "low",
            "了解": "low",
            "简单": "low"
        }
        
        for keyword, complexity in complexity_keywords.items():
            if keyword in goal_description:
                features["complexity"] = complexity
                features["keywords"].append(keyword)
                break
        
        # 分析结构类型
        structure_patterns = [
            (r"个.*汉字", "collection"),
            (r"个.*单词", "collection"),
            (r"个.*概念", "collection"),
            (r"系统.*学习", "hierarchical"),
            (r"掌握.*体系", "hierarchical"),
            (r"项目", "project"),
            (r"实践", "practical"),
            (r"理论", "theoretical"),
            (r"步骤", "sequential"),
            (r"流程", "sequential")
        ]
        
        for pattern, structure in structure_patterns:
            if re.search(pattern, goal_description):
                features["structure_type"] = structure
                break
        
        # 检测知识类型
        knowledge_type = self.time_model._detect_knowledge_type(goal_description)
        features["knowledge_type"] = knowledge_type
        
        # 检测学习深度
        learning_level = self.time_model._detect_learning_level(goal_description)
        features["learning_level"] = learning_level
        
        # 推荐思维导图深度
        if quantity:
            if quantity <= 10:
                features["suggested_mindmap_depth"] = 2
            elif quantity <= 100:
                features["suggested_mindmap_depth"] = 3
            elif quantity <= 1000:
                features["suggested_mindmap_depth"] = 4
            else:
                features["suggested_mindmap_depth"] = 5
        
        # 推荐思维导图风格
        if "系统" in goal_description or "体系" in goal_description:
            features["suggested_mindmap_style"] = MindMapStyle.STRUCTURED.value
        elif "创意" in goal_description or "设计" in goal_description:
            features["suggested_mindmap_style"] = MindMapStyle.CREATIVE.value
        elif "深入" in goal_description or "深度" in goal_description:
            features["suggested_mindmap_style"] = MindMapStyle.DEEP.value
        
        # 估算时间
        if quantity:
            # 创建临时目标进行时间估算
            temp_goal = LearningGoal(
                id="temp",
                description=goal_description,
                target_knowledge_count=quantity,
                scale=features["scale"]
            )
            features["estimated_time_minutes"] = self.time_model.estimate_for_goal(temp_goal)
        
        return features
    
    def extract_quantity(self, description: str) -> Optional[int]:
        """提取数量信息"""
        import re
        
        # 匹配数字模式
        patterns = [
            r'(\d+)\s*个',        # "3500个"
            r'(\d+)\s*项',        # "100项"
            r'(\d+)\s*种',        # "50种"
            r'(\d+)\s*门',        # "10门"
            r'(\d+)\s*节',        # "20节"
            r'(\d+)',             # 纯数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description)
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue
        
        # 匹配中文数字
        chinese_numbers = {
            "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
            "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
            "百": 100, "千": 1000, "万": 10000,
            "两": 2, "几": 5, "多": 10
        }
        
        # 简单中文数字匹配
        for chinese, value in chinese_numbers.items():
            if chinese in description:
                # 查找中文数字+量词模式
                pattern = f'{chinese}[个项种门节]'
                if re.search(pattern, description):
                    return value
        
        return None
    
    def determine_scale(self, quantity: Optional[int]) -> GoalScale:
        """确定目标规模"""
        if not quantity:
            return GoalScale.MICRO
        
        if quantity <= 10:
            return GoalScale.MICRO
        elif quantity <= 100:
            return GoalScale.SMALL
        elif quantity <= 1000:
            return GoalScale.MEDIUM
        elif quantity <= 10000:
            return GoalScale.LARGE
        else:
            return GoalScale.MASSIVE

# ========== 辅助函数 ==========

def generate_id(prefix: str = "", seed: str = "") -> str:
    """生成唯一ID"""
    import time
    timestamp = int(time.time() * 1000)
    random_part = hashlib.md5(f"{seed}{random.random()}".encode()).hexdigest()[:8]
    return f"{prefix}{timestamp}_{random_part}"

def create_mindmap_tree(items: List[Any], 
                       grouping_key: Optional[Callable] = None,
                       max_depth: int = 3,
                       max_children: int = 5) -> Optional[MindMapNode]:
    """
    创建思维导图树
    
    Args:
        items: 项目列表
        grouping_key: 分组函数，如果不提供则自动分组
        max_depth: 最大深度
        max_children: 每个节点最大子节点数
    
    Returns:
        思维导图根节点
    """
    if not items:
        return None
    
    # 创建根节点
    root = MindMapNode(
        id=generate_id("mindmap_root_"),
        title="知识体系",
        description=f"包含{len(items)}个项目的知识体系",
        depth=0
    )
    
    # 如果没有分组函数，按首字母分组
    if grouping_key is None:
        def default_grouping(item):
            if isinstance(item, str):
                return item[0].upper() if item else "OTHER"
            else:
                return str(item)[0].upper() if str(item) else "OTHER"
        grouping_key = default_grouping
    
    # 第一层分组
    groups = {}
    for item in items:
        group = grouping_key(item)
        if group not in groups:
            groups[group] = []
        groups[item].append(item)
    
    # 如果组太多，合并小分组
    if len(groups) > max_children:
        # 按组大小排序
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        # 保留前max_children-1个最大的组，其余合并为"其他"
        main_groups = dict(sorted_groups[:max_children-1])
        other_items = []
        for group_name, group_items in sorted_groups[max_children-1:]:
            other_items.extend(group_items)
        
        if other_items:
            main_groups["其他"] = other_items
        
        groups = main_groups
    
    # 创建第一层子节点
    for group_name, group_items in groups.items():
        child_node = MindMapNode(
            id=generate_id("mindmap_child_"),
            title=f"分组: {group_name}",
            description=f"包含{len(group_items)}个项目",
            depth=1,
            parent_id=root.id,
            importance=min(len(group_items) / len(items) * 0.8 + 0.2, 1.0)
        )
        
        root.children_ids.append(child_node.id)
        
        # 如果深度允许，创建第二层节点
        if max_depth > 1 and len(group_items) > 0:
            # 简单创建第二层：前几个项目
            for i, item in enumerate(group_items[:max_children]):
                if i >= max_children:
                    break
                
                leaf_node = MindMapNode(
                    id=generate_id("mindmap_leaf_"),
                    title=str(item)[:20],
                    description="具体知识点",
                    depth=2,
                    parent_id=child_node.id,
                    importance=0.5
                )
                
                child_node.children_ids.append(leaf_node.id)
    
    return root

def calculate_learning_curve(total_items: int, 
                           learning_rate: float = 0.7,
                           initial_speed: float = 10.0) -> List[float]:
    """
    计算学习曲线（项目/小时）
    
    Args:
        total_items: 总项目数
        learning_rate: 学习率（0-1），越高表示学习越快
        initial_speed: 初始速度（项目/小时）
    
    Returns:
        每个批次的学习速度列表
    """
    speeds = []
    
    # 计算批次数量
    batch_size = max(10, total_items // 50)
    batches = math.ceil(total_items / batch_size)
    
    for i in range(batches):
        # 随着学习深入，速度可能先提高后稳定
        if i < batches * 0.3:  # 前30%：加速期
            speed = initial_speed * (1 + learning_rate * i / (batches * 0.3))
        elif i < batches * 0.8:  # 30%-80%：稳定期
            speed = initial_speed * (1 + learning_rate)
        else:  # 后20%：复习巩固期，速度可能下降
            speed = initial_speed * (1 + learning_rate * 0.8)
        
        speeds.append(speed)
    
    return speeds

def mindmap_to_dict_tree(mindmap_root: MindMapNode, 
                        node_map: Dict[str, MindMapNode]) -> Dict[str, Any]:
    """将思维导图转换为树形字典结构"""
    def build_tree(node: MindMapNode) -> Dict[str, Any]:
        tree_node = {
            "id": node.id,
            "title": node.title,
            "description": node.description,
            "depth": node.depth,
            "importance": node.importance,
            "difficulty": node.difficulty,
            "learning_status": node.learning_status,
            "estimated_time_minutes": node.estimated_time_minutes,
            "children": []
        }
        
        for child_id in node.children_ids:
            child_node = node_map.get(child_id)
            if child_node:
                tree_node["children"].append(build_tree(child_node))
        
        return tree_node
    
    return build_tree(mindmap_root)

# ========== 单例管理器 ==========

class FoundationManager:
    """基础管理器 - 提供全局访问点"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.goal_analyzer = SimpleGoalAnalyzer()
            self.time_model = TimeEstimationModel()
            self._initialized = True
    
    def get_goal_analyzer(self) -> IGoalAnalyzer:
        """获取目标分析器"""
        return self.goal_analyzer
    
    def get_time_model(self) -> TimeEstimationModel:
        """获取时间估算模型"""
        return self.time_model
    
    def create_learning_goal(self, description: str) -> LearningGoal:
        """创建学习目标（带分析）"""
        analysis = self.goal_analyzer.analyze(description)
        
        goal = LearningGoal(
            id=generate_id("goal_", description),
            description=description,
            scale=analysis["scale"],
            target_knowledge_count=analysis["estimated_items"],
            complexity=0.5 if analysis["complexity"] == "medium" else 
                      0.3 if analysis["complexity"] == "low" else 0.7,
            mindmap_depth=analysis["suggested_mindmap_depth"],
            mindmap_style=MindMapStyle(analysis["suggested_mindmap_style"]),
            metadata={"analysis": analysis}
        )
        
        return goal
    
    def create_mindmap_node(self, 
                           title: str, 
                           description: str = "",
                           depth: int = 0,
                           parent_id: Optional[str] = None) -> MindMapNode:
        """创建思维导图节点"""
        return MindMapNode(
            id=generate_id("mindmap_", title),
            title=title,
            description=description,
            depth=depth,
            parent_id=parent_id,
            importance=0.5,
            difficulty=0.5
        )

# ========== 测试代码 ==========

if __name__ == "__main__":
    print("🧪 测试基础模块（思维导图增强版）...")
    print("=" * 70)
    
    # 测试基础管理器
    manager = FoundationManager()
    analyzer = manager.get_goal_analyzer()
    
    test_goals = [
        "学习3500个常用汉字",
        "掌握Python编程基础",
        "了解人工智能的100个核心概念",
        "系统学习机器学习算法体系",
        "完成一个网页开发项目",
        "熟悉50个日常英语会话场景"
    ]
    
    print("\n📊 目标分析测试:")
    print("-" * 50)
    
    for goal_desc in test_goals:
        print(f"\n目标: '{goal_desc}'")
        analysis = analyzer.analyze(goal_desc)
        
        print(f"  规模: {analysis['scale'].value}")
        print(f"  数量: {analysis.get('quantity', '无')}")
        print(f"  复杂度: {analysis['complexity']}")
        print(f"  推荐思维导图深度: {analysis['suggested_mindmap_depth']}")
        print(f"  推荐思维导图风格: {analysis['suggested_mindmap_style']}")
        print(f"  预估时间: {analysis.get('estimated_time_minutes', 0)}分钟")
    
    # 测试时间估算模型
    print("\n\n⏰ 时间估算测试:")
    print("-" * 50)
    
    time_model = manager.get_time_model()
    
    test_cases = [
        ("汉字学习", "characters", 3500, LearningLevel.UNDERSTANDING),
        ("单词学习", "words", 5000, LearningLevel.FAMILIARITY),
        ("概念学习", "concepts", 100, LearningLevel.UNDERSTANDING),
    ]
    
    for name, ktype, quantity, level in test_cases:
        # 创建测试目标
        test_goal = LearningGoal(
            id="test",
            description=name,
            target_knowledge_count=quantity,
            scale=GoalScale.MASSIVE if quantity > 10000 else 
                  GoalScale.LARGE if quantity > 1000 else
                  GoalScale.MEDIUM if quantity > 100 else
                  GoalScale.SMALL if quantity > 10 else GoalScale.MICRO
        )
        
        # 估算时间
        minutes = time_model.estimate_for_goal(test_goal)
        hours = minutes / 60
        
        print(f"{name} ({quantity}个): {minutes}分钟 ≈ {hours:.1f}小时")
        
        # 显示时间安排选项
        schedule_options = time_model.generate_schedule_options(minutes)
        for option, details in schedule_options.items():
            print(f"  {option}: {details['estimated_days']:.0f}天 ({details['description']})")
    
    # 测试思维导图创建
    print("\n\n🧠 思维导图测试:")
    print("-" * 50)
    
    test_items = ["苹果", "香蕉", "橙子", "西瓜", "葡萄", "草莓", "蓝莓", "芒果", 
                  "菠萝", "桃子", "梨子", "樱桃", "柠檬", "柚子", "猕猴桃"]
    
    mindmap = create_mindmap_tree(test_items, max_depth=2, max_children=3)
    
    if mindmap:
        print(f"思维导图根节点: {mindmap.title}")
        print(f"子节点数量: {len(mindmap.children_ids)}")
        
        # 创建节点映射
        node_map = {mindmap.id: mindmap}
        # 实际应用中需要填充所有节点
        
        # 转换为树形结构
        tree = mindmap_to_dict_tree(mindmap, node_map)
        print(f"树结构深度: {tree.get('depth', 0)}")
    
    # 测试思维导图节点时间估算
    print("\n\n[T] Mindmap node time estimation test:")
    print("-" * 50)
    
    test_node = MindMapNode(
        id="test_node",
        title="测试节点",
        description="用于时间估算测试",
        depth=2,
        importance=0.8,
        difficulty=0.6,
        estimated_time_minutes=45
    )
    
    estimated_time = time_model.estimate_for_mindmap_node(test_node)
    print(f"节点: {test_node.title}")
    print(f"  深度: {test_node.depth}, 重要性: {test_node.importance:.2f}, 难度: {test_node.difficulty:.2f}")
    print(f"  预估时间: {estimated_time}分钟")
    print(f"  节点自带预估: {test_node.estimated_time_minutes}分钟")
    
    print("\n✅ 基础模块测试完成（思维导图增强版）")
    print("=" * 70)