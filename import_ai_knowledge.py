#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionLearner — AI前沿与智能体架构知识体系导入脚本
涵盖：大模型原理、RAG、Agent、多模态、MCP、代码智能体、AI基础设施

用法：放到项目根目录，运行：
    python import_ai_knowledge.py

共 6 个目标，62 个单元，约 370 个知识节点
"""

import json, hashlib, re, sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("./learning_data")

def n(title, imp=0.8, t="concept"):
    return {"title": title, "importance": imp, "node_type": t}

# node_type: concept / fact / example / skill / question

KNOWLEDGE_PLAN = [

# ══════════════════════════════════════════════════════════════
# 1. 大语言模型原理与工程
# ══════════════════════════════════════════════════════════════
{
"description": "大语言模型：原理、训练与工程实践",
"goal_type": "general", "vision_score": 0.95,
"vision_pathway": "path_ai_core", "priority": 1,
"reason": "理解LLM是所有AI应用的基础，不懂原理就是黑盒调用",
"units": {

"Transformer架构：注意力机制的本质": [
    n("自注意力机制是什么，为什么有效", 1.0),
    n("Query/Key/Value 三个矩阵的直觉解释", 1.0),
    n("多头注意力：并行关注不同语义维度", 0.9),
    n("位置编码：让模型知道词的顺序", 0.9),
    n("Feed-Forward 层的作用", 0.8),
    n("LayerNorm 和残差连接为什么重要", 0.8),
    n("Encoder-Decoder vs Decoder-Only 的区别", 0.9, "fact"),
],

"预训练：LLM 的知识从哪里来": [
    n("下一个token预测（Next Token Prediction）任务", 1.0),
    n("预训练数据的规模和质量对模型的影响", 0.9, "fact"),
    n("Scaling Law：模型越大越好的规律和边界", 1.0, "fact"),
    n("涌现能力（Emergent Abilities）是什么", 0.9),
    n("预训练的计算成本：为什么贵", 0.8, "fact"),
    n("Common Crawl、Books、代码数据的作用", 0.7),
],

"指令微调与对齐：让模型听话": [
    n("SFT（监督微调）的基本流程", 1.0, "skill"),
    n("RLHF：用人类反馈强化学习的原理", 1.0),
    n("奖励模型（Reward Model）是什么", 0.9),
    n("PPO 算法在 RLHF 中的角色", 0.8),
    n("DPO（直接偏好优化）比 RLHF 简单在哪", 0.9),
    n("对齐税：为什么对齐后模型能力会下降", 0.8, "fact"),
    n("Constitutional AI（Claude的对齐方法）", 0.8),
],

"推理优化：让模型更快更便宜": [
    n("KV Cache：推理加速的核心机制", 1.0),
    n("量化（Quantization）：4bit/8bit 的原理和损失", 0.9),
    n("投机解码（Speculative Decoding）原理", 0.8),
    n("批处理（Batching）对吞吐量的影响", 0.8),
    n("vLLM/TensorRT-LLM 做了什么优化", 0.7, "example"),
    n("Flash Attention 解决了什么问题", 0.8),
],

"Prompt Engineering：工程师的核心技能": [
    n("System Prompt 的作用和最佳实践", 1.0, "skill"),
    n("Few-shot vs Zero-shot：什么时候用哪个", 0.9, "skill"),
    n("Chain-of-Thought（思维链）为什么有效", 1.0),
    n("ReAct 模式：推理+行动的结合", 0.9),
    n("温度（Temperature）和 Top-p 的调节", 0.8, "skill"),
    n("结构化输出：让模型稳定返回 JSON", 0.9, "skill"),
    n("Prompt 注入攻击和防御", 0.7),
],

"上下文窗口与长文本处理": [
    n("Context Window 的限制和发展趋势", 0.9, "fact"),
    n("Lost in the Middle：长文中间内容被遗忘的问题", 1.0, "fact"),
    n("RAG vs 长上下文：什么时候用哪个", 1.0),
    n("滑动窗口和分块处理的策略", 0.8, "skill"),
    n("Needle in Haystack 测试的意义", 0.7, "example"),
],

},
},


# ══════════════════════════════════════════════════════════════
# 2. RAG：检索增强生成
# ══════════════════════════════════════════════════════════════
{
"description": "RAG系统设计与工程实践",
"goal_type": "general", "vision_score": 0.93,
"vision_pathway": "path_ai_engineering", "priority": 2,
"reason": "RAG是当前AI应用最核心的工程模式，你的VisionLearner本质就是RAG系统",
"units": {

"向量数据库：语义检索的基础设施": [
    n("Embedding是什么：把文字变成向量", 1.0),
    n("余弦相似度 vs 欧氏距离：哪个更好", 0.8, "fact"),
    n("FAISS/ChromaDB/Weaviate 的适用场景", 0.9, "example"),
    n("ANN（近似最近邻）算法为什么比精确搜索快", 0.8),
    n("向量索引：HNSW/IVF 的原理", 0.7),
    n("Embedding 模型的选择：OpenAI vs 本地模型", 0.9, "skill"),
    n("多语言 Embedding 的挑战", 0.7, "fact"),
],

"文档分块策略：RAG 成败的关键": [
    n("固定大小分块 vs 语义分块的区别", 1.0),
    n("Chunk 大小对检索质量的影响", 1.0, "fact"),
    n("重叠分块（Overlap）为什么能改善效果", 0.9),
    n("按句子/段落/章节分块的适用场景", 0.9, "skill"),
    n("元数据（标题/来源/日期）对检索的帮助", 0.8),
    n("父子分块（Parent-Child Chunking）策略", 0.8),
],

"检索策略：从粗到精": [
    n("稀疏检索（BM25/TF-IDF）的原理和局限", 1.0),
    n("密集检索（Dense Retrieval）的原理", 1.0),
    n("混合检索（Hybrid Search）：两种方式结合", 0.9),
    n("重排序（Reranking）：二阶段检索提升精度", 0.9),
    n("查询扩展（Query Expansion）：让检索更准", 0.8),
    n("HyDE（假设文档嵌入）的原理", 0.7),
],

"RAG 系统评估与优化": [
    n("RAG 评估的三个维度：忠实度/相关性/完整性", 1.0),
    n("RAGAS 评估框架的使用", 0.8, "skill"),
    n("幻觉（Hallucination）的来源和检测", 1.0),
    n("检索召回率 vs 精确率的权衡", 0.9),
    n("上下文压缩：减少噪音提升质量", 0.8),
    n("RAG 常见失败模式和对策", 0.9, "fact"),
],

"Advanced RAG：前沿改进方向": [
    n("Self-RAG：模型自己判断是否需要检索", 0.9),
    n("CRAG（纠正性RAG）：检索质量自我评估", 0.8),
    n("Graph RAG：用知识图谱增强检索", 0.9),
    n("Multi-hop Retrieval：多步推理检索", 0.8),
    n("RAG Fusion：多路检索结果融合", 0.7),
    n("Agentic RAG：Agent 主动决策检索策略", 1.0),
],

},
},


# ══════════════════════════════════════════════════════════════
# 3. AI Agent 与多智能体系统
# ══════════════════════════════════════════════════════════════
{
"description": "AI Agent架构与多智能体系统设计",
"goal_type": "general", "vision_score": 0.96,
"vision_pathway": "path_ai_agent", "priority": 3,
"reason": "Agent是2024-2025年最热方向，是AI从问答走向自主行动的核心",
"units": {

"Agent 的本质：感知-决策-行动循环": [
    n("什么是 AI Agent，和普通 LLM 的区别", 1.0),
    n("感知（Perception）：Agent 能接收什么输入", 0.9),
    n("记忆（Memory）：短期/长期/外部记忆的设计", 1.0),
    n("规划（Planning）：如何把目标拆解成步骤", 1.0),
    n("工具使用（Tool Use）：Function Calling 的机制", 1.0, "skill"),
    n("行动（Action）：Agent 能操作什么", 0.9),
],

"ReAct 与思维链规划": [
    n("ReAct 框架：Reasoning + Acting 交替进行", 1.0),
    n("Thought-Action-Observation 循环的实现", 1.0, "skill"),
    n("Chain-of-Thought 在 Agent 中的应用", 0.9),
    n("Tree of Thought：分支探索策略", 0.8),
    n("Plan-and-Execute：先规划再执行的模式", 0.9),
    n("Self-Reflection：Agent 自我纠错机制", 0.9),
],

"工具调用与 Function Calling": [
    n("Function Calling 的标准格式（OpenAI规范）", 1.0, "skill"),
    n("工具定义：如何写好 JSON Schema", 1.0, "skill"),
    n("并行工具调用 vs 串行工具调用", 0.9),
    n("工具执行结果如何返回给模型", 0.9, "skill"),
    n("工具调用失败的处理策略", 0.8),
    n("工具的安全性：权限控制和沙箱", 0.8),
],

"MCP（模型上下文协议）：工具标准化": [
    n("MCP 是什么：Anthropic 提出的工具协议", 1.0),
    n("MCP Server 和 MCP Client 的职责", 1.0),
    n("MCP 的三种能力：Tools/Resources/Prompts", 1.0),
    n("MCP vs Function Calling 的区别", 0.9, "fact"),
    n("如何写一个 MCP Server（Python）", 0.9, "skill"),
    n("MCP 生态：现有的 MCP Server 有哪些", 0.8, "example"),
    n("MCP 的传输层：stdio vs SSE", 0.7),
],

"记忆系统设计": [
    n("短期记忆：对话上下文窗口的管理", 1.0),
    n("长期记忆：向量库存储经验", 1.0),
    n("情节记忆 vs 语义记忆的区别", 0.8),
    n("记忆压缩：如何保留重要信息", 0.9),
    n("记忆检索：什么时候调用什么记忆", 0.9),
    n("记忆遗忘策略：不是所有东西都该留", 0.7),
],

"多智能体系统（Multi-Agent）": [
    n("为什么需要多个 Agent 协作", 1.0),
    n("Orchestrator-Worker 模式：主从架构", 1.0),
    n("Agent 之间如何通信和传递任务", 0.9),
    n("并行 Agent：同时执行多个子任务", 0.9),
    n("角色专业化：不同 Agent 负责不同职责", 0.8),
    n("多 Agent 的一致性和冲突解决", 0.8),
    n("AutoGen/CrewAI/LangGraph 框架对比", 0.9, "example"),
],

"Agent 评估与可靠性": [
    n("Agent 任务成功率的定义和度量", 0.9),
    n("幻觉在 Agent 中的危害比对话更大", 1.0, "fact"),
    n("循环（Loop）检测：防止 Agent 无限重复", 0.9),
    n("人工确认节点（Human-in-the-loop）的设计", 1.0),
    n("Agent 测试框架的设计", 0.8, "skill"),
    n("Agent 失败的常见模式", 0.9, "fact"),
],

},
},


# ══════════════════════════════════════════════════════════════
# 4. 代码智能体与软件工程AI
# ══════════════════════════════════════════════════════════════
{
"description": "代码智能体：AI辅助软件工程",
"goal_type": "general", "vision_score": 0.94,
"vision_pathway": "path_ai_coding", "priority": 4,
"reason": "这是你系统最终要做到的目标：能看懂代码、发现问题、生成代码",
"units": {

"代码理解：AST与静态分析": [
    n("AST（抽象语法树）是什么，能提取什么信息", 1.0),
    n("Python ast 模块的核心用法", 1.0, "skill"),
    n("调用关系图的构建方法", 1.0, "skill"),
    n("数据流分析：变量从哪来到哪去", 0.8),
    n("控制流分析：代码执行路径", 0.8),
    n("代码相似度检测：找重复逻辑", 0.7),
],

"代码生成的工程实践": [
    n("代码生成的 Prompt 模式：上下文注入", 1.0, "skill"),
    n("项目上下文的提取和压缩", 1.0, "skill"),
    n("生成代码的自动验证：语法/类型/测试", 0.9),
    n("渐进式生成：先框架后细节", 0.8),
    n("代码风格一致性的保持", 0.8),
    n("如何让模型遵循现有代码规范", 0.9, "skill"),
],

"代码智能体的架构（Cursor/Claude Code的做法）": [
    n("LSP（语言服务器协议）：IDE 的信息来源", 0.9),
    n("文件系统工具：读/写/搜索代码文件", 1.0),
    n("终端执行工具：运行代码看结果", 1.0),
    n("错误反馈循环：看报错→分析→修改→重试", 1.0),
    n("代码搜索工具：grep/ast 搜索相关代码", 0.9),
    n("Git 工具：查看历史/diff/blame", 0.8),
],

"自动测试生成": [
    n("从函数签名自动生成测试用例", 0.9, "skill"),
    n("边界条件的自动识别", 0.8),
    n("Mutation Testing：验证测试有效性", 0.7),
    n("测试覆盖率分析与提升", 0.8, "skill"),
    n("LLM 生成测试的常见问题", 0.8, "fact"),
],

"代码审查与重构建议": [
    n("自动 Code Review 的实现思路", 0.9),
    n("重构模式识别：何时该提取函数/类", 1.0),
    n("技术债务的量化：如何计算复杂度", 0.8),
    n("基于项目历史的重构建议", 0.9),
    n("安全漏洞的静态检测", 0.8),
],

},
},


# ══════════════════════════════════════════════════════════════
# 5. 多模态与前沿模型
# ══════════════════════════════════════════════════════════════
{
"description": "多模态AI与前沿模型进展",
"goal_type": "general", "vision_score": 0.88,
"vision_pathway": "path_ai_frontier", "priority": 5,
"reason": "了解AI的整体发展方向，避免只会用文本模型",
"units": {

"视觉语言模型（VLM）": [
    n("图文对齐：CLIP 的训练方式", 0.9),
    n("视觉 Encoder 和语言 Decoder 如何连接", 0.9),
    n("GPT-4V/Claude/Gemini 的多模态能力对比", 0.8, "fact"),
    n("图像理解 vs 图像生成的区别", 0.8),
    n("VLM 的主要应用场景", 0.8, "example"),
    n("文档理解（PDF/表格/截图）的工程实现", 0.9, "skill"),
],

"扩散模型与图像生成": [
    n("扩散模型的核心原理：加噪和去噪", 0.9),
    n("Stable Diffusion 的架构组成", 0.8),
    n("LoRA 微调：用少量数据定制风格", 0.8, "skill"),
    n("ControlNet：精确控制生成结果", 0.7),
    n("文生图的 Prompt 工程技巧", 0.8, "skill"),
    n("FLUX/SD3 等新一代模型的改进", 0.7, "fact"),
],

"推理模型与慢思考": [
    n("o1/o3/DeepSeek-R1 的推理方式", 1.0),
    n("思维链扩展（Test-time Compute）的原理", 1.0),
    n("慢思考 vs 快思考：什么任务用哪个", 0.9, "skill"),
    n("强化学习训练推理能力的方法", 0.8),
    n("推理模型的 token 消耗问题", 0.8, "fact"),
    n("推理模型适合和不适合的任务", 0.9, "fact"),
],

"小型语言模型（SLM）与端侧部署": [
    n("为什么需要小模型：隐私/延迟/成本", 1.0),
    n("Qwen2.5/Phi/Gemma 等小模型的能力", 0.9, "fact"),
    n("模型蒸馏：从大模型提炼小模型", 0.8),
    n("端侧推理：llama.cpp/Ollama/MLC", 0.9, "skill"),
    n("量化对小模型的影响更显著", 0.7, "fact"),
    n("小模型的微调：LoRA 在本地的实践", 0.8, "skill"),
],

},
},


# ══════════════════════════════════════════════════════════════
# 6. AI 应用架构与工程
# ══════════════════════════════════════════════════════════════
{
"description": "AI应用工程：从原型到生产",
"goal_type": "general", "vision_score": 0.91,
"vision_pathway": "path_ai_engineering", "priority": 6,
"reason": "有了算法还需要工程落地，这是把AI变成真实产品的关键",
"units": {

"LLM 应用的系统设计": [
    n("LLM 应用的典型架构模式", 1.0),
    n("同步 vs 流式（Streaming）响应的选择", 0.9),
    n("Token 成本控制的工程策略", 1.0, "skill"),
    n("缓存层设计：相同问题不重复调用", 0.9, "skill"),
    n("降级策略：主模型挂了怎么办", 0.8),
    n("多模型路由：根据任务选择模型", 0.8),
],

"LangChain/LlamaIndex 框架解析": [
    n("LangChain 的核心抽象：Chain/Agent/Memory", 0.9),
    n("LangChain 适合什么场景，不适合什么", 0.9, "fact"),
    n("LlamaIndex 专注 RAG 的设计思路", 0.9),
    n("LangGraph：用图来编排复杂 Agent 流程", 1.0),
    n("什么时候自己写，什么时候用框架", 1.0, "skill"),
    n("框架的过度抽象问题和调试难度", 0.8, "fact"),
],

"生产环境的可观测性": [
    n("LLM 调用的 Tracing：每一步发生了什么", 1.0),
    n("Prompt 版本管理：像代码一样管理提示词", 0.9, "skill"),
    n("LangSmith/Langfuse 等观测工具", 0.8, "example"),
    n("延迟监控：P50/P95/P99 的意义", 0.8),
    n("成本追踪：每个功能花了多少钱", 0.9, "skill"),
    n("异常检测：幻觉和错误的自动发现", 0.8),
],

"AI 产品的评估体系": [
    n("离线评估 vs 在线评估的区别", 0.9),
    n("Golden Dataset（黄金测试集）的建立", 1.0, "skill"),
    n("LLM-as-Judge：用模型评估模型输出", 0.9),
    n("A/B 测试在 AI 产品中的应用", 0.8, "skill"),
    n("用户反馈数据的收集和利用", 0.9),
    n("回归测试：新版本不能比旧版差", 0.9, "skill"),
],

"AI 安全与对抗": [
    n("Prompt 注入攻击的原理和防御", 1.0),
    n("越狱（Jailbreak）的常见手段", 0.8),
    n("敏感信息泄露的防护", 0.9),
    n("输入/输出过滤的工程实现", 0.8, "skill"),
    n("速率限制和滥用检测", 0.7),
    n("AI 系统的审计日志设计", 0.7),
],

"本地部署与私有化": [
    n("Ollama 的架构和使用", 1.0, "skill"),
    n("vLLM 生产部署的配置", 0.8, "skill"),
    n("GPU 内存管理：多模型共享显存", 0.8),
    n("模型选型：能力/成本/延迟的三角权衡", 1.0, "skill"),
    n("API 兼容层：OpenAI 接口标准的意义", 0.9),
    n("边缘部署：在资源受限设备上运行模型", 0.7),
],

},
},

]


# ══════════════════════════════════════════════════════════════
# 以下是导入逻辑（和原脚本一致，不需要修改）
# ══════════════════════════════════════════════════════════════

def safe_key(text: str) -> str:
    prefix = re.sub(r'[^\w]', '', text[:4])
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{prefix}_{h}"

def make_node(title, importance=0.8, node_type="concept",
              depth=1, parent_id="", node_id=None):
    ts = datetime.now().isoformat()
    nid = node_id or f"node_{int(datetime.now().timestamp()*1000)}_{hash(title)&0xffff}"
    return {
        "id": nid, "title": title,
        "description": f"{title}的相关知识",
        "content": None, "collected": False,
        "collected_at": "", "collected_by": "",
        "depth": depth, "node_type": node_type,
        "importance": importance, "difficulty": 0.5,
        "prerequisite_score": 0.0,
        "learning_status": "pending",
        "estimated_time_minutes": 30,
        "actual_time_minutes": 0,
        "parent_id": parent_id,
        "children_ids": [], "sibling_ids": [],
        "knowledge_node_ids": [], "prerequisites": [],
        "related_nodes": [], "tags": [], "notes": "",
        "confidence": 0.8, "generated_by": "import_script",
        "generated_at": ts, "generation_prompt": "",
        "children": [],
    }

def build_tree(unit_name: str, child_specs: list) -> dict:
    ts = datetime.now().isoformat()
    root_id = f"node_{int(datetime.now().timestamp()*1000)}_root"
    root = make_node(unit_name, 1.0, "concept", 0, "", root_id)
    children = []
    for spec in child_specs:
        child_id = f"node_{int(datetime.now().timestamp()*1000)}_{hash(spec['title'])&0xffff}"
        child = make_node(
            spec["title"], spec.get("importance", 0.8),
            spec.get("node_type", "concept"), 1, root_id, child_id
        )
        root["children_ids"].append(child_id)
        children.append(child)
    root["children"] = children
    return root

def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")

def goal_exists(goal_id: str) -> bool:
    p = DATA_DIR / "goals" / f"{goal_id}.json"
    return p.exists()

def import_plan():
    print("\n" + "="*60)
    print("  VisionLearner — AI前沿知识体系导入")
    print("="*60)

    total_units = sum(len(g["units"]) for g in KNOWLEDGE_PLAN)
    total_nodes = sum(
        sum(len(nodes) for nodes in g["units"].values())
        for g in KNOWLEDGE_PLAN
    )
    print(f"\n计划导入：{len(KNOWLEDGE_PLAN)} 个目标，"
          f"{total_units} 个单元，约 {total_nodes} 个知识节点\n")

    for goal_spec in KNOWLEDGE_PLAN:
        desc = goal_spec["description"]
        goal_id = "goal_" + hashlib.md5(desc.encode()).hexdigest()[:12]

        print(f"{'─'*50}")
        print(f"目标 [{goal_spec['priority']}]：{desc}")
        print(f"  goal_id: {goal_id}")

        # ── 保存目标文件 ──
        goal_path = DATA_DIR / "goals" / f"{goal_id}.json"
        if goal_path.exists():
            print(f"  [!] 已存在，跳过")
            continue

        goal_data = {
            "id": goal_id,
            "description": desc,
            "goal_type": goal_spec.get("goal_type", "general"),
            "status": "active",
            "vision_score": goal_spec.get("vision_score", 0.9),
            "vision_pathway": goal_spec.get("vision_pathway", "path_ai"),
            "priority": goal_spec.get("priority", 5),
            "reason": goal_spec.get("reason", ""),
            "created_at": datetime.now().isoformat(),
            "_updated_at": datetime.now().isoformat(),
        }
        save_json(goal_path, goal_data)

        # ── 保存 unit 列表 ──
        units = list(goal_spec["units"].keys())
        unit_list_path = (DATA_DIR / "goal_units" /
                          f"{goal_id}_units.json")
        save_json(unit_list_path, {
            "goal_id": goal_id, "units": units,
            "created_at": datetime.now().isoformat(),
            "_updated_at": datetime.now().isoformat(),
        })

        # ── 保存每个 unit 的知识树 ──
        print(f"  导入 {len(units)} 个单元：")
        for unit_name, child_specs in goal_spec["units"].items():
            tree = build_tree(unit_name, child_specs)
            tree_key = f"{goal_id}_{safe_key(unit_name)}"
            tree_path = (DATA_DIR / "mindmap_trees" /
                         f"{tree_key}.json")
            if tree_path.exists():
                print(f"    [SKIP] {unit_name[:30]} 已存在")
                continue
            save_json(tree_path, {
                "goal_id": goal_id, "unit": unit_name,
                "tree": tree,
                "saved_at": datetime.now().isoformat(),
                "completion_rate": 0.0,
                "_updated_at": datetime.now().isoformat(),
            })
            print(f"    [OK] {unit_name[:40]} "
                  f"({len(child_specs)} 节点)")

        print(f"  [OK] {desc} 导入完成")

    print(f"\n{'='*60}")
    print("[OK] 全部导入完成！")
    print()
    print("下一步：")
    print("  1. 启动系统：python main.py --web")
    print("  2. 在目标列表选择想学习的目标")
    print("  3. 点击「批量学习」开始填充知识节点")
    print("  4. 学完后直接提问，系统从本地知识库回答")
    print("="*60 + "\n")

if __name__ == "__main__":
    # 检查目录
    if not Path("main.py").exists():
        print("[X] 请在 VisionLearner 项目根目录下运行此脚本")
        sys.exit(1)
    import_plan()
