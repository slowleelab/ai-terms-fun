---
title: Agentic RAG 智能体检索
slug: agentic-rag
category: 进阶专题
tags: [Agentic RAG, Agent, 多轮检索, 工具调用, 问题分解, ReAct]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# Agentic RAG 智能体检索

> 五层读懂一个词。这次拆的是：**Agentic RAG**--把 RAG 从"一次检索一次生成"升级为"LLM Agent 自主决定检索策略、多轮检索、工具调用、问题分解"，专治复杂问题。

---

## L1 · 一句话点破

**Agentic RAG = LLM Agent + 检索工具 + 多轮反思**。Agent 把检索当成工具，自己决定何时检索、检索几次、检索什么、要不要换工具，把 RAG 从死流程变成自主推理循环。

---

## L2 · 通俗类比

Naive RAG 是个只会"查一次资料就开答"的实习生。遇到简单问题够用，遇到复杂问题就拉胯：

- "对比 A、B、C 三家公司的 RLHF 方案" -- 一次检索召回不全
- "2024 年相比 2023 年，X 技术有什么进展" -- 需要分两次检索再对比
- "基于这份财报，公司明年的增长点在哪" -- 需要先检索财报，再检索行业数据，再综合

Agentic RAG 给这个实习生升级成一个**会自主规划的研究员**：

- 把复杂问题**拆解**成子问题（"对比 A/B/C" -> "查 A"、"查 B"、"查 C"）
- 每个子问题**独立检索**
- 检索结果**反思**够不够，不够再检
- 可以**调用多个工具**（向量检索、SQL 查询、web 搜索、计算器）
- 多轮迭代直到**自认为答得够好**

**和前几代 RAG 的区别**：

| 维度 | Naive RAG | Self-RAG / CRAG | Agentic RAG |
|------|-----------|-----------------|-------------|
| 检索次数 | 1 次 | 1 次 + 可能重检 | 多轮自主 |
| 工具 | 向量检索 | 向量 + web | 多工具组合 |
| 问题分解 | 无 | 无 | 有 |
| 规划 | 死流程 | 反思纠错 | 自主规划循环 |
| 复杂问题 | 差 | 中 | 好 |
| 成本 | 低 | 中 | 高（多轮 LLM） |

**代价**：多轮 LLM 调用，延迟从秒级涨到分钟级；token 开销大；Agent 可能陷入无效循环。所以 Agentic RAG 适合复杂问题，简单问题用 Naive RAG 更划算。

---

## L3 · 正经定义

**Agentic RAG**：把 RAG 包装为 LLM Agent 框架的应用，Agent 以检索、SQL 查询、web 搜索、计算等为工具，通过 ReAct / Plan-and-Solve / Reflection 等推理范式，自主决定检索时机、工具选择、多轮迭代，实现复杂问题的端到端解决。

**核心组件**：

1. **LLM 推理引擎**：Agent 的大脑，做规划、推理、反思
2. **工具集（Tools）**：
   - 向量检索（语义检索）
   - BM25 检索（关键词）
   - SQL 查询（结构化数据）
   - web 搜索（开放知识）
   - 计算器 / 代码执行
   - 知识图谱查询
3. **记忆（Memory）**：短期（本轮对话）+ 长期（跨轮积累）
4. **规划器（Planner）**：把复杂问题拆成子任务
5. **反思器（Reflector）**：评估中间结果，决定继续 / 重试 / 终止

**典型推理循环（ReAct 范式）**：

```
Thought: 这个问题需要对比三家公司的 RLHF 方案，
         我得分别检索每家。
Action: vector_search("OpenAI RLHF approach")
Observation: [chunk1, chunk2, ...]
Thought: OpenAI 的信息够了，接下来查 Anthropic
Action: vector_search("Anthropic constitutional AI")
Observation: [chunk3, chunk4, ...]
Thought: 两家都有了，查 Google DeepMind
Action: vector_search("DeepMind RLHF")
Observation: [chunk5, chunk6, ...]
Thought: 三家信息齐了，综合对比
Action: generate_answer(...)
Observation: 最终答案
```

**参考资料**：

- 📄 Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023, arXiv:2210.03629
- 📄 Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020（原始 RAG）
- 🔧 LangChain Agent 文档：https://python.langchain.com/docs/modules/agents/
- 🔧 LlamaIndex Agentic RAG：https://docs.llamaindex.ai/en/stable/optimizing/agentic_rag/
- 📄 Asai et al., *Self-RAG*, ICLR 2024（Agentic RAG 的反思能力来源）

---

## L4 · 原理深挖

### 4.1 从 RAG 到 Agentic RAG 的范式跃迁

Naive RAG 的控制流是**线性**的：`query -> retrieve -> generate`。Self-RAG / CRAG 加了反思 / 纠错，但仍是**单轮**：检一次，纠一次，答一次。

Agentic RAG 把控制流变成**循环**：

```
query
  ↓ 规划
子任务列表
  ↓ 循环:
     ├─ 选择工具
     ├─ 执行工具（检索/查询/计算）
     ├─ 观察结果
     ├─ 反思：够不够？要不要换工具？
     └─ 决策：继续 / 重试 / 完成
  ↓ 综合输出
最终答案
```

这个循环让 Agent 能处理：

- **多跳问题**（A 的导师的导师是谁）
- **对比问题**（A vs B vs C）
- **时序问题**（X 在 2023 vs 2024 的变化）
- **聚合问题**（基于多源数据综合判断）

### 4.2 推理范式：ReAct / Plan-and-Solve / Reflection

**ReAct**（Reasoning + Acting）：

```
Thought -> Action -> Observation -> Thought -> Action -> ...
```

每一步先思考（Thought），再决定动作（Action），观察结果（Observation），循环直到完成。ReAct 是 Agentic RAG 最常用的范式。

**Plan-and-Solve**：

```
Plan: 把问题拆成 [子任务1, 子任务2, 子任务3]
for 子任务 in Plan:
    Execute(子任务)
Synthesize(所有子任务结果)
```

先规划再执行，适合复杂多步问题。比 ReAct 更结构化，但规划错了整个流程就崩。

**Reflection**：

```
Generate -> Critique -> Revise -> Generate -> ...
```

生成后自我批评，发现不足就修订，循环到满意。Reflection 提升答案质量但增加延迟。

**实际系统常混合**：Plan-and-Solve 做整体规划，每个子任务用 ReAct 执行，最后用 Reflection 优化。

### 4.3 工具调用的实现

Agent 通过**函数调用（function calling）**机制使用工具：

```python
tools = [
    {
        "name": "vector_search",
        "description": "在知识库中语义检索相关 chunk",
        "parameters": {
            "query": {"type": "string", "description": "检索 query"},
            "top_k": {"type": "integer", "default": 5}
        }
    },
    {
        "name": "sql_query",
        "description": "查询结构化数据库",
        "parameters": {
            "sql": {"type": "string", "description": "SQL 语句"}
        }
    },
    {
        "name": "web_search",
        "description": "web 搜索开放知识",
        "parameters": {
            "query": {"type": "string"}
        }
    }
]

def agentic_rag(query, llm, tools, max_steps=10):
    messages = [{"role": "user", "content": query}]
    for step in range(max_steps):
        # LLM 决定下一步动作
        response = llm.chat(messages, tools=tools)
        if response.finish_reason == "stop":
            return response.content  # 最终答案
        # 执行工具调用
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call.name, tool_call.arguments)
            messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
    return "达到最大步数，未能完成"
```

**关键设计**：

- `max_steps` 防止无限循环
- 工具描述要清晰，否则 LLM 选错工具
- 工具返回要结构化（JSON），方便 LLM 解析

### 4.4 问题分解（Query Decomposition）

复杂问题先拆解：

```python
def decompose_query(query, llm):
    prompt = f"""把以下复杂问题拆成可独立检索的子问题：
    问题：{query}
    输出 JSON 数组：["子问题1", "子问题2", ...]
    """
    sub_queries = llm.generate(prompt)
    return json.loads(sub_queries)

# 示例
query = "对比 OpenAI、Anthropic、Google 的对齐方法"
sub_queries = [
    "OpenAI 的对齐方法是什么",
    "Anthropic 的对齐方法是什么",
    "Google 的对齐方法是什么"
]
```

**分解策略**：

- **并列分解**：对比类问题拆成多个独立子问题
- **递进分解**：多跳问题拆成依赖链（先查 A，再用 A 查 B）
- **混合**：先并列再递进

**坑**：分解过细导致检索爆炸，分解过粗没解决问题。要做分解质量评估。

### 4.5 记忆与上下文管理

Agentic RAG 多轮交互产生大量中间结果，记忆管理是关键：

**短期记忆**：本轮对话的 Thought / Action / Observation 序列。问题：序列长了 LLM context 爆炸。解决：摘要压缩 / 滑动窗口。

**长期记忆**：跨轮积累的知识。问题：如何持久化、如何检索。解决：向量库 + 元数据。

**记忆管理策略**：

```python
def manage_memory(messages, max_tokens=4000):
    total = sum(len(m.content) for m in messages)
    while total > max_tokens:
        # 压缩最早的中间步骤
        old = messages.pop(0)
        summary = llm.summarize(old.content)
        messages.insert(0, {"role": "system", "content": f"历史摘要：{summary}"})
        total = sum(len(m.content) for m in messages)
    return messages
```

### 4.6 Agentic RAG vs Self-RAG / CRAG 的关系

Agentic RAG 不是 Self-RAG / CRAG 的替代，而是**上层框架**，可以集成它们：

- **Agentic RAG + Self-RAG**：Agent 的反思能力用 Self-RAG 反思 token 实现
- **Agentic RAG + CRAG**：Agent 的检索工具内嵌 CRAG 纠错逻辑
- **Agentic RAG + GraphRAG**：Agent 把 GraphRAG 的 Global / Local Search 当工具调用

```
Agentic RAG（上层框架）
  ├─ 工具1: Naive 向量检索
  ├─ 工具2: CRAG 纠错检索
  ├─ 工具3: GraphRAG 全局检索
  ├─ 工具4: SQL 查询
  ├─ 工具5: web 搜索
  └─ 反思能力: Self-RAG 风格
```

### 4.7 成本与延迟分析

Agentic RAG 的成本是 Naive RAG 的 **10-100 倍**：

| 维度 | Naive RAG | Agentic RAG |
|------|-----------|-------------|
| LLM 调用次数 | 1 | 5-20（每步一次） |
| 检索次数 | 1 | 3-10 |
| 端到端延迟 | 1-3 秒 | 30 秒 - 5 分钟 |
| token 开销 | ~1000 | 10000-50000 |
| 复杂问题准确率 | 低 | 高 |

**降本策略**：

- 简单问题路由到 Naive RAG，复杂问题才上 Agentic
- 用小模型做规划，大模型做最终生成
- 缓存中间结果
- 限制 `max_steps`

### 4.8 何时用 Agentic RAG

**适合**：

- 多跳推理
- 多源数据综合
- 对比分析
- 需要工具组合（检索 + 计算 + SQL）
- 开放式研究问题

**不适合**：

- 简单事实查询（用 Naive RAG）
- 实时性要求高（延迟受不了）
- 成本敏感场景
- 简单单跳 QA

**路由策略**：

```python
def route_rag(query, classifier):
    complexity = classifier.predict(query)  # 简单 / 中等 / 复杂
    if complexity == "simple":
        return naive_rag(query)
    elif complexity == "medium":
        return self_rag(query)  # 或 CRAG
    else:
        return agentic_rag(query)
```

### 4.9 Agentic RAG 的工程挑战

**挑战 1：循环控制**。Agent 可能陷入无效循环（反复检索同样内容）。要设 `max_steps` + 重复检测。

**挑战 2：工具选择错误**。LLM 选错工具（该用 SQL 用了向量检索）。要工具描述清晰 + few-shot 示例。

**挑战 3：中间结果丢失**。多轮中间结果在 context 里被压缩丢失。要做关键信息持久化。

**挑战 4：评估困难**。Agentic RAG 的中间步骤多，端到端评估难归因。要分步评估 + 端到端评估结合。

**挑战 5：成本不可控**。复杂问题可能烧几十次 LLM 调用。要硬性 `max_steps` + 成本上限。

**挑战 6：可解释性**。Agent 的推理链长，用户难理解。要可视化 Thought / Action 链。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2022-10**：ReAct 论文提出 Thought-Action-Observation 范式
- **2023**：LangChain / AutoGPT 等 Agent 框架兴起
- **2023 下半年**：Agent + RAG 结合，Agentic RAG 概念成型
- **2024**：LlamaIndex 推出 Agentic RAG 官方支持；Self-RAG / CRAG 的反思思想被 Agent 吸收
- **2024 下半年**：多 Agent 协作 RAG（如 CrewAI、AutoGen）出现
- **2025**：Agentic RAG 成为企业级 RAG 标配，与 workflow 编排融合

### 5.2 常见坑

**坑 1：没有 max_steps**。Agent 陷入无限循环，烧光 token。必须设硬性步数上限。

**坑 2：工具描述模糊**。LLM 选错工具或参数填错。工具描述要详细 + 给 few-shot 示例。

**坑 3：忘了重复检测**。Agent 反复检索同样 query，浪费且无进展。要检测重复 Action 并打断。

**坑 4：context 爆炸**。多轮中间结果全塞 context，LLM 爆 context window。要做记忆压缩 / 滑动窗口。

**坑 5：简单问题也上 Agentic**。简单事实查询用 Agentic RAG 是杀鸡用牛刀，成本高延迟大。要路由分流。

**坑 6：评估只看最终答案**。Agentic RAG 中间步骤错了一步，最终答案可能歪。要分步评估。

**坑 7：反思不闭环**。Agent 反思"不够"但不触发重检索 / 重写，反思成摆设。反思要驱动 Action。

**坑 8：工具结果不结构化**。工具返回自然语言，LLM 解析错。工具返回要 JSON / 结构化。

**坑 9：多 Agent 协作无协调**。多 Agent 各干各的，结果冲突。要协调机制 / 投票 / 仲裁。

**坑 10：延迟没告知用户**。Agentic RAG 几十秒到几分钟，用户以为卡死。要流式输出 + 进度提示。

**坑 11：成本无上限**。复杂问题烧几十次 LLM 调用，账单爆炸。要 token / 调用次数预算。

**坑 12：Agent 幻觉传染**。Agent 中间步幻觉，后续基于错误信息继续推理，错误放大。要中间结果事实核查。

### 5.3 面试怎么考

1. **Agentic RAG 和 Naive RAG 的本质区别？** 答：Naive 单轮检索生成，Agentic 多轮自主推理循环，Agent 把检索当工具自主调度。
2. **Agentic RAG 的推理范式有哪些？** 答：ReAct（Thought-Action-Observation）、Plan-and-Solve（先规划再执行）、Reflection（生成-批评-修订）。
3. **Agentic RAG 和 Self-RAG / CRAG 的关系？** 答：Agentic 是上层框架，可集成 Self-RAG 的反思、CRAG 的纠错、GraphRAG 的图谱检索作为工具。
4. **Agentic RAG 的成本为什么高？** 答：多轮 LLM 调用（5-20 次）+ 多次检索（3-10 次）+ 记忆管理，延迟和 token 都是 Naive RAG 的 10-100 倍。
5. **Agentic RAG 什么时候不适用？** 答：简单事实查询、高实时性、成本敏感、简单单跳 QA。

---

## 速记卡

| 组件 | 作用 |
|------|------|
| LLM 推理引擎 | 规划、推理、反思 |
| 工具集 | 向量/BM25/SQL/web/计算 |
| 记忆 | 短期 + 长期 |
| 规划器 | 问题分解 |
| 反思器 | 评估 + 决策 |

**推理范式**：

| 范式 | 流程 | 适用 |
|------|------|------|
| ReAct | Thought-Action-Observation 循环 | 通用 |
| Plan-and-Solve | 先规划子任务再执行 | 复杂多步 |
| Reflection | 生成-批评-修订 | 质量优化 |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| `max_steps` | 5-15 | 防无限循环 |
| 工具数 | 3-8 | 选择难度 |
| 短期记忆 | 4k-8k token | context 限制 |
| 路由阈值 | 分类器决定 | 简单 vs 复杂分流 |

**一句话记忆**：Agentic RAG = LLM Agent 把检索当工具，ReAct / Plan-and-Solve / Reflection 多轮自主推理循环，专治复杂问题（多跳 / 对比 / 聚合）。代价是 LLM 调用 10-100x、延迟分钟级、成本爆炸。简单问题别用，路由分流是标配。Agentic 是上层框架，可集成 Self-RAG / CRAG / GraphRAG 作为工具。

---

> *上一篇：[Corrective RAG 检索纠错](./corrective-rag) -- 即插即用的检索质检 + web 兜底。*
> *下一篇：[Multi-modal RAG 多模态检索](./multimodal-rag) -- 文本 + 图像 + 表格统一向量空间，跨模态检索。*
