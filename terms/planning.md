---
title: Planning 任务规划
slug: planning
category: 进阶专题
tags: [Planning, Plan-and-Solve, 任务分解, Agent, Tree of Thought]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# Planning 任务规划

> 五层读懂一个词。这次拆的是：**Planning**--LLM Agent 处理复杂任务的关键能力。先规划再执行（Plan-and-Solve）、边执行边调整（Re-plan）、树搜索探索多路径（ToT/MCTS）。ReAct 边想边做适合简单任务，Planning 先想好全局再做适合复杂任务。

---

## L1 · 一句话点破

**Planning = 任务分解 + 全局规划 + 动态调整**。把复杂任务拆成子任务序列，先制定完整计划再执行（Plan-and-Solve），执行中根据反馈动态调整（Re-plan），或用树搜索探索多条路径选最优（ToT/MCTS/LATS）。ReAct 是「走一步看一步」，Planning 是「先看全程再走」。

---

## L2 · 通俗类比

ReAct 像**探险**：

- 走一步看一步（Thought -> Action -> Observation）
- 不知道终点在哪，靠反馈调整
- 简单任务（查个信息）够用
- 复杂任务（多步骤、有依赖）容易迷失

**Planning 像「先看地图再出发」**：

- 先规划完整路线（Plan）
- 按计划执行（Execute）
- 路上遇阻调整路线（Re-plan）

**举例**（任务：写一篇关于 LLM 量化的技术博客）：

**ReAct 方式**：

```
Thought 1: 我要写博客，先查资料
Action 1: Search["LLM 量化"]
Observation 1: ...
Thought 2: 查到资料了，开始写大纲
Action 2: WriteOutline[...]
...边写边想，可能反复返工
```

**Planning 方式**：

```
Plan:
  1. 搜索 LLM 量化相关资料
  2. 整理关键概念（GPTQ/AWQ/QLoRA）
  3. 写大纲（5 个章节）
  4. 逐章撰写
  5. 校对修改

Execute:
  Step 1: 搜索资料 -> 成功
  Step 2: 整理概念 -> 成功
  Step 3: 写大纲 -> 大纲如下
  Step 4: 撰写第 1 章 -> 发现需要更多资料
    Re-plan: 插入"补充搜索 LLM.int8()"步骤
  Step 5: 继续撰写...
```

**Planning 的三种范式**：

**1. Plan-and-Solve（一次性规划）**：

- 一次性生成完整计划
- 按计划顺序执行
- 简单但缺乏灵活性

**2. Re-plan（动态调整）**：

- 初始规划
- 执行中根据反馈调整
- 平衡规划质量和灵活性

**3. Tree Search（树搜索）**：

- 探索多条可能路径
- 评估每条路径
- 选最优（或剪枝）
- 计算成本高但效果好

**对比**：

| 范式 | 规划时机 | 灵活性 | 成本 | 适用 |
|------|---------|--------|------|------|
| ReAct | 边做边想 | 高 | 中 | 简单任务 |
| Plan-and-Solve | 一次性 | 低 | 低 | 中等任务 |
| Re-plan | 动态调整 | 中高 | 中 | 复杂任务 |
| Tree Search | 多路径探索 | 高 | 高 | 极复杂任务 |

**代价**：

- 规划本身消耗 token
- 计划可能错误，导致执行浪费
- Re-plan 需要平衡稳定性和灵活性
- 树搜索计算成本高

**适用**：

- 多步骤任务（写作、代码、研究）
- 有依赖关系的任务
- 需要全局视角的任务
- 需要回溯的复杂任务

---

## L3 · 正经定义

**Planning**（LLM Agent 任务规划）：在执行复杂任务前，先生成完整或部分的子任务计划（plan），再按计划执行。主流范式：

- **Plan-and-Solve**（Wang et al. 2023）：一次生成完整计划再执行
- **ReWOO**（Xu et al. 2023）：Planner 一次规划，Worker 并行执行，Solver 合成
- **LLM+P**（Liu et al. 2023）：LLM 翻译问题为 PDDL，经典规划器求解，再翻译回
- **Tree of Thoughts**（Yao et al. 2023）：树形探索多条推理路径
- **LATS**（Zhou et al. 2023）：ReAct + MCTS 蒙特卡洛树搜索
- **Re-plan**：执行中动态调整计划

**关键能力**：

- **任务分解**：把大任务拆成可执行的子任务
- **依赖识别**：识别子任务间的依赖关系
- **资源估算**：估算每个子任务的工具、时间
- **风险评估**：预判可能的失败点
- **动态调整**：根据执行反馈调整计划

**参考资料**：

- 📄 Wang et al., *Plan-and-Solve Prompting*, ACL 2023
- 📄 Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, NeurIPS 2023
- 📄 Zhou et al., *Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models*, 2023（LATS）
- 📄 Xu et al., *ReWOO: Decoupling Reasoning from Tools for Zero-Shot Tool-Integrated Reasoning*, 2023
- 📄 Liu et al., *LLM+P: Empowering Large Language Models with Optimal Planning Proficiency*, 2023
- 🔧 LangChain Plan-and-Execute：https://python.langchain.com/docs/modules/agents/agent_types/plan_and_execute

---

## L4 · 原理深挖

### 4.1 为什么需要 Planning

**ReAct 的局限**：

1. **局部最优**：每步只看当前，可能错过全局最优
2. **错误传播**：早期错误一路放大
3. **无法回溯**：走错了没法回头
4. **复杂任务迷失**：多步骤任务容易丢失目标

**Planning 的优势**：

1. **全局视角**：先看全貌再执行
2. **依赖管理**：识别子任务依赖
3. **可回溯**：计划可调整
4. **目标导向**：始终朝最终目标

**何时用 Planning**：

- 子任务 > 5 个
- 子任务有依赖
- 单步错误代价高
- 需要全局优化

### 4.2 Plan-and-Solve

**核心思想**：先让 LLM 生成完整计划，再逐步执行。

**Prompt**：

```
任务: <复杂任务>

第一步，制定计划:
Let's first understand the problem and devise a plan to solve it.
Plan:
1. <子任务 1>
2. <子任务 2>
...
N. <子任务 N>

第二步，执行计划:
Let's carry out the plan.
Step 1: <执行子任务 1>
...
Step N: <执行子任务 N>

最终答案: <合成>
```

**示例**（数学应用题）：

```
问题: 小明有 5 个苹果，小红比小明多 3 个，他们一共有多少苹果？

Plan:
1. 算小红有多少苹果（小明 + 3）
2. 算总数（小明 + 小红）

Step 1: 小红 = 5 + 3 = 8
Step 2: 总数 = 5 + 8 = 13

答案: 13
```

**效果**（GSM8K 数学题）：

| 方法 | 准确率 |
|------|--------|
| CoT | 78.0% |
| Plan-and-Solve | 80.5% |

**优势**：

- 减少 CoT 中的计算错误（先规划再算）
- 适合多步骤任务
- 实现简单

**劣势**：

- 计划可能错误
- 一次性规划，无法调整
- 简单任务过度规划

### 4.3 ReWOO: 分离规划与执行

**核心思想**：Planner 一次性生成所有工具调用（含变量引用），Worker 并行执行，Solver 合成答案。

**流程**：

```
Planner (一次性):
  Plan:
    #E1 = Search["LLM 量化"]
    #E2 = Search["GPTQ"]
    #E3 = LLM[#E1, #E2, "总结 LLM 量化方法"]
  
Worker (并行执行):
  #E1 = <搜索结果 1>
  #E2 = <搜索结果 2>
  #E3 = <LLM 基于 #E1, #E2 的总结>
  
Solver (合成):
  基于 #E1, #E2, #E3，生成最终答案
```

**关键创新**：

- **变量引用**：#E1, #E2 等作为占位符
- **并行执行**：无依赖的工具调用并行
- **减少 LLM 调用**：Planner 一次，Solver 一次，中间 Worker 不需要 LLM

**优势**：

- token 消耗降低 50%+（减少中间 Thought）
- 并行执行，延迟降低
- 适合工具调用密集任务

**劣势**：

- 规划阶段不能根据中间结果调整
- 变量引用依赖 LLM 正确生成
- 适合结构化任务，不适合需要灵活推理的

### 4.4 LLM+P: 经典规划器集成

**核心思想**：LLM 翻译问题为 PDDL（Planning Domain Definition Language），用经典规划器（如 Fast Downward）求解，再翻译回自然语言。

**流程**：

```
1. LLM 把自然语言问题翻译为 PDDL:
   Domain: blocks-world
   Problem: (on b1 b2), (on b2 b3), (clear b1) -> (on b3 b2), (on b2 b1)

2. 经典规划器求解:
   Plan: (unstack b1 b2), (put-down b1), (unstack b2 b3), (stack b2 b1), (stack b3 b2)

3. LLM 把 plan 翻译回自然语言:
   "先把 b1 从 b2 上拿下来放桌上，再把 b2 从 b3 上拿下来，..."
```

**优势**：

- 经典规划器保证最优解
- 适合结构化规划问题（如物流、调度）
- LLM 不需要规划能力

**劣势**：

- 需要 PDDL domain（人工设计）
- 只适合可形式化的问题
- 翻译可能出错

### 4.5 Tree of Thoughts (ToT)

**核心思想**：把推理过程建模为树，每个节点是一个「思维状态」，探索多条路径，用评估器剪枝，选最优路径。

**流程**：

```
1. 思维分解: 把问题拆成多个思维步骤
2. 思维生成: 每步生成 k 个候选思维
3. 状态评估: 评估每个思维的优劣（value/promise）
4. 搜索算法: BFS/DFS 搜索，剪枝

   Root
   / | \
  T1 T2 T3  (k=3 候选)
  |\  |
  ...
  最优路径
```

**示例**（24 点游戏）：

```
问题: 用 4 7 8 8 凑 24

ToT:
  Step 1: 候选操作
    - 8 / (4 - (8/7))  promise: 高
    - (7 - 8/8) * 4    promise: 高
    - 8 * (4 - 7/8)    promise: 高
    - 4 * 7 - 8 + 8    promise: 低
  
  Step 2: 沿高 promise 路径深入
    (7 - 8/8) * 4 = (7 - 1) * 4 = 6 * 4 = 24 ✓
```

**搜索算法**：

- **BFS**：广度优先，适合浅树
- **DFS**：深度优先，适合深树
- **Beam Search**：保留 top-k 路径
- **MCTS**：蒙特卡洛树搜索

**效果**（24 点游戏）：

| 方法 | 成功率 |
|------|--------|
| CoT | 4% |
| CoT-SC（自洽） | 9% |
| **ToT** | **74%** |

**优势**：

- 探索多条路径，避免局部最优
- 可回溯
- 适合需要搜索的问题

**劣势**：

- 计算成本高（多次 LLM 调用）
- 评估器质量影响大
- 适合搜索空间适中的问题

### 4.6 LATS: ReAct + MCTS

**核心思想**：把 ReAct 的「想-动-看」循环嵌入 MCTS 框架，用蒙特卡洛树搜索探索多条 Agent 轨迹。

**流程**：

```
1. 选择 (Selection): 从根节点用 UCB 选最有潜力的节点
2. 扩展 (Expansion): 用 LLM 生成 k 个候选 Action
3. 模拟 (Rollout): 执行 Action 得到新状态，评估
4. 回传 (Backprop): 把评估值回传到根节点
5. 重复直到找到解或预算耗尽
```

**关键创新**：

- ReAct 的 Thought-Action-Observation 作为节点
- 用 LLM 作为策略（生成 Action）和价值函数（评估状态）
- 多条路径探索，避免 ReAct 的局部最优

**效果**（HumanEval 编程）：

| 方法 | pass@1 |
|------|--------|
| CoT | 54.2% |
| ReAct | 56.8% |
| **LATS** | **72.5%** |

**优势**：

- 结合 ReAct 的灵活性和树搜索的全局性
- 适合复杂决策任务

**劣势**：

- 计算成本极高（10-100x 单次 ReAct）
- 需要好的价值函数
- 实时场景不适用

### 4.7 Re-plan: 动态调整

**核心思想**：执行中根据反馈调整计划，平衡 Plan-and-Solve 的稳定性和 ReAct 的灵活性。

**LangChain Plan-and-Execute 架构**：

```
1. Planner: 生成初始多步计划
2. Executor: 执行单个步骤（可能是 ReAct Agent）
3. Re-Planner: 根据执行结果，决定下一步或调整计划
   - 继续下一步
   - 重新规划剩余步骤
   - 标记完成
```

**Re-plan 决策**：

```python
def replan(current_plan, executed_step, result, original_task):
    prompt = f"""
    原任务: {original_task}
    当前计划: {current_plan}
    已执行: {executed_step}
    执行结果: {result}
    
    决定:
    (a) 计划仍有效，执行下一步: <下一步>
    (b) 需要调整，新计划: <新计划>
    (c) 任务完成: <最终答案>
    """
    return llm(prompt)
```

**触发 Re-plan 的条件**：

- 执行结果与预期不符
- 发现新的信息需要补充步骤
- 某步骤失败需要换路径
- 计划明显偏离原任务

**优势**：

- 适应动态环境
- 平衡稳定性和灵活性
- 适合真实世界复杂任务

**劣势**：

- Re-plan 本身消耗 token
- 可能频繁 Re-plan 导致不稳定
- 需要设计好的 Re-plan 触发条件

### 4.8 Planning 的评估

**评估指标**：

- **任务成功率**：完成任务的比率
- **步骤效率**：实际步骤数 / 最优步骤数
- **token 消耗**：总 token 数
- **延迟**：总时间
- **可恢复性**：失败后能否恢复

**基准测试**：

| 基准 | 任务类型 |
|------|---------|
| GSM8K | 数学（多步） |
| HotpotQA | 多跳问答 |
| ALFWorld | 具身决策 |
| WebArena | 网页操作 |
| HumanEval | 编程 |
| BabyAI | 导航规划 |

### 4.9 Planning 的局限

**局限 1: 规划质量依赖 LLM 能力**

- 弱模型规划质量差
- 复杂任务规划可能遗漏步骤
- 需要 GPT-4 级别模型

**局限 2: 计划可能错误**

- 初始规划错误导致执行浪费
- 需要执行-反馈-调整机制

**局限 3: token 消耗大**

- 规划本身消耗 token
- Re-plan 多次消耗
- ToT/LATS 多路径探索消耗巨大

**局限 4: 适合结构化任务**

- 创意任务（写作、设计）Planning 帮助小
- 适合有明确步骤的任务

**局限 5: 实时性差**

- 规划需要时间
- ToT/LATS 不适合实时场景

**局限 6: 评估困难**

- 计划的好坏难以自动评估
- 需要执行后才知道
- 价值函数设计困难

---

## L5 · 沿革与坑

### 5.1 沿革

- **2022-01**：CoT（Wei et al.），LLM 推理起点
- **2023-03**：ReAct（Yao et al.），推理+行动
- **2023-05**：Plan-and-Solve（Wang et al.），先规划再执行
- **2023-05**：Tree of Thoughts（Yao et al.），树搜索推理
- **2023-10**：ReWOO（Xu et al.），分离规划与执行
- **2023-10**：LATS（Zhou et al.），ReAct + MCTS
- **2023-10**：LLM+P（Liu et al.），经典规划器集成
- **2024**：LangChain Plan-and-Execute，Re-plan 成主流
- **2024-2025**：Planning 成为复杂 Agent 标配，Re-plan + Memory + Multi-Agent 组合

### 5.2 常见坑

**坑 1: 简单任务用 Planning**。一步能解决的任务硬要规划，过度工程。要按任务复杂度选范式。

**坑 2: 一次性规划期望完美**。Plan-and-Solve 初始规划可能错。要加 Re-plan 机制。

**坑 3: ToT/LATS 用在实时场景**。计算成本 10-100x，不适合实时。要离线或对延迟不敏感场景。

**坑 4: 规划粒度太细**。每步太细，规划本身爆炸。要合理粒度（3-7 步）。

**坑 5: 规划粒度太粗**。每步太粗，执行时还是要 ReAct。要平衡。

**坑 6: 没有反馈循环**。Plan-and-Solve 不调整，错误一路放大。要 Re-plan。

**坑 7: Re-plan 频率太高**。每步都 Re-plan，不稳定且成本高。要在关键节点 Re-plan。

**坑 8: 价值函数差**。ToT/LATS 的评估器质量决定效果。要精心设计评估 prompt。

**坑 9: 忽略任务依赖**。规划没识别子任务依赖，并行执行导致错误。要 DAG 建模。

**坑 10: 期望 LLM 规划最优**。LLM 规划是近似的，不是最优。复杂规划用 LLM+P（经典规划器）。

**坑 11: 没设最大规划深度**。Re-plan 可能无限循环。要设预算。

**坑 12: 规划和执行用同一 LLM**。规划需要强模型（GPT-4），执行可以用弱模型。要分级。

**坑 13: 忽略成本**。ToT 探索 k 条路径，每条 n 步，成本 k*n*单步成本。要算预算。

### 5.3 面试怎么考

1. **Planning 解决什么问题？** 答：ReAct 边想边做适合简单任务，复杂任务容易局部最优、错误传播、迷失目标。Planning 先全局规划再执行，识别依赖、可回溯、目标导向。
2. **Plan-and-Solve 和 ReAct 的区别？** 答：ReAct 每步 Thought-Action-Observation，边想边做；Plan-and-Solve 先生成完整计划再逐步执行，全局视角但缺灵活性。
3. **ToT 的核心思想？** 答：把推理建模为树，每步生成 k 个候选思维，用评估器剪枝，搜索算法（BFS/DFS/Beam/MCTS）选最优路径。适合需要搜索的问题（24 点等）。
4. **LATS 是什么？** 答：ReAct + MCTS。把 ReAct 的 Thought-Action-Observation 作为节点，用蒙特卡洛树搜索探索多条 Agent 轨迹，LLM 作为策略和价值函数。效果好但成本高。
5. **Re-plan 何时触发？** 答：执行结果与预期不符、发现新信息、步骤失败、计划偏离原任务。要平衡稳定性和灵活性，不能太频繁。
6. **Planning 的局限？** 答：规划质量依赖 LLM、计划可能错误、token 消耗大、适合结构化任务、实时性差、评估困难。

---

## 速记卡

**Planning 范式对比**：

| 范式 | 思想 | 成本 | 适用 |
|------|------|------|------|
| ReAct | 边想边做 | 中 | 简单任务 |
| Plan-and-Solve | 一次规划再执行 | 低 | 中等任务 |
| ReWOO | 规划/执行分离，并行 | 低 | 工具密集 |
| LLM+P | LLM + 经典规划器 | 中 | 结构化规划 |
| ToT | 树搜索思维 | 高 | 搜索问题 |
| LATS | ReAct + MCTS | 极高 | 复杂决策 |
| Re-plan | 动态调整 | 中 | 真实世界 |

**Plan-and-Solve Prompt**：

```
1. 制定计划
   Plan:
   1. <子任务>
   2. <子任务>
   ...
2. 执行计划
   Step 1: <执行>
   ...
3. 合成答案
```

**ToT 流程**：

```
1. 思维分解
2. 思维生成 (k 个候选)
3. 状态评估 (value/promise)
4. 搜索 (BFS/DFS/Beam/MCTS)
5. 剪枝 + 回溯
```

**Re-plan 触发条件**：

```
- 执行结果 ≠ 预期
- 发现新信息
- 步骤失败
- 计划偏离原任务
```

**效果对比**：

| 任务 | CoT | ReAct | Plan-and-Solve | ToT | LATS |
|------|-----|-------|----------------|-----|------|
| GSM8K | 78% | - | 80.5% | - | - |
| 24 点 | 4% | - | - | 74% | - |
| HumanEval | 54% | 57% | - | - | 72.5% |

**一句话记忆**：Planning = 任务分解 + 全局规划 + 动态调整。Plan-and-Solve 一次规划再执行（简单稳定），ReWOO 分离规划/执行并行（工具密集），LLM+P 用经典规划器（结构化），ToT 树搜索思维（搜索问题），LATS = ReAct + MCTS（复杂决策），Re-plan 动态调整（真实世界）。ReAct 边想边做适合简单任务，Planning 先想好再做适合复杂任务。局限：规划质量依赖 LLM、token 消耗大、ToT/LATS 成本高、实时性差。按任务复杂度选范式：简单用 ReAct，中等用 Plan-and-Solve，复杂用 Re-plan，搜索用 ToT。

---

> *上一篇：[Function Calling 函数调用](./function-calling) -- 工具调用接口，Planning 决定怎么用工具。*
> *下一篇：[Memory 记忆机制](./memory) -- Planning 决定做什么，Memory 记住做过什么。*
