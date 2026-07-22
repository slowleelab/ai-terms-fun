---
title: ReAct 推理与行动
slug: react
category: 进阶专题
tags: [ReAct, Reasoning, Acting, Agent, 思维链, 工具调用]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# ReAct 推理与行动

> 五层读懂一个词。这次拆的是：**ReAct**--LLM Agent 的奠基范式。Reasoning + Acting 交错进行：先想（Thought）再动（Action）看结果（Observation）再想，把 LLM 从「会聊天」变成「会干活」。CoT 只动脑不动手，ReAct 边动脑边动手。

---

## L1 · 一句话点破

**ReAct = Reasoning + Acting 交错循环**。LLM 在每一步先输出 Thought（推理）、再输出 Action（调用工具）、接收 Observation（工具返回）、再进入下一轮 Thought。把纯推理的 CoT 和纯行动的工具调用缝合成「想-动-看」闭环，是 LLM Agent 的奠基范式。

---

## L2 · 通俗类比

纯 CoT（Chain-of-Thought）像**闭卷考试**：

- 只能靠自己脑子里已有的知识推理
- 不能查资料、不能问别人、不能用计算器
- 模型知识截止后的事，一概不知
- 算术题算错就是算错，没法验证

纯工具调用（Acting only）像**无脑行动派**：

- 不思考直接调工具
- 调错了工具不知道
- 拿到结果不会推理下一步
- 撞了南墙不回头

**ReAct 像「带工具的侦探」**：

- **Thought**（思考）：根据已知线索，推理下一步该做什么
- **Action**（行动）：调用工具（搜索、计算器、数据库）
- **Observation**（观察）：看工具返回什么
- **回到 Thought**：根据新观察继续推理

**举例**（问题：2024 年诺贝尔物理学奖得主是哪个学校的？）：

```
Thought 1: 我需要先查 2024 年诺贝尔物理学奖得主是谁
Action 1: Search["2024 Nobel Prize Physics winner"]
Observation 1: 2024 年诺贝尔物理学奖授予 John Hopfield 和 Geoffrey Hinton
Thought 2: 我需要查 Hopfield 和 Hinton 的工作单位
Action 2: Search["John Hopfield affiliation"]
Observation 2: John Hopfield 是普林斯顿大学教授
Thought 3: 已经得到答案，可以回答
Action 3: Finish["普林斯顿大学"]
```

**关键洞察**：

- CoT 不会的问题（如 2024 年事），ReAct 通过工具调用能解决
- 工具调用错了，ReAct 通过 Thought 推理能纠正
- 把 LLM 的「推理能力」+「工具调用」组合，能力质变

**vs CoT**：

| 维度 | CoT | ReAct |
|------|-----|-------|
| 推理 | ✅ | ✅ |
| 工具调用 | ❌ | ✅ |
| 外部知识 | ❌ | ✅（通过工具） |
| 错误纠正 | ❌ | ✅（看 Observation 调整） |
| 适用场景 | 推理题 | 真实世界任务 |

**代价**：

- 多次工具调用，延迟高、成本高
- Thought 质量决定整体效果
- 错误传播（一步错步步错）
- token 消耗大（每步都带历史）

**适用**：

- 需要外部信息的任务（搜索、数据库）
- 多步推理任务（数学、规划）
- 需要验证的任务（事实核查）
- Agent 系统的基础范式

---

## L3 · 正经定义

**ReAct**（Reasoning and Acting）：LLM Agent 范式，由 Yao et al. 2022 提出。核心是让 LLM 在解决任务时交错生成 **Thought**（推理步骤）、**Action**（工具调用）、**Observation**（工具返回），形成「思考-行动-观察」循环，直到任务完成。

**Prompt 模板**（原始论文）：

```
Thought N: <推理当前状态，决定下一步>
Action N: <工具调用，如 Search[query] / Lookup[keyword] / Finish[answer]>
Observation N: <工具返回结果>

Thought N+1: ...
```

**关键特性**：

- **推理与行动交错**：不是先想完所有再动，也不是只动不想
- **外部信息获取**：通过 Action 调用工具获取外部知识
- **自我纠错**：Observation 反馈让模型调整下一步
- **可解释性**：Thought 链条暴露推理过程

**参考资料**：

- 📄 Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023
- 📄 Wei et al., *Chain-of-Thought Prompting*, NeurIPS 2022（CoT，ReAct 的推理基础）
- 📄 Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*, NeurIPS 2023
- 🔧 LangChain ReAct Agent：https://python.langchain.com/docs/modules/agents/agent_types/react
- 🔧 LlamaIndex ReAct：https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/

---

## L4 · 原理深挖

### 4.1 为什么需要 ReAct

**CoT 的局限**：

1. **知识截止**：模型不知道训练后的事，无法回答时效性问题
2. **幻觉**：推理时无法验证事实，可能一本正经胡说
3. **计算错误**：算术、统计等精确计算，LLM 容易错
4. **无法访问私有数据**：企业内部数据、实时数据库

**Acting-only 的局限**：

1. **不知何时调用**：没有推理，不知道当前该不该调工具
2. **不知调什么**：没有推理，无法选择合适工具
3. **不会处理结果**：拿到 Observation 不会推理下一步

**ReAct 的缝合**：

- Thought 决定何时、调什么工具
- Action 执行工具调用
- Observation 反馈，Thought 据此调整

### 4.2 ReAct 的工作流

```
任务: Q
上下文: C (含可用工具描述)

迭代 N:
    输入: Q + C + history(Thought_1, Action_1, Obs_1, ..., Thought_{N-1}, Action_{N-1}, Obs_{N-1})
    
    LLM 生成:
        Thought_N: <推理当前状态>
        Action_N: <工具调用>
    
    执行 Action_N，得到 Observation_N
    
    如果 Action_N == Finish[answer]:
        返回 answer
    否则:
        进入迭代 N+1
```

**停止条件**：

- 模型输出 `Finish[answer]`
- 达到最大迭代数
- 工具调用失败过多

### 4.3 Prompt 设计

**ReAct 的 prompt 结构**：

```
你是任务解决 Agent。可用工具：
- Search[query]: 搜索引擎查询
- Lookup[keyword]: 在当前文档中查找关键词
- Calculate[expr]: 数学计算
- Finish[answer]: 完成任务，返回答案

任务: <问题>

按以下格式输出（参考示例）：

Question: <问题>
Thought 1: <推理>
Action 1: <工具调用>
Observation 1: <工具返回>
...
Thought N: <最终推理>
Action N: Finish[<最终答案>]

示例:
Question: 科罗拉多造山运动延伸到的地区海拔高于富士山吗？
Thought 1: 我需要查科罗拉多造山运动延伸到的地区
Action 1: Search[科罗拉多造山运动]
Observation 1: 科罗拉多造山运动延伸到落基山脉地区
Thought 2: 我需要查落基山脉的海拔
Action 2: Search[落基山脉 海拔]
Observation 2: 落基山脉最高峰埃尔伯特峰海拔 4401 米
Thought 3: 我需要查富士山的海拔
Action 3: Search[富士山 海拔]
Observation 3: 富士山海拔 3776 米
Thought 4: 4401 > 3776，所以落基山脉海拔高于富士山
Action 4: Finish[是]

现在开始:
Question: <实际问题>
Thought 1:
```

**关键设计**：

- **工具描述**：清晰说明每个工具的输入输出
- **few-shot 示例**：用示例教模型 ReAct 格式
- **格式约束**：Thought / Action / Observation 严格交替
- **停止信号**：Finish[answer] 作为终止

### 4.4 工具调用的实现

**工具抽象**：

```python
class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func
    
    def __call__(self, input_str):
        return self.func(input_str)

# 定义工具
tools = {
    "Search": Tool("Search", "Search[query]", lambda q: web_search(q)),
    "Calculate": Tool("Calculate", "Calculate[expr]", lambda e: str(eval(e))),
    "Finish": Tool("Finish", "Finish[answer]", lambda a: a),
}
```

**ReAct Agent 主循环**：

```python
def react_agent(question, tools, llm, max_steps=10):
    history = f"Question: {question}\n"
    
    for step in range(1, max_steps + 1):
        # 1. 构造 prompt
        prompt = build_prompt(question, history, tools)
        
        # 2. LLM 生成 Thought + Action
        output = llm(prompt)
        thought, action = parse_output(output)  # 解析 Thought/Action
        
        # 3. 执行 Action
        tool_name, tool_input = parse_action(action)
        if tool_name == "Finish":
            return tool_input  # 返回最终答案
        
        observation = tools[tool_name](tool_input)
        
        # 4. 更新 history
        history += f"Thought {step}: {thought}\n"
        history += f"Action {step}: {action}\n"
        history += f"Observation {step}: {observation}\n"
    
    return "达到最大步数，未能完成任务"
```

### 4.5 ReAct 的效果

**HotpotQA**（多跳问答，论文数据）：

| 方法 | EM（精确匹配） |
|------|---------------|
| CoT（无工具） | 28.4 |
| Act-only（无推理） | 25.7 |
| **ReAct** | **35.1** |
| ReAct + CoT-SC（自洽） | 34.2 |

**关键观察**：

- ReAct 显著优于纯 CoT 和纯 Acting
- 在需要外部知识的任务上优势明显
- 在纯推理任务上（如数学），ReAct 不一定优于 CoT（工具调用反而干扰）

**ALFWorld**（具身决策任务）：

| 方法 | 成功率 |
|------|--------|
| Act-only | 28% |
| CoT | 33% |
| **ReAct** | **71%** |

**结论**：ReAct 在需要「推理+工具」的任务上效果最好。

### 4.6 ReAct 的问题

**问题 1：错误传播**

```
Thought 1: 错误推理 -> Action 1 错 -> Observation 1 无用 -> Thought 2 错 -> ...
```

一步错步步错，难以恢复。

**问题 2：Thought 冗余**

- 每步都生成 Thought，token 消耗大
- 简单任务不需要那么多推理
- 增加延迟和成本

**问题 3：工具调用格式不稳**

- LLM 可能输出格式错误的 Action
- 需要 robust 的解析
- 不同 LLM 对 prompt 格式的遵守度不同

**问题 4：迭代次数控制**

- 简单任务过度推理
- 复杂任务步数不够
- 难以预设合适的 max_steps

**问题 5：幻觉传染**

- LLM 可能在 Thought 中编造 Observation
- 不实际调用工具，直接「脑补」结果
- 需要强制工具调用

### 4.7 ReAct 的演进

**Plan-and-Solve**：先规划再执行

- ReAct 是「边想边做」，Plan-and-Solve 是「先想好再做」
- 适合复杂任务，减少中途错误
- 但缺乏灵活性，无法动态调整

**Reflexion**：失败后反思

- ReAct 失败后，反思原因
- 用反思指导下次尝试
- 多轮迭代提升成功率

**ReWOO**：分离规划和执行

- Planner 一次性生成所有工具调用
- Worker 执行，Solver 合成答案
- 减少中间 Thought 的 token 消耗

**LATS**（Language Agent Tree Search）：

- ReAct + 树搜索
- 探索多条路径，选最优
- 计算成本高但效果好

### 4.8 ReAct 的现代实现

**OpenAI Function Calling**：

- 不再用文本格式 Action，而是结构化 function call
- 模型直接输出 JSON 格式的工具调用
- 更稳定，无需解析

**LangChain ReActAgent**：

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

tools = [
    Tool(name="Search", func=web_search, description="搜索"),
    Tool(name="Calculator", func=calculate, description="计算"),
]

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "2024 诺奖物理学得主哪个学校？"})
```

**现代 Agent 框架对 ReAct 的扩展**：

- 多工具并行调用（ReAct 是串行）
- 流式 Thought 输出
- 工具调用结果缓存
- 失败自动重试

### 4.9 ReAct 的局限

**局限 1：串行执行**。每步依赖前一步，无法并行，延迟高。

**局限 2：上下文长度**。多步推理 token 累积，长任务可能超上下文。

**局限 3：Thought 质量依赖 LLM 能力**。弱模型 Thought 质量差，整体效果差。

**局限 4：工具描述敏感**。工具描述写得不好，模型选错工具。

**局限 5：成本高**。多轮调用，token 消耗大，延迟高。

**局限 6：可解释性 vs 性能 trade-off**。Thought 增加可解释性但消耗 token，去掉 Thought 可能更快但不可解释。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2022-01**：CoT 论文（Wei et al.），开启 LLM 推理研究
- **2022-03**：ReAct 论文（Yao et al.），首次提出推理+行动交错
- **2022-10**：Toolformer（Schick et al.），LLM 自学工具调用
- **2023-03**：OpenAI Function Calling，结构化工具调用
- **2023 下半年**：LangChain/LlamaIndex 等 Agent 框架以 ReAct 为基础范式
- **2024**：Plan-and-Solve / Reflexion / LATS 等改进，ReAct 仍是 Agent 基础
- **2024-2025**：ReAct 成为 Agent 教科书范式，所有 Agent 框架支持

### 5.2 常见坑

**坑 1：简单任务用 ReAct**。纯推理任务（数学、逻辑），ReAct 的工具调用反而干扰。CoT 更好。

**坑 2：Thought 过多**。每步都长篇 Thought，token 爆炸。要约束 Thought 长度。

**坑 3：工具描述模糊**。模型不知道何时用哪个工具。要清晰描述工具用途、输入、输出。

**坑 4：没设最大步数**。陷入死循环，token 烧光。要设 max_steps（通常 5-10）。

**坑 5：格式解析脆弱**。LLM 输出格式不严格，正则解析失败。要用结构化输出（Function Calling）或 robust 解析。

**坑 6：错误传播不处理**。一步错步步错。要加反思机制（Reflexion）或回溯。

**坑 7：工具调用结果不验证**。模型可能编造 Observation。要强制实际调用工具。

**坑 8：上下文超长**。多步推理 history 累积，超上下文窗口。要总结历史或用长上下文模型。

**坑 9：期望 ReAct 万能**。ReAct 适合「推理+工具」任务，不适合纯推理或纯检索任务。

**坑 10：成本低估**。ReAct 多轮调用，成本是单次 LLM 调用的 5-10 倍。要评估 ROI。

**坑 11：工具选择错**。模型选错工具，导致错误结果。要工具描述清晰 + few-shot 示例。

**坑 12：忽略并发**。ReAct 串行执行，延迟高。能并行的工具调用要并行。

### 5.3 面试怎么考

1. **ReAct 解决什么问题？** 答：CoT 只推理无工具、Acting-only 只行动无推理。ReAct 把两者缝合，Thought-Action-Observation 交错循环，让 LLM 边推理边用工具。
2. **ReAct 的核心循环？** 答：Thought（推理当前状态）-> Action（调用工具）-> Observation（工具返回）-> 下一轮 Thought，直到 Finish[answer]。
3. **ReAct vs CoT？** 答：CoT 闭卷推理，适合纯推理题；ReAct 开卷+工具，适合需要外部信息、验证的任务。纯推理 ReAct 不一定优于 CoT。
4. **ReAct 的常见问题？** 答：错误传播、Thought 冗余、格式不稳、迭代次数控制、幻觉传染。可用 Reflexion/结构化输出/最大步数缓解。
5. **ReAct 的现代演进？** 答：Plan-and-Solve（先规划）、Reflexion（反思）、ReWOO（分离规划执行）、LATS（树搜索）、Function Calling（结构化工具调用）。

---

## 速记卡

**核心循环**：

```
Thought N:  推理当前状态，决定下一步
Action N:   调用工具（Search/Calculate/Finish/...）
Observation N: 工具返回
-> 进入 N+1
终止: Finish[answer]
```

**Prompt 结构**：

```
工具描述 + few-shot 示例 + 实际问题 + Thought 1:
```

**效果对比**：

| 方法 | HotpotQA EM | ALFWorld 成功率 |
|------|-------------|----------------|
| CoT | 28.4 | 33% |
| Act-only | 25.7 | 28% |
| **ReAct** | **35.1** | **71%** |

**适用场景**：

| 场景 | ReAct 适用？ |
|------|-------------|
| 时效性问答 | ✅（工具获取新信息） |
| 多跳推理 | ✅（分步搜索） |
| 数学计算 | ⚠️（CoT 可能更好） |
| 纯逻辑推理 | ❌（CoT 更好） |
| 事实核查 | ✅（验证） |
| 企业知识库 | ✅（RAG + 推理） |

**演进**：

| 方法 | 思路 |
|------|------|
| ReAct | 边想边做 |
| Plan-and-Solve | 先规划再做 |
| Reflexion | 失败反思 |
| ReWOO | 分离规划/执行 |
| LATS | 树搜索 |
| Function Calling | 结构化工具调用 |

**一句话记忆**：ReAct = Reasoning + Acting 交错循环。Thought（推理）-> Action（工具调用）-> Observation（结果）-> 下一轮 Thought，直到 Finish[answer]。把 CoT 的推理能力和工具调用的行动能力缝合，是 LLM Agent 的奠基范式。适合需要外部信息、多步推理、事实验证的任务；纯推理任务 CoT 更好。常见问题：错误传播、Thought 冗余、格式不稳、成本高（5-10x 单次调用）。现代演进：Plan-and-Solve / Reflexion / Function Calling。

---

> *上一篇：[量化推理算法 GPTQ/AWQ](./quantization-inference) -- 推理工程末篇，Agent 专题从 ReAct 开始。*
> *下一篇：[Function Calling 函数调用](./function-calling) -- ReAct 的结构化演进，工具调用从文本解析变 JSON。*
