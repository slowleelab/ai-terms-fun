---
title: Multi-Agent 多智能体
slug: multi-agent
category: 进阶专题
tags: [Multi-Agent, Agent 协作, 角色分工, AutoGen, MetaGPT, CAMEL]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# Multi-Agent 多智能体

> 五层读懂一个词。这次拆的是：**Multi-Agent**--多个 LLM Agent 协作解决复杂任务。角色分工（PM/架构师/工程师/测试）、辩论（多个 Agent 互相批判）、层级（Manager-Worker）、网络（自由协作）。单 Agent 上下文和角色受限，Multi-Agent 用分工+协作突破限制。AutoGen/MetaGPT/CAMEL 是代表框架。

---

## L1 · 一句话点破

**Multi-Agent = 多个 LLM Agent 角色分工 + 协作沟通**。每个 Agent 扮演特定角色（如产品经理、架构师、工程师），通过消息传递协作完成任务。主流拓扑：**层级**（Manager 分配给 Worker）、**辩论**（多 Agent 互相批判）、**网络**（自由协作）、**流水线**（顺序传递）。突破单 Agent 上下文和角色能力限制，但协调成本高。

---

## L2 · 通俗类比

单 Agent 像**全能选手**：

- 一个人写需求、设计、编码、测试
- 上下文窗口有限，装不下所有信息
- 角色切换容易混乱
- 复杂任务力不从心

**Multi-Agent 像「团队」**：

- 产品经理写需求
- 架构师设计
- 工程师编码
- 测试工程师测试
- 互相沟通协作

**典型场景**（用 Multi-Agent 开发一个 Web 应用）：

```
User: 做一个待办事项 Web 应用

Product Manager Agent:
  - 分析需求
  - 写 PRD
  - 传给 Architect Agent

Architect Agent:
  - 设计技术方案
  - 选 React + Node.js
  - 传给 Engineer Agent

Engineer Agent:
  - 编码实现
  - 传给 QA Agent

QA Agent:
  - 写测试用例
  - 测试
  - 发现 bug 传回 Engineer
  - 修复后传给 Deploy Agent
```

**Multi-Agent 的四种拓扑**：

**1. 层级（Hierarchical）**

```
Manager Agent
  |-- Worker A
  |-- Worker B
  |-- Worker C
```

- Manager 分配任务
- 适合结构化任务
- 单点瓶颈（Manager）

**2. 辩论（Debate）**

```
Agent A <-> Agent B <-> Agent C
       Judge
```

- 多 Agent 互相批判
- 适合推理、决策
- 收敛慢

**3. 网络（Network）**

```
Agent A <--> Agent B
   ^           ^
   v           v
Agent C <--> Agent D
```

- 自由协作
- 适合探索性任务
- 协调复杂

**4. 流水线（Pipeline）**

```
Agent A -> Agent B -> Agent C -> Agent D
```

- 顺序传递
- 适合分阶段任务
- 不灵活

**为什么 Multi-Agent 比 Single-Agent 强**：

| 维度 | Single-Agent | Multi-Agent |
|------|--------------|-------------|
| 上下文 | 一个窗口 | 多个窗口，分担 |
| 角色 | 一人分饰多角 | 各司其职 |
| 专业化 | 通用 | 角色专精 |
| 错误纠正 | 自我反思 | 互相批判 |
| 复杂任务 | 容易迷失 | 分工协作 |

**代价**：

- 协调成本高（Agent 间通信）
- token 消耗大（多 Agent 都调用 LLM）
- 延迟高（多轮协作）
- 一致性问题（Agent 间认知不一致）
- 调试困难

**适用**：

- 软件开发（MetaGPT, ChatDev）
- 复杂研究（多角度探索）
- 决策（多角色辩论）
- 内容创作（多角色协作）
- 任务自动化（AutoGen）

---

## L3 · 正经定义

**Multi-Agent System**（多智能体系统）：多个 LLM Agent 协作完成任务的架构。每个 Agent 有特定角色、工具、记忆，通过消息传递协作。主流框架：

- **AutoGen**（Microsoft）：通用 Multi-Agent 框架，支持自由对话协作
- **MetaGPT**：模拟软件团队（PM/Architect/Engineer/QA），SOP 驱动
- **CAMEL**：Role-Playing，两个 Agent 角色扮演协作
- **ChatDev**：软件开发流水线，多 Agent 顺序协作
- **CrewAI**：角色分工 + 任务分配 + 流程编排
- **LangGraph**：图结构 Agent 编排，支持循环和条件

**关键设计**：

- **角色定义**：每个 Agent 的 system prompt 定义角色、职责、工具
- **通信协议**：Agent 间如何传递消息（直接/黑板/广播）
- **协调机制**：谁来决定下一步（中心化/去中心化）
- **终止条件**：何时结束协作

**参考资料**：

- 📄 Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*, 2023
- 📄 Hong et al., *MetaGPT: Meta Programming for Multi-Agent Collaborative Framework*, ICLR 2024
- 📄 Li et al., *CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society*, 2023
- 📄 Qian et al., *Communicative Agents for Software Development (ChatDev)*, 2023
- 📄 Du et al., *Improving Factuality and Reasoning in Language Models through Multiagent Debate*, 2023
- 🔧 LangGraph：https://langchain-ai.github.io/langgraph/
- 🔧 CrewAI：https://docs.crewai.com/

---

## L4 · 原理深挖

### 4.1 为什么需要 Multi-Agent

**Single-Agent 的局限**：

1. **上下文有限**：复杂任务需要大量信息，单窗口装不下
2. **角色混乱**：一个 Agent 扮演多角色，prompt 冲突
3. **专业度不足**：通用 prompt 不如专业 prompt
4. **自我纠错难**：自己的错误难以发现
5. **任务复杂度**：单 Agent 处理不了多阶段复杂任务

**Multi-Agent 的解决**：

1. **分担上下文**：每个 Agent 处理一部分，结果汇总
2. **角色专注**：每个 Agent 一个角色，prompt 精准
3. **专业 prompt**：针对角色定制
4. **互相纠错**：Agent 间互相批判
5. **分工协作**：复杂任务拆分

### 4.2 角色设计

**角色定义要素**：

```python
agent = Agent(
    name="Architect",
    role="软件架构师",
    system_prompt="""
    你是资深软件架构师，职责:
    1. 根据 PRD 设计技术方案
    2. 选择技术栈
    3. 定义模块划分和接口
    4. 输出架构设计文档
    
    输出格式: Markdown，含架构图描述
    """,
    tools=[search_tool, draw_diagram],
    memory=architect_memory,
)
```

**角色设计原则**：

- **职责单一**：每个 Agent 一个明确职责
- **输入输出清晰**：定义接收和产出
- **工具匹配**：给 Agent 合适的工具
- **避免重叠**：角色间职责不重叠
- **覆盖完整**：角色集合覆盖任务全流程

**MetaGPT 的角色**：

| 角色 | 职责 | 输入 | 输出 |
|------|------|------|------|
| Product Manager | 需求分析 | 用户原始需求 | PRD |
| Architect | 技术设计 | PRD | 设计文档 + 接口 |
| Project Manager | 任务分解 | 设计文档 | 任务列表 |
| Engineer | 编码 | 任务 | 代码 |
| QA Engineer | 测试 | 代码 | 测试报告 |

### 4.3 通信机制

**方案 1: 直接消息**

```python
# Agent A 直接发消息给 Agent B
agent_b.receive(agent_a.send("这是 PRD，请设计架构"))
```

- 简单直接
- 适合点对点
- 多 Agent 时复杂度 O(n²)

**方案 2: 黑板模式**

```python
# 共享工作区
blackboard = {
    "prd": None,
    "design": None,
    "code": None,
    "test": None,
}

# Agent 读写黑板
agent_pm.write(blackboard, "prd", prd)
agent_architect.read(blackboard, "prd")
agent_architect.write(blackboard, "design", design)
```

- 解耦
- 适合异步协作
- 需要并发控制

**方案 3: 发布订阅**

```python
# Agent 订阅话题
agent_a.subscribe("design_done")
agent_b.publish("design_done", design)
# agent_a 收到通知
```

- 松耦合
- 适合事件驱动
- 实现复杂

**方案 4: 群组对话**

```python
# AutoGen 的群组对话
group_chat = GroupChat(agents=[agent_a, agent_b, agent_c], messages=[])
manager = GroupChatManager(group_chat)

# 所有 Agent 共享对话历史
# Manager 决定谁发言
agent_a.initiate_chat(manager, message="开始讨论")
```

- AutoGen 风格
- 共享上下文
- Manager 调度

### 4.4 协调机制

**中心化协调**：

```
Manager Agent
  - 决定下一步谁发言
  - 决定何时终止
  - 汇总结果
```

- 简单可控
- 单点瓶颈
- 适合结构化任务

**去中心化协调**：

```
Agent 们平等协商
  - 轮流发言
  - 投票决策
  - 共识机制
```

- 灵活
- 协调复杂
- 适合探索性任务

**AutoGen 的 Manager 调度**：

```python
def manager_decide_next_speaker(group_chat):
    """Manager 决定下一个发言者"""
    last_speaker = group_chat.last_speaker
    last_message = group_chat.last_message
    
    # 简单策略: 轮询
    # 智能策略: LLM 决定
    prompt = f"""
    对话历史: {group_chat.messages}
    谁应该下一个发言？选择: {[a.name for a in group_chat.agents]}
    """
    next_speaker = llm(prompt)
    return next_speaker
```

### 4.5 辩论机制：互相批判

**核心思想**：多个 Agent 对同一问题给出答案，互相批判，迭代收敛。

**流程**（Du et al. 2023）：

```
Round 1:
  Agent A: 答案 A1
  Agent B: 答案 B1
  Agent C: 答案 C1

Round 2:
  Agent A 看到 B1, C1，修改: 答案 A2
  Agent B 看到 A1, C1，修改: 答案 B2
  Agent C 看到 A1, B1，修改: 答案 C2

...

收敛或达到最大轮数
最终: 多数投票 / Judge 决定
```

**效果**（事实性问答）：

| 方法 | 准确率 |
|------|--------|
| Single-Agent | 75% |
| 2 Agent 辩论 | 80% |
| 3 Agent 辩论 | 83% |
| 4 Agent 辩论 | 84% |

**收益递减**：3-4 个 Agent 后提升变小。

**优势**：

- 互相纠错
- 多角度思考
- 减少单 Agent 幻觉

**劣势**：

- 收敛慢
- 可能陷入僵局
- token 消耗大

### 4.6 MetaGPT: SOP 驱动的软件开发

**核心思想**：把人类软件团队的 SOP（Standard Operating Procedure）编码到 Multi-Agent 系统。

**SOP 示例**：

```
1. PM 收到需求，写 PRD
2. Architect 收到 PRD，写设计
3. PM 分解任务
4. Engineer 领任务，编码
5. QA 测试，反馈
6. Engineer 修复
7. 交付
```

**关键机制**：

- **标准化输出**：每个角色输出标准格式（PRD 模板、设计模板）
- **共享消息池**：所有 Agent 共享项目状态
- **结构化通信**：Agent 间用结构化消息
- **错误反馈循环**：QA -> Engineer -> QA

**效果**（HumanEval 编程）：

| 方法 | pass@1 |
|------|--------|
| GPT-4 单 Agent | 54% |
| MetaGPT | 75%+ |

### 4.7 AutoGen: 通用 Multi-Agent 框架

**核心特性**：

- **可定制的 Agent**：灵活定义角色、工具
- **群组对话**：多 Agent 自由对话
- **代码执行**：Agent 可执行代码
- **人类参与**：支持 Human-in-the-loop

**示例**（AutoGen 解决编码任务）：

```python
from autogen import AssistantAgent, UserProxyAgent

# 创建 Assistant
coder = AssistantAgent(
    name="coder",
    system_message="你是资深工程师，写 Python 代码",
    llm_config=llm_config,
)

# 创建 User Proxy（执行代码）
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding"},
)

# 开始对话
user_proxy.initiate_chat(
    coder,
    message="写一个快速排序的 Python 实现",
)
```

**工作流**：

```
1. user_proxy 发任务
2. coder 写代码
3. user_proxy 执行代码
4. 如果出错，反馈给 coder
5. coder 修改
6. 直到成功
```

### 4.8 LangGraph: 图结构编排

**核心思想**：用图结构（节点 + 边）编排 Agent，支持循环、条件、并行。

**示例**（多 Agent 协作研究）：

```python
from langgraph.graph import StateGraph, END

# 定义状态
class ResearchState(TypedDict):
    question: str
    search_results: list
    answer: str

# 定义节点（Agent）
def searcher(state):
    # 搜索 Agent
    results = search(state["question"])
    return {"search_results": results}

def writer(state):
    # 写作 Agent
    answer = write(state["question"], state["search_results"])
    return {"answer": answer}

def reviewer(state):
    # 审核 Agent
    if quality_good(state["answer"]):
        return END
    else:
        return "searcher"  # 回到搜索

# 构建图
workflow = StateGraph(ResearchState)
workflow.add_node("searcher", searcher)
workflow.add_node("writer", writer)
workflow.add_node("reviewer", reviewer)

workflow.add_edge("searcher", "writer")
workflow.add_conditional_edges("writer", reviewer)

workflow.set_entry_point("searcher")
app = workflow.compile()
```

**优势**：

- 可视化流程
- 支持循环、条件
- 状态管理
- 适合复杂工作流

### 4.9 Multi-Agent 的挑战

**挑战 1: 协调成本**

- Agent 间通信消耗 token
- 多轮对话延迟高
- 协调逻辑复杂

**挑战 2: 一致性**

- 不同 Agent 认知不一致
- 上下文不共享，可能矛盾
- 需要 shared memory

**挑战 3: 终止条件**

- 何时结束协作？
- 辩论可能不收敛
- 需要明确终止信号

**挑战 4: 错误传播**

- 一个 Agent 错误影响下游
- 需要错误检测和恢复
- QA Agent 兜底

**挑战 5: 调试困难**

- 多 Agent 交互复杂
- 难以定位问题
- 需要详细日志

**挑战 6: 成本**

- Multi-Agent token 消耗是单 Agent 的 N 倍
- 延迟是单 Agent 的 N 倍
- 要评估 ROI

**挑战 7: 角色设计**

- 角色划分不合理导致效率低
- 角色重叠导致冲突
- 需要精心设计

**挑战 8: 评估**

- Multi-Agent 系统评估困难
- 单 Agent 表现 vs 系统表现
- 需要端到端评估

### 4.10 Multi-Agent vs Single-Agent

**何时用 Multi-Agent**：

- 任务复杂度高（多阶段、多角色）
- 需要专业化（每个角色专精）
- 需要互相批判（辩论）
- 上下文超出单窗口

**何时用 Single-Agent**：

- 任务简单
- 延迟敏感
- 成本敏感
- 不需要多角度

**经验法则**：

- 先试 Single-Agent
- 如果效果不够，再加 Agent
- 角色 3-5 个最佳，太多反而乱
- 辩论 3-4 个 Agent 收益最大

---

## L5 · 沿革与坑

### 5.1 沿革

- **2023-03**：CAMEL（Role-Playing 双 Agent）论文
- **2023-05**：ChatDev（软件开发 Multi-Agent）
- **2023-08**：AutoGen（Microsoft，通用 Multi-Agent 框架）
- **2023-08**：MetaGPT 论文，SOP 驱动软件开发
- **2023-09**：Multi-Agent Debate（Du et al.），辩论提升事实性
- **2023-10**：CrewAI、LangGraph 等框架兴起
- **2024**：Multi-Agent 成为复杂任务标配，框架成熟
- **2024-2025**：Multi-Agent + Memory + Planning + Function Calling 组合，应用场景扩展（研究、写作、编程、决策）

### 5.2 常见坑

**坑 1: 简单任务用 Multi-Agent**。简单任务 Multi-Agent 过度工程，成本高。要先试 Single-Agent。

**坑 2: 角色太多**。10+ Agent 协调爆炸。3-5 个最佳。

**坑 3: 角色重叠**。两个 Agent 职责重叠，互相抢活。要职责清晰。

**坑 4: 没有终止条件**。辩论不收敛，无限循环。要设最大轮数。

**坑 5: 上下文不共享**。Agent 间认知不一致，矛盾。要 shared memory 或群组对话。

**坑 6: 错误传播不处理**。一个 Agent 错误一路传，最后输出垃圾。要 QA Agent 兜底 + 错误检测。

**坑 7: 协调逻辑复杂**。中心化 Manager 成瓶颈，去中心化难收敛。要按场景选协调机制。

**坑 8: 成本失控**。N 个 Agent，每轮 N 次 LLM 调用，多轮后 token 爆炸。要监控和限预算。

**坑 9: 调试困难**。多 Agent 交互复杂，问题难定位。要详细日志 + 可视化。

**坑 10: 辩论期望太高**。辩论提升有上限，4+ Agent 收益递减。要 3-4 个。

**坑 11: 角色 prompt 不专业**。角色定义模糊，Agent 表现差。要详细 system prompt。

**坑 12: 期望 Multi-Agent 自动协作**。Multi-Agent 需要精心设计工作流，不是丢几个 Agent 就行。

**坑 13: 忽略人类参与**。关键决策没人把关，Multi-Agent 跑偏。要 Human-in-the-loop。

**坑 14: 评估只看最终结果**。最终结果好但中间过程有问题。要过程评估。

### 5.3 面试怎么考

1. **Multi-Agent 解决什么问题？** 答：Single-Agent 上下文有限、角色混乱、自我纠错难。Multi-Agent 分担上下文、角色专注、互相批判，适合复杂任务。
2. **Multi-Agent 的四种拓扑？** 答：层级（Manager-Worker）、辩论（互相批判）、网络（自由协作）、流水线（顺序传递）。按任务选。
3. **MetaGPT 的核心思想？** 答：把人类软件团队 SOP 编码到 Multi-Agent，角色（PM/Architect/Engineer/QA）+ 标准化输出 + 共享消息池 + 错误反馈循环。
4. **AutoGen 的特性？** 答：通用 Multi-Agent 框架，可定制 Agent、群组对话、代码执行、Human-in-the-loop。UserProxy + Assistant 协作。
5. **Multi-Agent 辩论的效果？** 答：多 Agent 互相批判，提升事实性和推理。3-4 个 Agent 收益最大，4+ 递减。收敛慢、成本高。
6. **Multi-Agent 的挑战？** 答：协调成本、一致性、终止条件、错误传播、调试困难、成本高、角色设计、评估困难。

---

## 速记卡

**四种拓扑**：

| 拓扑 | 结构 | 适用 |
|------|------|------|
| 层级 | Manager -> Workers | 结构化任务 |
| 辩论 | Agent <-> Agent | 推理、决策 |
| 网络 | 自由协作 | 探索性任务 |
| 流水线 | A -> B -> C | 分阶段任务 |

**角色设计**：

```
name + role + system_prompt + tools + memory
- 职责单一
- 输入输出清晰
- 工具匹配
- 避免重叠
- 覆盖完整
```

**通信机制**：

| 方案 | 特点 |
|------|------|
| 直接消息 | 简单，O(n²) |
| 黑板 | 解耦，异步 |
| 发布订阅 | 松耦合，事件驱动 |
| 群组对话 | 共享上下文，Manager 调度 |

**主流框架**：

| 框架 | 特点 |
|------|------|
| AutoGen | 通用，群组对话 |
| MetaGPT | SOP 驱动软件开发 |
| CAMEL | Role-Playing |
| ChatDev | 软件开发流水线 |
| CrewAI | 角色分工 + 任务编排 |
| LangGraph | 图结构编排 |

**Multi-Agent vs Single-Agent**：

| 维度 | Single | Multi |
|------|--------|-------|
| 上下文 | 单窗口 | 多窗口分担 |
| 角色 | 通用 | 专精 |
| 纠错 | 自我反思 | 互相批判 |
| 成本 | 低 | 高（N 倍） |
| 延迟 | 低 | 高（N 倍） |
| 适用 | 简单任务 | 复杂任务 |

**辩论效果**：

| Agent 数 | 准确率 |
|----------|--------|
| 1（基线） | 75% |
| 2 | 80% |
| 3 | 83% |
| 4 | 84%（递减） |

**经验法则**：

- 先试 Single-Agent
- 角色 3-5 个最佳
- 辩论 3-4 个 Agent
- 设最大轮数
- 加 QA Agent 兜底
- 监控 token 预算

**一句话记忆**：Multi-Agent = 多个 LLM Agent 角色分工 + 协作沟通。突破单 Agent 上下文、角色、纠错限制。四种拓扑：层级（Manager-Worker）、辩论（互相批判）、网络（自由协作）、流水线（顺序）。主流框架：AutoGen（通用群组对话）、MetaGPT（SOP 驱动软件开发）、CAMEL（Role-Playing）、ChatDev、CrewAI、LangGraph（图编排）。角色设计：职责单一、输入输出清晰、工具匹配。辩论 3-4 个 Agent 收益最大。挑战：协调成本、一致性、终止条件、错误传播、调试、成本（N 倍 Single）。先试 Single-Agent，不够再加 Agent，3-5 个最佳。

---

> *上一篇：[Memory 记忆机制](./memory) -- 单 Agent + Memory 后，Multi-Agent 协作解决更复杂任务。*
> *下一篇预告：长上下文专题 -- RoPE / Ring Attention / Lost in the Middle / Long-context RAG，从模型层突破上下文限制。*
