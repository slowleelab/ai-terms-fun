---
title: Memory 记忆机制
slug: memory
category: 进阶专题
tags: [Memory, 短期记忆, 长期记忆, RAG, Agent, MemGPT]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# Memory 记忆机制

> 五层读懂一个词。这次拆的是：**Memory**--LLM Agent 跨轮次、跨会话记住信息的能力。短期记忆靠上下文窗口，长期记忆靠向量库 + 摘要 + 反思。MemGPT 借鉴 OS 虚拟内存做分层管理，Generative Agents 用反思+检索实现「像人一样记忆」。没有 Memory，Agent 就是金鱼。

---

## L1 · 一句话点破

**Memory = Agent 跨时间保留和调用信息的能力**。短期记忆（Working Memory）用上下文窗口存当前对话；长期记忆（Long-term Memory）用向量库存历史事实 + 摘要存抽象经验 + 反思存高阶洞察。MemGPT 借鉴 OS 分层内存（主存+磁盘）做 LLM 记忆管理，是 Agent 摆脱「金鱼记忆」的关键。

---

## L2 · 通俗类比

LLM 本身像**金鱼**：

- 每次调用是无状态的
- 只记得这次 prompt 里给的内容
- 上下文窗口装不下就丢
- 下次调用又是空白

**Memory 让 Agent 像人一样记忆**：

**短期记忆（Working Memory）= 工作台**

- 当前正在处理的对话/任务
- 放在上下文窗口里
- 容量有限（如 128k token）
- 任务结束就清空

**长期记忆（Long-term Memory）= 笔记本 + 文件柜**

- 笔记本（Summary）：对话摘要、任务总结
- 文件柜（Vector DB）：原始对话、事实、文档
- 反思日记（Reflection）：从经验中提炼的高阶洞察
- 按需检索，注入工作台

**举例**（用户用 Agent 写小说，跨多天）：

**Day 1**:

```
User: 我要写一本关于 AI 量化的小说，主角叫李明
Agent: 好的，李明是量化交易员... [存入长期记忆: 主角=李明, 主题=AI量化]
```

**Day 7**:

```
User: 继续写李明的故事
Agent: [检索长期记忆: 主角=李明, 已写章节=5, 性格=内向]
Agent: 接上第 5 章，李明...
```

**关键**：没有长期记忆，Agent 不知道「李明是谁」「写到哪了」。

**三层记忆架构**（Generative Agents / MemGPT）：

| 层级 | 类比 | 实现 | 容量 | 检索 |
|------|------|------|------|------|
| 工作记忆 | 工作台 | 上下文窗口 | 小（128k） | 直接访问 |
| 短期记忆 | 便签 | 最近 N 轮摘要 | 中 | 时间衰减 |
| 长期记忆 | 文件柜 | 向量库 + 反思 | 大 | 语义检索 |

**Memory 操作**：

- **写入**（Write）：把新信息存入记忆
- **检索**（Retrieve）：按相关性召回记忆
- **反思**（Reflect）：从记忆中提炼高阶洞察
- **遗忘**（Forget）：淘汰过时/无用记忆
- **更新**（Update）：修改已有记忆

**代价**：

- 检索增加延迟
- 向量库存储成本
- 摘要/反思消耗 token
- 记忆冲突处理复杂
- 隐私和安全风险

**适用**：

- 长期对话助手（Personal AI）
- 跨会话任务（项目管理）
- 需要用户画像的个性化
- 多 Agent 协作（共享记忆）

---

## L3 · 正经定义

**Memory**（LLM Agent 记忆机制）：让 Agent 跨轮次、跨会话保留和调用信息的能力。主流架构：

- **短期记忆 / 工作记忆**：当前对话上下文，存在 LLM 上下文窗口
- **长期记忆**：跨会话信息，存外部存储（向量库、知识图谱、数据库）
- **反思机制**（Reflection）：从历史记忆中提炼高阶洞察
- **分层管理**（MemGPT）：借鉴 OS 虚拟内存，主存（上下文）+ 磁盘（外部存储），自动 page in/out

**主流方案**：

| 方案 | 思路 | 代表 |
|------|------|------|
| 上下文拼接 | 把历史直接放进 prompt | 早期 ChatBot |
| 滑动窗口 | 只保留最近 N 轮 | 简单对话系统 |
| 摘要压缩 | 老对话压缩成摘要 | LangChain ConversationSummaryMemory |
| 向量检索 | 历史向量化，按相关性检索 | RAG-based Memory |
| 知识图谱 | 提取实体关系存图 | GraphRAG-style Memory |
| 反思+检索 | 周期性反思，存洞察 | Generative Agents |
| 分层管理 | OS 式主存+磁盘 | MemGPT |

**参考资料**：

- 📄 Park et al., *Generative Agents: Interactive Simulacra of Human Behavior*, UIST 2023（反思+检索记忆）
- 📄 Packer et al., *MemGPT: Towards LLMs as Operating Systems*, 2023（OS 式分层记忆）
- 📄 Zhong et al., *MemoryBank: Enhancing Large Language Models with Long-Term Memory*, 2023
- 📄 Wang et al., *A Survey on the Memory Mechanism of Large Language Model based Agents*, 2024
- 🔧 LangChain Memory：https://python.langchain.com/docs/modules/memory/
- 🔧 LlamaIndex Memory：https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat/

---

## L4 · 原理深挖

### 4.1 为什么需要 Memory

**LLM 的无状态性**：

- 每次调用是独立的
- 上下文窗口有限（4k-200k）
- 跨会话无法保留
- 长对话上下文爆炸

**问题场景**：

1. **长对话**：对话超过上下文窗口，老内容丢失
2. **跨会话**：用户明天回来，Agent 不记得昨天
3. **个性化**：Agent 不记得用户偏好
4. **多 Agent**：Agent 间无法共享信息
5. **任务连续性**：复杂任务跨多轮，需要记住进度

**Memory 的价值**：

- **连续性**：跨轮次、跨会话保持上下文
- **个性化**：记住用户偏好、历史
- **学习**：从经验中积累
- **效率**：避免重复提问/计算

### 4.2 短期记忆：上下文管理

**方案 1: 全量上下文**

```
messages = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ... # 所有历史
]
```

- 简单
- 上下文爆炸风险
- 老对话 token 浪费

**方案 2: 滑动窗口**

```python
# 只保留最近 K 轮
messages = messages[-K*2:]  # K 轮 user+assistant
```

- 简单高效
- 丢失早期信息
- 适合短对话

**方案 3: 摘要压缩**

```python
# 老对话压缩成摘要
if len(messages) > threshold:
    old_messages = messages[:-K*2]
    summary = llm.summarize(old_messages)
    messages = [{"role": "system", "content": f"对话历史摘要: {summary}"}] + messages[-K*2:]
```

- 保留长期信息
- 摘要损失细节
- 摘要本身消耗 token

**方案 4: 实体记忆**

```python
# 提取并跟踪实体
entities = {
    "李明": {"age": 30, "job": "量化交易员", "hobby": "AI"},
    "北京": {"weather": "晴"},
}
# 每轮更新实体，注入 system prompt
```

- 结构化
- 适合跟踪用户信息
- 实体识别可能错

**LangChain 的 Memory 类型**：

| Memory 类型 | 思路 |
|------------|------|
| ConversationBufferMemory | 全量保留 |
| ConversationBufferWindowMemory | 滑动窗口 |
| ConversationSummaryMemory | 摘要压缩 |
| ConversationSummaryBufferMemory | 摘要 + 滑动 |
| ConversationEntityMemory | 实体跟踪 |
| VectorStoreRetrieverMemory | 向量检索 |

### 4.3 长期记忆：向量库 + 检索

**核心思想**：把历史对话/事实向量化存入向量库，需要时按语义检索 top-k，注入上下文。

**架构**：

```
写入:
  对话/事实 -> Embedding -> 向量库 (含原文 + metadata)

检索:
  当前 query -> Embedding -> 向量库相似度搜索 -> top-k 注入上下文
```

**实现**：

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

memory = VectorStoreRetrieverMemory(retriever=retriever)

# 自动写入和检索
# 每轮对话自动存入向量库
# 下次对话自动检索相关历史
```

**检索策略**：

- **语义检索**：query 与历史 embedding 相似度
- **时间衰减**：最近的历史权重高
- **重要性加权**：重要的记忆权重高（Generative Agents）
- **混合检索**：语义 + 时间 + 重要性

**Generative Agents 的记忆评分**：

$$
\text{score} = \alpha \cdot \text{recency} + \beta \cdot \text{importance} + \gamma \cdot \text{relevance}
$$

- **recency**：时间衰减（指数衰减）
- **importance**：LLM 评估的重要性（1-10）
- **relevance**：与当前 query 的语义相似度

### 4.4 反思机制：从记忆中提炼洞察

**核心思想**：周期性地让 Agent 反思历史记忆，提炼高阶洞察（如「李明喜欢在压力大时听古典音乐」），存入记忆库。

**Generative Agents 的反思流程**：

```
1. 触发反思（每 N 条新记忆或定期）
2. 让 LLM 提出反思问题:
   "关于李明，最近有哪些值得总结的？"
   -> "李明的压力应对方式？"
3. 检索相关记忆:
   查询 "李明 压力" -> top-k 记忆
4. LLM 生成洞察:
   "李明在压力大时倾向于听古典音乐和散步"
5. 存入记忆库（作为新的高阶记忆）
```

**反思的层次**：

- **L1 记忆**：原始观察（"李明今天听了贝多芬"）
- **L2 反思**：从 L1 提炼（"李明喜欢古典音乐"）
- **L3 元反思**：从 L2 提炼（"李明用音乐调节情绪"）

**反思的价值**：

- 抽象出模式，指导未来行为
- 减少记忆冗余
- 支持复杂推理

**反思的成本**：

- 消耗 token（LLM 调用）
- 反思质量依赖 LLM
- 反思频率难调

### 4.5 MemGPT: OS 式分层记忆

**核心思想**：借鉴 OS 虚拟内存，把 LLM 上下文窗口当「主存」，外部存储当「磁盘」，自动 page in/out。

**架构**：

```
Main Context (上下文窗口):
  - System instructions
  - Working context (当前任务相关)
  - FIFO queue (最近对话)

External Context (磁盘):
  - Recall storage (所有对话历史)
  - Archival storage (向量化的事实/笔记)

Page in/out:
  - 主存满时，把老内容 page out 到磁盘
  - 需要时从磁盘 page in 到主存
```

**关键操作**：

- **main_context**: 当前上下文窗口
- **recall_storage**: 所有历史对话（数据库）
- **archival_storage**: 向量化笔记（向量库）
- **search_archival**: 检索 archival
- **insert_archival**: 写入 archival
- **evict**: 主存满时移出老内容

**LLM 自主管理**：

- LLM 通过 function call 决定何时检索 archival
- LLM 决定何时把主存内容存入 archival
- LLM 决定何时 evict 主存

**示例**：

```
User: 还记得我们上周讨论的量化策略吗？

Agent (内部):
  Thought: 主存里没有，需要检索 archival
  Action: search_archival("量化策略 上周")
  Observation: 找到 3 条相关记忆
  
Agent (回复): 是的，我们讨论了均值回归策略...
```

**优势**：

- 突破上下文窗口限制
- LLM 自主管理记忆
- 适合超长对话

**劣势**：

- LLM 决策何时检索，可能漏检
- function call 增加 token 消耗
- 实现复杂

### 4.6 记忆的写入与更新

**写入策略**：

- **全量写入**：每轮对话都存
- **重要写入**：只存 LLM 判断为重要的
- **摘要写入**：存摘要而非原文
- **实体写入**：提取实体存入

**更新策略**：

- **覆盖**：新信息覆盖旧（适合事实）
- **追加**：新信息追加（适合时间序列）
- **合并**：LLM 合并新旧（适合复杂更新）

**遗忘策略**：

- **TTL**：到期自动删除
- **LRU**：最近最少使用删除
- **重要性淘汰**：删除不重要的
- **冲突解决**：新旧冲突时保留新的

**示例**（用户偏好更新）：

```
旧记忆: 用户喜欢科幻
新对话: 用户说"最近开始读历史"

更新策略:
  LLM 合并: "用户喜欢科幻，最近也读历史"
  或覆盖: "用户喜欢历史"（如果用户明确改变偏好）
```

### 4.7 多 Agent 共享记忆

**场景**：多个 Agent 协作，需要共享信息。

**架构 1: 共享记忆库**

```
Agent A, Agent B, Agent C
       |
   共享记忆库
   (向量库 + 知识图谱)
```

- 所有 Agent 读写同一个记忆库
- 简单，但有写冲突
- 适合协作型任务

**架构 2: 消息传递**

```
Agent A -> 消息 -> Agent B
Agent B -> 消息 -> Agent C
```

- Agent 间显式通信
- 类似 Multi-Agent 系统
- 适合分工型任务

**架构 3: 黑板模式**

```
黑板（共享工作区）
  |
  |-- Agent A 写入
  |-- Agent B 读取
  |-- Agent C 修改
```

- 异步共享
- 适合复杂协作

### 4.8 记忆的评估

**评估维度**：

- **准确性**：记忆是否正确
- **完整性**：是否遗漏关键信息
- **时效性**：记忆是否过时
- **检索精度**：能否检索到相关记忆
- **检索召回**：是否遗漏相关记忆
- **效率**：检索延迟、token 消耗

**基准测试**：

- **LoCoMo**：长对话记忆基准
- **LongMemEval**：长期记忆评估
- **MemoryBench**：Agent 记忆测试

### 4.9 Memory 的局限

**局限 1: 检索精度有限**

- 向量检索可能召回无关记忆
- 漏召相关记忆
- 需要混合检索 + 重排序

**局限 2: 记忆冲突**

- 新旧记忆冲突，如何取舍？
- 不同来源记忆冲突
- 需要 LLM 仲裁

**局限 3: 反思质量**

- 反思可能过度概括
- 反思可能引入幻觉
- 反思频率难调

**局限 4: token 消耗**

- 检索的记忆要注入上下文
- 反思消耗 token
- 长期记忆管理本身有成本

**局限 5: 隐私和安全**

- 记忆库含敏感信息
- 跨用户记忆泄漏风险
- 需要 access control

**局限 6: 一致性**

- 记忆和当前对话可能不一致
- 记忆更新不及时
- 需要 consistency check

**局限 7: 冷启动**

- 新用户无历史记忆
- 个性化效果差
- 需要 onboarding

**局限 8: 规模化**

- 记忆库膨胀
- 检索延迟增加
- 需要分片、索引优化

---

## L5 · 沿革与坑

### 5.1 沿革

- **2023-03**：LangChain Memory 模块（ConversationBufferMemory 等）
- **2023-04**：Generative Agents（Park et al.），反思+检索记忆，斯坦福小镇
- **2023-10**：MemGPT（Packer et al.），OS 式分层记忆
- **2023-10**：MemoryBank，长期记忆+遗忘
- **2024-01**：Letta（MemGPT 商业版）
- **2024 下半年**：OpenAI Memory 功能上线，Claude Projects 记忆
- **2024-2025**：Memory 成为 Agent 标配，反思+检索+分层组合，跨 Agent 共享记忆

### 5.2 常见坑

**坑 1: 全量上下文不管理**。长对话上下文爆炸，token 烧光。要滑动窗口或摘要压缩。

**坑 2: 摘要损失细节**。摘要可能丢关键信息（如数字、人名）。要关键信息保留 + 摘要。

**坑 3: 向量检索召回差**。纯向量检索可能漏相关记忆。要混合检索（向量+关键词）+ 重排序。

**坑 4: 反思过度概括**。反思可能过度抽象，丢失具体信息。要保留原始记忆 + 反思。

**坑 5: 记忆冲突不处理**。新旧记忆冲突，Agent 不知道用哪个。要冲突检测 + LLM 仲裁。

**坑 6: 隐私泄漏**。跨用户记忆泄漏（如向量库未隔离）。要用户隔离 + access control。

**坑 7: 反思频率太高**。频繁反思消耗 token，且可能反复改写记忆。要合理频率。

**坑 8: 检索不注入元数据**。检索结果没时间戳、来源，Agent 不知道记忆新旧。要保留 metadata。

**坑 9: 记忆库无限增长**。长期记忆库膨胀，检索变慢。要遗忘机制 + 分片。

**坑 10: 期望 Memory 万能**。Memory 提升连续性，但不提升推理能力。弱模型+Memory 不如强模型。

**坑 11: 不区分记忆类型**。事实、偏好、事件混在一起。要分类管理（事实用图、事件用时间线、偏好用结构化）。

**坑 12: 冷启动不处理**。新用户无记忆，体验差。要 onboarding 引导 + 通用默认值。

**坑 13: 多 Agent 记忆冲突**。多 Agent 写同一记忆库，互相覆盖。要写锁 + 版本控制。

**坑 14: MemGPT 式自主管理漏检**。LLM 决定何时检索，可能漏检。要自动检索 + LLM 主动检索结合。

### 5.3 面试怎么考

1. **Agent 为什么需要 Memory？** 答：LLM 无状态、上下文有限、跨会话无法保留。Memory 让 Agent 跨轮次/跨会话保留信息，实现连续性、个性化、学习。
2. **短期记忆和长期记忆的区别？** 答：短期记忆是当前对话上下文（工作台，有限容量），长期记忆是跨会话的外部存储（文件柜，向量库+摘要+反思）。
3. **Generative Agents 的记忆机制？** 答：三层：观察记忆（原始）+ 反思记忆（高阶洞察）+ 检索（recency + importance + relevance 加权）。
4. **MemGPT 的核心思想？** 答：借鉴 OS 虚拟内存，主存=上下文窗口，磁盘=外部存储，LLM 通过 function call 自主 page in/out，突破上下文限制。
5. **记忆冲突怎么处理？** 答：检测冲突（新旧记忆矛盾）+ LLM 仲裁（合并/覆盖/保留新）+ 时间戳优先。要设计合理的更新策略。
6. **Memory 的局限？** 答：检索精度有限、记忆冲突、反思质量、token 消耗、隐私安全、一致性、冷启动、规模化。

---

## 速记卡

**三层记忆架构**：

| 层级 | 类比 | 实现 | 容量 | 检索 |
|------|------|------|------|------|
| 工作记忆 | 工作台 | 上下文窗口 | 小（128k） | 直接 |
| 短期记忆 | 便签 | 最近 N 轮摘要 | 中 | 时间衰减 |
| 长期记忆 | 文件柜 | 向量库 + 反思 | 大 | 语义检索 |

**Memory 操作**：

```
Write:    新信息 -> 向量库
Retrieve: query -> 向量检索 -> top-k 注入上下文
Reflect:  周期性 -> LLM 提炼洞察 -> 存入记忆
Update:   新旧冲突 -> LLM 仲裁
Forget:   TTL/LRU/重要性淘汰
```

**Generative Agents 评分**：

$$
\text{score} = \alpha \cdot \text{recency} + \beta \cdot \text{importance} + \gamma \cdot \text{relevance}
$$

**MemGPT 分层**：

```
Main Context (主存):
  - System instructions
  - Working context
  - FIFO queue (最近对话)

External Context (磁盘):
  - Recall storage (全历史)
  - Archival storage (向量化笔记)

LLM 通过 function call:
  - search_archival
  - insert_archival
  - evict
```

**主流方案对比**：

| 方案 | 思路 | 优势 | 劣势 |
|------|------|------|------|
| 滑动窗口 | 最近 N 轮 | 简单 | 丢早期 |
| 摘要压缩 | 老对话摘要 | 保留长期 | 损失细节 |
| 向量检索 | 语义召回 | 精准 | 检索延迟 |
| 实体记忆 | 结构化跟踪 | 适合用户画像 | 实体识别错 |
| 反思+检索 | 高阶洞察 | 抽象模式 | 消耗 token |
| MemGPT 分层 | OS 式管理 | 突破上下文 | 实现复杂 |

**反思层次**：

```
L1 观察: "李明今天听了贝多芬"
L2 反思: "李明喜欢古典音乐"
L3 元反思: "李明用音乐调节情绪"
```

**一句话记忆**：Memory = Agent 跨时间保留和调用信息的能力。短期记忆用上下文窗口（工作台），长期记忆用向量库+摘要+反思（文件柜）。Generative Agents 用 recency+importance+relevance 三因子检索 + 周期性反思提炼高阶洞察。MemGPT 借鉴 OS 虚拟内存，主存=上下文、磁盘=外部存储，LLM 通过 function call 自主 page in/out 突破上下文限制。操作：Write/Retrieve/Reflect/Update/Forget。局限：检索精度、记忆冲突、反思质量、token 消耗、隐私安全。没有 Memory，Agent 是金鱼；有 Memory，Agent 才能连续、个性化、学习。

---

> *上一篇：[Planning 任务规划](./planning) -- Planning 决定做什么，Memory 记住做过什么。*
> *下一篇：[Multi-Agent 多智能体](./multi-agent) -- 单 Agent + Memory 后，多 Agent 协作解决更复杂任务。*
