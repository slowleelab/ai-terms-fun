---
title: Self-RAG 自反思检索
slug: self-rag
category: 进阶专题
tags: [Self-RAG, 反思 token, 检索决策, RAG, ICLR 2024]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# Self-RAG 自反思检索

> 五层读懂一个词。这次拆的是：**Self-RAG**--让 LLM 自己决定要不要检索、检索结果用不用、答案要不要重写，用"反思 token"把 RAG 从死流程变成自适应流程。

---

## L1 · 一句话点破

**Self-RAG = 检索决策 token + 检索评价 token + 答案评价 token**。LLM 边生成边反思，按需检索、按质取舍、按需重写，用训练出来的"反思能力"替代固定的检索流水线。

---

## L2 · 通俗类比

Naive RAG 是死板的流水线：每次提问都先检索 top-k，再把 chunk 塞给 LLM 让它答。问题在于：

- 简单问题（"1+1 等于几"）也被强行检索，浪费且可能误导
- 检索回来的垃圾 chunk，LLM 照样硬答，产生幻觉
- 答案有没有依据，没人检查

Self-RAG 给 LLM 配了一个**内心独白**机制。LLM 边想边自言自语：

- `[Retrieve]`：这个问题我要不要去查资料？
- `[IsRel]`：查回来的资料跟问题相关吗？
- `[IsSup]`：我写的答案有没有被资料支持？
- `[IsUse]`：这个答案对用户有用吗？

每一步都自己打分，分数低的就重写或重检索。像一个会自我反省的助手，而不是只会机械查资料的实习生。

**核心差异**：

| 维度 | Naive RAG | Self-RAG |
|------|-----------|----------|
| 检索触发 | 每次都检 | 按需检 |
| 检索质量判断 | 无 | 有（IsRel） |
| 答案依据判断 | 无 | 有（IsSup） |
| 答案质量判断 | 无 | 有（IsUse） |
| 失败处理 | 硬答 | 重检索 / 重写 |

**代价**：反思 token 是训练出来的，需要专门的数据集做 SFT + RL；推理时 token 开销变大；训练复杂度高。所以 Self-RAG 是"用训练成本换推理质量"的范式。

---

## L3 · 正经定义

**Self-RAG**（Self-Reflective Retrieval-Augmented Generation）：Asai et al. (ICLR 2024) 提出，通过引入**反思 token（reflection tokens）**让 LLM 在生成过程中自主控制检索时机、评价检索质量、评估答案质量，实现按需检索与自我纠错。

**反思 token 类型**：

1. **`[Retrieve]`**（检索决策）：判断当前生成是否需要检索，取值 `yes / no / continue`
2. **`[IsRel]`**（相关性判断）：判断检索的 chunk 是否与 query 相关，取值 `relevant / irrelevant`
3. **`[IsSup]`**（支持度判断）：判断生成的段落是否被 chunk 支持，取值 `fully / partially / no`
4. **`[IsUse]`**（有用性判断）：判断最终答案对用户是否有用，取值 1-5 分

**生成流程**：

```
1. 读 query -> 输出 [Retrieve]
   ├─ no: 直接生成（不检索）
   └─ yes:
      2. 检索 top-k chunks
      3. 对每个 chunk 生成一段答案 + [IsRel] + [IsSup]
      4. 用 [IsUse] 给每段答案打分
      5. 选 [IsUse] 最高的段落作为最终答案
```

**训练方法**：

- **SFT 阶段**：用 GPT-4 生成带反思 token 的训练数据，SFT 一个基础模型
- **DPO / RL 阶段**：用反思 token 的分数作为奖励信号，强化"检索准、答案有据、答案有用"的行为

**参考资料**：

- 📄 Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, ICLR 2024, arXiv:2310.11511
- 🔧 Self-RAG 官方实现：https://github.com/AkariAsai/self-rag
- 📄 Asai et al., *Adaptive RAG: Learning to Adapt Retrieval-Augmented Large Language Models*, NAACL 2024

---

## L4 · 原理深挖

### 4.1 反思 token 的设计

Self-RAG 的核心创新是把 RAG 的"控制流"编码进 token 序列。LLM 不再是被动的"检索-生成"执行者，而是主动的决策者。

**token 序列示例**：

```
Query: RLHF 的损失函数是什么？

[Retrieve] yes
[Relevant] relevant (chunk: "RLHF 的损失函数是 PPO...")
[IsSup] fully supported
Response: RLHF 的损失函数是 PPO，目标是最大化奖励同时...
[IsUse] 5
```

简单问题：

```
Query: 1+1 等于几？

[Retrieve] no
Response: 1+1=2
[IsUse] 5
```

不相关检索：

```
Query: 首都北京有多少人口？
[Retrieve] yes
[Relevant] irrelevant (chunk: "上海是经济中心...")
[IsSup] no support
Response: [重写或重检索]
```

**关键点**：反思 token 是模型**自己生成**的，不是外部规则强加的。模型学会了"什么时候该检、检回来好不好、答得对不对"的判断能力。

### 4.2 训练数据构造

Self-RAG 训练需要**带反思 token 的监督数据**。Asai et al. 用以下流程构造：

**1. 收集指令数据**：从 Alpaca、FLAN 等指令数据集采样

**2. 用 GPT-4 生成反思 token**：

```
Prompt: 给定问题 Q 和检索结果 D，生成答案 A，
并标注 [Retrieve] / [IsRel] / [IsSup] / [IsUse]。
```

GPT-4 充当"老师"，标注出理想反思行为。

**3. 数据增强**：

- 对需要检索的问题：强制检索并标注
- 对不需要检索的问题：标注 `[Retrieve] no`
- 对检索噪声：插入不相关 chunk，标注 `[IsRel] irrelevant`
- 对答案不完整：标注低 `[IsSup]` / `[IsUse]`

**4. SFT 训练**：把反思 token 当作普通 token 拼进序列，SFT 一个基础 LLM（Llama 2 / Llama 3）。

### 4.3 DPO 强化反思能力

SFT 后的模型会生成反思 token，但质量参差。用 DPO 进一步强化：

**偏好对构造**：

- 正例：`[Retrieve] yes`（需要检索时）+ 高 `[IsSup]` / `[IsUse]`
- 负例：`[Retrieve] no`（应检索但没检）+ 低 `[IsSup]` / `[IsUse]`

**DPO 损失**：

$$
\mathcal{L}_{DPO} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right)
$$

其中 $y_w$ / $y_l$ 是偏好对的正负样本，$\pi_\theta$ 是策略模型，$\pi_{ref}$ 是参考模型。DPO 让模型倾向于生成"高 IsSup + 高 IsUse"的反思行为。

### 4.4 推理时的分段生成

Self-RAG 推理不是一次性生成，而是**分段**：

```python
def self_rag_generate(query, retriever, llm, max_segments=5):
    # 1. 判断是否检索
    retrieve_token = llm.generate(f"{query}\n[Retrieve]")
    if retrieve_token == "no":
        return llm.generate(query)
    
    # 2. 检索
    chunks = retriever.search(query, top_k=4)
    
    # 3. 对每个 chunk 生成候选答案 + 评价
    candidates = []
    for chunk in chunks:
        is_rel = llm.generate(f"{query}\n{chunk}\n[IsRel]")
        if is_rel == "irrelevant":
            continue
        response = llm.generate(f"{query}\n{chunk}\nResponse:")
        is_sup = llm.generate(f"{response}\n{chunk}\n[IsSup]")
        is_use = llm.generate(f"{response}\n[IsUse]")
        candidates.append((response, is_sup, int(is_use)))
    
    # 4. 选 IsUse 最高的
    best = max(candidates, key=lambda x: x[2])
    return best[0]
```

**关键工程点**：

- 多 chunk 并行生成候选，提升吞吐
- `[IsUse]` 排序替代复杂 rerank
- 低分段触发重检索或重写

### 4.5 Self-RAG vs Adaptive RAG vs Naive RAG

| 维度 | Naive RAG | Adaptive RAG | Self-RAG |
|------|-----------|--------------|----------|
| 检索决策 | 总是检 | 分类器决定 | LLM 自决 |
| 检索质量判断 | 无 | 无 | 有 |
| 答案评价 | 无 | 无 | 有 |
| 训练 | 不需要 | 训分类器 | SFT + DPO |
| 推理开销 | 低 | 低 | 高（多段） |
| 答案忠实度 | 中 | 中 | 高 |

**Adaptive RAG**（同作者后续工作）用一个轻量分类器决定"不检索 / 单次检索 / 多次检索"，比 Self-RAG 轻量但能力弱。Self-RAG 把所有决策都交给 LLM 自身。

### 4.6 反思 token 的局限

**局限 1：反思能力受基础模型限制**。小模型（<7B）反思 token 质量差，误判频繁。Self-RAG 需要 7B+ 模型才能稳定。

**局限 2：推理 token 开销大**。每段都要生成 `[IsRel]` `[IsSup]` `[IsUse]`，token 数翻倍。

**局限 3：训练数据依赖 GPT-4**。反思 token 的"标准答案"是 GPT-4 生成的，存在 teacher 模型的偏差。

**局限 4：反思 token 可能被绕过**。模型生成 `[IsRel] relevant` 但实际不相关，反思本身也会幻觉。

**局限 5：复杂多轮场景**。多轮对话中反思 token 怎么传递、累积，论文没充分讨论。

### 4.7 Self-RAG 的工程化

实际部署常做简化：

- **只保留 `[Retrieve]`**：用一个小分类器替代，省 token
- **离线反思**：生成后再用单独模型评价，不嵌入生成流程
- **混合**：检索用 Naive RAG，答案评价用 Self-RAG 风格的 `[IsSup]` 检查

完整 Self-RAG 适合高质量要求场景（医疗、法律），简化版适合通用 QA。

### 4.8 与后续工作的关系

- **CRAG / Corrective RAG**：用外部评估器替代反思 token，更轻量（见下一篇）
- **Agentic RAG**：把 Self-RAG 的"自决"思想扩展到 Agent 框架，多轮反思
- **Self-Refine / Reflexion**：不涉及检索，但同样用"自我反思"思想改进生成

---

## L5 · 沿革与坑

### 5.1 沿革

- **2023-10**：Asai et al. 发 Self-RAG 论文，提出反思 token 范式
- **2024-02**：同组发 Adaptive RAG，简化检索决策
- **2024-05**：CRAG 提出"外部评估器"替代反思 token
- **2024 下半年**：Self-RAG 思想被 Agentic RAG 吸收，反思成为 Agent 核心能力
- **2025**：反思 token 与 RL（RLAIF）结合，反思能力进一步增强

### 5.2 常见坑

**坑 1：小模型硬上 Self-RAG**。7B 以下模型反思 token 质量差，误判频繁。至少 7B，理想 13B+。

**坑 2：训练数据反思标签不准**。GPT-4 生成的反思标签本身有噪声，要做人工抽检 + 清洗。

**坑 3：推理 token 开销没估**。每段 4 个反思 token，输出 token 数翻倍，成本估算容易漏。

**坑 4：把反思 token 当万能解**。反思 token 也会幻觉，`[IsSup] fully` 不代表真有依据。要配合外部事实核查。

**坑 5：多轮场景反思状态丢失**。多轮对话中反思 token 怎么跨轮累积，工程上没标准方案，需自己设计。

**坑 6：[IsUse] 评分不准**。"有用性"是主观的，模型评分与人类偏好对齐难，常需 RLHF 微调。

**坑 7：简化版反思效果打折**。只保留 `[Retrieve]` 省事，但丢了 `[IsSup]` 的幻觉抑制能力，忠实度回落到 Naive RAG 水平。

**坑 8：评估指标不匹配**。用 BLEU / ROUGE 评估 Self-RAG 不公平，要用忠实度（faithfulness）+ 答案准确率。

**坑 9：检索器和反思模型耦合**。检索器差，反思模型再强也救不回来。检索器和反思能力要协同优化。

**坑 10：忘了闭环重写**。低 `[IsSup]` 分段要触发重检索或重写，不做闭环就退化成"会打分的 Naive RAG"。

### 5.3 面试怎么考

1. **Self-RAG 的反思 token 有哪几种？** 答：`[Retrieve]` / `[IsRel]` / `[IsSup]` / `[IsUse]`。
2. **Self-RAG 怎么训练？** 答：GPT-4 生成带反思 token 的监督数据，SFT + DPO 强化。
3. **Self-RAG 比 Naive RAG 强在哪？** 答：按需检索、检索质量判断、答案依据判断、答案质量判断、失败重写。
4. **反思 token 的局限？** 答：依赖大模型、推理开销大、训练数据依赖 GPT-4、反思本身会幻觉。
5. **Self-RAG 和 CRAG 的区别？** 答：Self-RAG 用 LLM 自身反思 token，CRAG 用外部评估器；Self-RAG 端到端训练，CRAG 即插即用。

---

## 速记卡

| 反思 token | 作用 | 取值 |
|-----------|------|------|
| `[Retrieve]` | 是否检索 | yes / no / continue |
| `[IsRel]` | chunk 相关性 | relevant / irrelevant |
| `[IsSup]` | 答案支持度 | fully / partially / no |
| `[IsUse]` | 答案有用性 | 1-5 分 |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| 基础模型规模 | 7B+ | 反思质量 |
| 检索 top_k | 3-5 | 候选数 |
| `[IsUse]` 阈值 | ≥3 | 重写触发 |
| SFT 数据量 | 10k-100k | 反思能力 |

**一句话记忆**：Self-RAG = LLM 自生成反思 token（检索决策 / 相关性 / 支持度 / 有用性），把 RAG 从死流程变自适应流程。训练贵（SFT + DPO）、推理贵（token 翻倍）、但忠实度和相关性显著提升。需要 7B+ 模型，小模型反思不稳。

---

> *上一篇：[GraphRAG 图谱增强检索](./graphrag) -- 知识图谱 + 社区聚类，专治全局性问题。*
> *下一篇：[Corrective RAG 检索纠错](./corrective-rag) -- 用外部评估器替代反思 token，即插即用的检索纠错。*
