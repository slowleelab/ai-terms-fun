---
title: 贪心解码（Greedy Decoding）
slug: greedy-decoding
category: 推理与生成
tags: [贪心解码, 解码策略, 重复退化, 确定性生成]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 贪心解码（Greedy Decoding）

> **一句话 TL;DR**：贪心解码是 [自回归生成](./autoregressive) 最简单的选词策略--每步选概率最高的 token。优点：快、确定、易复现。缺点：容易陷入重复退化（"我我我我我"）、错过全局最优序列。它是理解 [beam search](./beam-search)、[Top-k](./top-k-sampling) 等更复杂解码策略的基准线。

---

## L1 · 一句话点破

贪心解码：**每一步都选当前概率最高的 token，不再回头。**

形式化：给定已生成序列 $x_{<t}$ 和模型输出的下一 token 分布 $P_t$：

$$
x_t = \arg\max_{v \in V} P_t(v | x_{<t})
$$

简单直接。模型说"很"概率最高，就选"很"；下一步模型说"好"概率最高，就选"好"。一路选最高。

## L2 · 通俗类比

走迷宫时，每到一个路口就选"眼前看起来最近出口的方向"，不回头。这是贪心。

问题：

- **局部最优 ≠ 全局最优**：眼前最近出口的方向可能是个死胡同。回头看，刚才选另一个路口更好。
- **可能陷入循环**：在某些迷宫结构下，贪心会让你在几个路口绕圈出不来。

贪心解码同理：

- **错过更优序列**：当前最高概率 token 可能让你后面只能选低概率 token，全局看不如选次高 token + 后续高概率 token
- **重复退化**：模型陷入"我我我我我"或"重复重复重复重复"的循环，因为"重复"在已重复的上下文下概率最高

贪心解码的"省事"换来的是"质量上限低"。它适合：**确定性要求高、对多样性无要求的场景**，如代码补全、事实问答（需要稳定答案）。

## L3 · 正经定义

**贪心解码（Greedy Decoding）**：自回归生成的最简解码策略，每步选择条件概率最大的 token。

```
def greedy_decode(model, prompt, max_len):
    seq = prompt
    for _ in range(max_len):
        logits = model(seq)[-1]      # 最后一个位置的 logits
        next_token = argmax(logits)
        if next_token == EOS: break
        seq.append(next_token)
    return seq
```

**性质**：

- **确定性**：同一 prompt 永远生成同一输出（除非模型本身有不确定性）
- **不可复水**：每步只看当前最优，不考虑后续
- **复杂度**：$O(T)$（每步 $O(1)$ 选择），最便宜

**典型问题**：

| 问题 | 表现 | 原因 |
|------|------|------|
| 重复退化 | "我我我我我"、"然后然后然后" | 模型对已重复 token 给高概率 |
| 短视 | 错过更优全局序列 | 不回溯，局部最优 |
| 枯燥 | 输出机械、无创意 | 永远选最安全选项 |

**对比 [beam search](./beam-search)**：beam search 每步保留 top-B 个候选序列，扩展探索。能避开贪心的局部最优陷阱，但更贵、仍有重复退化问题。

**参考资料**：
- [Jurafsky & Martin, Speech and Language Processing 第 13 章](https://web.stanford.edu/~jurafsky/slp3/) - 经典解码策略
- [Holtzman et al., 2019 - The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 重复退化与 nucleus sampling
- [Vijayakumar et al., 2016 - Diverse Beam Search](https://arxiv.org/abs/1610.02424)

## L4 · 原理深挖

### 4.1 为什么贪心不是最优：序列概率的视角

自回归生成目标是最大化整个序列的概率：

$$
P(x_{1:T}) = \prod_t P(x_t | x_{<t})
$$

贪心每步最大化 $P(x_t | x_{<t})$，但不一定最大化乘积。反例：

```
路径 A: P=0.5 * 0.5 * 0.5 = 0.125  (贪心会选，第一步 0.5 最高)
路径 B: P=0.4 * 0.9 * 0.9 = 0.324  (整体更高，但贪心不会选)
```

贪心选了第一步最高的 0.5，但后续概率低；路径 B 第一步次高，但后续高，整体更优。

**这正是 [beam search](./beam-search) 解决的问题**：保留多条候选路径，用累计概率比较，找到更接近全局最优的序列。

但 beam search 也不是真正的全局最优（仍是近似），且计算成本是贪心的 B 倍。

### 4.2 重复退化：贪心的标志性失败

[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 系统研究"神经文本退化"现象。贪心解码（以及 beam search）的典型失败模式：

**① 简单重复**

```
"我去了公园，然后我去了公园，然后我去了公园..."
```

模型生成"我去了公园"后，"我去了公园"作为已知上下文，下一步模型给"我去了公园"高概率（因为它刚见过这个 pattern）。

**② 词汇退化**

模型反复使用一小撮高频词，丢失词汇多样性。Holtzman et al. 统计：人类文本中下一 token 的概率分布通常较平（多个候选词概率接近），而贪心 / beam search 生成的文本概率分布越来越尖（少数词主导）。

**③ 语义重复**

```
"这件事很重要。这件事非常重要。这件事的重要性无法估量。"
```

不同字面但同义重复，模型陷入"语义循环"。

**为什么贪心容易退化？** 模型对"已出现的 pattern"给高概率（自回归的特性），贪心强化这个偏向，越重复越重复。采样的随机性（[Top-k](./top-k-sampling)、[Top-p](./top-p-sampling)）能打破循环。

### 4.3 贪心什么时候够用

贪心不是一无是处。它在以下场景是合理选择：

**① 确定性输出任务**

- 代码补全：同一上下文应生成同一补全，不需要多样性
- 事实问答：要"标准答案"，不要创意
- 分类任务的 token 输出：如"答案是 A/B/C"

**② 高温度会损害质量的场景**

数学推理、代码生成、结构化输出（如 JSON）等任务，引入随机性反而降低质量。贪心或低温采样更稳。

**③ 实时性要求高的场景**

贪心最便宜。流式输出、低延迟场景可能优先贪心（再配合 speculative decoding 加速）。

**④ 评测复现**

学术论文评测常要求贪心（temperature=0），保证结果可复现。

### 4.4 贪心 vs 采样：确定性 vs 多样性的权衡

贪心本质是"采样温度=0"的极端情况。引入温度的采样策略谱：

| 策略 | 确定性 | 多样性 | 适用 |
|------|--------|--------|------|
| 贪心（temp=0） | 最强 | 无 | 代码、推理、评测 |
| 低温采样（temp=0.3） | 强 | 弱 | 助手回答、事实性任务 |
| 中温采样（temp=0.7） | 中 | 中 | 通用对话（默认） |
| 高温采样（temp=1.0+） | 弱 | 强 | 创意写作、brainstorming |

实际使用：默认 0.7 左右，根据任务调。[温度](./temperature) 词条详述机制。

### 4.5 贪心和 beam search 的关系

[beam search](./beam-search) 是贪心的推广：保留 B 条候选路径，每步扩展所有候选。

- B=1 退化为贪心
- B>1 探索更多，可能找到更优序列
- B 越大越接近全局最优，但成本越高

但 beam search 在开放式生成（对话、写作）效果不一定好，甚至更糟：

- **更严重的重复退化**：beam search 偏好"高概率"序列，而高概率序列常是重复的
- **缺乏多样性**：B 条候选相似，没真正探索
- **不匹配人类语言分布**：人类语言不是"高概率"的，是"合理但适度意外"的

[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 提出 nucleus sampling (Top-p)，主张开放式生成用采样而非搜索。这是 ChatGPT 等对话模型默认用采样而非 beam search 的原因。

但在翻译、摘要等"有标准答案"的任务，beam search 仍优于采样。任务性质决定策略选择。

### 4.6 重复退化的对策

针对贪心/beam search 的重复退化，常见对策：

**① 采样**：[Top-k](./top-k-sampling)、[Top-p](./top-p-sampling)、[温度](./temperature) 引入随机性打破循环

**② Repetition Penalty**：对已出现 token 的概率打折（如 [CTRL 论文](https://arxiv.org/abs/1909.05858)）。HuggingFace transformers 默认 1.0（无惩罚），可调到 1.1-1.3

**③ No-repeat-ngram**：禁止重复 n-gram，硬约束

**④ Beam Diversification**：[Diverse Beam Search](https://arxiv.org/abs/1610.02424) 在 beam 内加多样性惩罚

**⑤ 后处理**：检测重复 pattern，重写或截断

实际工程中常用组合：**Top-p (0.9) + 温度 (0.7) + 轻度 repetition penalty (1.1)**，平衡多样性和质量。

## L5 · 沿革与坑

### 沿革

- **1980s-1990s**：贪心解码在 HMM、统计机器翻译中是默认基线。
- **2014-2017**：seq2seq + attention 时代，beam search 取代贪心成为翻译、摘要的主流。
- **2017-2019**：GPT-2 等开放式生成模型出现，beam search 暴露重复退化问题。
- **2019**：[Holtzman et al. - Nucleus Sampling](https://arxiv.org/abs/1904.09751) 系统研究退化，提出 Top-p。采样成为开放式生成主流。
- **2020-2022**：GPT-3、ChatGPT 等大模型默认用 Top-p + 温度，beam search 在对话场景式微。
- **2023-2024**：推理模型（如 o1、DeepSeek-R1）对确定性输出要求高，贪心/低温回归。同时 Code LLM、数学 LLM 评测倾向贪心保证复现。
- **2025**：对比研究表明，"采样 + self-consistency"（多次采样取多数）在推理任务上优于贪心，重新评估采样价值。

### 常见误解

- ❌ **误解**：贪心解码效果最差，不该用。
  ✅ **真相**：在确定性、推理、评测复现场景，贪心是合理甚至最优选择。它的"缺点"（无多样性）在那些场景是优点（4.3）。

- ❌ **误解**：贪心一定能找到最优序列。
  ✅ **真相**：贪心找的是局部最优（每步最高概率），不是全局最优（整个序列最高概率）。反例见 4.1。

- ❌ **误解**：贪心 = temperature=0。
  ✅ **真相**：等价。temperature=0 时 softmax 退化为 argmax，与贪心完全相同。框架里常通过设 temperature=0 实现。

- ❌ **误解**：贪心的重复退化是模型 bug。
  ✅ **真相**：是自回归 + 贪心策略的结构性问题。模型对"已见 pattern"给高概率是正常的（自回归特性），贪心强化这个偏向导致循环。用采样即可缓解（4.6）。

- ❌ **误解**：beam search 总比贪心好。
  ✅ **真相**：在翻译、摘要等"有标准答案"任务上 beam search 通常更好；但在开放式生成（对话、写作）上 beam search 反而更易退化，采样更合适（4.5）。

### 面试怎么考

1. **"什么是贪心解码？有什么优缺点？"** --每步选最高概率 token。优点：快、确定、可复现。缺点：局部最优、重复退化（L1、L3）。
2. **"贪心为什么不是全局最优？"** --序列概率是乘积，贪心最大化每步单概率不一定最大化乘积。反例：0.5*0.5*0.5 < 0.4*0.9*0.9（4.1）。
3. **"什么是重复退化？怎么解决？"** --模型陷入"重复重复重复"循环。对策：采样、repetition penalty、no-repeat-ngram（4.2、4.6）。
4. **"贪心和 beam search 的关系？"** --beam search 保留 B 条候选，B=1 退化为贪心。beam search 在有标准答案任务上更好，开放式生成未必（4.5）。
5. **"什么场景用贪心？"** --代码、推理、评测复现等需要确定性的场景（4.3）。

## 延伸阅读

- 📄 [Holtzman et al., 2019 - The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- 📄 [Vijayakumar et al., 2016 - Diverse Beam Search](https://arxiv.org/abs/1610.02424)
- 📄 [Keskar et al., 2019 - CTRL](https://arxiv.org/abs/1909.05858) - repetition penalty
- 📝 [Jurafsky & Martin - SLP3 第 13 章](https://web.stanford.edu/~jurafsky/slp3/)

---

> *上一篇：[自回归生成](./autoregressive) -- GPT 为什么一个字一个字往外吐。*
> *下一篇：[Beam Search](./beam-search) -- 保留多条候选路径的解码策略。*
