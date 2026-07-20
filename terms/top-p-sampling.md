---
title: Top-p 采样（Nucleus Sampling）
slug: top-p-sampling
category: 推理与生成
tags: [Top-p, Nucleus Sampling, 解码策略, 自适应窗口, 概率分布]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# Top-p 采样（Nucleus Sampling）

> **一句话 TL;DR**：Top-p 采样（nucleus sampling）是 [Top-k 采样](./top-k-sampling) 的改进版--不固定窗口大小 k，而是按累计概率 p 自适应截断：把概率从高到低累加，到达 p 时停止，只在这些"核心 token"里采样。它解决了 Top-k 固定窗口的缺陷，是当前大模型开放式生成的主流解码策略。

---

## L1 · 一句话点破

Top-p 采样（nucleus sampling）：**每步把 token 按概率从高到低排序，累加概率直到达到阈值 p，只在这个"核心集合"（nucleus）里重新归一化采样。**

```
原始分布（排序后）: P(A)=0.5, P(B)=0.3, P(C)=0.1, P(D)=0.05, P(E)=0.03, ...
Top-p (p=0.9):
  累加: 0.5 (A) -> 0.8 (A+B) -> 0.9 (A+B+C) ✓ 达到 0.9
  核心集合: {A, B, C}
  重新归一化: P(A)=0.556, P(B)=0.333, P(C)=0.111
  采样: 按新分布随机选
```

p 是"累计概率阈值"。p=0 退化为 [贪心解码](./greedy-decoding)；p=1 退化为纯随机采样。

典型 p=0.9-0.95。

## L2 · 通俗类比

继续自助餐类比（见 [Top-k 采样](./top-k-sampling)）：

- **[贪心](./greedy-decoding)**：永远只拿第一名那道菜。
- **[Top-k](./top-k-sampling)**：永远在前 k 名里随机拿。问题：有时前 3 名就占了 95% 人气，第 4-40 名都是冷门；有时前 40 名人气都差不多。固定 k 不能适应。
- **Top-p**：拿"累计人气达到 90%"的那些菜。如果前 3 道就占了 90%，就只在这 3 道里选；如果前 40 道才占 90%，就在 40 道里选。**窗口大小自适应分布形状**。

具体例子：

**情况 A：模型很确定**

```
P(好)=0.95, P(差)=0.03, P(中)=0.01, ...
Top-p (p=0.9): 累加 0.95 已超 0.9，核心集合 = {好}
→ 退化为贪心，选"好"
```

模型确定时，Top-p 自动变贪心，避免稀释。

**情况 B：模型很犹豫**

```
P(苹果)=0.15, P(梨)=0.13, P(香蕉)=0.12, P(橘子)=0.11, P(葡萄)=0.10, P(西瓜)=0.09, ...
Top-p (p=0.9): 累加 0.15+0.13+0.12+0.11+0.10+0.09+... 约需 7-8 个 token 才达 0.9
→ 核心集合 = {苹果, 梨, 香蕉, 橘子, 葡萄, 西瓜, ...}
→ 在这些合理候选里随机选
```

模型犹豫时，Top-p 自动扩大窗口，增加多样性。

这就是 Top-p 比 [Top-k](./top-k-sampling) 优秀的地方：**窗口大小随分布自适应**。

## L3 · 正经定义

**Top-p 采样 / Nucleus Sampling**（[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)）：每步定义"核心集合" $\mathcal{V}^{(p)}$ 为概率从高到低累加达到 p 的最小 token 集合：

$$
\mathcal{V}^{(p)} = \min \left\{ S \subseteq V : \sum_{v \in S} P(v) \geq p \right\}
$$

在 $\mathcal{V}^{(p)}$ 上重新归一化并采样：

$$
P'(v) = \frac{P(v)}{\sum_{v' \in \mathcal{V}^{(p)}} P(v')}, \quad v \in \mathcal{V}^{(p)}
$$

```
def top_p_sample(model, prompt, p, max_len):
    seq = prompt
    for _ in range(max_len):
        logits = model(seq)[-1]
        probs = softmax(logits)
        # 按概率降序排序
        sorted_probs, sorted_idx = probs.sort(descending=True)
        # 累计概率
        cum_probs = sorted_probs.cumsum()
        # 找到第一个累计 >= p 的位置，保留它及之前的
        cutoff = (cum_probs > p).nonzero()[0][0] + 1
        nucleus = sorted_probs[:cutoff]
        nucleus_idx = sorted_idx[:cutoff]
        # 重新归一化并采样
        nucleus = nucleus / nucleus.sum()
        next_token = nucleus_idx[sample(nucleus)]
        if next_token == EOS: break
        seq.append(next_token)
    return seq
```

**典型参数**：p=0.9（[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 推荐）、p=0.92、p=0.95。

**与其他解码的关系**：

| 策略 | 窗口 | 自适应 | 主流度 |
|------|------|--------|--------|
| [贪心](./greedy-decoding) | 1 | - | 确定性场景 |
| [Top-k](./top-k-sampling) | 固定 k | 否 | 较少 |
| **Top-p** | 累计 p | 是 | **主流** |
| [温度](./temperature) | 全词表 | - | 配合 Top-p |

**参考资料**：
- [Holtzman et al., 2019 - The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - Top-p 提出，必读
- [Fan et al., 2018 - Top-k Sampling](https://arxiv.org/abs/1805.04833) - 前作
- [Zhang et al., 2020 - Trade-off Between Diversity and Quality](https://arxiv.org/abs/2002.00655) - 采样策略对比

## L4 · 原理深挖

### 4.1 为什么 Top-p 比 Top-k 好：分布形状的洞察

[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 的核心洞察：**自然语言的下一 token 概率分布形状不固定，固定窗口（Top-k）无法适应。**

他们实证发现，自然语言文本的下一 token 分布有两种典型形态：

**① 锐利分布（sharp distribution）**

模型很确定，少数 token 占绝大部分概率。例如标点前的 token："今天天气很___"，"好"占 0.9+。

这种情况下应该 k=1 或 k=2（贪心或近似贪心）。但 [Top-k](./top-k-sampling) (k=40) 会强行纳入 39 个低概率 token，稀释确定性的输出。

**② 平坦分布（flat distribution）**

模型很犹豫，多个 token 概率接近。例如创意写作："她推开门，看到___"，可能有"一片森林""一个老人""一只猫"等多个合理候选。

这种情况下应该 k=10-20（大窗口，允许多样性）。但 Top-k (k=40) 可能纳入太多不相关 token。

**Top-p 的自适应**：p=0.9 时

- 锐利分布：核心集合小（1-3 个），趋近贪心
- 平坦分布：核心集合大（10-20 个），允许多样性

这种"窗口随分布形状自动调节"是 Top-p 优于 Top-k 的根本原因。

### 4.2 Nucleus 的概率解释：高概率"质量"

"nucleus"（核）这个词的选择有深意。在概率论中，分布的"核"是占据绝大部分概率质量的最小集合。

对自然语言，[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 实证：人类文本的下一 token 几乎总是落在 nucleus（p=0.9 的核心集合）内。换言之：

- **nucleus 内的 token**：人类可能写的（合理候选）
- **nucleus 外的 token**：人类几乎不会写的（错误、罕见、不相关）

Top-p 采样的本质：**只在"人类合理候选"里随机选，排除"不合理候选"**。这既保证质量（不会选离谱 token），又保留多样性（在合理候选里随机）。

这与 [贪心解码](./greedy-decoding)（永远选最高，枯燥）和纯随机（可能选离谱 token）形成对比。Top-p 找到了"合理但适度意外"的平衡点。

### 4.3 Top-p 与人类语言分布的匹配

[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 的另一关键发现：**人类语言的下一 token 概率分布是双峰的（bimodal）。**

- **第一峰**：少数高概率 token（如 0.3-0.5 的"好""的""了"）
- **第二峰**：长尾低概率 token（每个 0.001 以下，但总数多）
- **中间**：几乎没有

这意味着人类语言不是"一个最优 token"，而是"少数合理 token + 大量不可能 token"。Top-p 的 nucleus 恰好捕捉第一峰（合理 token），排除第二峰（不可能 token）。

[贪心](./greedy-decoding) 和 [beam search](./beam-search) 永远选第一峰最高的，丢失了第一峰内的多样性。纯随机把第二峰也纳入，引入不合理 token。Top-p 在第一峰内随机，最匹配人类语言分布。

这是 ChatGPT 等对话模型默认用 Top-p 的根本理论依据。

### 4.4 p 的选择：质量 vs 多样性

p 调节质量与多样性的权衡：

| p 值 | 行为 | 适用 |
|------|------|------|
| 0.0 | 退化为贪心 | 确定性输出（代码、推理） |
| 0.5 | 窗口很小 | 高质量、低多样性（事实问答） |
| 0.9 | 平衡 | **通用对话默认** |
| 0.95 | 窗口较大 | 创意写作、brainstorming |
| 1.0 | 退化为纯随机 | 几乎不用 |

实务：**p=0.9 是通用默认**。需要更确定降到 0.7-0.8；需要更多创意升到 0.95。

注意 p 和 [温度](./temperature) 都调节多样性，但机制不同（见 4.5）。实际常组合使用。

### 4.5 Top-p vs 温度：两种调节方式

[温度](./temperature) 通过调节 softmax 的"陡峭度"改变整个分布形状：

$$
P_T(v) = \frac{\exp(\text{logit}_v / T)}{\sum \exp(\text{logit}_{v'} / T)}
$$

- T 低：分布变陡，高概率 token 概率更高
- T 高：分布变平，低概率 token 概率更高

**Top-p 是"硬截断"**（nucleus 外概率直接置 0），**温度是"软调节"**（所有 token 概率都调整，不截断）。

两者可组合：

```
logits = model(seq)[-1]
logits = logits / temperature       # 先调温度
probs = softmax(logits)
# 再 Top-p
sorted_probs, sorted_idx = probs.sort(descending=True)
cum_probs = sorted_probs.cumsum()
cutoff = (cum_probs > p).nonzero()[0][0] + 1
nucleus = sorted_probs[:cutoff] / sorted_probs[:cutoff].sum()
next_token = sorted_idx[:cutoff][sample(nucleus)]
```

**OpenAI / Anthropic API 默认**：temperature=1.0, top_p=1.0（不限制），用户按需调。实务常用组合：

- **通用对话**：temperature=0.7, top_p=0.9
- **事实问答**：temperature=0.3, top_p=0.8
- **创意写作**：temperature=0.9, top_p=0.95
- **代码/推理**：temperature=0, top_p=1（纯贪心）

### 4.6 Top-p 的局限

尽管 Top-p 是主流，它仍有局限：

**① 长尾 token 永远被排除**

nucleus 外的 token 概率被置 0。但有时低概率 token 是合理的（罕见但正确的词）。Top-p 可能让模型永远学不会用罕见词。

**② p 阈值仍需手动调**

p 不是万能参数，不同任务、不同模型最优 p 不同。仍需实验调参。

**③ 错误累积问题未解决**

[自回归生成](./autoregressive) 的错误累积在 Top-p 下仍存在：采样选错一个 token，后续基于错字继续。Top-p 缓解了 [贪心](./greedy-decoding) 的重复退化，但没解决错误累积。

**④ 对长序列生成效果下降**

长序列生成时，早期一个"合理但走偏"的采样会让后续都偏。Top-p 在短生成（对话、短文）效果好，长篇生成（小说、长文）仍需配合其他机制（如大纲引导、self-consistency）。

### 4.7 采样策略的演化

Top-p 之后还有演化：

**① Top-a 采样**：动态调整 p，基于当前分布的熵

**② Typical Sampling**（[Meister et al., 2022](https://arxiv.org/abs/2202.00666)）：选"典型"token（条件熵附近的），更匹配信息论视角

**③ Mirostat**（[Basu et al., 2020](https://arxiv.org/abs/2007.14966)）：动态调节温度保持目标困惑度

**④ ETA Sampling**：综合多种策略

但这些都没撼动 Top-p 的主流地位。Top-p 简单、有效、可解释，是工程友好性胜出的典型例子。

## L5 · 沿革与坑

### 沿革

- **2018**：[Fan et al. - Top-k Sampling](https://arxiv.org/abs/1805.04833) 提出，解决贪心/beam search 枯燥。
- **2019**：[Holtzman et al. - Nucleus Sampling (Top-p)](https://arxiv.org/abs/1904.09751) 提出，实证优于 Top-k。论文同时揭示神经文本退化现象。
- **2020-2022**：GPT-3、ChatGPT 等大模型默认用 Top-p，成为开放式生成事实标准。
- **2022**：[Meister et al. - Typical Sampling](https://arxiv.org/abs/2202.00666) 等改进出现，但未撼动 Top-p。
- **2023-2024**：best-of-n、self-consistency 等基于采样的方法在推理任务流行，Top-p 作为基础采样器。
- **2025**：推理模型（o1、DeepSeek-R1）用 best-of-n + 投票，Top-p 仍是基础采样策略。

### 常见误解

- ❌ **误解**：Top-p 总比 Top-k 好。
  ✅ **真相**：多数场景 Top-p 更好（自适应窗口），但在分布稳定的场景（如代码补全）Top-k 的固定窗口是优点（可预测）。两者各有适用（4.1）。

- ❌ **误解**：p 越大多样性越好，效果越好。
  ✅ **真相**：p 大引入更多低概率 token，可能让输出质量下降。p 应根据任务调，典型 0.9-0.95。

- ❌ **误解**：Top-p 能解决幻觉。
  ✅ **真相**：Top-p 解决的是"枯燥重复"问题，不是"幻觉"问题。幻觉的根源是模型不知道自己不知道什么，采样策略无法解决（见 [幻觉](./hallucination)）。

- ❌ **误解**：Top-p 和温度是一回事，二选一。
  ✅ **真相**：机制不同。Top-p 是硬截断，温度是软调节。可组合使用，实务中常组合（4.5）。

- ❌ **误解**：Top-p 采样质量一定比贪心好。
  ✅ **真相**：在开放式生成（对话、写作）Top-p 通常更好；在确定性任务（代码、推理）贪心更稳。任务性质决定（4.4）。

- ❌ **误解**：nucleus 是论文发明的概念。
  ✅ **真相**：nucleus（核）是概率论已有概念，指占据绝大部分概率质量的最小集合。论文借用这个概念命名采样策略（4.2）。

### 面试怎么考

1. **"什么是 Top-p 采样？和 Top-k 的区别？"** --按累计概率 p 自适应截断，vs Top-k 固定窗口。Top-p 多数场景更优（L1、4.1）。
2. **"为什么 Top-p 比 Top-k 好？"** --自然语言分布形状不固定，Top-p 窗口随分布自适应，Top-k 固定窗口不能适应（4.1）。
3. **"什么是 nucleus？为什么这么命名？"** --概率论中"占据绝大部分概率质量的最小集合"。Top-p 只在 nucleus 内采样，排除不合理 token（4.2）。
4. **"Top-p 和温度有什么区别？"** --Top-p 是硬截断，温度是软调节全分布。可组合使用（4.5）。
5. **"p 怎么选？"** --通用默认 0.9。需要确定性降到 0.7-0.8；需要创意升到 0.95（4.4）。
6. **"Top-p 有什么局限？"** --长尾 token 永远被排除、p 需手调、错误累积未解决、长序列效果下降（4.6）。

## 延伸阅读

- 📄 [Holtzman et al., 2019 - The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 必读
- 📄 [Fan et al., 2018 - Top-k Sampling](https://arxiv.org/abs/1805.04833)
- 📄 [Meister et al., 2022 - Typical Sampling](https://arxiv.org/abs/2202.00666)
- 📝 [HuggingFace - How to generate text](https://huggingface.co/blog/how-to-generate)

---

> *上一篇：[Top-k 采样](./top-k-sampling) -- 只在高概率候选里随机。*
> *下一篇：[温度 Temperature](./temperature) -- 一个参数怎么调节"创造力"。*
