---
title: Top-k 采样
slug: top-k-sampling
category: 推理与生成
tags: [Top-k, 采样, Fanout, 解码策略, 多样性]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# Top-k 采样

> **一句话 TL;DR**：Top-k 采样是 [自回归生成](./autoregressive) 的解码策略--每步只在前 k 个最高概率的 token 里随机采样，其余低概率 token 直接排除。它解决 [贪心解码](./greedy-decoding) 的枯燥重复，又避免纯随机采样的胡言乱语，是"多样性 vs 质量"的折中。后被 [Top-p 采样](./top-p-sampling) 在多数场景超越。

---

## L1 · 一句话点破

Top-k 采样：**每步把模型输出的概率分布截断到前 k 个 token，重新归一化后随机采样。**

```
原始分布: P(好)=0.5, P(差)=0.2, P(中)=0.1, P(坏)=0.05, P(烂)=0.05, ... (长尾)
Top-3 截断: 只保留 P(好), P(差), P(中)
重新归一化: P(好)=0.625, P(差)=0.25, P(中)=0.125
采样: 按新分布随机选一个
```

k 是固定的"窗口大小"。k=1 退化为 [贪心解码](./greedy-decoding)；k=|V|（词表大小）退化为纯随机采样。

典型 k=40-50。

## L2 · 通俗类比

自助餐选菜：

- **贪心解码**：永远只拿人气第一名那道菜。天天一样，吃腻。
- **纯随机采样**：在所有菜（包括没人要的剩菜）里随机拿。可能拿到难吃的。
- **Top-k 采样**：只在"人气前 k 名"里随机拿。既保证不会太差（前 k 已经是受欢迎的），又有变化（不是固定一道）。

k 的权衡：

- k 小（如 k=5）：选择少，质量稳，多样性低（接近贪心）
- k 大（如 k=100）：选择多，多样性高，质量可能下降（接近纯随机）
- 中间（k=40-50）：平衡

问题：k 是固定的，但不同位置"该选多少候选"不一样。有时模型很确定（如标点前），就该 k 小；有时模型很犹豫（如创意写作），就该 k 大。Top-k 用固定 k 不够灵活，这是 [Top-p 采样](./top-p-sampling) 出现的原因。

## L3 · 正经定义

**Top-k 采样**（[Fan et al., 2018](https://arxiv.org/abs/1805.04833)）：自回归生成时，每步只在概率最高的 k 个 token 上重新归一化并采样。

```
def top_k_sample(model, prompt, k, max_len):
    seq = prompt
    for _ in range(max_len):
        logits = model(seq)[-1]
        probs = softmax(logits)
        # 取 top-k token
        topk_probs, topk_idx = probs.topk(k)
        # 重新归一化
        topk_probs = topk_probs / topk_probs.sum()
        # 采样
        next_token = topk_idx[sample(topk_probs)]
        if next_token == EOS: break
        seq.append(next_token)
    return seq
```

**性质**：

- **多样性可控**：k 调节多样性。k 小趋近贪心，k 大趋近随机
- **排除长尾噪声**：低概率 token（错别字、罕见词）被排除
- **固定窗口**：不随分布形状自适应

**典型参数**：k=40（[Fan et al., 2018](https://arxiv.org/abs/1805.04833) 默认）、k=50、k=100。

**与其他解码的关系**：

| 策略 | 多样性来源 | 适用 |
|------|-----------|------|
| [贪心](./greedy-decoding) | 无 | 确定性输出 |
| [Beam search](./beam-search) | 无（保留多候选但不随机） | 翻译、摘要 |
| Top-k | 固定窗口内随机 | 通用生成 |
| [Top-p](./top-p-sampling) | 自适应窗口内随机 | 主流，多数场景更优 |
| 纯随机 | 全词表随机 | 几乎不用 |

**参考资料**：
- [Fan et al., 2018 - Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) - Top-k 采样提出
- [Holtzman et al., 2019 - Nucleus Sampling](https://arxiv.org/abs/1904.09751) - Top-p，对比 Top-k
- [Ackley et al., 1985 - Boltzmann sampling](https://link.springer.com/article/10.1007/BF00341431) - 早期温度采样

## L4 · 原理深挖

### 4.1 为什么需要采样：贪心和 beam search 的局限

[贪心解码](./greedy-decoding) 和 [beam search](./beam-search) 都是"确定性地选最优"，导致：

- **重复退化**：永远选最高概率，模型对已见 pattern 给高概率，陷入循环
- **枯燥**：同一 prompt 永远生成同一输出，缺乏多样性
- **不匹配人类语言**：人类语言有适度随机性，不是"每步最优"

[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 实证：人类文本的下一 token 概率分布是双峰的（少数高概率 + 长尾），最优生成应在高概率区随机选择，而非固定选最高。

Top-k 采样就是引入这种"高概率区的随机性"。

### 4.2 Top-k 的固定窗口问题

Top-k 的核心缺陷：**k 是固定的，不随分布形状自适应。**

考虑两种情况：

**情况 A：模型很确定**

```
P(好)=0.95, P(差)=0.02, P(中)=0.01, ...
```

模型 95% 确定是"好"。但 Top-k (k=40) 会把 39 个低概率 token 也纳入采样范围，重新归一化后变成 P(好)=0.6, ... 39 个低概率 token 各 0.01。原本确定的输出被稀释。

这种情况下应该 k=1 或 k=2（贪心或近似贪心）。

**情况 B：模型很犹豫**

```
P(苹果)=0.15, P(梨)=0.13, P(香蕉)=0.12, P(橘子)=0.11, P(葡萄)=0.10, ...
```

模型很不确定，多个候选概率接近。Top-k (k=40) 会包含很多低概率 token，但前 5 个才是真正合理的候选。

这种情况下应该 k=5（窄窗口），不是 k=40。

**Top-k 用固定 k 无法区分两种情况**，要么过度稀释（情况 A），要么不够聚焦（情况 B）。

[Top-p 采样](./top-p-sampling)（nucleus sampling）解决这个问题：根据累计概率自适应确定窗口大小。

### 4.3 Top-k 的优点：简单可控

尽管有缺陷，Top-k 仍有价值：

**① 简单**

实现简单，调参容易（一个 k）。比 Top-p 直观。

**② 排除长尾噪声**

低概率 token（错别字、不相关词）被截断。在词表大、长尾脏的模型上有效。

**③ 行为可预测**

固定 k 让行为可预测。k=10 永远是 10 个候选；k=50 永远是 50 个。便于调试和复现。

**④ 适合分布稳定的场景**

当模型分布形状相对一致（如代码补全），Top-k 的固定窗口不是问题。

### 4.4 Top-k vs Top-p：什么时候用哪个

| 维度 | Top-k | Top-p |
|------|-------|-------|
| 窗口 | 固定 k | 自适应（累计概率 p） |
| 适应性 | 不随分布变化 | 随分布自适应 |
| 直觉 | "前 N 个候选" | "累计概率 p 的候选" |
| 主流度 | 较少 | 主流 |
| 典型参数 | k=40 | p=0.9 |

**多数场景 Top-p 更好**：它能根据模型确定性自动调整窗口大小。

**Top-k 仍有用**：

- 简单调试、复现
- 分布稳定场景（如代码补全）
- 与 Top-p 组合使用（如 Top-k=50 + Top-p=0.95，双重约束）

实际 API（OpenAI、Anthropic）默认用 Top-p，但保留 Top-k 作为可选。

### 4.5 Top-k 与温度的关系

[温度](./temperature) 是另一种调节分布"陡峭度"的方式：

$$
P_T(v) = \frac{\exp(\text{logit}_v / T)}{\sum \exp(\text{logit}_{v'} / T)}
$$

- T 低：分布变陡，高概率 token 概率更高（趋近贪心）
- T 高：分布变平，低概率 token 概率更高（趋近随机）

温度是"全局调节"，Top-k 是"硬截断"。两者可组合：

```
logits = model(seq)[-1]
logits = logits / temperature       # 先调温度
probs = softmax(logits)
topk_probs, topk_idx = probs.topk(k)  # 再 Top-k
topk_probs = topk_probs / topk_probs.sum()
next_token = topk_idx[sample(topk_probs)]
```

实际使用：**Top-p + 温度** 是主流组合（OpenAI 默认 Top-p=1, temperature=1，按需调）。Top-k 通常设很大（如 100）作为上限，主要靠 Top-p 和温度控制。

### 4.6 采样的根本局限：质量上限

采样策略（Top-k、Top-p）解决多样性问题，但带来新问题：

**① 质量波动**

随机性让输出质量不稳定。同一 prompt 多次生成，质量可能差异大。差的采样会生成不通顺、错误的输出。

**② 不可复现**

随机性让结果不可预测。调试、评测、生产环境可能需要固定 seed。

**③ 错误累积**

[自回归生成](./autoregressive) 的错误累积问题在采样下更严重：随机选错一个 token，后续基于错字继续，越错越远。

对策：

- **多次采样取最优**：best-of-n，多次采样后用 reward model 或投票选最好的
- **Self-consistency**：多次采样，取多数答案（适合推理任务）
- **约束解码**：限制 token 必须满足某种约束（语法、JSON 格式）

## L5 · 沿革与坑

### 沿革

- **2018**：[Fan et al. - Top-k Sampling](https://arxiv.org/abs/1805.04833) 提出，用于故事生成，解决贪心/beam search 的枯燥。
- **2019**：[Holtzman et al. - Nucleus Sampling (Top-p)](https://arxiv.org/abs/1904.09751) 提出，实证优于 Top-k。
- **2020-2022**：GPT-3、ChatGPT 等默认用 Top-p，Top-k 作为可选。
- **2023-2024**：best-of-n、self-consistency 等基于采样的方法流行，采样成为推理任务的关键工具。
- **2025**：推理模型（o1、DeepSeek-R1）用 best-of-n + 投票替代贪心，采样地位回升。

### 常见误解

- ❌ **误解**：k 越大多样性越好，效果越好。
  ✅ **真相**：k 大引入更多低概率 token，可能让输出质量下降。k 应该根据任务调，典型 40-50。

- ❌ **误解**：Top-k 和 Top-p 二选一。
  ✅ **真相**：可组合使用（Top-k + Top-p 双重约束）。Top-k 设大作为上限，Top-p 作为主要控制。

- ❌ **误解**：Top-k 总比贪心好。
  ✅ **真相**：在确定性要求高的场景（代码、推理），贪心更稳。Top-k 的随机性反而损害质量。

- ❌ **误解**：Top-k 是固定窗口，所以不如 Top-p。
  ✅ **真相**：在分布稳定的场景（如代码补全），固定 k 是优点（可预测）。Top-p 不是普适更优。

- ❌ **误解**：采样质量一定比贪心好。
  ✅ **真相**：采样带来多样性，但牺牲了稳定性。质量取决于任务和参数。开放式生成采样好，确定性任务贪心更好。

### 面试怎么考

1. **"什么是 Top-k 采样？"** --每步在前 k 个高概率 token 中重新归一化采样（L1、L3）。
2. **"Top-k 的缺陷是什么？"** --固定 k 不随分布自适应。模型确定时过度稀释，犹豫时不够聚焦（4.2）。
3. **"Top-k 和 Top-p 的区别？"** --Top-k 固定窗口，Top-p 自适应窗口（累计概率）。Top-p 多数场景更优（4.4）。
4. **"Top-k 和温度的关系？"** --温度是全局调节分布陡峭度，Top-k 是硬截断。可组合使用（4.5）。
5. **"采样有什么根本问题？"** --质量波动、不可复现、错误累积（4.6）。

## 延伸阅读

- 📄 [Fan et al., 2018 - Top-k Sampling](https://arxiv.org/abs/1805.04833)
- 📄 [Holtzman et al., 2019 - Nucleus Sampling](https://arxiv.org/abs/1904.09751)
- 📝 [HuggingFace - How to generate text](https://huggingface.co/blog/how-to-generate)

---

> *上一篇：[Beam Search](./beam-search) -- 保留多条候选路径的解码策略。*
> *下一篇：[Top-p 采样](./top-p-sampling) -- 自适应窗口的采样策略。*
