---
title: 多头注意力（Multi-Head Attention）
slug: multi-head-attention
category: 模型架构与训练
tags: [Transformer, 注意力机制, 架构, NLP]
author: ai-terms-fun
created: 2026-07-19
updated: 2026-07-19
---

# 多头注意力（Multi-Head Attention）

> **一句话 TL;DR**：把一个大的自注意力拆成 $h$ 个并行的小注意力（每个维度 $d/h$），各自学习不同的关系模式，最后拼接投影回原维度。代价是参数量略增，换来的是模型能同时建模多种不同的依赖关系。

---

## L1 · 一句话点破

多头注意力的本质是：**用 $h$ 个独立的、维度更小的自注意力并行计算，让模型从多个"视角"同时观察序列，每个头专注于一种关系模式。**

它不是"多个头比一个头强"这么简单--真正的动机是**解耦**：一个头学语法依赖，另一个头学语义关联，再一个头学长距离指代。把这些视角硬塞进一个注意力里会互相干扰，拆开反而干净。

## L2 · 通俗类比

想象一篇文章有多个读者：

- **语法老师**看的时候，眼睛在主谓宾之间跳，关注句法结构。
- **内容编辑**看的时候，眼睛在论点和论据之间跳，关注论证关系。
- **校对员**看的时候，眼睛在前后文的用词一致性上跳，关注指代和呼应。

每个人关注的是不同的"关系"。如果你逼一个人同时做这三件事，他的注意力会顾此失彼。多头注意力就是**派 $h$ 个专家同时读，各读各的，最后汇总**。

关键细节：每个专家的"带宽"（维度）比单干时小。$h=8$ 时每个头只有 $d/8$ 的维度。总计算量和单头差不多，但视角多了 $h$ 个。这就是为什么它划算。

## L3 · 正经定义

**多头注意力（Multi-Head Attention, MHA）** 是 [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) 提出的自注意力扩展形式。它将单个 $d$ 维的注意力拆成 $h$ 个并行的 $d_k = d/h$ 维注意力（"头"），每个头独立做 Q/K/V 投影和缩放点积注意力，最后把所有头的输出拼接后做一次线性投影：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

其中投影矩阵 $W_i^Q \in \mathbb{R}^{d \times d_k}$、$W_i^K \in \mathbb{R}^{d \times d_k}$、$W_i^V \in \mathbb{R}^{d \times d_v}$（通常 $d_k = d_v = d/h$），输出投影 $W^O \in \mathbb{R}^{hd_v \times d}$。

原论文使用 $d=512$、$h=8$，因此每个头 $d_k = 64$。这个选择让多头注意力的总计算量与单头 $d_k=512$ 的注意力基本相当，但能学到更丰富的关系。

**参考资料**：
- [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 第 3.2.2 节定义了多头注意力
- [Voita et al., 2019 - Analyzing Multi-Head Self-Attention](https://arxiv.org/abs/1905.09418) - 实证分析不同头学到了什么
- [Michel et al., 2019 - Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) - 头数冗余性的经典研究

## L4 · 原理深挖

### 4.1 为什么要"多头"，而不是"一个大头"？

理论上一个 $d_k = d$ 的单头注意力已经能表示任意加权和，表达能力上不弱于多头。那为什么 Transformer 要拆？

**原因 1：优化的便利性。**

单个注意力头学到的权重分布倾向于"尖锐化"--softmax 容易收敛到只关注一两个 token，丢失其他关系。多头机制让模型不必在"关注 A 还是关注 B"之间二选一，而是头 1 关注 A、头 2 关注 B，各得其所。这本质上是给优化器更多的"独立通道"，降低相互干扰。

**原因 2：关系类型的多样性。**

实证研究（[Voita et al., 2019](https://arxiv.org/abs/1905.09418)）发现，训练后的 Transformer 里不同头确实学到了可解释的不同功能：

- **位置头**：关注固定相对距离的 token（如"前一个词""句首"）
- **句法头**：关注句法上的依赖（如动词关注其主语/宾语）
- **罕见词头**：关注句中的生僻词，把它们的信息扩散开

这些功能无法被单个注意力头同时高效表达--它们对 Q/K 投影的要求是冲突的。

**原因 3：计算开销几乎不增。**

单头 $d_k = d$ 的计算量是 $O(n^2 d)$。多头 $h$ 个 $d_k = d/h$ 的总计算量也是 $O(n^2 d)$（$h \times n^2 \cdot d/h = n^2 d$）。也就是说，**多头几乎是"免费的"--同样的 FLOPs，换来更丰富的关系建模**。这是 Transformer 设计中最划算的一笔交易之一。

### 4.2 实现细节：为什么用一个大矩阵投影，而不是循环 $h$ 次

朴素的实现会循环 $h$ 次调用 `SelfAttention`。但生产实现（包括 PyTorch 的 `nn.MultiheadAttention`）用一个 $d \times d$ 的大矩阵一次性投影，再 reshape 成 $[h, n, d/h]$：

```python
# 高效实现：一次大投影 + reshape，避免 h 次循环
Q = X @ W_Q            # [n, d]
Q = Q.view(n, h, d_k)  # [n, h, d_k]
Q = Q.transpose(0, 1)  # [h, n, d_k]  每个头是一行

# 对所有头并行做注意力
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)  # [h, n, n]
weights = softmax(scores, dim=-1)
out = weights @ V      # [h, n, d_k]

out = out.transpose(0, 1).reshape(n, d)  # 拼回 [n, d]
out = out @ W_O        # 输出投影
```

这个技巧让多头注意力的 GPU 利用率远高于循环实现，是实际训练速度的关键。

### 4.3 头数的选择：8 是怎么来的，能不能改

原论文选 $h=8$，$d_k=64$。这不是神圣数字，而是经验选择：

- **$d_k$ 太小**（如 $<8$）：每个头的表达能力不足，softmax 容易退化。
- **$h$ 太大**（如 $d=512$ 时 $h=32$，$d_k=16$）：头之间冗余，浪费参数。
- **$h$ 太小**（如 $h=1$）：退化为单头，失去关系解耦优势。

现代模型的趋势：
- **小模型**（$d \le 1024$）：$h=8$ 或 $h=12$，$d_k \approx 64\text{-}128$。
- **大模型**（$d \ge 4096$）：$h=32$ 或更多，但 $d_k$ 保持 128 左右，不再严格等于 $d/h$。LLaMA-2 70B 用 $h=64$，$d_k=128$。
- **GQA / MQA**（分组查询注意力、多查询注意力）：推理优化，让多个头共享 K/V，减少 KV cache 显存。LLaMA-2 70B 用 GQA-8。这是当前大模型的事实标准，详见 L4.5。

### 4.4 头的冗余性：很多头其实没用

[Michel et al., 2019](https://arxiv.org/abs/1905.10650) 的研究发现一个反直觉的事实：**Transformer 的头高度冗余**。

- 在 BERT-base（12 层 × 12 头 = 144 个头）里，去掉 40% 的头对效果几乎无影响。
- 去掉某些"重要头"则会显著掉点。
- 头的重要性随层变化：编码器高层和底层各有一些关键头，中间层头普遍不重要。

这个发现催生了两个方向：

1. **剪枝**：训练后去掉不重要的头，模型压缩（见 [剪枝](./pruning)）。
2. **架构简化**：既然头冗余，能不能从设计阶段就减少头数？-> GQA/MQA 应运而生。

### 4.5 从 MHA 到 MQA / GQA：推理时的 KV cache 优化

标准 MHA 的痛点在推理：自回归生成时，每生成一个 token 都要存前面所有 token 的 K 和 V（KV cache），显存占用 $O(n \cdot h \cdot d_k)$。$h$ 越大，KV cache 越大。

- **MQA（Multi-Query Attention, [Shazeer 2019](https://arxiv.org/abs/1911.02150)）**：所有头共享同一组 K/V，只有 Q 是多头的。KV cache 缩小 $h$ 倍，但效果下降明显。
- **GQA（Grouped-Query Attention, [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245)）**：折中方案，把 $h$ 个头分成 $g$ 组，组内共享 K/V。$g=1$ 退化为 MQA，$g=h$ 退化为 MHA。LLaMA-2 70B 用 GQA-8（8 组），效果接近 MHA，推理快得多。

当前主流大模型（LLaMA-2/3、Mistral、Qwen2 等）几乎都用 GQA。这是"多头注意力"在工程实践中的最新演化。

### 4.6 最小可运行 Demo

参见 [`demos/multi-head-attention/`](../demos/multi-head-attention/) -- 30 行 PyTorch 实现多头注意力，并对比单头，直观看到"多个头学到了不同的注意力模式"。

## L5 · 沿革与坑

### 沿革

- **2017 年**，Vaswani 等人在 Transformer 原论文中提出 MHA，$h=8$ 是经验选择，论文没有详细论证为什么是 8。
- **2019 年**，两篇关键论文改变认知：Voita 等人证明头有功能分化，Michel 等人证明头高度冗余。多头注意力从"必要组件"变成"可压缩的冗余设计"。
- **2019 年**，Shazeer 提出 MQA，首次为推理效率牺牲头的独立性。
- **2023 年**，Ainslie 等人提出 GQA，在 MHA 和 MQA 之间找到平衡点。LLaMA-2 70B 采用 GQA-8，标志着 GQA 成为工业标准。
- **2024 年起**，主流开源大模型普遍采用 GQA，标准 MHA 在新模型中已少见，但作为概念基础仍是理解一切变体的前提。

### 常见误解

- ❌ **误解**：头越多越好，模型能学到更多关系。
  ✅ **真相**：头数有冗余上限。超过某个点后，新增的头是冗余的，浪费参数和显存（见 4.4）。Michel 等人证明 BERT 可以去掉 40% 的头而不掉点。

- ❌ **误解**：每个头都学到了有意义、可解释的功能。
  ✅ **真相**：Voita 等人的研究表明，只有少数头（约 20-30%）有清晰可解释的功能（位置、句法等），其余多数头的作用模糊甚至冗余。可视化注意力头时要警惕过度解读。

- ❌ **误解**：多头注意力和卷积神经网络的多通道是一回事。
  ✅ **真相**：表面相似（都是多组并行特征），但机制不同。CNN 的通道是局部共享的权重，多头注意力是全局的、内容相关的加权。类比有助于建立直觉，但不要延伸太远。

- ❌ **误解**：GQA/MQA 是 MHA 的"简化版"，效果一定更差。
  ✅ **真相**：在精心调参下，GQA 的效果可以非常接近 MHA，而推理速度和显存占用显著优于 MHA。在长上下文场景下，GQA 的优势压倒性。这是工程权衡的胜利，不是"降级"。

### 面试怎么考

1. **"写一下多头注意力的公式，解释每个符号。"** --必考。要能默写 $\text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$ 并说清 $d_k = d/h$。
2. **"为什么用多头，而不是一个大的单头？"** --高频区分度题。答出"关系解耦 + 优化便利 + 计算量几乎不增"三点（见 4.1）。
3. **"多头注意力的计算量和单头比怎么样？"** --几乎相等（$O(n^2 d)$ 不变），这是它划算的原因。
4. **"头数怎么选？太多太少各有什么问题？"** --见 4.3。$d_k$ 太小表达力不足，$h$ 太大冗余浪费。
5. **"什么是 GQA？为什么大模型都用它？"** --推理时 KV cache 显存优化，MHA 和 MQA 的折中（见 4.5）。能答出这个说明跟得上最新实践。
6. **"Transformer 的头都能去掉吗？"** --考冗余性理解（见 4.4）。答出"40% 可剪枝，少数关键头不能动"即可。

## 延伸阅读

- 📄 [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 第 3.2.2 节
- 📄 [Voita et al., 2019 - Analyzing Multi-Head Self-Attention](https://arxiv.org/abs/1905.09418) - 头的功能分化
- 📄 [Michel et al., 2019 - Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) - 头的冗余性
- 📄 [Shazeer, 2019 - Fast Transformer Decoding with MQA](https://arxiv.org/abs/1911.02150) - MQA 起源
- 📄 [Ainslie et al., 2023 - GQA](https://arxiv.org/abs/2305.13245) - 当前主流方案
- 📝 [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/) - 图解多头部分

---

> *上一篇：[自注意力](./self-attention) -- 多头注意力的基础。*
> *下一篇：[位置编码](./positional-encoding) -- 自注意力没有顺序概念，所以需要它。*
