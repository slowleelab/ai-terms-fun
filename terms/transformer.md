---
title: Transformer
slug: transformer
category: 模型架构与训练
tags: [Transformer, 架构, 序列建模, NLP]
author: ai-terms-fun
created: 2026-07-19
updated: 2026-07-19
---

# Transformer

> **一句话 TL;DR**：一个完全由注意力机制和前馈网络堆叠而成的序列模型架构。它扔掉了 RNN 的循环结构，靠自注意力一次建立全局依赖，从此改写了 NLP 的基本盘，并在之后统一了几乎所有模态。

---

## L1 · 一句话点破

Transformer 是一个**纯由注意力机制 + 前馈网络 + 残差连接 + 层归一化堆叠而成的架构**，没有任何循环或卷积。它的核心主张是：**用自注意力替代 RNN 来建模序列依赖**，从而获得并行计算能力和任意距离的直接交互。

它是现代大模型（BERT、GPT、LLaMA、Claude、Gemini）共同的祖先。理解 Transformer，就理解了这一代 AI 的地基。

## L2 · 通俗类比

把 Transformer 想成一条**流水线上的多道工序**：

- 每道工序（一个 Transformer 层）都做两件事：先让所有零件互相"沟通"（自注意力，交换信息），再让每个零件自己"加工"（前馈网络，变换表示）。
- 工序之间用残差连接串联：每道工序的输出 = 输入 + 加工结果。信息能沿着这条"高速公路"直达深层，不会被层层加工磨没。
- 整条流水线跑完，最初的一串 token 就被层层加工成了携带全局语义的深度表示。

和 RNN 的区别：RNN 是**一个工人按顺序逐个处理零件**，处理到第 100 个时早已忘了第 1 个；Transformer 是**每道工序里所有零件同时沟通**，第 1 个和第 100 个可以直接对话。代价是每道工序的计算量大，但能全并行。

## L3 · 正经定义

**Transformer** 是 Vaswani 等人在 2017 年论文 [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) 中提出的序列到序列架构。它完全基于注意力机制，不包含循环（RNN）或卷积（CNN）结构。

原始 Transformer 采用**编码器-解码器**结构：

- **编码器**（Encoder）：$N$ 层（原论文 $N=6$），每层包含一个多头自注意力子层和一个前馈网络（FFN）子层，每个子层外面包残差连接 + 层归一化。编码器把输入序列编码成连续表示。
- **解码器**（Decoder）：同样 $N$ 层，每层包含三个子层：**掩码自注意力**（防止看到未来 token）、**交叉注意力**（关注编码器输出）、**前馈网络**。解码器自回归地生成输出序列。

一个完整的前馈子层：

$$
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$

即两层线性变换夹一个 ReLU（现代实现多用 GeLU/GELU 或 SwiGLU）。

后续的 BERT 只用编码器（理解型任务），GPT 只用解码器（生成型任务），形成了"仅编码器""仅解码器""编码器-解码器"三大变体，详见 [编码器-解码器](./encoder-decoder)。

**参考资料**：
- [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始论文
- [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/) - 最广为流传的图解
- [The Annotated Transformer - Harvard NLP](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - 逐行代码注解

## L4 · 原理深挖

### 4.1 一个 Transformer 层到底做了什么

以编码器层为例，给定输入 $X \in \mathbb{R}^{n \times d}$，一层做的事是：

$$
\begin{aligned}
A &= \text{LayerNorm}\big(X + \text{MultiHeadAttn}(X)\big) \\
Y &= \text{LayerNorm}\big(A + \text{FFN}(A)\big)
\end{aligned}
$$

注意三个细节：

1. **残差连接**：每个子层的输入直接加到输出上（$X + \text{...}$）。这让梯度能绕过变换层直达深处，是训练深层 Transformer 的前提。
2. **LayerNorm 在外**：原论文的写法是"先残差，再 LayerNorm"（Post-LN）。后来的实践（GPT 系列）发现"先 LayerNorm，再残差"（Pre-LN）训练更稳定，成为现代主流。
3. **多头注意力**：不是做一个大注意力，而是做 $h$ 个小注意力（每个 $d_k = d/h$）再拼接。详见 [多头注意力](./multi-head-attention)，这里只要知道它让模型能同时关注不同维度的关系。

### 4.2 解码器的两个特殊之处

**① 掩码自注意力（Masked Self-Attention）**

生成任务是自回归的：预测第 $t$ 个 token 时，只能看到前 $t-1$ 个。但自注意力天然是全局的，会把整个序列都看一遍。

解法：在注意力分数矩阵的上三角部分填 $-\infty$，softmax 后这些位置变成 0，相当于"看不见未来"：

$$
\text{Mask}[i, j] = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

$$
\text{scores} = \frac{Q K^\top}{\sqrt{d_k}} + \text{Mask}
$$

这是为什么 GPT 类模型只能"从左往右"看、不能双向的原因。

**② 交叉注意力（Cross-Attention）**

解码器的第二个注意力子层，Q 来自解码端，K/V 来自编码端。这让生成时能动态参考输入序列--典型场景是翻译：生成目标语言时关注源语言的哪些词。

纯解码器模型（GPT）没有这一层，因为它不编码外部输入，只做自回归生成。

### 4.3 位置编码：填补缺失的顺序信息

自注意力对输入顺序完全不变（置换等变），打乱 token 顺序，输出只是对应打乱。这意味着 Transformer 本身**不知道词的先后**。

解法是额外注入位置信息，两种主流方案：

- **绝对位置编码**（原论文、BERT、GPT-2）：给每个位置一个固定的向量 $p_i$，加到 token 的 embedding 上：$X_i = \text{emb}(w_i) + p_i$。原论文用正弦/余弦函数生成，BERT/GPT 用可学习的位置向量。
- **相对位置编码**（T5、LLaMA 的 RoPE）：不编码绝对位置，而是编码"两个 token 相距多远"，让注意力分数本身带位置信息。RoPE 是当前主流大模型的事实标准。

详见 [位置编码](./positional-encoding)。

### 4.4 计算流总览（编码器-解码器）

```
输入序列
   │
   ▼
[Embedding + 位置编码]
   │
   ▼
┌─────────────────────┐
│ 编码器 × N 层        │  每层: 多头自注意力 + FFN, 都带残差+LayerNorm
└─────────────────────┘
   │ 编码器输出（K, V 给交叉注意力用）
   ▼
┌─────────────────────┐
│ 解码器 × N 层        │  每层: 掩码自注意力 + 交叉注意力 + FFN
└─────────────────────┘
   │
   ▼
[线性层 + softmax]
   │
   ▼
输出序列（下一个 token 的概率分布）
```

### 4.5 为什么 Transformer 赢了

| 维度 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 并行性 | 必须按时间步串行 | 全序列可并行 |
| 长程依赖 | 信息要传 $n$ 步，衰减严重 | 任意两 token 直接交互，1 步 |
| 显存 | $O(n)$ | $O(n^2)$（注意力矩阵） |
| 扩展性 | 难堆深，梯度消失 | 残差连接让堆深容易 |
| 硬件友好 | 不适合 GPU 并行 | 全是矩阵乘法，GPU 完美适配 |

Transformer 用 $O(n^2)$ 的代价换来了并行性和长程依赖建模能力，在 GPU 时代这笔交易极其划算。这是它干掉 RNN 的根本原因--不是"效果更好"这么简单，而是**计算范式的胜利**。

### 4.6 最小可运行 Demo

参见 [`demos/self-attention/`](../demos/self-attention/) -- 自注意力的实现。一个完整的 Transformer 层只需在它外面套上 FFN + 残差 + LayerNorm，约 30 行。完整可训练的 Transformer 参见 [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)。

## L5 · 沿革与坑

### 沿革

- **2017 年 6 月**，Google 的 Vaswani 等人发表 *Attention Is All You Need*。最初目标很朴素：在机器翻译上刷过当时的 SOTA（基于 RNN+注意力）。论文标题嚣张，但当时没人意识到它会改写整个领域。
- **2017-2018 年**，关键转变发生：研究者发现把 Transformer 的编码器单独拿出来，预训练后做下游分类任务，效果惊人。
- **2018 年 10 月**，Google 发表 BERT（仅编码器），11 项 NLP 任务刷榜；OpenAI 发表 GPT（仅解码器）。Transformer 正式分裂为三大变体。
- **2020 年 GPT-3 之后**，纯解码器架构在 scaling law 加持下逐渐成为主流，编码器-解码器结构在通用大模型里式微（但翻译、T5 等场景仍在用）。
- **2023 年起**，Transformer 统一了几乎所有模态：ViT（视觉）、Whisper（语音）、Sora（视频）都建立在它之上。"Transformer 是通用架构"从口号变成事实。

### 常见误解

- ❌ **误解**：Transformer = 自注意力。
  ✅ **真相**：自注意力只是其中一个组件。完整的 Transformer 还包括多头机制、前馈网络、残差连接、LayerNorm、位置编码。把 Transformer 等同于自注意力，就像把汽车等同于发动机。

- ❌ **误解**：Transformer 没有位置信息，所以不擅长顺序任务。
  ✅ **真相**：自注意力本身确实置换等变，但加了位置编码后，Transformer 完全能建模顺序。RoPE 等现代方案在长序列外推上表现优秀。真正限制它的是 $O(n^2)$ 显存，不是顺序建模能力。

- ❌ **误解**：编码器-解码器是 Transformer 的"标准"形态，GPT 是"简化版"。
  ✅ **真相**：原始论文确实是编码器-解码器，但后续发展表明，纯解码器架构在生成任务上更具扩展性，GPT 系列不是"简化"，而是另一条成功路线。选择哪种架构取决于任务，没有高下之分。

- ❌ **误解**：层数越多越好。
  ✅ **真相**：深层 Transformer 需要配合残差、Pre-LN、学习率预热等技巧才能训稳。盲目堆深会导致训练崩溃。模型容量是宽度（$d$）和深度的平衡，不是单维追求。

### 面试怎么考

1. **"画一下 Transformer 的结构。"** --必考题。要能画出编码器-解码器、各子层、残差连接、位置编码，并解释数据流。
2. **"解码器为什么需要掩码？"** --考自回归生成的理解（见 4.2①）。
3. **"为什么需要残差连接？"** --深层网络梯度传播，没有它训不动深 Transformer。
4. **"Pre-LN 和 Post-LN 有什么区别？为什么现代模型用 Pre-LN？"** --训练稳定性，Post-LN 在深层时梯度方差大，需要 warmup；Pre-LN 更稳定。
5. **"Transformer 相比 RNN 的优劣？"** --并行性 + 长程依赖 + 显存 $O(n^2)$，三点都要说全（见 4.5）。
6. **"为什么 GPT 用纯解码器，BERT 用纯编码器？"** --任务决定的：生成任务需要自回归 + 掩码，理解任务需要双向上下文。

## 延伸阅读

- 📄 [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 必读原论文
- 📝 [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/) - 图解经典，入门首选
- 📝 [The Annotated Transformer - Harvard NLP](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - 原论文逐行代码
- 📄 [On Layer Normalization in the Transformer Architecture (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745) - Pre-LN vs Post-LN 的深度分析
- 📝 [The Transformer Family Version 2.0 - Lilian Weng](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) - Transformer 变体全景

---

> *上一篇：[自注意力](./self-attention) -- Transformer 的核心机制。*
> *下一篇：[多头注意力](./multi-head-attention) -- 一个头不够，那就来八个。*
