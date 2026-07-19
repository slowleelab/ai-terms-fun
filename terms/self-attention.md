---
title: 自注意力（Self-Attention）
slug: self-attention
category: 模型架构与训练
tags: [Transformer, 注意力机制, 序列建模, NLP]
author: ai-terms-fun
created: 2026-07-19
updated: 2026-07-19
---

# 自注意力（Self-Attention）

> **一句话 TL;DR**：序列里每个位置都去"看"其他所有位置，按相关度加权求和，得到自己的新表示。这套操作让模型一次就能建立任意两个 token 之间的依赖，不再像 RNN 那样逐字往后传。

---

## L1 · 一句话点破

自注意力的本质是：**对一个序列中的每个位置 $i$，用它与所有位置 $j$ 的相似度作为权重，把所有位置的向量加权求和，作为位置 $i$ 的新表示。**

"注意力"这个名字是误导性的--它既不模拟人的注意力，也不是"选择性地关注"。它就是一次**全连接的加权聚合**，权重由内容相似度决定。叫它"自相关聚合"可能更准确，但"注意力"已经约定俗成。

## L2 · 通俗类比

想象你在读一句英文："*The animal didn't cross the street because **it** was too tired.*"

读到 **it** 的时候，你的大脑会自动把 **it** 和前面的 *animal* 关联起来，而不是和 *street*。人脑怎么做到的？靠语义和上下文。

自注意力就是让模型学会做这件事：对 **it** 这个位置，计算它和句子里每个词的"相关度分数"，分数高的词（如 *animal*、*tired*）贡献大，分数低的词（如 *the*、*street*）贡献小，然后把所有词的向量按这个分数加权求和，得到 **it** 的新表示。这个新表示里，已经"混入"了 *animal* 的信息。

**关键点**：这个"相关度"不是人工设定的规则，是模型从数据里学出来的。模型通过大量阅读，自己学会了"看到代词就去找它指代的名词"这类模式。

## L3 · 正经定义

**自注意力（Self-Attention）** 是一种将输入序列映射到输出序列的变换，输出序列中每个位置是输入序列所有位置的加权和，权重由位置间的相似度决定。它最早在 [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中作为 Transformer 的核心机制提出。

形式化地，给定输入序列的表示 $X \in \mathbb{R}^{n \times d}$（$n$ 为序列长度，$d$ 为模型维度），自注意力通过三个可学习的投影矩阵将其映射为查询（Query）、键（Key）、值（Value）：

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

其中 $W_Q, W_K \in \mathbb{R}^{d \times d_k}$，$W_V \in \mathbb{R}^{d \times d_v}$。注意力输出为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

这里 $Q K^\top \in \mathbb{R}^{n \times n}$ 是位置间的相似度矩阵（点积），$\text{softmax}$ 沿最后一维归一化得到权重，$\sqrt{d_k}$ 是缩放因子防止点积过大导致 softmax 饱和。

"Self"指 Q、K、V 都来自同一个输入序列。若 Q 来自另一个序列（如解码端），则称为交叉注意力（cross-attention）。

**参考资料**：
- [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [Bahdanau et al., 2014 - Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - 注意力机制的起源（用于 seq2seq）
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - 原论文的逐行代码注解

## L4 · 原理深挖

### 4.1 为什么需要 Q、K、V 三个角色，而不是直接用 X 算相似度？

最朴素的注意力是 $\text{softmax}(X X^\top) X$，即"输入和自己算点积再加权"。为什么 Transformer 要多此一举引入三个投影？

因为一个向量同时承担"我是谁"（被查询的身份）、"我匹配什么"（作为键被比较）、"我提供什么"（作为值被聚合）三种角色会互相打架。

举个具体场景：一个词既要表达自己的语义（供别人查询匹配），又要表达自己能贡献的信息（值）。如果用同一个向量，"匹配信号"和"内容信号"就被耦合了，模型无法独立优化两者。引入 $W_Q, W_K, W_V$ 三个投影，就是把"查询视角""被匹配视角""内容视角"解耦，让模型分别学习。

代价是参数量增加（3 个 $d \times d$ 矩阵），但带来的灵活性远超成本。这是 Transformer 设计里最关键的一步。

### 4.2 为什么除以 $\sqrt{d_k}$

点积 $Q K^\top$ 的方差随 $d_k$ 线性增长：若 $q, k$ 的分量是均值 0、方差 1 的独立随机变量，$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$。

当 $d_k$ 较大（如 64）时，点积的数值会比较大，导致 softmax 的输入落入饱和区，梯度接近 0，训练停滞。除以 $\sqrt{d_k}$ 把方差拉回 1 量级，让 softmax 工作在敏感区间。

这是一个工程细节，但很重要--不除以 $\sqrt{d_k}$，大模型根本训不动。

### 4.3 计算复杂度：自注意力的阿喀琉斯之踵

自注意力的时间复杂度是 $O(n^2 d)$，空间复杂度是 $O(n^2)$，都源于那个 $n \times n$ 的注意力矩阵。

| 序列长度 $n$ | 注意力矩阵大小 | 显存（FP16） |
|--------------|----------------|--------------|
| 512          | 512 × 512      | 0.5 MB       |
| 4,096        | 4K × 4K        | 32 MB        |
| 32,768       | 32K × 32K      | 2 GB         |
| 131,072      | 128K × 128K    | 32 GB        |

这就是为什么长上下文模型（128K+ token）极其吃显存。整个领域的工程努力--FlashAttention、稀疏注意力、线性注意力、滑动窗口--本质上都在绕开这个 $n^2$。

### 4.4 一次完整的计算流（伪代码）

```python
def self_attention(X, W_Q, W_K, W_V):
    # X: [n, d]  输入序列
    Q = X @ W_Q   # [n, d_k]  查询
    K = X @ W_K   # [n, d_k]  键
    V = X @ W_V   # [n, d_v]  值

    # 计算相似度并缩放
    scores = Q @ K.T / sqrt(d_k)   # [n, n]

    # 归一化得到权重（每行求和为 1）
    weights = softmax(scores, dim=-1)   # [n, n]

    # 加权求和
    output = weights @ V   # [n, d_v]
    return output
```

整个过程没有任何循环，全是矩阵乘法--这就是它能被 GPU 高效并行的原因，也是它干掉 RNN 的根本理由。

### 4.5 最小可运行 Demo

参见 [`demos/self-attention/`](../demos/self-attention/) -- 20 行 PyTorch 实现一个能跑的自注意力，并打印出注意力矩阵，直观看到"代词 it 找到 animal"这件事真的发生了。

## L5 · 沿革与坑

### 沿革

- **2014 年**，Bahdanau 等人在机器翻译中提出注意力机制，用于解决 seq2seq 的瓶颈：编码器把整句压成一个固定向量，长句信息丢失严重。注意力让解码器在每一步"回看"编码器的所有位置。
- **2015 年**，Luong 等人提出乘性注意力（点积形式），比 Bahdanau 的加性形式更简洁高效。
- **2017 年**，Vaswani 等人（Google）发表 *Attention Is All You Need*，关键创新是**去掉 RNN，让注意力自己干所有事**，Q/K/V 全部来自同一序列（self-attention）。论文标题嚣张但准确--确实只需要注意力。
- **2018 年起**，BERT、GPT 系列证明 self-attention 是通用基础设施，Transformer 彻底统一了 NLP 架构，并随后席卷 CV、语音、多模态。

### 常见误解

- ❌ **误解**：自注意力模拟了人类大脑的注意力机制。
  ✅ **真相**：完全无关。人的注意力是认知科学概念，自注意力是加权求和的矩阵运算。名字相同纯属类比启发，把它当神经科学来理解会误入歧途。

- ❌ **误解**：注意力权重高 = 模型在"关注"那个词。
  ✅ **真相**：权重高只代表"那次点积大"，可能因为语义相关，也可能因为某些 token 的 $K$ 向量模长大、或训练中出现了捷径。权重可视化能提供线索，但不能直接当作"模型推理过程"。此外，多头注意力下不同头学到的东西不同，单个头的权重不代表整体行为。

- ❌ **误解**：自注意力是"局部"操作，只关注邻近词。
  ✅ **真相**：自注意力是全局的，任意两个 token 都能直接交互，不管隔多远。这是它相对 RNN 的核心优势--RNN 要传 $n$ 步才能让首尾交互，自注意力 1 步到位。

- ❌ **误解**：自注意力没有顺序概念，所以需要位置编码。
  ✅ **真相**：这条不是误解，是事实。自注意力的计算对输入顺序完全不变（置换等变），打乱 token 顺序结果只是对应打乱--必须额外注入位置信息，这就是 [位置编码](./positional-encoding) 存在的理由。

### 面试怎么考

1. **"写一下 self-attention 的公式，解释每个符号。"** --送分题，必须能默写 $\text{softmax}(QK^\top/\sqrt{d_k})V$ 并说清 Q/K/V 的来源。
2. **"为什么除以 $\sqrt{d_k}$？"** --考工程理解（见 4.2）。答出"方差随 $d_k$ 增长，防 softmax 饱和"即可。
3. **"自注意力的时间/空间复杂度是多少？为什么长上下文难做？"** --$O(n^2)$，注意力矩阵（见 4.3）。能顺势提到 FlashAttention / 稀疏注意力的，加分。
4. **"为什么需要 Q、K、V 三个矩阵，而不是直接用 X？"** --区分度题（见 4.1）。答出"解耦查询/匹配/内容三种视角"的人，说明真理解了。
5. **"自注意力相比 RNN 的优势是什么？"** --并行计算 + 全局依赖建模。两点都要说到。

## 延伸阅读

- 📄 [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 必读原始论文
- 📄 [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) - 注意力机制起源
- 📝 [The Illustrated Transformer - Jay Alammar](https://jalammar.github.io/illustrated-transformer/) - 最经典的图解
- 📝 [The Annotated Transformer - Harvard NLP](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - 原论文逐行代码注解
- 📄 [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) - 解决 $O(n^2)$ 显存问题的里程碑

---

> *下一篇：[多头注意力](./multi-head-attention) -- 既然一个注意力头能干，为什么要搞 8 个？*
