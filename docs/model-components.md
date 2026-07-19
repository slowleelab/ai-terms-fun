---
title: 模型组件：参数 / 层 / 激活函数
slug: model-components
category: 模型架构与训练
tags: [参数, 层, 激活函数, ReLU, GeLU, 基础]
author: ai-terms-fun
created: 2026-07-19
updated: 2026-07-19
---

# 模型组件：参数 / 层 / 激活函数

> **一句话 TL;DR**：神经网络的最小积木。**参数**是模型学到的知识，**层**是参数的组织单元，**激活函数**是让网络能学非线性关系的开关。三者组合起来，构成从 [CNN](./cnn-rnn-lstm) 到 [Transformer](./transformer) 的所有架构。

---

## L1 · 一句话点破

- **参数（Parameters）**：模型在训练中通过反向梯度更新学到的一组数字（权重矩阵和偏置），是模型"知识"的载体。"175B 参数"就是指这组数字有 1750 亿个。
- **层（Layers）**：把参数按功能分组并顺序串联的结构单元。数据从输入层流过若干隐藏层到输出层，每层做一次变换。
- **激活函数（Activation Function）**：夹在层的线性变换之间的非线性函数。没有它，多层网络等价于单层线性回归；有了它，网络才能学复杂的非线性关系。

这三者关系：**层是容器，参数是容器里的内容，激活函数是层之间的非线性开关。**

## L2 · 通俗类比

把神经网络想象成一条流水线上的多个加工站：

- **层 = 加工站**：原料（输入）从第一站流到最后一站，每站对原料做一次处理。
- **参数 = 每站工人的手艺**：每个工人怎么操作（切多厚、加多少料）由一组数字决定。这些数字是工人长期训练（模型训练）练出来的，不是出厂设定。
- **激活函数 = 工人的判断开关**：原料加工后，工人要判断"这个结果值不值得往下传"。线性变换只是机械地加工，激活函数决定"超过某个阈值才放行，否则丢弃"。

如果没有激活函数，所有加工站串起来等价于一个"超级加工站"（线性组合可合并），流水线再长也没用。激活函数让每个站都有"判断"，整个流水线才能学到复杂的、非线性的加工逻辑。

ReLU 这种激活函数就是最朴素的判断："负数直接扔掉，正数原样放行"。

## L3 · 正经定义

**参数（Parameters）**：神经网络中通过训练学习的可调变量，包括权重矩阵 $W$ 和偏置向量 $b$。一个线性层 $y = Wx + b$ 中，$W \in \mathbb{R}^{m \times n}$ 和 $b \in \mathbb{R}^m$ 是参数，共 $m \cdot n + m$ 个。模型总参数量决定其容量（表达能力上限）。

**层（Layers）**：神经网络的基本结构单元，对输入做一次可微变换。常见类型：
- **全连接层（Linear/Dense）**：$y = Wx + b$
- **卷积层**：局部加权求和
- **注意力层**：见 [自注意力](./self-attention)
- **归一化层**：LayerNorm、RMSNorm
- **激活层**：施加非线性

深度 = 层数。深网络能学到层次化特征（浅层学局部、深层学抽象），但训练更难（梯度消失）。

**激活函数（Activation Function）**：施加在层输出上的非线性函数 $f$，使网络能逼近任意非线性映射（万能逼近定理）。常见激活函数：

| 名称 | 公式 | 特点 |
|------|------|------|
| Sigmoid | $\sigma(x) = 1/(1+e^{-x})$ | 输出 (0,1)，易梯度饱和 |
| Tanh | $\tanh(x)$ | 输出 (-1,1)，仍会饱和 |
| ReLU | $\max(0, x)$ | 简单、不饱和、有死区 |
| GeLU | $x \cdot \Phi(x)$ | 平滑，BERT/GPT 默认 |
| SwiGLU | $\text{Swish}(xW_1) \otimes xW_2$ | 带门控，LLaMA 起 |

**参考资料**：
- [Glorot et al., 2011 - Deep Sparse Rectifier Networks](https://proceedings.mlr.press/v15/glorot11a.html) - ReLU 的理论基础
- [Hendrycks & Gimpel, 2016 - Gaussian Error Linear Units (GeLU)](https://arxiv.org/abs/1606.08415)
- [Shazeer, 2020 - GLU Variants](https://arxiv.org/abs/2002.05202) - SwiGLU

## L4 · 原理深挖

### 4.1 为什么没有激活函数，多层等于单层

这是理解激活函数价值的关键。考虑一个两层无激活函数的网络：

$$
y = W_2 (W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2) = W' x + b'
$$

其中 $W' = W_2 W_1$、$b' = W_2 b_1 + b_2$。两层线性变换的组合仍是线性变换，等价于一个单层。无论堆多少层，整体仍是一个线性映射。

**线性映射的表达能力极其有限**：它只能学直线（或高维超平面）划分，无法学曲线边界。但现实世界的数据（图像、语言）几乎都是非线性可分的。

激活函数的作用就是**打破线性性**：在每层线性变换后施加非线性 $f$：

$$
y = W_2 f(W_1 x + b_1) + b_2
$$

此时 $W_2 f(W_1 x)$ 不能合并成单一线性变换，网络的层数才有意义。[万能逼近定理](https://en.wikipedia.org/wiki/Universal_approximation_theorem)证明，带非线性的单隐层网络足够宽时能逼近任意连续函数。深度则让这个逼近更高效（用指数少的参数达到同样效果）。

### 4.2 ReLU 为什么成为默认选择：简单到极致的优势

ReLU（Rectified Linear Unit）的定义极其简单：

$$
\text{ReLU}(x) = \max(0, x)
$$

它在 2011 年前后取代 sigmoid/tanh 成为深度学习默认激活函数，原因有三个：

**① 计算极简**

ReLU 只需一次比较和置零，比 sigmoid 的指数运算快得多。在大规模训练里，这点速度差异被放大成显著优势。

**② 缓解梯度饱和**

sigmoid/tanh 在输入绝对值较大时梯度接近 0（饱和区），反向传播时梯度消失，深层网络训不动。ReLU 在正区间梯度恒为 1，不会饱和，让深层网络可训练。这是 ReLU 推动深度学习爆发的核心原因。

**③ 产生稀疏激活**

ReLU 把所有负输入置零，让一部分神经元"沉默"。这种稀疏激活类似大脑神经元的特性，有正则化效果，且让表示更具解释性。

ReLU 的缺点是**死区问题**（dying ReLU）：如果一个神经元的输入恒为负，它的梯度永远为 0，参数永远不更新，神经元"死掉"。Leaky ReLU、PReLU 等变体用"负区间给小斜率"缓解这个问题。

### 4.3 GeLU：为什么 Transformer 用它而不是 ReLU

GeLU（Gaussian Error Linear Unit）的定义：

$$
\text{GeLU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。直观上，GeLU 以输入的概率大小加权输入：$x$ 越大越可能被保留，越小越可能被置零。

**ReLU vs GeLU 的区别**：

- ReLU 在 $x=0$ 处硬切换（不可导），GeLU 在 0 附近平滑过渡。
- ReLU 对所有 $x > 0$ 原样保留，GeLU 对小的正数也有轻微抑制（$x \cdot \Phi(x) < x$ 当 $x$ 较小时）。
- GeLU 在 NLP 任务上经验性略好于 ReLU，BERT、GPT 系列默认用它。

近似公式（常用于工程实现）：

$$
\text{GeLU}(x) \approx 0.5x \left(1 + \tanh\!\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)
$$

为什么 Transformer 偏偏选 GeLU 而不是 ReLU？没有强理论解释，更多是经验选择--在 Transformer 的训练动态下，GeLU 的平滑性似乎让收敛更稳。原论文没详细论证，但后续实践证明它效果稳定，成为默认。

### 4.4 SwiGLU：现代大模型的新标配

SwiGLU（[Shazeer 2020](https://arxiv.org/abs/2002.05202)）是 LLaMA 起的现代大模型标配，它的定义带门控：

$$
\text{SwiGLU}(x) = \text{Swish}(x W_1) \otimes (x W_2)
$$

其中 $\text{Swish}(x) = x \cdot \sigma(x)$ 是 Swish 激活（也叫 SiLU），$\otimes$ 是逐元素乘。

理解它的结构：把 FFN 原本的 $f(xW_1) W_2$ 改造成"门控"形式--一路做 Swish 激活，另一路做线性投影，两路逐元素相乘。一路决定"激活模式"，另一路决定"内容"，相乘实现"内容受门控调节"。

为什么比 GeLU 好？经验上 SwiGLU 在大模型上效果略好，但代价是 FFN 参数增加 50%（多一个 $W_2$ 投影）。LLaMA、PaLM、Qwen 等都采用它，已成为事实标准。

注意 SwiGLU 的命名：**Swi**sh + **G**ated **L**inear **U**nit。理解命名就理解了它的结构来源。

### 4.5 参数量怎么算：从层到模型大小

以 LLaMA-2 7B 为例，估算参数量：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Embedding 层 | vocab_size × d = 32000 × 4096 ≈ 131M | 1.9% |
| 注意力层（每层 Q/K/V/O） | 4 × d² = 4 × 4096² ≈ 67M | - |
| FFN（每层，SwiGLU 3 个投影） | 3 × d × hidden = 3 × 4096 × 11008 ≈ 135M | - |
| 每层小计 | ≈ 202M | - |
| 32 层合计 | 32 × 202M ≈ 6.5B | 93% |
| 总计 | ≈ 7B | 100% |

可见 Transformer 的参数绝大部分在**注意力层和 FFN**里，Embedding 占比很小（但推理时是访存瓶颈）。理解参数分布，对量化、剪枝、推理优化都有用。

### 4.6 最小可运行 Demo

参见 [`demos/self-attention/`](../demos/self-attention/) 里的 `SelfAttention` 类--它的 `nn.Linear` 就是带参数的全连接层，没有显式激活（注意力的 softmax 是归一化，不算激活）。一个完整 Transformer 层只需在它外面套上 `nn.Linear` + GeLU 作为 FFN。

## L5 · 沿革与坑

### 沿革

- **1950s-1980s**：感知机时代。sigmoid 作为可导激活函数被广泛使用，把生物神经元的"放电"建模成平滑概率。
- **1986 年**：反向传播算法普及。sigmoid/tanh 因可导成为默认，但深层网络训练困难（梯度饱和）。
- **2011 年**：[Glorot et al.](https://proceedings.mlr.press/v15/glorot11a.html) 系统论证 ReLU 在深度网络中的优势，ReLU 迅速取代 sigmoid 成为默认。这一年被视为深度学习爆发的关键技术节点之一。
- **2016 年**：Hendrycks & Gimpel 提出 GeLU，在 BERT、GPT 系列中成为默认。
- **2017 年**：Transformer 发表，FFN 用 ReLU（原论文）。后续 BERT 改用 GeLU。
- **2020 年**：Shazeer 提出 SwiGLU，LLaMA 起成为大模型标配。
- **2023 年后**：SwiGLU + RMSNorm + RoPE + GQA 成为现代大模型的"标准四件套"。

### 常见误解

- ❌ **误解**：参数越多模型越聪明。
  ✅ **真相**：参数量决定容量上限，但能否用好取决于训练数据质量和训练方法。Chinchilla 法则表明，参数和数据要匹配增长，只堆参数不堆数据会饱和。LLaMA-7B 参数远少于 GPT-3 175B，但用更多更好数据训练，效果不输。

- ❌ **误解**：激活函数只是个小细节，不影响大局。
  ✅ **真相**：激活函数的选择影响训练稳定性和最终效果。ReLU 推动 deep learning 爆发、GeLU 成为 Transformer 默认、SwiGLU 成为 LLaMA 标配，这些都是激活函数改变格局的实例。它不是细节，是架构选择的一部分。

- ❌ **误解**：ReLU 在负区间梯度为 0 是 bug。
  ✅ **真相**：这是设计特性，带来稀疏激活。但确实有"死区"问题（神经元永久沉默），Leaky ReLU 等变体缓解了它。现代大模型已很少用纯 ReLU，转向 GeLU/SwiGLU。

- ❌ **误解**：层的深度越深越好。
  ✅ **真相**：深网络表达能力更强，但训练更难（梯度消失、过拟合）。残差连接、LayerNorm 等技巧让堆深成为可能，但不是"越深越好"，而是"在算力预算下找深度和宽度的平衡"。

- ❌ **误解**：GeLU 是 ReLU 的"升级版"。
  ✅ **真相**：GeLU 和 ReLU 是不同的设计哲学。ReLU 是硬开关（简单、稀疏），GeLU 是概率加权（平滑、连续）。两者各有优势，GeLU 在 NLP 上经验性更好，但 CV 上 ReLU 仍广泛使用。不是简单的升级关系。

### 面试怎么考

1. **"为什么神经网络需要激活函数？没有会怎样？"** --必考。没有激活函数，多层等价于单层线性变换（见 4.1 的数学推导）。
2. **"ReLU 相比 sigmoid 有什么优势？"** --计算简单、不饱和（梯度恒为 1）、稀疏激活（见 4.2）。
3. **"ReLU 的死区问题是什么？怎么解决？"** --负输入梯度为 0，神经元永久沉默。Leaky ReLU 给负区间小斜率。
4. **"GeLU 和 ReLU 的区别？为什么 Transformer 用 GeLU？"** --平滑过渡 vs 硬切换，经验上 NLP 任务更好（见 4.3）。
5. **"什么是 SwiGLU？为什么 LLaMA 用它？"** --Swish + 门控线性单元，带门控的 FFN，效果略好但参数增加 50%（见 4.4）。
6. **"给定一个 Transformer 层，估算它的参数量。"** --注意力 4d² + FFN（看激活函数，GeLU 是 2dh，SwiGLU 是 3dh），见 4.5。

## 延伸阅读

- 📄 [Glorot et al., 2011 - ReLU 理论基础](https://proceedings.mlr.press/v15/glorot11a.html)
- 📄 [Hendrycks & Gimpel, 2016 - GeLU](https://arxiv.org/abs/1606.08415)
- 📄 [Shazeer, 2020 - GLU Variants (SwiGLU)](https://arxiv.org/abs/2002.05202)
- 📝 [Universal Approximation Theorem - Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem)
- 📝 [Activation Functions in Neural Networks - Vishal](https://www.v7labs.com/blog/neural-networks-activation-functions)

---

> *上一篇：[传统模型：CNN / RNN / LSTM](./cnn-rnn-lstm) -- 架构演化史。*
> *下一篇：[预训练](./pre-training) -- 进入训练范式分类。*
