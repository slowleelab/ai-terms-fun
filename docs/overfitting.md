---
title: 过拟合 & 正则化（Overfitting & Regularization）
slug: overfitting
category: 模型架构与训练
tags: [过拟合, 正则化, Dropout, 权重衰减, 偏差方差, 早停]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 过拟合 & 正则化（Overfitting & Regularization）

> **一句话 TL;DR**：过拟合是模型"把训练数据背下来了"而不是"学到规律"--训练 loss 低、验证 loss 高。正则化是一切抑制过拟合手段的统称：权重衰减、Dropout、早停、数据增强、BatchNorm 等。大模型时代有个反直觉现象：模型越大、数据越多，反而越不容易过拟合（double descent），挑战了经典偏差-方差权衡。

---

## L1 · 一句话点破

**过拟合（Overfitting）**：模型在训练集上表现极好，但在验证/测试集上表现差。本质是模型容量过大，把训练数据里的噪声和细节都"记住"了，没学到真正的规律。

**正则化（Regularization）**：所有抑制过拟合、提升泛化能力的手段。核心思想：**给模型加点约束或噪声，逼它学"通用规律"而非"记答案"。**

判断过拟合的金标准：**训练 loss 持续下降，验证 loss 开始上升**。这时模型不是在学规律，而是在背题。

## L2 · 通俗类比

学生备考有两种极端：

- **欠拟合（Underfitting）**：考前只看了 3 页书，连基础题都不会。模型容量不够，没学到规律。表现为训练 loss 和验证 loss 都高。
- **过拟合（Overfitting）**：把历年真题每道题的答案都背下来了，但题型一变就懵。模型容量过大，把题目和答案死记硬背，没学到解题方法。表现为训练 loss 极低、验证 loss 高。
- **理想**：学了核心解题方法，遇到新题也能做。模型学到了规律，泛化好。

正则化像"防背题机制"：

- **权重衰减（Weight Decay）**：限制笔记厚度。不让你抄太多细节，逼你只记关键点。
- **Dropout**：考试时随机抽走你 50% 的笔记页。逼你不依赖任何单一页，每页内容都足够独立支撑解题。
- **早停（Early Stopping）**：背到一定程度就强制停止，再多背就开始死记硬背了。
- **数据增强（Data Augmentation）**：把题目做各种变形再让你练，逼你学方法而非记题面。

这些手段的共同目标：**让模型学到"题目的规律"而非"题目的答案"。**

## L3 · 正经定义

**过拟合**：模型 $f_\theta$ 在训练集 $D_{train}$ 上损失极低，但在独立同分布的测试集 $D_{test}$ 上损失显著更高。形式化：

$$
\mathcal{L}_{train}(\theta) \ll \mathcal{L}_{test}(\theta)
$$

**偏差-方差分解**（经典理论）：

$$
\mathbb{E}[(y - \hat{y})^2] = \underbrace{\text{Bias}^2}_{\text{欠拟合}} + \underbrace{\text{Variance}}_{\text{过拟合}} + \underbrace{\sigma^2}_{\text{噪声}}
$$

- **Bias（偏差）**：模型假设太简单导致的系统性误差（欠拟合）
- **Variance（方差）**：模型对训练数据敏感导致的波动（过拟合）
- 经典权衡：模型越复杂，bias 越低，variance 越高。最优复杂度在中间。

**正则化**主要手段：

| 方法 | 机制 | 典型用途 |
|------|------|---------|
| L2 / 权重衰减 | 限制参数范数 | 通用，[AdamW](./optimizer) 内置 |
| L1 | 产生稀疏参数 | 特征选择 |
| Dropout | 随机失活神经元 | Transformer、CNN |
| Early Stopping | 验证 loss 上升即停 | 通用 |
| Data Augmentation | 扩充训练数据 | CV、NLP |
| BatchNorm / LayerNorm | 平滑损失曲面 | 深层网络 |
| Label Smoothing | 软化标签 | 分类（见 [损失函数](./loss-function)） |
| Weight Tying | 共享参数 | [Embedding](./embedding)、Transformer |

**参考资料**：
- [Goodfellow et al. - Deep Learning 第 5、7 章](https://www.deeplearningbook.org/) - 偏差方差、正则化
- [Srivastava et al., 2014 - Dropout](https://arxiv.org/abs/1207.0580)
- [Belkin et al., 2019 - Reconciling modern ML and bias-variance tradeoff (Double Descent)](https://arxiv.org/abs/1812.11118)
- [Zhang et al., 2017 - Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530)

## L4 · 原理深挖

### 4.1 过拟合的本质：记忆 vs 泛化

过拟合的根源是**模型容量大于数据信息量**。当模型参数远多于"真实规律所需的参数"，多余容量用来记训练数据的噪声和细节，这些记忆在新数据上没用甚至有害。

[Zhang et al., 2017](https://arxiv.org/abs/1611.03530) 做了一个震撼实验：把 ImageNet 训练集的标签完全随机打乱，现代 CNN 仍能把训练 loss 降到 0--它把随机标签也"记住"了。这说明：

- 现代神经网络容量极大，足以记住任意标签
- 训练 loss 低 ≠ 学到规律
- 泛化能力（在测试集表现好）不是"训练好"的自然结果，需要正则化

**为什么模型不总是过拟合？** 经典理论说"模型大就该过拟合"，但实际 SGD、Adam 等优化器自带"隐式正则"：它们倾向找到"平坦极小值"而非"尖锐极小值"，平坦极小值泛化更好。这是深度学习最神秘的现象之一。

### 4.2 权重衰减：L2 正则的概率解释

L2 正则在损失上加 $\frac{\lambda}{2}\|\theta\|^2$：

$$
\mathcal{J}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2
$$

梯度：$\nabla \mathcal{J} = \nabla \mathcal{L} + \lambda \theta$。每步把参数向 0 拉一点（衰减）。

**概率解释**：L2 正则等效于参数的高斯先验。最大后验估计（MAP）：

$$
\max_\theta P(\theta|D) \propto P(D|\theta) \cdot P(\theta)
$$

若 $P(\theta) = \mathcal{N}(0, 1/\lambda)$，则 $\log P(\theta) = -\frac{\lambda}{2}\|\theta\|^2 + \text{const}$，对数后验 = 对数似然 - L2 正则。所以**L2 正则 = 假设参数服从 0 均值高斯先验，限制参数远离 0**。

L1 正则类似，对应参数的拉普拉斯先验，产生稀疏（很多参数变 0），适合特征选择。

[AdamW](./optimizer) 的关键修复就是把 L2 正则从梯度里解耦，避免被自适应学习率稀释。

### 4.3 Dropout：训练时集成、推理时平均

Dropout（[Srivastava et al., 2014](https://arxiv.org/abs/1207.0580)）：训练时随机把一部分神经元置 0（按概率 $p$），推理时用全部神经元（输出乘 $1-p$ 或训练时缩放）。

直觉：

- **防止共适应**：神经元之间不能太依赖彼此。任何一部分神经元被"删掉"，剩下的也得能工作，逼出鲁棒特征。
- **隐式集成**：每次 dropout 等效训练一个子网络，推理时是所有子网络的平均。$n$ 个神经元的网络有 $2^n$ 个子网络，Dropout 是它们的近似集成。

数学上，Dropout 接近于在每个层上做贝叶斯近似（[Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142)），可以用于不确定性估计。

Transformer 中 dropout 用在注意力权重、FFN 输出、残差连接等位置，是稳定训练的关键。

### 4.4 早停：简单但有效

早停：监控验证 loss，当它连续 $N$ 个 epoch 不下降（甚至上升）时停止训练。

为什么有效：训练初期模型先学"通用规律"（验证 loss 下降），后期才开始"记训练数据"（验证 loss 上升）。早停在分界点。

早停的优势：简单、免费、对几乎所有任务有效。劣势：需要验证集，且需要谨慎设置 patience（容忍多少 epoch 不下降）。

实务中早停常和正则化叠加使用，是"保险丝"。

### 4.5 数据增强：最便宜的正则

数据增强（Data Augmentation）通过对训练数据做变换扩充数据量：

- **CV**：翻转、裁剪、旋转、颜色抖动、CutMix、MixUp
- **NLP**：回译、同义词替换、EDA、Token dropout
- **语音**：加噪、变速、SpecAugment

为什么算正则？数据增强等效于告诉模型"$x$ 和它的变形 $T(x)$ 应该有相同标签"，这是个强先验。它强迫模型学到"对变换不变的特征"（如识别猫不管它朝哪边）。

数据增强是性价比最高的正则：不增加模型复杂度，不增加推理成本，效果显著。ResNet、ViT 等都重度依赖数据增强。

**注意**：数据增强必须保持标签语义。对"区分左右"的任务做水平翻转就破坏标签。增强策略要匹配任务。

### 4.6 Double Descent：颠覆经典权衡

经典偏差-方差权衡说：模型复杂度有最优值，过了就过拟合。但 [Belkin et al., 2019](https://arxiv.org/abs/1812.11118) 发现现代深度学习存在**双重下降（Double Descent）**：

模型复杂度从低到高时：

1. **欠拟合区**：复杂度低，bias 高，测试误差高
2. **过拟合峰值（插值阈值）**：复杂度刚好能拟合训练集，测试误差最高
3. **现代过参数化区**：复杂度继续增加，测试误差反而下降

这颠覆了"模型越大越容易过拟合"的直觉。在过参数化区，模型有多个能拟合训练集的解，优化器（SGD 等）倾向选泛化好的那个。

大模型时代的现象（[Nakkiran et al., 2021 - Deep Double Descent](https://arxiv.org/abs/1912.02292)）：

- **模型 double descent**：固定数据，增参数，测试误差双重下降
- **数据 double descent**：固定模型，增数据，测试误差也可能先升后降
- **Epoch double descent**：训练时间也有 double descent

这对实践的启示：**别因为"模型太大可能过拟合"就缩模型**。在过参数化区，更大模型反而泛化更好。这是 GPT-3 等"大力出奇迹"的理论支撑之一。

### 4.7 大模型时代的过拟合新特征

大模型时代过拟合有几个新现象：

**① 训练 loss 不再是过拟合指标**

大模型预训练 loss 单调下降，没有典型过拟合曲线。但模型可能在"记住训练数据的具体内容"（如能背出训练集里的句子）。这种过拟合不表现为 loss 上升，而表现为"创造性下降"或"训练数据泄露"。

**② 记忆 vs 泛化的研究**

[Carlini et al., 2021](https://arxiv.org/abs/2012.07805) 量化了大模型对训练数据的记忆：GPT-2 能逐字背出训练集里的特定句子。记忆不全是坏事（知识需要记忆），但过多记忆 = 过拟合。

**③ 数据质量 > 数据量**

[Chinchilla scaling law](https://arxiv.org/abs/2203.15556) 之后的研究发现：低质数据让模型记垃圾，是隐式过拟合。数据筛选、去重、清洗比堆量更重要。

**④ SFT/RLHF 的过拟合**

[指令微调](./instruction-tuning) 和 [RLHF](./rlhf) 阶段数据少，极易过拟合。表现为模型回答变得机械、模式化。需要早停、低学习率、少量 epoch。

## L5 · 沿革与坑

### 沿革

- **1960s**：偏差-方差分解在统计学建立，奠定过拟合理论。
- **1980s-1990s**：神经网络兴起，权重衰减、早停成为标配正则化。
- **2012**：AlexNet 用 Dropout + 数据增强刷新 ImageNet，Dropout 流行。
- **2014**：[Srivastava et al. - Dropout](https://arxiv.org/abs/1207.0580) 系统化 Dropout 理论。
- **2015**：BatchNorm 提出，间接正则化（平滑损失曲面）。
- **2017**：[AdamW](https://arxiv.org/abs/1711.05101) 修正权重衰减，大模型标准正则之一。
- **2017**：[Zhang et al. - Rethinking Generalization](https://arxiv.org/abs/1611.03530) 震撼实验，挑战经典泛化理论。
- **2019**：[Belkin et al. - Double Descent](https://arxiv.org/abs/1812.11118) 颠覆偏差-方差权衡。
- **2021**：[Nakkiran et al. - Deep Double Descent](https://arxiv.org/abs/1912.02292) 在大模型实证。
- **2021**：[Carlini et al.](https://arxiv.org/abs/2012.07805) 量化大模型对训练数据的记忆。
- **2022-2025**：研究焦点转向"数据质量、数据去重、记忆 vs 泛化"，传统正则化在大模型中作用被重新评估。

### 常见误解

- ❌ **误解**：模型越大越容易过拟合。
  ✅ **真相**：经典理论如此，但 double descent 表明过参数化区反而泛化更好（4.6）。大模型时代的现象挑战了这个直觉。

- ❌ **误解**：训练 loss 越低越好。
  ✅ **真相**：训练 loss 极低常意味着过拟合。要看验证 loss、要看泛化。Zhang et al. 2017 证明模型能把随机标签 loss 训到 0，但毫无意义。

- ❌ **误解**：Dropout 越大越好。
  ✅ **真相**：Dropout 过大会欠拟合（信息被丢太多）。典型 $p=0.1\sim0.5$，需实验调。Transformer 中通常 0.1。

- ❌ **误解**：正则化就是加 L2。
  ✅ **真相**：正则化是泛指一切抑制过拟合的手段。L2、Dropout、早停、数据增强、BatchNorm、Label Smoothing 都是正则化（L3 表格）。

- ❌ **误解**：大模型不需要正则化。
  ✅ **真相**：大模型仍需正则化，只是形式变了。数据去重、数据筛选、权重衰减、Dropout、训练时长控制都是大模型正则化。"隐式正则"（SGD 偏好平坦极小值）也起作用。

- ❌ **误解**：验证 loss 上升就是过拟合。
  ✅ **真相**：多数情况是，但也可能是数据分布漂移、学习率过大、bug。需要排查。Epoch double descent 现象中，验证 loss 可能先升后降，过早停止反而损失。

- ❌ **误解**：数据增强总是有益。
  ✅ **真相**：增强必须保持标签语义。错误的增强（如对区分左右的任务做镜像）会破坏标签，让模型学错。增强策略要匹配任务。

### 面试怎么考

1. **"什么是过拟合？怎么判断？"** --训练 loss 低、验证 loss 高，验证 loss 上升时尤其明显。金标准是 train/val loss 分叉（L1、L3）。
2. **"偏差-方差权衡是什么？"** --总误差 = bias² + variance + 噪声。模型复杂度增加，bias 降、variance 升，最优在中间（L3）。
3. **"Dropout 的原理？训练和推理有什么不同？"** --训练随机失活，推理用全部。防止共适应、隐式集成（4.3）。
4. **"L2 正则为什么有效？概率解释？"** --限制参数范数，等效参数高斯先验。AdamW 解耦权重衰减修正了 L2 在自适应优化器里被稀释的问题（4.2）。
5. **"什么是 double descent？它颠覆了什么？"** --过参数化区测试误差反而下降。颠覆"模型越大越易过拟合"的经典直觉（4.6）。
6. **"大模型时代还需要正则化吗？"** --需要，形式变了。数据质量、去重、Dropout、权重衰减、训练时长控制都是（4.7）。

## 延伸阅读

- 📄 [Goodfellow et al. - Deep Learning 第 5、7 章](https://www.deeplearningbook.org/)
- 📄 [Srivastava et al., 2014 - Dropout](https://arxiv.org/abs/1207.0580)
- 📄 [Zhang et al., 2017 - Rethinking Generalization](https://arxiv.org/abs/1611.03530)
- 📄 [Belkin et al., 2019 - Double Descent](https://arxiv.org/abs/1812.11118)
- 📄 [Nakkiran et al., 2021 - Deep Double Descent](https://arxiv.org/abs/1912.02292)
- 📄 [Carlini et al., 2021 - Extracting Training Data](https://arxiv.org/abs/2012.07805)

---

> *上一篇：[优化器（Adam / AdamW）](./optimizer) -- 怎么沿着损失函数下山。*
> *下一篇：[自回归生成](./autoregressive) -- GPT 为什么一个字一个字往外吐。*
