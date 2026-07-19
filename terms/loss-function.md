---
title: 损失函数（Loss Function）
slug: loss-function
category: 模型架构与训练
tags: [损失函数, 交叉熵, MSE, 梯度下降, 大模型损失]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 损失函数（Loss Function）

> **一句话 TL;DR**：损失函数是衡量"模型预测"和"真实答案"差距的标量函数。它把"模型好不好"翻译成"[优化器](./optimizer)能下山的方向"。大模型预训练用的是**交叉熵损失**，它本质是"模型给真实下一个 token 分配的概率的负对数"--概率越高，损失越低。

---

## L1 · 一句话点破

损失函数（Loss Function，也称目标函数 / 代价函数）做一件事：**给定模型预测 $\hat{y}$ 和真实标签 $y$，输出一个标量 $L(\hat{y}, y)$，数值越大表示错得越离谱。**

训练就是在参数空间里找一组参数，让所有样本的平均损失最小化：

$$
\theta^* = \arg\min_\theta \mathbb{E}_{(x,y) \sim D} \left[ L(f_\theta(x), y) \right]
$$

[优化器](./optimizer)（Adam、SGD）就是下山算法，而损失函数定义了"山"的形状。山怎么定义，决定模型学到什么。

## L2 · 通俗类比

想象你是射箭教练，学生每次射完一组箭，你要给一个"分数"告诉他射得有多偏：

- **MSE（均方误差）**：量靶心到箭的距离，平方后取平均。离得越远扣分越多（平方放大远距离错误）。
- **MAE（平均绝对误差）**：量靶心到箭的距离，不平方，直接平均。远距离错误扣分线性增长。
- **交叉熵**：学生每次射箭前要押"会射中哪里"。押中靶心给高分；押错地方还射错，重罚。惩罚的是"自信地错"。
- **Hinge Loss**：押对就行，押多准无所谓，但押错且押得不够远会被罚。鼓励"留足余量"。

不同"评分方式"引导学生练成不同风格：

- MSE 教出的学生追求"平均偏最小"，对离群箭很敏感
- MAE 教出的学生追求"中位数偏最小"，对离群箭不敏感
- 交叉熵教出的学生追求"押对概率最大"，鼓励明确表态
- Hinge 教出的学生追求"留安全余量"

损失函数不是客观真理，是设计选择。同一个任务，不同损失训出风格各异的模型。

## L3 · 正经定义

**损失函数 $L(\hat{y}, y) \in \mathbb{R}$** 度量预测 $\hat{y}$ 和真值 $y$ 的不相似度。训练目标是最小化经验损失（通常加正则项）：

$$
\mathcal{J}(\theta) = \frac{1}{N}\sum_{i=1}^N L(f_\theta(x_i), y_i) + \lambda R(\theta)
$$

按任务类型分类：

| 任务类型 | 典型损失 | 公式（核心项） | 用途 |
|---------|---------|---------------|------|
| 回归 | MSE | $\frac{1}{N}\sum (\hat{y}-y)^2$ | 房价、温度预测 |
| 回归 | MAE | $\frac{1}{N}\sum \|\hat{y}-y\|$ | 离群点鲁棒回归 |
| 二分类 | Binary Cross-Entropy | $-[y\log\hat{y}+(1-y)\log(1-\hat{y})]$ | 情感、广告点击 |
| 多分类 | Cross-Entropy | $-\sum_k y_k \log \hat{y}_k$ | 图像分类、NLP 分类 |
| 大模型预训练 | Next-Token CE | $-\log P_\theta(y_t \| x_{<t})$ | GPT/LLaMA 预训练 |
| 大模型对齐 | DPO 损失 | 见 [RLHF](./rlhf) | 偏好对齐 |
| 排序 | ListMLE / NDCG-loss | $-\log P_\theta(\pi \| x)$ | 搜索、推荐 |
| 对比学习 | InfoNCE | $-\log \frac{e^{s_+/\tau}}{\sum e^{s_i/\tau}}$ | [Embedding](./embedding) 训练 |
| 检测 | Focal Loss | $-(1-\hat{y})^\gamma \log \hat{y}$ | 类别不均衡检测 |

**参考资料**：
- [Goodfellow et al., 2016 - Deep Learning 第 5-8 章](https://www.deeplearningbook.org/) - 损失函数系统论述
- [Lin et al., 2017 - Focal Loss](https://arxiv.org/abs/1708.02002)
- [Oord et al., 2018 - InfoNCE](https://arxiv.org/abs/1807.03748)
- [Rafailov et al., 2023 - DPO 损失](https://arxiv.org/abs/2305.18290)

## L4 · 原理深挖

### 4.1 为什么是交叉熵：最大似然的视角

大模型预训练用 next-token cross-entropy，不是随便选的，它来自**最大似然估计（MLE）**。

假设语料是某个真实分布 $P_{data}$ 采样的序列，模型 $P_\theta$ 试图逼近它。最大似然目标是：

$$
\max_\theta \mathbb{E}_{x \sim P_{data}} [\log P_\theta(x)]
$$

对自回归模型，$P_\theta(x) = \prod_t P_\theta(x_t | x_{<t})$，取对数：

$$
\log P_\theta(x) = \sum_t \log P_\theta(x_t | x_{<t})
$$

最大化对数似然 = 最小化负对数似然：

$$
\mathcal{L} = -\sum_t \log P_\theta(x_t | x_{<t}) = \sum_t \text{CE}_t
$$

这就是 next-token cross-entropy。它有清晰的概率解释：**让模型给真实数据分配的概率最大化。**

### 4.2 交叉熵和 KL 散度的关系

交叉熵和信息论紧密相关。对真实分布 $p$（one-hot，即真实 token）和预测分布 $q$（模型输出）：

$$
H(p, q) = -\sum_k p_k \log q_k = H(p) + D_{KL}(p \| q)
$$

- $H(p)$：真实分布的熵，与模型无关（常数）
- $D_{KL}(p \| q)$：真实分布到预测分布的 KL 散度

所以**最小化交叉熵 = 最小化 KL 散度 = 让预测分布逼近真实分布**。这是从信息论角度解释为什么 CE 是"自然"的损失。

注意 one-hot 真实分布时 $H(p) = 0$，CE 退化为 $-\log q_{true\_token}$。这就是为什么 next-token loss 看起来这么简单：就是真实 token 概率的负对数。

### 4.3 为什么不用 MSE 做分类：梯度饱和

直觉上分类也能用 MSE（让 logits 接近 one-hot），但实际不这么做。原因：

**梯度饱和**。MSE 配 sigmoid 输出时，梯度形式：

$$
\frac{\partial L}{\partial z} = (\sigma(z) - y) \cdot \sigma'(z)
$$

当 $z$ 远离真值（错得离谱）时，$\sigma(z)$ 趋近 0 或 1，$\sigma'(z)$ 趋近 0--**错得越离谱，梯度越小**，学习越慢。这是早期神经网络训练慢的核心原因。

交叉熵配 softmax 时梯度形式：

$$
\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k
$$

干净利落：错得越远，梯度越大。这就是为什么分类任务几乎清一色用 CE。

### 4.4 Label Smoothing：CE 的软化

标准 CE 用 one-hot 真实分布，强迫模型把所有概率给真实类。这容易让模型过度自信（over-confident），且在标签可能有噪声时脆弱。

**Label Smoothing** 把 one-hot 改成软标签：

$$
y_k^{smooth} = (1-\epsilon) \cdot y_k + \frac{\epsilon}{K}
$$

$\epsilon$ 通常 0.1，$K$ 是类别数。效果：真实类概率上限 $1-\epsilon+\epsilon/K$（如 0.913），其他类至少 $\epsilon/K$（如 0.0003）。

Label Smoothing 的作用：

- **防止过度自信**：模型不被逼到 softmax 极端值
- **提升校准**：预测置信度更接近真实准确率
- **正则化**：隐式约束 logits，类似 L2

[原 Transformer 论文](https://arxiv.org/abs/1706.03762) 就用了 label smoothing $\epsilon=0.1$，是提升翻译质量的关键技巧之一。

### 4.5 大模型预训练损失的特征

观察大模型预训练 loss 曲线的几个特征：

**① Loss 不是 0**

即使训练到极致，next-token CE 也不会到 0。原因：自然语言有固有不可预测性。例如"今天天气很___"，下一个词可能是"好""热""冷"，都有可能。完美预测需要知道说话者意图、上下文，本质上是不可压缩的熵。这个下限叫**条件熵** $H(y|x)$。

**② Loss 与 perplexity 的关系**

Perplexity（困惑度）= $\exp(\text{CE})$。它表示"模型在每个位置平均犹豫几个词"。CE 2.3 对应 PPL 10，CE 1.6 对应 PPL 5。PPL 更直观但本质是 CE 的指数变换。

**③ Scaling Law 的 loss 形式**

[Chinchilla scaling law](https://arxiv.org/abs/2203.15556) 把 loss 拟合为参数量 $N$、数据量 $D$、计算量 $C$ 的幂律：

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

$E$ 是不可压缩熵（下限），$A/N^\alpha$ 是模型容量不足项，$B/D^\beta$ 是数据不足项。这个公式指导了"模型该多大、数据该多少"的核心决策。

**④ Loss 平台期 vs 性能跃升**

训练 loss 看起来平滑下降，但下游能力常出现"涌现"--某个能力在 loss 降到某阈值后突然出现。这是大模型最神秘的现象之一，至今没有完全的解释。

### 4.6 对比损失：从监督到自监督

[Embedding](./embedding) 训练、CLIP、SimCLR 等用**对比损失（Contrastive Loss）**，特别是 InfoNCE：

$$
\mathcal{L} = -\log \frac{e^{s(x, x^+)/\tau}}{\sum_{x' \in \{x^+, x^-_1, ..., x^-_K\}} e^{s(x, x')/\tau}}
$$

- $x^+$：与 $x$ 相似的正样本
- $x^-_i$：负样本
- $\tau$：温度
- $s$：相似度（如余弦）

InfoNCE 形式上像 CE，但"类别"是"哪个是正样本"。它强迫模型把正样本和负样本在表示空间拉开。

对比学习的核心难点是负样本选择：太简单学不到东西，太难（伪负样本）会破坏表示。这是 [Embedding](./embedding) 训练的关键工程问题。

## L5 · 沿革与坑

### 沿革

- **1940s-1950s**：MSE 在统计学回归中是标准损失，源于高斯-马尔可夫定理。
- **1950s-1980s**：感知机、logistic 回归用 CE / log loss，统计学习理论奠基。
- **1990s-2000s**：SVM 用 hinge loss，结构风险最小化理论成熟。
- **2012-2015**：深度学习时代，CE 在分类、MSE 在回归成为默认。[Lin et al., 2017 - Focal Loss](https://arxiv.org/abs/1708.02002) 解决类别不均衡。
- **2017**：[Transformer 原论文](https://arxiv.org/abs/1706.03762) 用 label smoothing，成为 NLP 标配。
- **2018-2020**：[BERT](https://arxiv.org/abs/1810.04805)、GPT 系列用 next-token CE，确认大模型预训练损失标准。
- **2020**：[Chinchilla scaling law](https://arxiv.org/abs/2203.15556) 把 loss 拟合成参数-数据的幂律。
- **2023**：[DPO](https://arxiv.org/abs/2305.18290) 提出偏好对齐的新损失形式，取代 RLHF 的奖励-策略两阶段。
- **2024-2025**：研究焦点转向"过程奖励"（每步 reward）和推理类损失，纯 next-token CE 的局限显现。

### 常见误解

- ❌ **误解**：损失越低，模型越聪明。
  ✅ **真相**：训练 loss 低只说明模型拟合了训练分布。泛化能力、下游任务表现需要单独评估。Loss 降到 0.5 和 0.4 在某些任务上可能没区别，在另一些任务上天差地别。

- ❌ **误解**：所有任务都该用 MSE。
  ✅ **真相**：MSE 对离群点敏感（平方放大误差），且做分类时梯度饱和。分类用 CE，回归用 MSE/MAE，排名用排序损失，各有适用场景（L3 表格）。

- ❌ **误解**：交叉熵就是负对数似然。
  ✅ **真相**：在 one-hot 标签下两者等价，但一般情况交叉熵 $H(p,q)$ 包含真实分布熵 $H(p)$。最小化 CE 等价于最小化 NLL + 常数，所以优化上等价，但语义不同。

- ❌ **误解**：loss 曲线平滑下降就是健康训练。
  ✅ **真相**：loss 平滑下降不一定代表学到有用特征。可能模型在记训练集（过拟合），可能学到 shortcut feature。要看 val loss、下游任务表现、loss 曲线突然的台阶（可能学到新能力或崩溃）。

- ❌ **误解**：label smoothing 总是有益。
  ✅ **真相**：在标签干净、需要明确置信度的场景，label smoothing 可能让模型不够自信，反而损害精确率。它适合标签有噪声、需要校准、类别多的场景。

- ❌ **误解**：大模型预训练 loss 到 0 就完美了。
  ✅ **真相**：自然语言有不可压缩的熵下限 $H(y|x)$，loss 不可能到 0。即使到下限，模型也只是"完美预测了分布"，不代表能生成有意义的内容（生成和预测分布是两件事）。

### 面试怎么考

1. **"大模型预训练用什么损失？为什么？"** --Next-token cross-entropy，来自最大似然估计。最小化 NLL = 最大化数据概率（4.1）。
2. **"为什么分类用 CE 不用 MSE？"** --MSE 配 sigmoid 梯度饱和，错得越远学得越慢；CE 配 softmax 梯度干净（4.3）。
3. **"什么是 label smoothing？有什么用？"** --软化 one-hot 标签，防过度自信、提升校准、正则化（4.4）。
4. **"InfoNCE 损失的形式和作用？"** --把正样本和负样本在表示空间拉开，是对比学习核心。形式像 CE，"类别"是哪个是正样本（4.6）。
5. **"为什么训练 loss 到不了 0？"** --自然语言有不可压缩的条件熵下限 $H(y|x)$（4.5）。
6. **"loss 和 perplexity 的关系？"** --PPL = exp(CE)，是 CE 的指数变换，更直观（4.5）。

## 延伸阅读

- 📄 [Goodfellow et al. - Deep Learning 第 5-8 章](https://www.deeplearningbook.org/)
- 📄 [Lin et al., 2017 - Focal Loss](https://arxiv.org/abs/1708.02002)
- 📄 [Oord et al., 2018 - InfoNCE / CPC](https://arxiv.org/abs/1807.03748)
- 📄 [Hoffmann et al., 2022 - Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)
- 📄 [Rafailov et al., 2023 - DPO](https://arxiv.org/abs/2305.18290)

---

> *上一篇：[迁移学习](./transfer-learning) -- 预训练能迁移的底层机理。*
> *下一篇：[优化器（Adam / AdamW）](./optimizer) -- 怎么沿着损失函数下山。*
