---
title: 优化器（Adam / AdamW）
slug: optimizer
category: 模型架构与训练
tags: [优化器, Adam, AdamW, SGD, 学习率调度]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 优化器（Adam / AdamW）

> **一句话 TL;DR**：优化器是把[损失函数](./loss-function)的梯度变成"参数怎么更新"的算法。大模型训练几乎清一色用 **AdamW**（Adam + 解耦权重衰减），它结合了动量（惯性）和自适应学习率（每个参数用自己的步长）。AdamW 是 2017 年提出后统治大模型训练至今的标准选择。

---

## L1 · 一句话点破

优化器做一件事：**给定损失 $\mathcal{L}(\theta)$ 对参数 $\theta$ 的梯度 $\nabla \mathcal{L}$，决定参数怎么走一步：**

$$
\theta \leftarrow \theta - \eta \cdot \Delta\theta
$$

最简单的 SGD 直接 $\Delta\theta = \nabla \mathcal{L}$。Adam 在此基础上加了两件事：**动量**（用历史梯度的滑动平均稳住方向）和**自适应学习率**（每个参数用自己的有效步长）。AdamW 又修正了 Adam 把权重衰减耦合进动量里的 bug。

优化器的选择直接决定训练是否稳定、收敛多快、能不能逃出鞍点。损失函数定义"山"，优化器是"下山策略"。

## L2 · 通俗类比

下山有几种走法：

- **SGD（随机梯度下降）**：每一步只看脚下，按当前坡度走。坡陡走快、坡缓走慢。问题：到了山脊会左右横跳（梯度震荡），到了平原就走不动。
- **SGD + Momentum**：带惯性。你不仅看脚下，还延续上一脚的方向。像滑雪--遇到小坑小坡能借惯性冲过去，减少震荡。
- **AdaGrad**：每个方向有自己的步长。经常走的陡坡走熟了步子变小，少走的缓坡保持大步子。问题：步子只减不增，最后所有人停下来。
- **RMSProp**：AdaGrad 的"记忆衰减版"。步长按近期梯度调整，不会无脑衰减到 0。
- **Adam**：Momentum + RMSProp 的合体。既有惯性（稳方向），又有自适应步长（每个参数自己调）。下山像带着导航+滑雪板的登山者。
- **AdamW**：Adam + 修正权重衰减的 bug。原 Adam 把权重衰减混进动量里，效果打折。AdamW 把权重衰减独立出来，正则化效果更好。

直觉理解 Adam 为何强：神经网络有几十亿参数，有些参数梯度一直很大（活跃），有些一直很小（懒）。SGD 用同一学习率，活跃的参数早该小步走（已经接近最优），懒的参数该大步走（还很远）。Adam 让每个参数用自己的步长，整体协调得多。

## L3 · 正经定义

**优化器**是迭代算法，给定参数 $\theta_t$、梯度 $g_t = \nabla \mathcal{L}(\theta_t)$、学习率 $\eta$，输出 $\theta_{t+1}$。

### SGD

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$

### Momentum SGD

$$
v_t = \beta v_{t-1} + g_t, \quad \theta_{t+1} = \theta_t - \eta \cdot v_t
$$

$\beta \approx 0.9$ 是动量系数。

### Adam ([Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980))

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(一阶矩估计)}
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(二阶矩估计)}
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(偏差修正)}
$$
$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

典型参数：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。

### AdamW ([Loshchilov & Hutter, 2017](https://arxiv.org/abs/1711.05101))

Adam 把权重衰减耦合进 $m_t$，破坏了 L2 正则的效果。AdamW 解耦：

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
$$

权重衰减 $\lambda \theta_t$ 独立施加，不进入动量。这是大模型训练的事实标准。

### 学习率调度（重要程度不亚于优化器本身）

| 调度 | 形式 | 用途 |
|------|------|------|
| Warmup | 前 N 步线性升到 $\eta_{max}$ | 大模型标配，稳定初期 |
| Cosine | 余弦衰减到 $\eta_{min}$ | 主流，平滑 |
| Step Decay | 每 K 步乘 $\gamma$ | 经典 CV |
| Inverse Sqrt | $\eta / \sqrt{t}$ | Transformer 原论文 |
| WSD | Warmup-Stable-Decay | 2024 流行，可重启续训 |

**参考资料**：
- [Kingma & Ba, 2014 - Adam](https://arxiv.org/abs/1412.6980)
- [Loshchilov & Hutter, 2017 - AdamW](https://arxiv.org/abs/1711.05101)
- [Reddi et al., 2018 - On the Convergence of Adam](https://arxiv.org/abs/1904.09237) - Adam 收敛性问题与 AMSGrad
- [Liu et al., 2019 - On the Variance of Adaptive Learning Rate](https://arxiv.org/abs/1908.03265) - RAdam
- [Kaplan et al., 2020 - GPT-3 scaling paper](https://arxiv.org/abs/2001.08361)

## L4 · 原理深挖

### 4.1 动量为什么有用：减少震荡

考虑一个椭圆等高线的损失（一个方向陡、一个方向缓）。SGD 在陡方向上来回震荡，缓方向上进展慢。动量平均历史梯度，**陡方向的震荡被平均掉（正负抵消），缓方向的进展被累积**，整体走向更直。

数学上，动量等效于指数加权平均（EMA）的梯度，相当于在历史窗口内平均。它能：

- **加速收敛**：一致方向上的梯度累积，等效大步长
- **逃出局部极小**：惯性让优化器冲过浅坑
- **减少震荡**：高频震荡被平均抑制

$\beta = 0.9$ 意味着等效窗口约 $1/(1-\beta) = 10$ 步。

### 4.2 自适应学习率：每个参数自己的步长

AdaGrad/RMSProp/Adam 的核心洞察：**梯度大的参数说明它在频繁变化，应该小步走防止过冲；梯度小的参数说明它很少动，应该大步走推动进展。**

Adam 用 $v_t$（梯度平方的 EMA）作为分母：

$$
\Delta \theta = \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

- $g_t$ 大：$v_t$ 大，分母大，有效步长小
- $g_t$ 小：$v_t$ 小，分母小，有效步长大

这给每个参数一个"自适应"步长。对深度网络中不同层梯度量级差异大的情况特别有用。

### 4.3 偏差修正：为什么 Adam 需要

Adam 用 EMA 估计 $m_t$、$v_t$，但 EMA 初始为 0，前几步会严重低估。$\beta_2 = 0.999$ 时尤其严重：第一步 $v_1 = 0.001 g_1^2$，分母 $\sqrt{v_1} \approx 0.03 |g_1|$，远小于真实 $|g_1|$。

偏差修正 $\hat{v}_t = v_t / (1-\beta_2^t)$ 在初期把估计放大到接近真实值。$t \to \infty$ 时 $\beta_2^t \to 0$，修正项消失。

没有偏差修正，训练初期会出现超大步长（因为分母被低估），导致参数爆炸。这是 Adam 实现的关键细节。

### 4.4 AdamW 修正了什么：解耦权重衰减

[原 Adam 论文](https://arxiv.org/abs/1412.6980) 把 L2 正则写成"在梯度上加 $\lambda \theta$"：

$$
g_t \leftarrow \nabla \mathcal{L} + \lambda \theta_t
$$

然后 $g_t$ 进入 $m_t$ 和 $v_t$。问题：$\lambda \theta$ 的平方进入 $v_t$，让分母变大，**等效降低了权重衰减的强度**。在自适应优化器里，L2 正则的效果被稀释。

[Loshchilov & Hutter, 2017](https://arxiv.org/abs/1711.05101) 指出这个问题，提出**解耦权重衰减**：

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t
$$

权重衰减 $\lambda \theta_t$ 独立施加，不进入动量和方差估计。这样：

- L2 正则的效果不被稀释
- 学习率和权重衰减可以独立调
- 大模型泛化更好

实证：在 ResNet、Transformer 等网络上，AdamW 比 Adam（带 L2）泛化更好。AdamW 成为大模型训练标准。

### 4.5 学习率调度：和优化器同等重要

实际训练中，**学习率调度的影响往往超过优化器选择**。常见模式：

**① Warmup：稳定初期**

训练初期参数随机，梯度方向噪声大。直接用大学习率会震荡甚至发散。Warmup 在前 $N$ 步（如 2000 步）线性升到 $\eta_{max}$，让模型先稳定下来再大步前进。

GPT-3、LLaMA 等大模型都用 warmup，是稳定训练的关键。

**② Cosine Decay：平滑收敛**

Warmup 后用余弦曲线衰减到 $\eta_{min}$：

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{t\pi}{T}\right)
$$

平滑过渡，末期小步精修。是当前主流调度。

**③ WSD（Warmup-Stable-Decay）**

2024 年流行的新调度：Warmup -> 长时间 Stable（保持 $\eta_{max}$）-> 末段 Decay。优点是可以从 Stable 阶段任意点重启续训，适合"训一点用一点"的迭代开发。

**④ 学习率上限的实务**

大模型的学习率通常在 $1\text{e-}4$ 到 $6\text{e-}4$ 之间。参数量越大，最优学习率越小（但不是线性）。Chinchilla 等研究给出了不同规模模型的推荐学习率。

### 4.6 Adam 的局限和后续改进

Adam 不是万能的：

**① 收敛性问题**

[Reddi et al., 2018](https://arxiv.org/abs/1904.09237) 指出 Adam 在某些凸优化问题上不收敛，原因是 $v_t$ 的 EMA 可能"忘记"早期的大梯度。AMSGrad 修正：取 $v_t$ 的历史最大值。

**② RAdam：无 warmup 的 Adam**

[Liu et al., 2019](https://arxiv.org/abs/1908.03265) 指出 warmup 的本质是弥补 $v_t$ 估计初期的方差。RAdam 引入方差修正项，无需 warmup 也能稳定训练。

**③ Lion：2023 年的新选择**

[Lion (Google, 2023)](https://arxiv.org/abs/2302.00652) 用符号函数 + 动量，比 AdamW 内存少一半，部分任务效果更好。但目前仍未撼动 AdamW 的统治地位。

**④ Sophia：二阶优化器**

[Liu et al., 2023 - Sophia](https://arxiv.org/abs/2305.14342) 用对角 Hessian 估计，比一阶 Adam 在 LLM 训练上少 50% 步数。但实现复杂，工程友好度不如 AdamW。

AdamW 的统治力来自：**简单、稳定、调参容易、生态成熟**。学术上更好的优化器很多，但替代 AdamW 需要工程生态的整体迁移，难度大。

## L5 · 沿革与坑

### 沿革

- **1951**：Robbins & Monro 提出 SGD（随机逼近），优化算法起点。
- **1964**：Rumelhart 等人提出反向传播 + 动量，神经网络训练奠基。
- **2011**：Duchi 提出 AdaGrad，自适应学习率开端。
- **2012**：Hinton 提出 RMSProp，修正 AdaGrad 衰减过快。
- **2014**：[Kingma & Ba 提出 Adam](https://arxiv.org/abs/1412.6980)，集大成，迅速成为主流。
- **2017**：[Loshchilov & Hutter 提出 AdamW](https://arxiv.org/abs/1711.05101)，修正权重衰减耦合问题。
- **2018**：[Reddi et al.](https://arxiv.org/abs/1904.09237) 指出 Adam 不收敛，提出 AMSGrad。
- **2019**：[RAdam](https://arxiv.org/abs/1908.03265) 提出无 warmup 的 Adam。
- **2020-2023**：LLaMA、GPT-3、PaLM 等大模型清一色用 AdamW + Warmup + Cosine，标准组合确立。
- **2023**：[Lion](https://arxiv.org/abs/2302.00652)、[Sophia](https://arxiv.org/abs/2305.14342) 挑战 AdamW，但生态未迁移。
- **2024-2025**：AdamW 仍是大模型训练事实标准，研究焦点转向"如何在大 batch 下稳定 AdamW"。

### 常见误解

- ❌ **误解**：Adam 一定比 SGD 好。
  ✅ **真相**：在泛化能力上 SGD + Momentum 在很多 CV 任务上仍优于 Adam。Adam 收敛快、调参易，但泛化上有时不如 SGD。选择取决于任务、数据、模型。大模型领域 AdamW 是标准，但不是普适最优。

- ❌ **误解**：学习率越大收敛越快。
  ✅ **真相**：学习率过大训练发散，过小收敛慢。最优学习率通常需要 grid search。大模型通常在 $1\text{e-}4$ 到 $6\text{e-}4$，warmup + cosine 调度。

- ❌ **误解**：Adam 自动调学习率，所以不用调 $\eta$。
  ✅ **真相**：Adam 的"自适应"是每个参数的相对步长，全局 $\eta$ 仍需调。$\eta$ 选错（太大发散、太小慢）效果天差地别。

- ❌ **误解**：Adam 和 AdamW 差不多。
  ✅ **真相**：在大模型 + 权重衰减场景，AdamW 明显更好。原 Adam 把权重衰减耦合进动量，正则效果被稀释。AdamW 解耦是关键修复（4.4）。

- ❌ **误解**：warmup 是为了让模型"慢慢启动"。
  ✅ **真相**：warmup 的本质是弥补 Adam $v_t$ 估计初期的方差问题。初期 $v_t$ 严重低估，导致有效步长爆炸。Warmup 给 $v_t$ 时间收敛到真实值。RAdam 通过数学修正实现无 warmup 的稳定训练。

- ❌ **误解**：优化器选对了模型就能训好。
  ✅ **真相**：优化器只是训练的一环。学习率调度、batch size、初始化、正则化、数据质量，每个都同等重要。AdamW + 烂学习率调度 vs SGD + 完美调度，后者可能更好。

- ❌ **误解**：Adam 的 $\beta_1, \beta_2$ 要精调。
  ✅ **真相**：$\beta_1 = 0.9$、$\beta_2 = 0.999$ 是几乎所有任务的默认值。少数场景（如梯度很稀疏）会调 $\beta_2$，但绝大多数情况用默认即可。值得花时间调的是学习率、batch size、调度。

### 面试怎么考

1. **"Adam 的两个一阶/二阶矩估计分别是什么？"** --$m_t$ 是梯度的 EMA（动量），$v_t$ 是梯度平方的 EMA（自适应分母）（L3）。
2. **"为什么 Adam 需要偏差修正？"** --EMA 初始为 0，前几步严重低估，导致分母过小、步长爆炸。修正项放大初期估计（4.3）。
3. **"AdamW 修正了 Adam 的什么问题？"** --原 Adam 把权重衰减耦合进动量，正则被稀释。AdamW 解耦权重衰减，独立施加（4.4）。
4. **"为什么大模型训练要 warmup？"** --稳定初期，弥补 $v_t$ 估计的方差问题（4.5）。
5. **"Adam 和 SGD+Momentum 的区别？"** --SGD 用统一学习率；Adam 每个参数自适应。Adam 收敛快调参易，SGD 泛化有时更好（4.1、4.2、5）。
6. **"Adam 的 $\beta_1, \beta_2$ 怎么选？"** --默认 0.9/0.999 适用于绝大多数场景，不精调。重点调学习率、batch size、调度（5）。

## 延伸阅读

- 📄 [Kingma & Ba, 2014 - Adam](https://arxiv.org/abs/1412.6980)
- 📄 [Loshchilov & Hutter, 2017 - AdamW](https://arxiv.org/abs/1711.05101)
- 📄 [Reddi et al., 2018 - AMSGrad](https://arxiv.org/abs/1904.09237)
- 📄 [Liu et al., 2019 - RAdam](https://arxiv.org/abs/1908.03265)
- 📄 [Lion (2023)](https://arxiv.org/abs/2302.00652)
- 📝 [Sebastian Ruder - Overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)

---

> *上一篇：[损失函数](./loss-function) -- 模型怎么知道自己错没错。*
> *下一篇：[过拟合 & 正则化](./overfitting) -- Dropout / 权重衰减为什么有效。*
