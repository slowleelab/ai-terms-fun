---
title: RoPE 旋转位置编码
slug: rope
category: 进阶专题
tags: [RoPE, 位置编码, 旋转位置嵌入, 长上下文, 注意力机制]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# RoPE 旋转位置编码

> 五层读懂一个词。这次拆的是：**RoPE**--LLM 位置编码的事实标准。用旋转矩阵编码位置信息，让两个 token 的注意力分数只依赖它们的位置差（相对位置），而非绝对位置。天然支持外推（训练 2k 拓展到 32k+），是 LLaMA/Qwen/Mistral/DeepSeek 等的共同选择。

---

## L1 · 一句话点破

**RoPE = 用旋转矩阵编码位置**。绝对位置（Sinusoidal）+ 相对位置（Attention Bias）的优雅融合：对每个 token 的 Query 和 Key 向量按位置 $\theta$ 旋转，点积结果自然包含相对位置差 $m-n$。优点是天然支持相对位置（不受绝对位置影响）和外推（训练短序列，推理长序列），是 LLaMA 及几乎所有主流开源 LLM 的位置编码方案。

---

## L2 · 通俗类比

Transformer 的注意力机制本身**不知道 token 顺序**：

- "我爱你" 和 "你爱我" 对注意力来说是同一组 token
- 需要「位置编码」告诉模型哪个 token 在哪个位置

**传统位置编码的两种思路**：

**绝对位置**（Sinusoidal / Learned Positional Encoding）：

- 每个位置有一个唯一编码
- "第 3 个 token" 的编码和 "第 100 个 token" 不同
- 问题：训练只见过 2k 位置，推理 4k+ 位置没学过，效果崩塌

**相对位置**（Attention with Relative Position Bias）：

- 不在乎 token 在哪，而在乎两个 token 之间差多少
- 点积只依赖 $m-n$（位置差）
- 问题：实现复杂，性能开销大

**RoPE = 用旋转来实现「看似绝对，实则相对」**：

像**表盘上的指针旋转**：

- 每个位置对应一个旋转角度
- token 在位置 $m$，Q 和 K 向量旋转 $m\theta$
- token 在位置 $n$，Q 和 K 向量旋转 $n\theta$
- 两个 token 做点积时，旋转角度自动相减：$m\theta - n\theta = (m-n)\theta$
- 所以点积只依赖位置差 $m-n$！

**关键洞察**：

- RoPE 表面上是绝对位置编码（每个位置旋转不同角度）
- 但点积结果只依赖相对位置（旋转角度差）
- 兼具绝对位置的简单 + 相对位置的泛化性
- 更重要的是：可以外推！（训练 2k，推理 32k+）

**为什么能外推**：

- 相对位置只依赖位置差 $|m-n|$
- 训练时见过的位置差覆盖了大部分推理场景
- 配合 NTK/线性插值，平滑扩展频率

**RoPE 的使用**：

| 模型 | 位置编码 |
|------|---------|
| LLaMA / LLaMA 2 / LLaMA 3 | RoPE |
| Qwen / Qwen 2 / Qwen 2.5 | RoPE |
| Mistral / Mixtral | RoPE |
| DeepSeek V2 / V3 | RoPE |
| ChatGLM 3/4 | RoPE |
| Phi-3 / Phi-4 | RoPE |
| Gemma 2 | RoPE |

几乎**所有主流开源 LLM 都用 RoPE**。

**代价**：

- 实现复杂（复数旋转，注意实现细节）
- 外推需要插值（NTK/线性/YaRN）
- 某些长上下文任务仍有衰减

**适用**：

- 几乎所有 Transformer 模型
- 需要长上下文外推
- 需要相对位置编码泛化能力

---

## L3 · 正经定义

**RoPE**（Rotary Position Embedding，旋转位置嵌入）：由 Su et al. 2021 提出的位置编码方法。核心思想：将 position $m$ 的 Query/Key 向量按 $m\theta$ 角度旋转（通过 2D 旋转矩阵的块对角形式作用在高维向量上），使得自注意力的点积 $q_m \cdot k_n$ 仅依赖于相对位置差 $m-n$：

$$
(q_m^{\text{RoPE}})^\top k_n^{\text{RoPE}} = (R_{\Theta,m} q_m)^\top (R_{\Theta,n} k_n) = q_m^\top R_{\Theta,m-n} k_n
$$

其中 $R_{\Theta,m}$ 是分块旋转矩阵，$\Theta$ 是频率参数集。

**关键特性**：

- **相对位置**：注意力分数自然包含相对位置差
- **能外推**：配合 NTK/线性/YaRN 插值，训练短序列可拓展到长序列
- **效率高**：可融入现有 Attention 实现（对 Q/K 做旋转），计算开销小
- **递减权重**：远程依赖权重自然衰减（符合语言局部性）

**参考资料**：

- 📄 Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021
- 📄 Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, 2023
- 📄 Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation*, 2023（PI）
- 📝 Blog: https://blog.eleuther.ai/rotary-embeddings/
- 🔧 HuggingFace RoPE：https://huggingface.co/docs/transformers/model_doc/llama

---

## L4 · 原理深挖

### 4.1 为什么需要位置编码

**Self-Attention 的位置不变性**：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

- 这个公式对 token 排列顺序不敏感
- 交换两个 token 位置，输出也跟着交换
- 但不会因此产生「不同顺序不同语义」的理解

**位置编码的目的**：

- 在输入中注入位置信息
- 让 Attention 知道哪个 token 在哪个位置
- "我爱你" ≠ "你爱我"

**两种路径**：

1. **加到输入上**（Additive）：pos encoding + token embedding → 进入模型
2. **融入 Attention**（Multiplicative）：在 Attention 计算中引入位置

RoPE 属于后者（融入 Attention），在 Q/K 上操作。

### 4.2 RoPE 的数学

**核心公式**（对位置 $m$ 的向量 $x \in \mathbb{R}^d$）：

将 $x$ 的维度按 $(x_0, x_1), (x_2, x_3), ..., (x_{d-2}, x_{d-1})$ 成对分组。对第 $i$ 对 $(x_{2i}, x_{2i+1})$：

$$
\begin{pmatrix}
x_{2i}' \\ x_{2i+1}'
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
x_{2i} \\ x_{2i+1}
\end{pmatrix}
$$

其中 $\theta_i = 10000^{-2i/d}$（与 Sinusoidal 编码相同的频率）。

**旋转矩阵** $R_{\Theta,m}$ 是这些 2×2 旋转矩阵的块对角拼接：

$$
R_{\Theta,m} =
\begin{bmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

**RoPE 的 Attention**：

$$
\text{Attention}(Q, K, V)_m = \sum_n \text{softmax}\left(\frac{(R_{\Theta,m} q_m)^\top (R_{\Theta,n} k_n)}{\sqrt{d}}\right) v_n
$$

**为什么是相对位置**：

$$
(R_{\Theta,m} q_m)^\top (R_{\Theta,n} k_n) = q_m^\top R_{\Theta,m}^\top R_{\Theta,n} k_n = q_m^\top R_{\Theta,n-m} k_n
$$

因为旋转矩阵的正交性：$R_{\Theta,m}^\top R_{\Theta,n} = R_{\Theta,n-m}$，点积只依赖 $n-m$。

### 4.3 RoPE 的特性

**特性 1: 远程衰减**

随着 $|m-n|$ 增大，RoPE 的点积期望变小：

```
期望注意力权重:
Δ=0:   1.00
Δ=10:  0.85
Δ=100: 0.50
Δ=1000: 0.20
Δ=10000: 0.05
```

这符合语言直觉：越近的 token 越相关。

**特性 2: 外推能力**

训练时只见过位置 0-2047，推理时位置 0-32767。

**为什么能**：

- RoPE 编码的是相对位置 $m-n$，而非绝对位置
- 训练时见过位置差 0-2047
- 推理时位置差大部分仍在 0-2047 范围内
- 但远距离位置差的编码质量差

**需要插值来改善**：

- **线性插值**（PI）：$\tilde{\theta_i} = \theta_i / \lambda$，把长序列压缩到训练范围
- **NTK-Aware**：不同频率不同缩放，高频少缩、低频多缩
- **YaRN**：NTK + 温度调整，最佳外推效果

### 4.4 外推技术详解

**问题**：Llama-2 训练 4k，推理 32k，RoPE 如何适应？

**方法 1: 线性插值（Positional Interpolation, PI）**

把位置索引线性压缩：

$$
m' = m / \lambda, \quad \lambda = L_{\text{new}} / L_{\text{old}}
$$

- 简单，但所有频率一起缩放
- 效果一般

**方法 2: NTK-Aware Scaling**

基于神经正切核（NTK）理论，不同频率不同缩放：

$$
\theta_i' = \theta_i \cdot (b)^{-2i/d}, \quad b = (L_{\text{new}} / L_{\text{old}})^{2d/(d-2)}
$$

- 高频（近程依赖）几乎不动
- 低频（远程依赖）缩放更多
- 效果优于线性插值

**方法 3: YaRN**

NTK + 温度参数 $\tau$：

$$
\text{softmax}((qk^\top) / \sqrt{d} / \tau)
$$

- $\tau > 1$：降低注意力集中度，适合长序列
- 最佳外推效果

**对比**（Llama-2-7B，从 4k 到 32k，困惑度）：

| 方法 | PPL@16k | PPL@32k |
|------|---------|---------|
| 无外推 | 发散 | 发散 |
| PI（线性） | 5.8 | 7.2 |
| NTK | 5.3 | 6.1 |
| **YaRN** | **5.1** | **5.8** |

### 4.5 RoPE 的实现

**PyTorch 实现**：

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    q, k: [batch, heads, seq, dim]
    cos, sin: [seq, dim]
    position_ids: [batch, seq]
    """
    # 获取对应位置的 cos/sin
    cos = cos[position_ids].unsqueeze(2)  # [batch, seq, 1, dim]
    sin = sin[position_ids].unsqueeze(2)
    
    # 对半旋转: 把维度分成两半
    # q_rot = q * cos + rotate_half(q) * sin
    # 其中 rotate_half 交换两半并取反
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
```

**关键要点**：

- 只对 Q 和 K 做旋转，V 不动
- 旋转前一半和后一半维度成对
- cos/sin 预计算，推理时查表

### 4.6 RoPE vs 其他位置编码

| 方法 | 原理 | 相对位置 | 外推 | 开销 | 代表模型 |
|------|------|---------|------|------|---------|
| Sinusoidal | 正弦函数绝对编码 | ❌ | ❌ | 零 | Transformer |
| Learned | 可学习绝对编码 | ❌ | ❌ | 零 | GPT-2/BERT |
| T5 Relative | 相对偏置 bucket | ✅ | ⚠️ | 中 | T5 |
| ALiBi | 线性偏置衰减 | ✅ | ✅ | 零 | BLOOM |
| **RoPE** | **旋转矩阵** | **✅** | **✅** | **小** | **LLaMA/Qwen/Mistral** |
| NoPE | 不用位置编码 | - | - | 零 | - |

**RoPE 为什么胜出**：

- 比 Sinusoidal/Learned 有相对位置和外推
- 比 T5 Relative 简单、高效
- 比 ALiBi 效果好（ALiBi 远距离衰减太快）
- 实现小改动即可融入现有 Attention

### 4.7 RoPE 的变体

**Multimodal RoPE**：

- 图像不同 patch 有 2D 位置，RoPE 需要扩展到 2D
- 分别在 x 和 y 方向旋转

**Interleaved RoPE**：

- 对 Q 和 K 的某些维度不旋转，只旋转部分维度
- 保留部分内容信息不混合位置

**Dynamic NTK RoPE**：

- 推理时动态调整 NTK scale
- 根据输入长度自动选择最佳 scale

### 4.8 RoPE 的局限

**局限 1: 不是无限外推**。需要插值，超长序列（>128k）仍有衰减。

**局限 2: 旋转维度只能配对**。维度数必须是偶数。

**局限 3: 不适合序列到序列的偏移**。适应 token-level 相对位置，但不适合 cross-attention 的偏移（如机器翻译的源-目标偏移）。

**局限 4: 远距离衰减可能过度**。某些任务需要远程依赖（如长文推理），衰减太强妨碍。

**局限 5: 复数实现易出错**。cos/sin 计算、半精度（FP16/BF16）下数值精度问题。

**局限 6: Multimodal 需要调整**。图像 2D、视频 3D 位置需要专门的 RoPE 变体。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2017**：Transformer 提出 Sinusoidal Positional Encoding
- **2018**：GPT-2/BERT 用 Learned Positional Encoding
- **2019**：T5 用 Relative Position Bias（相对位置偏置）
- **2021**：ALiBi（线性偏置，简单外推）
- **2021-04**：RoPE 论文（Su et al.），旋转位置编码
- **2023-02**：LLaMA 用 RoPE，引爆社区
- **2023 中**：PI（线性插值）、NTK-Aware、YaRN 外推技术出现
- **2023 下**：几乎所有开源 LLM 切换到 RoPE
- **2024**：Dynamic NTK RoPE、Multimodal RoPE、NoPE 探索
- **2024-2025**：RoPE 成为 LLM 位置编码事实标准

### 5.2 常见坑

**坑 1: 期望 RoPE 天然无限外推**。RoPE 需要配合插值技术（NTK/YaRN）。直接外推效果差。

**坑 2: 插值方法选错**。线性插值简单但效果一般，NTK/YaRN 效果好。长上下文任务不要省。

**坑 3: 忘记对 Q 和 K 都做旋转**。只转一个得到奇怪的效果。必须两个都转。

**坑 4: 半精度下 cos/sin 精度问题**。FP16 下三角函数误差积累。用 BF16 或在 FP32 下计算旋转。

**坑 5: 维数不是偶数**。RoPE 要求 dim 能配对。奇数维度会报错。

**坑 6: 混合位置编码乱用**。有的模型用 Learnable + RoPE 不好。选一个统一用。

**坑 7: 微调时改 RoPE 参数**。预训练好的 RoPE 频率是固定的，不建议在微调时改。

**坑 8: 推理时忘了设 use_cache**。RoPE 对 KV-Cache 的 cos/sin 要用动态位置。cache 要一起旋转。

**坑 9: YaRN 参数调错**。α/β 参数影响大，乱调效果差。按论文推荐值：α=1, β=32/512。

**坑 10: 期望 RoPE 解决所有位置问题**。RoPE 是 token-level 位置，不支持结构化位置（如 DOM 树、代码 AST）。需要专门位置编码。

**坑 11: 评估只看 PPL**。PPL 好了不一定下游任务好。要在具体任务上评估外推效果（如 LongBench）。

**坑 12: 长上下文任务不验证**。训练 4k 推理 32k，要验证 8k/16k/32k 各档效果，不是一档而过。

### 5.3 面试怎么考

1. **RoPE 的核心思想？** 答：用旋转矩阵编码位置。对 Q 和 K 向量按位置旋转，点积自然包含相对位置差。表面是绝对编码，本质是相对编码。
2. **为什么 RoPE 的 Attention 是相对位置？** 答：旋转矩阵的正交性：$R_m^\top R_n = R_{n-m}$，所以 $q_m^\top R_m^\top R_n k_n = q_m^\top R_{n-m} k_n$，只依赖 $n-m$。
3. **RoPE 为什么能外推？** 答：只依赖相对位置差，训练见过的位置差大部分覆盖推理。配合 NTK/线性/YaRN 插值，平滑扩展。
4. **RoPE vs Sinusoidal vs ALiBi？** 答：Sinusoidal 无相对位置无外推；ALiBi 有相对位置和简单外推但远程衰减太快；RoPE 兼具相对位置、外推、效果好。
5. **NTK-Aware vs YaRN？** 答：NTK 按频率分比例缩放（高频少缩、低频多缩）；YaRN 在 NTK 基础上加温度参数调整注意力分布。YaRN 外推效果最好。

---

## 速记卡

**RoPE 公式**（对位置 $m$ 的第 $i$ 对维度）：

$$
\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

其中 $\theta_i = 10000^{-2i/d}$。

**相对位置**：

$$
(R_{\Theta,m} q_m)^\top (R_{\Theta,n} k_n) = q_m^\top R_{\Theta,n-m} k_n
$$

**外推方法对比**：

| 方法 | 原理 | PPL@32k（Llama-2-7B） |
|------|------|----------------------|
| 无外推 | - | 发散 |
| PI 线性插值 | 统一压缩 | 7.2 |
| NTK | 按频率不同缩放 | 6.1 |
| **YaRN** | NTK + 温度 | **5.8** |

**位置编码对比**：

| 方法 | 相对位置 | 外推 | 开销 | 代表 |
|------|---------|------|------|------|
| Sinusoidal | ❌ | ❌ | 零 | Transformer |
| Learned | ❌ | ❌ | 零 | GPT-2 |
| ALiBi | ✅ | ✅ | 零 | BLOOM |
| **RoPE** | **✅** | **✅** | **小** | **LLaMA/Qwen** |

**RoPE 模型**：LLaMA / Qwen / Mistral / Mixtral / DeepSeek V2/V3 / ChatGLM / Phi-3/4 / Gemma 2

**代码要点**：

```python
def apply_rope(q, k, cos, sin):
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k
```

**一句话记忆**：RoPE = 旋转矩阵位置编码。对 Q 和 K 按位置 $\theta$ 旋转，点积自然依赖相对位置差 $m-n$：表面绝对、本质相对。LLaMA/Qwen/Mistral/DeepSeek 等几乎所有主流 LLM 的标准选择。天然外推，配合 NTK/YaRN 插值从 4k 到 32k+。远程衰减符合语言直觉。局限：需要插值才能远距离外推，半精度下三角函数要小心，多维位置（2D/3D）需变体。

---

> *上一篇：[Multi-Agent 多智能体](./multi-agent) -- Agent 专题末篇，长上下文专题是系统层的能力支撑。*
> *下一篇：[Ring Attention 环注意力](./ring-attention) -- RoPE 解决了位置编码，Ring Attention 解决超长序列的显存和并行。*
