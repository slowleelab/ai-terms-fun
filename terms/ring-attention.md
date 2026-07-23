---
title: Ring Attention 环注意力
slug: ring-attention
category: 进阶专题
tags: [Ring Attention, 序列并行, 超长上下文, 分布式注意力, FlashAttention]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# Ring Attention 环注意力

> 五层读懂一个词。这次拆的是：**Ring Attention**--多 GPU 分布式计算超长序列注意力的核心范式。把序列均匀切分到多张 GPU，GPU 排列成环形，每个 GPU 持有自己的 Q 分片，K/V 沿环旋转传递。每张卡只算一段注意力，显存随 GPU 数线性扩展——4 张卡≈4 倍上下文长度。

---

## L1 · 一句话点破

**Ring Attention = 序列并行 + 环形通信**。把长序列均匀切分到 N 张 GPU，GPU 成环形拓扑。每个 GPU 持有自己的 Q 块，K/V 块沿环传递（每步传一个块），每步算一段局部注意力。显存从 $O(L^2)$（全序列都在单卡上算）变为 $O(L^2/N)$（每卡只需一个块）。PagedAttention 管理单卡内 KV-Cache，Ring Attention 管理跨卡序列——两者配合，百万 token 上下文不再遥不可及。

---

## L2 · 通俗类比

Transformer 的 Self-Attention 是 **$O(L^2)$ 的显存杀手**：

- 序列长度翻倍，注意力矩阵翻 4 倍（$L \times L$）
- 1M token 的注意力矩阵：$10^{12}$ 元素 ≈ 4TB（FP32），单卡根本装不下

**Ring Attention 像「环形流水线组装」**：

- 把长长的序列切分成 N 段，像拆成 N 段流水线
- N 个工人（GPU）站成一个环
- 每个工人手持一段 Q（自己的问题），K/V 块沿环传递
- 每人每步收到一个 K/V 块，算一段注意力，再传给下一个
- 所有块传一圈后，每人算出自己的完整注意力输出

**类比：传阅文件批注**

想象一个环形的会议室，每人手上有自己的「问题列表」（Q），一份「参考材料」（K/V）在圈内传递：

- 第 1 步：A 手持 K1,V1，A 用自己的 Q 和 K1,V1 算一份注意力
- 第 2 步：A 把 K1,V1 传给 B，A 收到 K2,V2，A 继续算
- ...
- 转完一圈，A 用所有 K/V 算完了完整注意力

**关键：每个人同时在工作**，只是用不同的 K/V 块——没有任何人闲着。

**为什么显存降了 N 倍**：

- 单卡：全序列 K/V 都在一张卡上 → $O(L^2)$
- Ring Attention：每卡只有 $1/N$ 的 K/V → $O(L^2 / N)$
- 4 张卡 ≈ 4 倍上下文长度

**配合 Blockwise 计算（FlashAttention 风格）**：

- 不在显存中建 $L \times L$ 注意力矩阵
- 每个块在线计算 softmax（用 Online Softmax / Tiling）
- 进一步压缩显存

**对比**：

| 方案 | 显存（注意力部分） | 通信 | 适用 |
|------|-------------------|------|------|
| 全序列单卡 | $O(L^2)$ | 无 | 短序列 |
| 序列并行（All-Gather） | $O(L^2)$ | 大 | 中长序列 |
| **Ring Attention** | **$O(L^2/N)$** | **小（环形）** | **超长序列** |

**代价**：

- 环形通信延迟（每步等 K/V 块传过来）
- 负载均衡：各卡序列长度要均匀
- 实现复杂：要配合 FlashAttention 的 Online Softmax

**适用**：

- 超长序列（>32k token）
- 多 GPU 分布式推理/训练
- 配合 FlashAttention 和 PagedAttention

---

## L3 · 正经定义

**Ring Attention**（环注意力）：由 Liu et al. 2023 提出的分布式注意力计算方法。将输入序列沿长度维度切分成 N 个块，分配到 N 个 GPU（排列成逻辑环）。每个 GPU 持有本地 Q 块作为「驻留」数据，K/V 块沿环传递。第 $t$ 步，GPU $i$ 收到来自 GPU $i-1$ 的 K/V 块 $K_{(i-t)\%N}, V_{(i-t)\%N}$，用本地 Q 块与该 K/V 块计算局部注意力，然后将此 K/V 块发送给 GPU $i+1$。N 步后所有 GPU 完成全序列注意力。

**核心思想**：

- **序列分片**：$L \rightarrow N$ 个等长块
- **环形传递**：K/V 块顺时针传递，每卡每步收到一个新块
- **局部计算**：每卡每步只算 $Q_{\text{local}} \times K_{\text{received}}$ 的局部注意力
- **Online Softmax**：配合 FlashAttention 的 Tiling 机制，在线累积统计量

**参考资料**：

- 📄 Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context*, 2023
- 📄 Liu et al., *World Record in Long-Context: 10M Token Generation with RingAttention*, 2024
- 📄 Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention*, 2022（基础组件）
- 🔧 FlashAttention：https://github.com/Dao-AILab/flash-attention
- 🔧 RingFlashAttention：https://github.com/lhao499/RingAttention

---

## L4 · 原理深挖

### 4.1 全序列 Attention 的显存困境

**Standard Attention 的显存**（训练/推理）：

- $Q, K, V \in \mathbb{R}^{L \times d}$，各 $Ld$ 元素
- 注意力分数 $S = QK^\top \in \mathbb{R}^{L \times L}$，$L^2$ 元素
- 显存 = $O(L^2 + Ld)$，当 $L$ 很大时 $L^2$ 主导

**例子**（FP16）：

| L | 注意力显存 |
|---|----------|
| 4k | 32 MB |
| 32k | 2 GB |
| 128k | 32 GB |
| 1M | 2 TB |

**FlashAttention 解决了单卡上不显式存储 $S$ 的问题**，但仍需要存取所有 K/V（$Ld$）。

**Ring Attention 解决的是**：K/V 太大，单卡放不下（或放得下但浪费）。

### 4.2 Ring Attention 的算法

**设置**：

- N 个 GPU，编号 0 到 N-1
- 序列长度 L，均匀分成 N 个块，每个块长度 $l = L/N$
- GPU $i$ 持有 $Q_i, K_i, V_i$（第 $i$ 块）

**算法**：

```
For step t = 0, 1, ..., N-1:
    1. GPU i 收到 K/V 块:
       接收来自 GPU (i-1)%N 的 K_send, V_send
    
    2. 计算局部注意力:
       O_i += FlashAttention(Q_i, K_send, V_send)
       (Online Softmax 累积统计量)
    
    3. 传递 K/V 块:
       如果是 t == 0: 发送 K_i, V_i
       否则: 发送刚收到的 K_send, V_send
       发给 GPU (i+1)%N
```

**可视化**（N=4, 每个 K/V 块沿环传递）：

```
Step 0: GPU 0 用 Q_0, K_0, V_0 算  | K_0 传给 GPU 1
        GPU 1 用 Q_1, K_1, V_1 算  | K_1 传给 GPU 2
        GPU 2 用 Q_2, K_2, V_2 算  | K_2 传给 GPU 3
        GPU 3 用 Q_3, K_3, V_3 算  | K_3 传给 GPU 0

Step 1: GPU 0 用 Q_0, K_3, V_3 算  | K_3 传给 GPU 1
        GPU 1 用 Q_1, K_0, V_0 算  | K_0 传给 GPU 2
        GPU 2 用 Q_2, K_1, V_1 算  | K_1 传给 GPU 3
        GPU 3 用 Q_3, K_2, V_2 算  | K_2 传给 GPU 0

... N 步后，所有 GPU 完成全序列 Attention
```

**关键性质**：

- N 步后，每张 GPU 的 $Q_i$ 和所有 K/V 块都算过 → 完整的 Attention 输出
- 每张 GPU 每步同时计算 → 无空闲 GPU
- 通信和计算可重叠（异步）

### 4.3 和 FlashAttention 的融合

Ring Attention 需要 FlashAttention 的 Tiling 机制来处理 Online Softmax。

**Online Softmax 回顾**：

标准 softmax 需要全局 max，分块计算时需要保存统计量：

$$
\text{softmax}(x) = \frac{\exp(x - m)}{\sum \exp(x - m)}
$$

FlashAttention 的分块策略：

```
对每个 K/V 块:
  计算局部 max, 局部 sum
  重缩放之前累积的输出
  累积新块
```

Ring Attention 复用这个分块策略，每个 step 传入的 K/V 块就是一个「分块」：

```python
def ring_attention_step(Q_i, K_received, V_received, O_i, m_i, l_i):
    """
    Q_i: 本地 Q 块
    K_received, V_received: 当前 step 收到的 K/V
    O_i, m_i, l_i: 累积的输出、max、sum
    """
    # FlashAttention 分块计算
    S_i = Q_i @ K_received.T / sqrt(d)  # [l, l]
    
    m_new = max(m_i, S_i.max())
    l_new = exp(m_i - m_new) * l_i + S_i.exp().sum()
    
    O_i = exp(m_i - m_new) * O_i + S_i.exp() @ V_received
    O_i = O_i / l_new  # 最后的 rescale
    
    return O_i, m_new, l_new
```

**最终输出**：O_i 是完整的 Attention 输出（$Q_i$ 对全序列 K 的注意力）。

### 4.4 通信分析

**通信量**：

- 每步每张 GPU 发送和接收一个 K/V 块
- 每个块大小：$l \times d$ 元素
- N 步总通信：每卡发送和接收 $(N-1) \times l \times d$ 元素 = 全序列 K/V 各一遍

**通信复杂度**：

- 总通信量 $\approx 2Ld$ / GPU（接收 K + V）
- 无额外内存开销

**对比 All-Gather**：

- All-Gather K/V：每卡接收全部 K/V（$Ld$），但发送自己的 K/V 给所有卡（总通信 $NLd$）
- Ring Attention：每卡接收和发送 $Ld$ 元素，总通信 $2Ld$ / GPU（与 N 无关）
- Ring Attention 通信量更优，尤其 N 大时

**通信-计算重叠**：

- 在发送/接收 K/V 的同时计算局部注意力
- 环形拓扑天然支持异步通信

### 4.5 Ring Attention 的局限性

**局限 1: 计算-通信不完美平衡**

- Attention 的计算复杂度 $O(l^2 d)$ 随 l 增大
- l 太大（大块）-> 计算重，通信轻
- l 太小（小块）-> 通信重，计算轻
- 需要选择合适的块大小

**局限 2: 因果注意力浪费**

- Causal Attention 中 $Q_i$ 只能 attend 到 $K_{≤i}$
- 环形传递时，后面的 K 块对前面的 Q 无用
- 浪费 50% 通信/计算（收发了用不上的 K/V）

**优化**：Striped Attention 或 Zigzag Ring（只发送需要的 K/V）

**局限 3: 负载均衡**

- 各 GPU 的序列长度要均匀，否则有些卡空闲等别人
- Padding 到等长浪费计算

**局限 4: 只解决了 Attention**

- 前馈网络（FFN）仍需完整序列长度
- FFN 在各卡独立计算（无通信）
- 但每卡的序列长度仍是 $l$

**局限 5: 实现复杂**

- 需要自定义 CUDA kernel（环形传递 + FlashAttention Tiling）
- 相比标准 Attention 实现，工程量大

### 4.6 实际效果

**单卡 vs Ring Attention（8xA100 80G, Llama-2-7B）**：

| 上下文长度 | 单卡显存 | Ring Attention (8 GPU) | 加速比 |
|-----------|---------|----------------------|--------|
| 32k | 45 GB | 12 GB/卡 | 1x |
| 128k | OOM | 25 GB/卡 | - |
| 256k | OOM | 45 GB/卡 | - |
| 1M | OOM | 72 GB/卡 | - |

**世界纪录**：Liu et al. 2024 用 Ring Attention 在 256 GPU 上实现 1,000 万 token 上下文生成。

### 4.7 与 PagedAttention / FlashAttention 的关系

| 技术 | 解决的问题 | 层次 |
|------|-----------|------|
| FlashAttention | 单卡 Attention 的显存和速度 | Kernel 层 |
| PagedAttention | 单卡内 KV-Cache 的内存碎片化 | 内存管理层 |
| **Ring Attention** | **跨多 GPU 的序列分布式** | **分布式层** |

三者可组合：FlashAttention 做分块计算，PagedAttention 管理 KV-Cache 块，Ring Attention 管理跨卡块调度——共同支撑百万 token 上下文。

### 4.8 变体与进阶

**Striped Attention**：Ring Attention + Striped 排列（交错分片），减少因果浪费。

**Tree Attention**：图拓扑替代环形，减少通信步数（$O(\log N)$ 步 vs $O(N)$ 步）。

**DistFlashAttention**：基于 FlashAttention 的序列并行（LightSeq, Megatron-LM 等使用）。

**LongContextServe**：推理场景的 Ring Attention + PagedAttention + Continuous Batching 联合优化。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2022-05**：FlashAttention（Dao et al.），单卡高效 Attention kernel
- **2023-03**：FlashAttention-2，提升 GPU 利用率至 70%
- **2023-10**：Ring Attention 论文（Liu et al.），环形序列并行
- **2023-11**：Striped Attention，改进因果注意力浪费
- **2024-02**：10M Token 上下文世界纪录（Liu et al.）
- **2024 中**：Ring Attention 集成到主流框架（vLLM, SGLang, Megatron-LM）
- **2024-2025**：Ring Attention + PagedAttention + Continuous Batching 联合方案成为超长上下文推理标配

### 5.2 常见坑

**坑 1: 短序列用 Ring Attention**。序列 < 16k 时，单卡够用，Ring Attention 通信开销反而不划算。要评估 ROI。

**坑 2: 块大小不合适**。块太大计算多、通信少；块太小通信多、计算少。要按 GPU 数、带宽、计算力调。

**坑 3: 因果注意力浪费**。标准 Ring Attention 收发了 50% 无用的 K/V（Q 不能 attend 后面的 K）。用 Striped Attention 或其他变体。

**坑 4: 没有通信-计算重叠**。同步收发 K/V 时 GPU 闲置。要异步通信 + CUDA Stream 重叠。

**坑 5: 认为 Ring Attention 解决一切长上下文**。它只解决 Attention 的显存。FFN、Embedding、Loss 都可能成瓶颈。

**坑 6: GPU 数不是 2 的幂**。环形拓扑对任意 N 有效，但 N 是 2 的幂时通信最规整、效率最高。

**坑 7: 序列不能均匀切分**。长句/短句混排导致负载不均。要有动态 padding 或动态分片。

**坑 8: 推理和训练 Ring Attention 需要不同的优化**。推理有 KV-Cache、prefill/decode 特点，不能直接用训练版本的 Ring Attention。

**坑 9: FP16 累积误差**。多步累积 Online Softmax 统计量，FP16 下精度可能不够。关键统计量用 FP32。

**坑 10: 期望 PagedAttention + Ring Attention 自动配合**。两者需要专门整合：PagedAttention 管理块表在单卡内，Ring Attention 要知道哪些块在自己的 K/V 分片中。

**坑 11: 忽略通信带宽**。NVIDIA NVLink（900 GB/s）与 PCIe（64 GB/s）差异巨大。Ring Attention 在 PCIe 环境下通信成为瓶颈。

**坑 12: 超长上下文任务不验证实际效果**。能跑 1M token 不一定在 1M token 上有用。要配合 Needle-in-a-Haystack 等长上下文评测。

### 5.3 面试怎么考

1. **Ring Attention 解决什么问题？** 答：超长序列 Attention 单卡显存放不下。Ring Attention 序列切分到 N 张 GPU，K/V 沿环传递，每卡只算一段注意力。显存从 $O(L^2)$ 降到 $O(L^2/N)$。
2. **Ring Attention 的工作流？** 答：N 个 GPU 环排列，每卡持有 Q 块 + K/V 块。每步收发一个 K/V 块，算一段局部注意力（FlashAttention Tiling），N 步后完成全序列。
3. **为什么说 Ring Attention 显存降 N 倍？** 答：每卡只持有 $1/N$ 的 K/V，注意力计算只涉及当前块。不需要存储完整 K/V。
4. **Ring Attention 和 FlashAttention 的关系？** 答：FlashAttention 做单卡内的分块计算和 Online Softmax；Ring Attention 做跨卡序列分片和环形通信。Ring Attention 用 FlashAttention 作为每个 step 的计算 kernel。
5. **因果注意力下 Ring Attention 的浪费？** 答：Q_i 不能 attend K_{j>i}，所以发送到前面的 K 块对后面的 Q 无用。Striped Attention 优化了这一点。

---

## 速记卡

**Ring Attention 算法**：

```
N 卡环形排列，序列 L 均分 N 块

For step t = 0 to N-1:
    1. 接收 K/V 块 (来自前一张卡)
    2. 局部注意力: O_i += FlashAttn(Q_i, K_recv, V_recv)
    3. 发送 K/V 块 (给下一张卡)
```

**显存对比**：

| 方法 | 显存（Attention） |
|------|------------------|
| 全序列单卡 | $O(L^2)$ |
| Ring Attention (N GPU) | $O(L^2 / N)$ |

**通信量**：

- 每卡总发送/接收 ≈ $2Ld$（K+V 各一遍）
- 与 GPU 数 N 无关
- 通信和计算可重叠

**实际效果（Llama-2-7B）**：

| 上下文 | 单卡 | 8卡 Ring |
|--------|------|---------|
| 32k | 45 GB | 12 GB/卡 |
| 128k | OOM | 25 GB/卡 |
| 1M | OOM | 72 GB/卡 |

**与前序技术的配合**：

| 技术 | 层次 |
|------|------|
| FlashAttention | 单卡 kernel（Tiling + Online Softmax） |
| PagedAttention | 单卡内存管理（KV-Cache 分页） |
| **Ring Attention** | **跨卡分布式（序列并行）** |

**因果注意力浪费**：标准 Ring Attention 50% 通信/计算浪费（Q 不能 attend 后面的 K）。用 Striped Attention / Zigzag 改进。

**一句话记忆**：Ring Attention = 环形序列并行。N 个 GPU 成环，K/V 块沿环传递，每卡每步算一段局部注意力（FlashAttention 风格），N 步完成全序列。显存 $O(L^2/N)$，4 张卡≈4 倍上下文。配合 FlashAttention（Tiling）+ PagedAttention（KV-Cache 管理）成为超长上下文推理的标配。挑战：因果注意力浪费 50%、通信-计算平衡、负载均衡。世界纪录：10M token。超长序列（>32k）多 GPU 部署的首选方案。

---

> *上一篇：[RoPE 旋转位置编码](./rope) -- RoPE 定位置，Ring Attention 解决超长序列的显存和并行。*
> *下一篇：[Lost in the Middle 中间迷失](./lost-in-the-middle) -- 长上下文有了，但模型能有效利用吗？中间的信息容易被忽略。*
