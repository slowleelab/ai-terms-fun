---
title: KV-Cache 键值缓存
slug: kv-cache
category: 进阶专题
tags: [KV-Cache, 推理加速, 自回归, 显存, LLM 推理]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# KV-Cache 键值缓存

> 五层读懂一个词。这次拆的是：**KV-Cache**--LLM 自回归推理的命根子。缓存已计算的 K/V，避免重复计算，把推理从 $O(n^2)$ 降到 $O(n)$。但显存占用爆炸，是所有推理优化的基础。

---

## L1 · 一句话点破

**KV-Cache = 缓存历史 token 的 Key/Value 矩阵**。自回归生成时，新 token 只需算新 K/V，不重算历史。把每步从 $O(n^2)$ 降到 $O(n)$，但显存占用 $O(n \cdot d \cdot L)$，是 LLM 推理的核心 trade-off。

---

## L2 · 通俗类比

LLM 生成是**自回归**：一个 token 一个 token 往外蹦。每生成一个新 token，要算 attention，而 attention 需要**所有历史 token 的 K 和 V**。

**没 KV-Cache 的情况**：

生成第 1000 个 token 时，要把前 999 个 token 的 K/V 全部重算一遍。每步 $O(n^2)$，生成 1000 个 token 总共 $O(n^3)$。70B 模型生成长文本，慢到不可用。

**有 KV-Cache 的情况**：

前 999 个 token 的 K/V 算过一次就缓存，生成第 1000 个 token 时只算新 token 的 K/V，拼到缓存末尾。每步 $O(n)$（只算新 token），总复杂度 $O(n^2)$。

**数字感受**（Llama-70B，batch=1，seq=2048）：

| 场景 | 无 KV-Cache | 有 KV-Cache |
|------|------------|-------------|
| 每步 FLOPs | ~10^13 | ~10^10 |
| 每步延迟 | ~1 秒 | ~1 毫秒 |
| 总生成延迟（1000 token） | ~1000 秒 | ~1 秒 |

KV-Cache 把推理速度提升 **1000 倍**。没有它，LLM 实时推理不可能。

**代价**：显存占用爆炸。

**KV-Cache 显存**（Llama-70B，batch=1，seq=4096）：

```
每层每 token：2 (K+V) × num_heads × head_dim × 2 bytes (BF16)
= 2 × 64 × 128 × 2 = 32 KB

每 token 全部层：32 KB × 80 层 = 2.56 MB

4096 token：2.56 MB × 4096 = 10.5 GB
```

单个请求的 KV-Cache 就要 10GB！batch=32 时 336GB，比模型权重（140GB）还大。

**核心 trade-off**：用显存换算力。KV-Cache 省了计算（FLOPs 降 1000 倍），但吃了显存（KV-Cache 可能比模型还大）。后续所有推理优化（PagedAttention、量化、推测解码）都围绕这个 trade-off 展开。

---

## L3 · 正经定义

**KV-Cache**：LLM 自回归推理中，缓存已计算 token 的 Key 和 Value 矩阵，避免每步重算。每个 token 的 K/V 算一次后存入缓存，新 token 只算自己的 K/V 并拼接到缓存末尾。

**Attention 计算**（带 KV-Cache）：

$$
\text{Attention}(q_{new}, K_{cached}, V_{cached}) = \text{softmax}\left(\frac{q_{new} K_{cached}^T}{\sqrt{d_k}}\right) V_{cached}
$$

其中 $q_{new}$ 是新 token 的 query，$K_{cached}$、$V_{cached}$ 是历史所有 token 的 K/V。

**KV-Cache 的显存占用**：

$$
\text{Memory}_{KV} = 2 \cdot L \cdot n_{kv\_heads} \cdot d_{head} \cdot s \cdot b \cdot \text{dtype\_size}
$$

其中：

- $L$：层数
- $n_{kv\_heads}$：KV head 数（GQA/MQA 下小于 query head 数）
- $d_{head}$：每个 head 维度
- $s$：序列长度
- $b$：batch size
- $\text{dtype\_size}$：2 bytes（FP16/BF16）

**参考资料**：

- 📄 Vaswani et al., *Attention Is All You Need*, NeurIPS 2017（Transformer 原始论文，KV-Cache 隐含其中）
- 📄 Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023（GQA 减少 KV head）
- 📄 Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019（MQA）
- 🔧 vLLM 文档：https://docs.vllm.ai/

---

## L4 · 原理深挖

### 4.1 为什么需要 KV-Cache

自回归生成的本质：每生成一个新 token，attention 要看所有历史 token。

**Attention 公式**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

生成第 $t$ 个 token 时：

- $Q$：第 $t$ 个 token 的 query（1 个向量）
- $K$：前 $t-1$ 个 token 的 key（$t-1$ 个向量）
- $V$：前 $t-1$ 个 token 的 value（$t-1$ 个向量）

**无 KV-Cache**：每步重算前 $t-1$ 个 token 的 K/V，复杂度 $O(t \cdot d)$ per step，总 $O(n^2 d)$。

**有 KV-Cache**：前 $t-1$ 个 token 的 K/V 已缓存，只算第 $t$ 个 token 的 K/V，复杂度 $O(d)$ per step，总 $O(n d)$。

### 4.2 KV-Cache 的前向流程

```python
class AttentionWithKVCache:
    def __init__(self, d_model, num_heads):
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.cache_k = None  # 缓存的历史 K
        self.cache_v = None  # 缓存的历史 V
    
    def forward(self, x_new, use_cache=True):
        # x_new: [batch, 1, d]  只输入新 token
        
        q = self.W_q(x_new)  # [batch, 1, d]
        k_new = self.W_k(x_new)  # [batch, 1, d]
        v_new = self.W_v(x_new)  # [batch, 1, d]
        
        if use_cache and self.cache_k is not None:
            # 拼接历史缓存
            k = torch.cat([self.cache_k, k_new], dim=1)  # [batch, t, d]
            v = torch.cat([self.cache_v, v_new], dim=1)  # [batch, t, d]
        else:
            k, v = k_new, v_new
        
        # 更新缓存
        self.cache_k = k
        self.cache_v = v
        
        # attention
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out
```

**关键点**：

- 每步只算新 token 的 Q/K/V，不重算历史
- 历史 K/V 拼到缓存末尾
- Attention 用新 Q 对全部缓存 K 做 attention

### 4.3 KV-Cache 的显存爆炸

**显存公式**：

$$
\text{Memory}_{KV} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot s \cdot b \cdot \text{dtype}
$$

**Llama-2-70B 示例**：

- $L = 80$ 层
- $n_{kv} = 64$ head（GQA 后）
- $d_h = 128$ 维
- $\text{dtype} = 2$ bytes（BF16）

**单 token KV-Cache**：

$$
2 \times 80 \times 64 \times 128 \times 2 = 2.6 \text{ MB}
$$

**不同序列长度 + batch size**：

| 场景 | seq | batch | KV-Cache 显存 |
|------|-----|-------|--------------|
| 单请求短文本 | 512 | 1 | 1.3 GB |
| 单请求长文本 | 4096 | 1 | 10.5 GB |
| 多请求中等 | 2048 | 32 | 168 GB |
| 多请求长文本 | 4096 | 32 | 336 GB |

**对比模型权重**：70B 模型 BF16 权重 140GB。KV-Cache 在长序列 + 大 batch 时**比模型权重还大**。

### 4.4 GQA / MQA：减少 KV head

KV-Cache 显存和 KV head 数成正比。减少 KV head 能直接省显存。

**MHA（Multi-Head Attention）**：query head 数 = KV head 数（如 64:64）

**MQA（Multi-Query Attention）**：所有 query head 共享 1 个 KV head（64:1）

**GQA（Grouped-Query Attention）**：query head 分组共享 KV head（如 64:8）

**KV-Cache 显存对比**：

| 方案 | KV head | KV-Cache 显存 | 效果 |
|------|---------|--------------|------|
| MHA | 64 | 100% | 基线 |
| GQA (64:8) | 8 | 12.5% | 接近 MHA |
| MQA (64:1) | 1 | 1.5% | 略低于 GQA |

**Llama-2-70B 用 GQA（8 KV head）**，KV-Cache 省 8 倍。Llama-3 也用 GQA。

### 4.5 KV-Cache 的量化

进一步省显存：把 KV-Cache 量化到低精度。

**FP16/BF16 KV-Cache**：基线，2 bytes/element

**INT8 KV-Cache**：1 byte/element，省 50%

**INT4 KV-Cache**：0.5 byte/element，省 75%

**精度损失**：

| 量化 | 显存 | 精度损失 |
|------|------|---------|
| FP16 | 100% | 0% |
| INT8 | 50% | <0.5% |
| INT4 | 25% | ~1-2% |

**实践**：INT8 KV-Cache 几乎无损，INT4 在长序列上有轻微掉点。

### 4.6 KV-Cache 的内存管理

**问题**：不同请求序列长度不同，KV-Cache 大小不一。预分配最大长度浪费显存，动态分配碎片化。

**朴素方案**：为每个请求预分配 `max_seq_len` 的 KV-Cache 空间。

- 浪费：大部分请求用不到 max_seq_len
- 不灵活：无法动态扩容

**PagedAttention 方案**（vLLM）：

- KV-Cache 分成固定大小的 page（如 16 token/page）
- 按需分配 page，类似操作系统的虚拟内存
- 显存利用率从 ~30% 提升到 ~95%

详见下一篇 PagedAttention。

### 4.7 KV-Cache 的生命周期

```
1. Prefill 阶段（处理 prompt）：
   - 计算 prompt 所有 token 的 K/V
   - 填充 KV-Cache
   - 计算密集型（一次算很多 token）

2. Decode 阶段（生成新 token）：
   - 每步算 1 个新 token 的 K/V
   - 拼接到 KV-Cache 末尾
   - 内存密集型（频繁读写缓存）

3. 释放阶段（请求结束）：
   - 释放 KV-Cache 显存
```

**Prefill vs Decode 的不同特性**：

| 阶段 | 计算量 | 显存访问 | 瓶颈 |
|------|--------|---------|------|
| Prefill | 大（一次算 prompt 全部 token） | 少 | 计算 |
| Decode | 小（每次算 1 token） | 大（读全部缓存） | 内存 |

**优化方向**：

- Prefill：算子融合、FlashAttention
- Decode：PagedAttention、 batching、量化 KV-Cache

### 4.8 KV-Cache 的复用

**Prefix Caching**：多个请求共享相同 prefix（如系统提示），prefix 的 KV-Cache 复用。

```
请求1: [系统提示] + 用户问题1
请求2: [系统提示] + 用户问题2
请求3: [系统提示] + 用户问题3
```

系统提示的 KV-Cache 算一次，三个请求复用。省 prefill 计算和显存。

**实践**：vLLM、SGLang 等推理引擎支持 prefix caching，对多请求同系统提示场景效果显著。

### 4.9 KV-Cache 的局限

**局限 1：显存爆炸**。长序列 + 大 batch 时 KV-Cache 比模型还大。

**局限 2：不支持训练**。KV-Cache 是推理专用，训练时 attention 要反向传播，不能缓存。

**局限 3：长序列退化**。序列越长，KV-Cache 越大，decode 阶段读取越慢。

**局限 4：多请求隔离**。不同请求的 KV-Cache 要隔离，管理复杂。

**局限 5：精度量化风险**。INT4/INT8 量化在长序列上可能掉点。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2017**：Transformer 论文，KV-Cache 隐含在自回归解码中
- **2019**：Shazeer 提出 MQA，减少 KV head 省 KV-Cache
- **2023**：GQA（Llama-2）普及，KV-Cache 省 8 倍
- **2023-06**：vLLM 发布 PagedAttention，KV-Cache 内存管理革命
- **2024**：KV-Cache 量化（INT8/INT4）普及，prefix caching 成为标配
- **2025**：KV-Cache 卸载到 CPU/SSD（长序列场景），多级缓存出现

### 5.2 常见坑

**坑 1：忘算 KV-Cache 显存**。只算模型权重显存，部署时 KV-Cache OOM。要预算 KV-Cache = $2 L n_{kv} d_h s b \cdot \text{dtype}$。

**坑 2：batch size 太大 OOM**。KV-Cache 随 batch 线性增长，batch=64 时 KV-Cache 可能比模型大。要限制 batch size。

**坑 3：没用 GQA/MQA**。MHA 模型 KV-Cache 大，用 GQA/MQA 模型（Llama-2/3）省 8-64 倍。

**坑 4：KV-Cache 量化没校准**。INT4 量化直接用，长序列掉点。要校准或用 INT8。

**坑 5：prefill 和 decode 不分流**。prefill 计算密集、decode 内存密集，用同一 batch 策略效率低。要分别优化。

**坑 6：prefix 没复用**。多请求同系统提示，每次重算 prefix KV-Cache。要用 prefix caching。

**坑 7：KV-Cache 碎片化**。朴素内存管理碎片化严重，显存利用率低。要用 PagedAttention。

**坑 8：长序列 decode 慢**。序列越长 decode 越慢（读 KV-Cache 多）。要限制 max_seq_len 或用 sliding window attention。

**坑 9：KV-Cache 卸载延迟大**。卸载到 CPU/SSD 省显存但延迟增加。要权衡。

**坑 10：多请求 KV-Cache 隔离错**。请求间 KV-Cache 串了，生成乱码。要严格隔离。

**坑 11：KV-Cache 没释放**。请求结束后 KV-Cache 没释放，显存泄漏。要有生命周期管理。

**坑 12：batch 内序列长度差异大**。短序列和长序列同 batch，短序列的 KV-Cache 浪费。要用 dynamic batching 或 PagedAttention。

### 5.3 面试怎么考

1. **KV-Cache 为什么能加速？** 答：自回归生成时，历史 token 的 K/V 算过一次就缓存，新 token 只算自己的 K/V，每步从 $O(n^2)$ 降到 $O(n)$。
2. **KV-Cache 的显存占用？** 答：$2 \cdot L \cdot n_{kv} \cdot d_h \cdot s \cdot b \cdot \text{dtype}$，长序列 + 大 batch 时可能比模型权重还大。
3. **GQA/MQA 怎么省 KV-Cache？** 答：减少 KV head 数（64:8 或 64:1），KV-Cache 随 KV head 数线性减少，效果几乎无损。
4. **KV-Cache 的生命周期？** 答：Prefill（填缓存，计算密集）-> Decode（拼新 K/V，内存密集）-> 释放。
5. **KV-Cache 的主要问题？** 答：显存爆炸（长序列 + 大 batch），用 PagedAttention 内存管理 + GQA/MQA 减 head + 量化降精度。

---

## 速记卡

| 阶段 | 操作 | 瓶颈 |
|------|------|------|
| Prefill | 算 prompt 全部 K/V 填缓存 | 计算 |
| Decode | 算新 token K/V 拼缓存 | 内存 |
| 释放 | 释放缓存 | - |

**显存公式**：

$$
\text{Memory}_{KV} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot s \cdot b \cdot \text{dtype}
$$

**KV head 优化**：

| 方案 | KV head | KV-Cache | 效果 |
|------|---------|---------|------|
| MHA | 64 | 100% | 基线 |
| GQA | 8 | 12.5% | 接近 MHA |
| MQA | 1 | 1.5% | 略低 |

**Llama-70B KV-Cache 显存**（BF16）：

| 场景 | 显存 |
|------|------|
| 单 token | 2.6 MB |
| 4096 token × 1 batch | 10.5 GB |
| 4096 token × 32 batch | 336 GB |

**一句话记忆**：KV-Cache = 缓存历史 token 的 K/V，自回归推理每步只算新 token，复杂度从 $O(n^2)$ 降到 $O(n)$，提速 1000 倍。代价是显存爆炸（长序列 + 大 batch 时比模型还大），用 GQA/MQA 减 KV head、量化降精度、PagedAttention 管内存。Prefill 计算密集、Decode 内存密集，是所有推理优化的基础。

---

> *上一篇：[KTO / SimPO 变体](./kto-simpo) -- 对齐专题末篇，推理工程是部署侧的优化。*
> *下一篇：[PagedAttention 分页注意力](./paged-attention) -- KV-Cache 的内存管理革命，vLLM 的核心。*
