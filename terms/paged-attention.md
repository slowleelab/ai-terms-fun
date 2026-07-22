---
title: PagedAttention 分页注意力
slug: paged-attention
category: 进阶专题
tags: [PagedAttention, vLLM, KV-Cache, 虚拟内存, 分页, 显存管理]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# PagedAttention 分页注意力

> 五层读懂一个词。这次拆的是：**PagedAttention**--vLLM 的核心创新，把操作系统的虚拟内存分页思想搬到 KV-Cache 管理上。显存利用率从 ~30% 提升到 ~95%，吞吐量翻 2-4 倍。

---

## L1 · 一句话点破

**PagedAttention = KV-Cache 分成固定大小 page + 按需分配 + 非连续存储**。借鉴 OS 虚拟内存，KV-Cache 不再预分配连续大块，而是分成 16 token/page 的小块按需分配，消除碎片化，显存利用率从 30% 飙到 95%。

---

## L2 · 通俗类比

KV-Cache 的传统管理方式像**给每个请求预订单间会议室**：

- 请求 1 可能只需要 100 token 空间，但预定了 2048 token 的房间（怕不够用）
- 请求 2 需要 2000 token，又预定了 2048 token
- 大量空间浪费（请求 1 用 100，浪费 1948）

更糟的是**碎片化**：请求 1 结束释放 2048 空间，但新请求 3 要 3000，放不下（虽然总空闲够，但不连续）。

PagedAttention 的思路像**操作系统的虚拟内存**：

- 不再预订单间大房间，而是**按需租用小隔间**（16 token/隔间）
- 请求 1 用 100 token = 7 个隔间（向上取整），用多少租多少
- 隔间可以**不连续**（逻辑上连续，物理上分散）
- 请求结束释放隔间，立即可被其他请求复用

**显存利用率对比**：

| 方案 | 预分配 | 实际利用 | 碎片化 |
|------|--------|---------|--------|
| 传统连续分配 | max_seq_len | ~30% | 严重 |
| PagedAttention | 按需 16 token/page | ~95% | 无 |

**vLLM 的实测**（Llama-7B，A10G）：

- 传统 HF Transformers：~30% 显存利用率
- PagedAttention：~95% 显存利用率
- 吞吐量提升：2-4 倍

**额外好处**：

- **Copy-on-write**：多请求共享相同 prefix（如系统提示），只读共享，修改时才复制
- **快速前缀复用**：相同 prefix 的请求共享 KV-Cache page
- **精确内存预算**：知道每个时刻实际用了多少显存

**代价**：page 表管理开销；非连续内存访问稍慢（但 GPU 上的影响小）；实现复杂。

---

## L3 · 正经定义

**PagedAttention**：Kwon et al. (SOSP 2023) 提出（vLLM 论文），借鉴操作系统虚拟内存的分页机制管理 KV-Cache。将 KV-Cache 分成固定大小的 block（通常 16 token/block），按需分配，允许非连续存储，消除内部碎片和外部碎片。

**核心数据结构**：

- **Block（页）**：固定大小（如 16 token）的 KV-Cache 存储单元
- **Block Table（页表）**：每个请求维护一个 block table，记录逻辑 block 到物理 block 的映射
- **Physical Blocks（物理页）**：GPU 上实际存储 KV-Cache 的物理块

**逻辑视图 vs 物理视图**：

```
逻辑视图（请求看到的）：
[token_0, token_1, ..., token_99]  连续

物理视图（实际存储）：
block_0: [token_0..15]   物理地址 A
block_1: [token_16..31]  物理地址 B
block_2: [token_32..47]  物理地址 C
...
block_6: [token_96..99]  物理地址 G（部分填充）
```

**Attention 计算**：通过 block table 查找每个逻辑 block 对应的物理 block，在物理 block 上计算 attention，结果逻辑上等价于连续 KV-Cache。

**参考资料**：

- 📄 Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023, arXiv:2309.06180
- 🔧 vLLM 官方实现：https://github.com/vllm-project/vllm
- 📄 OS 虚拟内存分页机制（PagedAttention 的灵感来源）

---

## L4 · 原理深挖

### 4.1 传统 KV-Cache 管理的问题

**方案 1：预分配 max_seq_len 连续空间**

```python
# 每个请求预分配最大序列长度的连续 KV-Cache
kv_cache = torch.empty(batch_size, max_seq_len, num_layers, 2, num_heads, head_dim)
```

**问题 1：内部碎片**。请求实际只用 100 token，预分配 2048，浪费 1948 token 空间。显存利用率 ~30%。

**问题 2：外部碎片**。请求 1 释放 2048 空间，但新请求要 3000，放不下（不连续）。虽然总空闲够，但无法分配。

**问题 3：无法动态扩容**。预分配后不能扩容，超长请求直接失败。

**方案 2：动态分配连续空间**

```python
# 每个请求按实际长度分配
kv_cache = torch.empty(batch_size, actual_seq_len, ...)
```

**问题**：动态分配在 GPU 上慢，且仍可能外部碎片化。

### 4.2 PagedAttention 的分页机制

借鉴 OS 虚拟内存：

**OS 分页**：

- 进程的虚拟地址空间分成固定大小的页（如 4KB）
- 物理内存分成相同大小的页框
- 页表映射虚拟页到物理页框
- 物理页框可以不连续

**PagedAttention**：

- 请求的逻辑 KV-Cache 分成固定大小的 block（如 16 token）
- GPU 物理显存分成相同大小的物理 block
- Block table 映射逻辑 block 到物理 block
- 物理 block 可以不连续

**Block Table 示例**：

```
请求 A 的 block table: [物理块 #7, #2, #15, #8]
请求 B 的 block table: [物理块 #3, #9, #2]  ← #2 和 A 共享
```

**优势**：

- 按需分配：用多少 token 分多少 block，无内部碎片
- 非连续存储：物理 block 可以分散，无外部碎片
- 动态扩容：序列变长就加 block，无需预分配

### 4.3 PagedAttention 的 Attention 计算

传统 attention 假设 K/V 连续存储。PagedAttention 要在非连续 block 上算 attention：

```python
def paged_attention(q_new, block_table, physical_blocks, num_tokens):
    """
    q_new: [batch, 1, num_heads, head_dim] - 新 token 的 query
    block_table: [batch, num_blocks] - 逻辑到物理 block 映射
    physical_blocks: [total_physical_blocks, block_size, num_heads, head_dim]
    num_tokens: [batch] - 每个请求的实际 token 数
    """
    output = torch.zeros_like(q_new)
    
    for block_idx in range(num_blocks):
        # 1. 查 block table，获取物理 block
        physical_idx = block_table[:, block_idx]  # [batch]
        # 2. 取物理 block 的 K/V
        k_block = physical_blocks[physical_idx]  # [batch, block_size, num_heads, head_dim]
        v_block = physical_blocks[physical_idx]
        # 3. 在 block 上算 attention
        attn = torch.einsum('bhd,bshd->bhs', q_new, k_block) / math.sqrt(head_dim)
        attn = F.softmax(attn, dim=-1)
        output += torch.einsum('bhs,bshd->bhd', attn, v_block)
    
    return output
```

**关键**：attention 在每个 block 上独立算，结果累加。数学上等价于连续 KV-Cache 的 attention（softmax 的分母是各 block 的指数和）。

**实现优化**：用 CUDA kernel 融合 block 遍历和 attention，减少 kernel launch 开销。

### 4.4 Copy-on-Write 与 Prefix Sharing

**Prefix Sharing**：多个请求共享相同 prefix（如系统提示）的 KV-Cache。

```
请求 A: [系统提示] + 问题 A
请求 B: [系统提示] + 问题 B
```

系统提示的 KV-Cache 算一次，A 和 B 共享这些 block（block table 指向同一物理 block）。

**Copy-on-Write（COW）**：

- 共享 block 是只读的
- 某个请求要修改共享 block 时，复制一份再改
- 类似 OS 的 fork() + COW

**Beam Search 的应用**：

- Beam search 的多个候选共享前缀
- 用 COW 避免重复存储
- 显存省 10-100 倍

### 4.5 PagedAttention 的显存利用率

**传统方案**：

- 预分配 max_seq_len，实际用 ~30%
- 碎片化损失 ~20%
- 总利用率 ~30%

**PagedAttention**：

- 按需分配，无内部碎片
- 非连续存储，无外部碎片
- block 大小 16 token，最后不满一个 block 的浪费 ~8 token
- 总利用率 ~95%

**实测**（vLLM 论文，Llama-7B）：

| 方案 | 显存利用率 | 吞吐量 |
|------|-----------|--------|
| HF Transformers | ~30% | 基线 |
| TGI（动态分配） | ~60% | 1.5x |
| **PagedAttention** | **~95%** | **2-4x** |

### 4.6 PagedAttention 的 block 大小选择

block 大小是关键超参：

**block 太大（如 256 token）**：

- 最后不满一个 block 的浪费多
- 内部碎片增大
- 但 block table 小，管理开销低

**block 太小（如 1 token）**：

- 几乎无浪费
- 但 block table 巨大，管理开销高
- kernel launch 开销大

**经验值**：16 token 是 vLLM 默认，平衡利用率和开销。

**不同 block 大小的效果**：

| block size | 利用率 | 开销 |
|-----------|--------|------|
| 1 | ~100% | 极高 |
| 16 | ~95% | 低 |
| 64 | ~90% | 更低 |
| 256 | ~80% | 最低 |

### 4.7 PagedAttention 的实现：vLLM

vLLM 是 PagedAttention 的 flagship 实现：

**架构**：

```
用户请求
    ↓
Scheduler（调度器，决定哪些请求进 batch）
    ↓
Block Manager（管理物理 block 分配）
    ↓
PagedAttention Kernel（CUDA kernel，在 block 上算 attention）
    ↓
生成 token
```

**关键组件**：

1. **Scheduler**：决定哪些请求在当前 step 处理，优先级调度
2. **Block Manager**：分配/释放物理 block，维护 block table
3. **PagedAttention Kernel**：CUDA kernel，高效在 block 上算 attention
4. **Prefix Cache**：自动检测共享 prefix，复用 block

### 4.8 PagedAttention 的局限

**局限 1：kernel 实现复杂**。PagedAttention 的 CUDA kernel 比传统 attention 复杂，要处理非连续访问。

**局限 2：小 batch 效果不显著**。batch=1 时，KV-Cache 碎片化问题不严重，PagedAttention 优势不明显。

**局限 3：block 大小调参**。block 太大浪费，太小开销高，要按场景调。

**局限 4：不支持所有 attention 变体**。某些 attention 变体（如 sliding window）在 paged 实现上更复杂。

**局限 5：CPU offload 场景复杂**。KV-Cache offload 到 CPU 时，block 管理更复杂。

### 4.9 PagedAttention 的影响

PagedAttention 是 LLM 推理服务的**里程碑**：

- vLLM 成为开源 LLM 推理服务事实标准
- TGI、SGLang、TensorRT-LLM 等纷纷借鉴分页思想
- 商业服务（Anthropic、OpenAI）也在用类似优化
- 推理成本降低 2-4 倍，加速 LLM 普及

---

## L5 · 沿革与坑

### 5.1 沿革

- **2023-06**：vLLM 项目启动，PagedAttention 概念成型
- **2023-09**：SOSP 论文发表，PagedAttention 引爆社区
- **2023 下半年**：vLLM 成为开源 LLM 推理服务首选
- **2024**：TGI、SGLang 等推理引擎借鉴分页思想
- **2024-2025**：PagedAttention 成为 LLM 推理标配，商业服务也采用类似优化

### 5.2 常见坑

**坑 1：block size 设错**。太大浪费显存，太小开销高。16 是默认，长序列场景可调到 32-64。

**坑 2：小 batch 用 PagedAttention 期望大提升**。batch=1 时碎片化不严重，PagedAttention 优势小。它是为多 batch 高并发设计的。

**坑 3：prefix 没复用**。多请求同系统提示没用 prefix caching，白白重算。要开 prefix cache。

**坑 4：显存预算没算 block table**。block table 本身占显存，大 batch + 长序列时不可忽略。

**坑 5：动态 batch 没配合**。PagedAttention 要配合 continuous batching（下一篇），用静态 batch 浪费优势。

**坑 6：attention 变体不兼容**。某些 attention 变体（sliding window、ALiBi）在 PagedAttention 实现上不完整。要确认支持。

**坑 7：CPU offload 场景没优化**。KV-Cache offload 到 CPU 时，block 管理开销放大。要专门优化。

**坑 8：只看吞吐不看延迟**。PagedAttention 提升吞吐，但单请求延迟可能略增（block 管理开销）。要按场景权衡。

**坑 9：多模型混部 block 冲突**。多个模型共享 GPU 时，block 池冲突。要隔离。

**坑 10：COW 没触发**。以为共享了 prefix，实际因为请求顺序问题没触发 COW。要预热 prefix cache。

### 5.3 面试怎么考

1. **PagedAttention 解决什么问题？** 答：KV-Cache 的内部碎片（预分配浪费）和外部碎片（不连续无法分配），用分页机制按需分配，利用率从 30% 提到 95%。
2. **PagedAttention 的核心思想？** 答：借鉴 OS 虚拟内存，KV-Cache 分成固定大小 block，按需分配，非连续存储，block table 映射逻辑到物理。
3. **PagedAttention 怎么在非连续 block 上算 attention？** 答：每个 block 独立算 attention（softmax 分母是各 block 指数和），结果累加，数学等价于连续 attention。
4. **PagedAttention 的 prefix sharing？** 答：多请求共享相同 prefix 的 KV-Cache block，block table 指向同一物理 block，修改时 copy-on-write。
5. **PagedAttention 的 block size 怎么选？** 答：16 是默认，平衡利用率（大 block 浪费）和管理开销（小 block 开销高）。

---

## 速记卡

| 概念 | 类比 OS |
|------|---------|
| 逻辑 block | 虚拟页 |
| 物理 block | 物理页框 |
| block table | 页表 |
| 按需分配 | 动态分配 |
| 非连续存储 | 虚拟内存 |
| Copy-on-Write | fork() COW |

**显存利用率对比**：

| 方案 | 利用率 | 吞吐量 |
|------|--------|--------|
| HF Transformers | ~30% | 基线 |
| TGI | ~60% | 1.5x |
| **PagedAttention** | **~95%** | **2-4x** |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| block size | 16 token | 利用率 vs 开销 |
| GPU 内存 | 物理block池 | 容量 |
| prefix cache | 开启 | 复用共享前缀 |

**一句话记忆**：PagedAttention = KV-Cache 分页（16 token/block）+ 按需分配 + 非连续存储 + block table 映射。借鉴 OS 虚拟内存，消除内部/外部碎片，显存利用率从 30% 飙到 95%，吞吐量 2-4 倍。vLLM 的核心创新，LLM 推理服务里程碑。配合 prefix sharing + COW + continuous batching 效果最佳。

---

> *上一篇：[KV-Cache 键值缓存](./kv-cache) -- PagedAttention 是 KV-Cache 的内存管理革命。*
> *下一篇：[Continuous Batching 连续批处理](./continuous-batching) -- 配合 PagedAttention，动态拼 batch 提升吞吐。*
