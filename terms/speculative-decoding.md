---
title: Speculative Decoding 推测解码
slug: speculative-decoding
category: 进阶专题
tags: [Speculative Decoding, 投机解码, 延迟优化, 草稿模型, 推理加速]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# Speculative Decoding 推测解码

> 五层读懂一个词。这次拆的是：**Speculative Decoding**--用小模型猜、大模型批改，把自回归推理的串行魔咒打破，单请求延迟降 2-3 倍，吞吐不降反升。

---

## L1 · 一句话点破

**Speculative Decoding = 小模型草拟 + 大模型并行批改**。用一个小的「draft model」快速猜 γ 个 token，大模型一次前向并行验证，命中的全部白嫖。把自回归的串行瓶颈改成局部并行，单请求延迟降 2-3x。

---

## L2 · 通俗类比

LLM 自回归生成像**一个人一字一字写文章**：

- 每写一个字都要查字典、斟酌（一次前向）
- 想写 100 字要查 100 次字典
- GPU 大部分时间在等这一次次的串行前向

**Speculative Decoding 像秘书和老板配合**：

- **秘书**（draft model，小模型）：快速打草稿，一次写 γ 个字
- **老板**（target model，大模型）：看草稿，一次批改 γ 个字
- 草稿写得对，老板直接签字（接受）
- 草稿写错，老板从第一个错处改起，后面的全废
- 因为老板是「并行批改 γ 个字」（一次前向算 γ 个位置的 logits），成本和批改 1 个字差不多

**关键洞察**：

- 大模型一次前向的成本，算 1 个 token 和算 γ 个 token 几乎一样（计算利用率更高）
- 小模型猜得快，命中率 50-80%
- 平均下来，一次大模型前向能产出 2-4 个有效 token

**举个数**（γ=4，命中率 75%）：

- 小模型花 0.2 个大模型 step 的时间，猜出 4 个 token
- 大模型一次前向验证 4 个 token，成本 = 1 step
- 命中 3 个，第 4 个被否，大模型补 1 个正确 token
- 总产出：3 + 1 = 4 token，成本：1.2 step
- 对比传统：4 token 要 4 step
- **加速比 4 / 1.2 ≈ 3.3x**

**代价**：

- 小模型可能猜错，浪费大模型算力（但批改本身不亏）
- 需要一个相关的小模型（蒸馏、同家族）
- 实现复杂，要处理接受/拒绝、采样一致性
- 小模型和大模型分布差异大时命中率低

**适用**：

- 单请求低延迟场景（对话、实时翻译）
- 有可用小模型（同家族小版本、蒸馏模型）
- 小模型和大模型输出分布相近

---

## L3 · 正经定义

**Speculative Decoding**（又称 Speculative Sampling / 投机解码）：LLM 推理加速技术。用一个小的 **draft model** $q$ 自回归生成 γ 个候选 token（speculative phase），再用大的 **target model** $p$ 一次前向并行计算这 γ+1 个位置的 logits（verification phase）。按接受规则决定接受前 k 个 token（k ≤ γ），并从第 k+1 个位置采样一个 token 作为修正。

**核心特性**：

- **数学等价**：输出分布与纯用大模型完全一致（lossless）
- **延迟降低**：单请求 2-3x
- **吞吐不降**：大模型前向计算利用率更高
- **无需训练**：现成模型即可，不改变参数

**两阶段**：

```
Speculative Phase（草拟）:
    draft model q 自回归生成 γ 个 token: x_1, x_2, ..., x_γ

Verification Phase（验证）:
    target model p 一次前向算 [prompt, x_1, ..., x_γ] 的 logits
    得到 p(x_{i+1} | x_{≤i}) 对每个位置

Accept/Reject（接受规则）:
    for i = 1 to γ:
        r = uniform(0, 1)
        if r < min(1, p(x_i) / q(x_i)):  # 接受
            keep x_i
        else:                              # 拒绝
            从 max(0, p - q) 分布采样修正 token
            break
    如果全部接受，从 p 分布多采一个 token（bonus）
```

**参考资料**：

- 📄 Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023（Google，原始论文）
- 📄 Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023（DeepMind，独立提出）
- 📄 Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, 2024（多头并行草拟）
- 📄 Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, 2024（特征级草拟）
- 🔧 vLLM spec decoding：https://docs.vllm.ai/en/latest/features/spec_decode.html

---

## L4 · 原理深挖

### 4.1 自回归的串行瓶颈

LLM 生成是**严格串行**：

```
x_1 = sample(p(· | prompt))
x_2 = sample(p(· | prompt, x_1))
x_3 = sample(p(· | prompt, x_1, x_2))
...
```

每个 token 依赖前一个 token，无法并行。

**GPU 利用率问题**：

- decode 阶段 batch=1，每步只算 1 个 token
- GPU 计算单元大量闲置（memory-bound）
- 一次前向算 1 个 token 和算 8 个 token 的时间几乎一样

**核心洞察**：如果能让大模型一次前向算多个位置，就能白嫖并行度。

### 4.2 Speculative Decoding 的两阶段

**阶段 1：Speculative（草拟）**

draft model $q$ 自回归生成 γ 个 token：

```
x_1 ~ q(· | prompt)
x_2 ~ q(· | prompt, x_1)
...
x_γ ~ q(· | prompt, x_1, ..., x_{γ-1})

draft 序列: [x_1, x_2, ..., x_γ]
```

draft model 小，每步快（例如 0.2x 大模型 step 时间）。

**阶段 2：Verification（验证）**

target model $p$ 一次前向，并行计算所有 γ+1 个位置：

```
输入: [prompt, x_1, x_2, ..., x_γ]
大模型一次前向，得到每个位置的 logits:
    p(· | prompt)        -> 用于验证 x_1
    p(· | prompt, x_1)   -> 用于验证 x_2
    ...
    p(· | prompt, x_1, ..., x_γ)  -> bonus 采样
```

**关键**：一次前向，γ+1 个位置的 logits 全部得到，成本约等于 1 step。

### 4.3 接受规则（数学等价的关键）

为什么 spec decoding 的输出分布和纯用大模型**完全一致**？

**接受规则**（对每个候选 token $x_i$）：

$$
\text{accept } x_i \text{ with prob } \min\left(1, \frac{p(x_i)}{q(x_i)}\right)
$$

**拒绝时**：从 $\max(0, p(\cdot) - q(\cdot))$ 归一化分布采样修正 token。

**证明思路**（rejection sampling 的变体）：

- 接受概率正比于 $p(x_i)$（大模型概率）
- 拒绝后修正采样填补缺口
- 最终分布：$p(\cdot)$

**数学等价性**：对任意 $x$，

$$
P(\text{output} = x) = p(x)
$$

即输出分布严格等于大模型分布，**无损加速**。

### 4.4 加速比分析

**符号**：

- $\gamma$：draft 长度
- $\alpha$：每个 token 的平均接受概率
- $c_d$：draft model 单 step 成本
- $c_t$：target model 单 step 成本

**期望产出**：

$$
E[\text{tokens per iteration}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

**期望成本**：

$$
E[\text{cost}] = \gamma \cdot c_d + c_t
$$

**加速比**（vs 纯大模型）：

$$
\text{speedup} \approx \frac{E[\text{tokens}]}{E[\text{cost}] / c_t} = \frac{E[\text{tokens}] \cdot c_t}{\gamma \cdot c_d + c_t}
$$

**典型场景**（$\gamma=4, \alpha=0.7, c_d = 0.1 c_t$）：

- $E[\text{tokens}] = (1 - 0.7^5) / 0.3 ≈ 3.0$
- $E[\text{cost}] = 4 \times 0.1 + 1 = 1.4$ step
- 加速比 ≈ 3.0 / 1.4 ≈ **2.1x**

**实测**（论文数据，LLaMA-65B + LLaMA-7B draft）：

- 文本摘要：2.3x
- 代码生成：2.0x
- 对话：2.5x

### 4.5 Draft Model 的选择

**选项 1：同家族小模型**

- 例如 LLaMA-65B + LLaMA-7B
- 分布相近，命中率高
- 无需额外训练

**选项 2：蒸馏小模型**

- 用大模型蒸馏一个小模型
- 分布更接近，命中率更高
- 训练成本高

**选项 3：Medusa 多头**

- 在大模型上额外训练多个 prediction head
- 每个 head 预测不同深度的 token
- 无需单独 draft model
- 训练成本中等

**选项 4：EAGLE 特征级草拟**

- 在大模型的隐空间草拟，不是 token 级
- 命中率更高（70-80%）
- 训练一个小 autoregressive head

**选项 5：Lookup（N-gram）**

- 用 prompt 中的 N-gram 匹配
- 零成本，但只在 prompt 有重复时有效
- 适合代码、长文档续写

**选择建议**：

| 场景 | draft model | γ | 期望加速 |
|------|------------|---|---------|
| 同家族可用 | 小版本 | 4-8 | 2-3x |
| 需要高命中率 | 蒸馏 | 4-8 | 2.5-3x |
| 无小模型 | Medusa | 4-8 | 2-2.5x |
| 极致加速 | EAGLE | 4-8 | 3-4x |
| 代码/长文 | Lookup | 2-4 | 1.5-2x |

### 4.6 实现细节

**vLLM 中的实现**：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-70B",
    speculative_model="meta-llama/Llama-2-7B",  # draft model
    num_speculative_tokens=5,                    # γ
)

# 其余使用方式不变
outputs = llm.generate(prompts, sampling_params)
```

**工作流**：

```
1. 接收请求，draft model 自回归生成 γ 个 token
2. target model 一次前向验证 γ+1 个位置
3. 按接受规则处理，得到 k 个接受 token + 1 个修正/bonus token
4. 如果 k == γ（全接受），可选再生成下一轮
5. 如果 k < γ，从第 k+1 位置修正，draft model 从此处继续
6. 重复直到生成 EOS
```

### 4.7 与 Continuous Batching 的配合

**挑战**：

- Continuous Batching 多请求共享 batch
- 不同请求的 spec decoding 状态不同（draft 长度、接受数）
- 如何在 batch 内调度 spec decoding？

**vLLM 的方案**：

- 每个 iteration 统一 γ（所有请求 spec 同样长度）
- batch 内 spec decoding 和非 spec decoding 请求混合
- 接受的 token 数不同，用 padding 补齐

**效果**：

- 高并发下 spec decoding 仍能 1.5-2x 加速
- 低并发下 2-3x 加速

### 4.8 局限与挑战

**局限 1：命中率敏感**。draft model 差，命中率低，浪费算力。$\alpha < 0.5$ 时收益不明显。

**局限 2：需要好 draft model**。不是所有场景都有合适的小模型。

**局限 3：实现复杂**。要处理采样一致性、batch 调度、KV-Cache 管理。

**局限 4：高并发场景收益小**。batch 大时 GPU 已满载，spec decoding 的并行优势减弱。

**局限 5：γ 调参**。γ 太小加速有限，γ 太大浪费 draft 时间。要按场景调。

**局限 6：sampling 一致性**。draft 和 target 用不同 sampling 时，数学等价性破坏。要用相同 temperature/top-p。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2022-202 speculative decoding 雏形**：Blockwise Parallel Decoding（斯坦福，并行预测多 token）
- **2023-01**：DeepMind 和 Google 同时独立发表 spec decoding 论文（rejection sampling 思路）
- **2023-05**：开源社区实现 spec decoding，vLLM/TGI 跟进
- **2024-01**：Medusa 论文（多头草拟，无需独立 draft model）
- **2024-02**：EAGLE 论文（特征级草拟，命中率 80%+）
- **2024-2025**：spec decoding 成为主流推理引擎标配，加速比稳定在 2-3x

### 5.2 常见坑

**坑 1：draft model 选错**。分布差异大的小模型命中率低，反而变慢。要选同家族或蒸馏。

**坑 2：γ 设太大**。γ=16 时 draft 时间长，命中率衰减（越远越难猜）。γ=4-8 通常最优。

**坑 3：sampling 不一致**。draft 用 greedy、target 用 sampling，等价性破坏。要统一 sampling 参数。

**坑 4：高并发场景期望 3x**。高并发下 GPU 已满载，spec decoding 只能 1.3-1.8x。它是低并发延迟优化。

**坑 5：没考虑 KV-Cache**。draft 和 target 都要 KV-Cache，显存翻倍。要管理好。

**坑 6：实现忽略 bonus token**。全接受时要从 $p$ 分布多采一个 token，否则少产出。

**坑 7：接受规则的实现错**。拒绝后修正采样分布 $\max(0, p - q)$ 归一化错，破坏等价性。

**坑 8：温度高时命中率低**。高温 sampling 下 draft 和 target 分布差异大，命中率降低。spec decoding 更适合低温或 greedy。

**坑 9：batch 内 spec 长度不一**。不同请求 spec 不同长度，要 padding 或统一 γ。

**坑 10：以为 spec decoding 能提升吞吐**。它是延迟优化，吞吐通常持平或略降（高并发下）。别用错场景。

**坑 11：忽略 draft model 加载成本**。额外加载 draft model 占显存，可能影响 batch size。

**坑 12：期望 spec decoding 解决一切**。它是局部优化，搭配 KV-Cache、PagedAttention、Continuous Batching 才能整体最优。

### 5.3 面试怎么考

1. **Speculative Decoding 的核心思想？** 答：小模型草拟 γ 个 token，大模型一次前向并行验证，按接受规则处理，输出分布数学等价于纯大模型。延迟降 2-3x。
2. **为什么数学等价？** 答：接受概率 $\min(1, p/q)$ + 拒绝时从 $\max(0, p-q)$ 采样，是 rejection sampling 的变体，输出分布严格等于 $p$。
3. **加速比受什么影响？** 答：命中率 $\alpha$、draft 长度 $\gamma$、draft/target 成本比。$\alpha$ 高、$\gamma$ 适中、draft 小时加速最大。
4. **draft model 怎么选？** 答：同家族小模型（LLaMA-7B 给 LLaMA-70B）、蒸馏模型、Medusa 多头、EAGLE 特征级、Lookup N-gram。按场景选。
5. **Spec Decoding 和 Continuous Batching 的关系？** 答：可配合，但 spec decoding 是延迟优化，高并发下 Continuous Batching 已满载 GPU，spec decoding 收益减弱。低并发延迟敏感场景首选 spec decoding。

---

## 速记卡

**两阶段**：

```
Speculative:  draft q 自回归生成 γ token
Verification: target p 一次前向验证 γ+1 位置
Accept:       r < min(1, p/q) 接受
Reject:       从 max(0, p-q) 采样修正
Bonus:        全接受时从 p 多采 1 个
```

**接受规则**：

$$
P(\text{accept } x_i) = \min\left(1, \frac{p(x_i)}{q(x_i)}\right)
$$

**加速比公式**：

$$
\text{speedup} \approx \frac{E[\text{tokens}] \cdot c_t}{\gamma \cdot c_d + c_t}, \quad E[\text{tokens}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

**典型参数**：

| 参数 | 典型值 | 说明 |
|------|--------|------|
| γ | 4-8 | draft 长度 |
| α | 0.5-0.8 | 接受率 |
| c_d / c_t | 0.05-0.2 | draft/target 成本比 |
| 加速比 | 2-3x | 单请求延迟 |

**Draft model 选项**：

| 方案 | 训练成本 | 命中率 | 适用 |
|------|---------|--------|------|
| 同家族小模型 | 无 | 中 | 通用 |
| 蒸馏 | 高 | 高 | 高端场景 |
| Medusa | 中 | 中高 | 无独立 draft |
| EAGLE | 中 | 高 | 特征级草拟 |
| Lookup | 无 | 低 | 代码/长文 |

**一句话记忆**：Speculative Decoding = 小模型猜、大模型并行批改。draft 自回归生成 γ 个候选 token，target 一次前向算 γ+1 个位置 logits，按 $\min(1, p/q)$ 接受规则处理，输出分布数学等价于纯大模型（lossless）。单请求延迟降 2-3x，是低并发延迟敏感场景的利器；高并发下 Continuous Batching 已满载 GPU，收益减弱。draft model 选同家族或 EAGLE，γ 取 4-8，sampling 要一致。

---

> *上一篇：[Continuous Batching 连续批处理](./continuous-batching) -- Continuous Batching 提吞吐，Speculative Decoding 降单请求延迟。*
> *下一篇：[量化推理算法 GPTQ/AWQ](./quantization-inference) -- 推理工程末篇，权重量化进一步压缩显存。*
