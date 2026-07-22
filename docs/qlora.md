---
title: QLoRA 量化低秩适配
slug: qlora
category: 进阶专题
tags: [QLoRA, 4bit 量化, NF4, 双量化, LoRA, PEFT]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# QLoRA 量化低秩适配

> 五层读懂一个词。这次拆的是：**QLoRA**--把基座模型量化到 4bit 省显存，再用 LoRA 训 BF16 的适配器，让消费级 GPU（48GB）也能微调 70B 模型。

---

## L1 · 一句话点破

**QLoRA = 4bit NF4 量化基座 + BF16 LoRA 适配器 + 双量化 + 分页优化器**。基座 4bit 冻结省显存，LoRA 16bit 训练保精度，单卡 48GB 微调 70B，效果逼近 16bit 全参微调。

---

## L2 · 通俗类比

LoRA 解决了"参数量"问题（只训 0.1% 参数），但没解决"显存"问题--基座模型还得原样加载。70B 模型 BF16 要 140GB，光是把模型塞进显存就要 2-3 张 A100，LoRA 省下的那点梯度显存杯水车薪。

QLoRA 的洞察：**基座是冻结的，不参与梯度更新，那它的精度可以狠压**。把基座量化到 4bit，70B 从 140GB 压到 35GB，单卡塞得下。但 LoRA 适配器要训练，精度不能压，保持 BF16。

三个关键创新让"4bit 基座 + 16bit 训练"不崩：

- **NF4 量化**：针对正态分布权重设计的 4bit 数据类型，比普通 INT4 精度高
- **双量化（Double Quantization）**：连量化常数本身也量化一次，再省 0.4bit/参数
- **分页优化器（Paged Optimizer）**：用 CPU 内存兜底显存峰值，防 OOM

**数字感受**（LLaMA-65B）：

| 方法 | 基座显存 | 可训显存 | 总显存 | 硬件 |
|------|---------|---------|--------|------|
| 16bit 全参 | 130 GB | 500+ GB | ~650 GB | 8×A100 |
| 16bit LoRA | 130 GB | 20 GB | ~150 GB | 2×A100 |
| **4bit QLoRA** | **33 GB** | **15 GB** | **~48 GB** | **单卡 A100** |

**代价**：4bit 量化有精度损失（虽然 NF4 把损失压到很小）；训练速度比 16bit LoRA 慢 ~30%（4bit 反量化开销）；合并部署时要小心量化误差叠加。

---

## L3 · 正经定义

**QLoRA（Quantized LoRA）**：Dettmers et al. (NeurIPS 2023) 提出，在 LoRA 基础上引入 4bit 量化基座，实现消费级 GPU 微调大模型。三大创新：

1. **NF4（NormalFloat 4-bit）**：基于权重正态分布特性设计的 4bit 数据类型，信息论最优
2. **双量化（Double Quantization）**：对量化常数本身再做一次量化，节省额外显存
3. **分页优化器（Paged Optimizer）**：利用 NVIDIA Unified Memory，显存峰值时自动把优化器状态页换到 CPU

**训练流程**：

```
基座 W (BF16) ──量化──> W_4bit (NF4) ──冻结──> 前向时反量化回 BF16 计算
                                              ↑
                                    LoRA A,B (BF16) ──训练──> 梯度更新
```

**核心数据流**：

- 前向：$W_{4bit}$ 反量化到 BF16 → 与 $x$ 相乘 → 加上 $BAx$ → 输出
- 反向：梯度只流过 $A, B$，$W_{4bit}$ 无梯度
- 优化器：只更新 $A, B$（BF16），$W_{4bit}$ 冻结

**参考资料**：

- 📄 Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023, arXiv:2305.14314
- 🔧 QLoRA 官方实现：https://github.com/artidoro/qlora
- 🔧 bitsandbytes 库（NF4 量化实现）：https://github.com/bitsandbytes-foundation/bitsandbytes
- 📄 Frantar et al., *GPTQ: Accurate Post-Training Quantization*（对比量化方案）

---

## L4 · 原理深挖

### 4.1 NF4：为正态权重设计的 4bit

普通 INT4 量化假设数据均匀分布，但 LLM 权重实际是**正态分布**（中心密集、尾部稀疏）。INT4 的等间距量化点在尾部浪费、中心不够密。

**NF4 的做法**：用正态分布的分位数（quantile）作为量化点，让每个量化区间包含等概率密度的权重。

**NF4 的 16 个量化值**（针对标准正态 $N(0,1)$ 归一化后）：

```
[-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

这些值是标准正态分布的 16 等分分位数，保证每个量化区间的权重数量大致相等。

**信息论最优**：对于正态分布数据，分数量化使量化误差的期望最小。论文证明 NF4 在信息论意义下是 4bit 量化的最优解。

**量化流程**：

```python
def quantize_nf4(W):
    # 1. 归一化到标准正态范围
    abs_max = W.abs().max()
    W_normalized = W / abs_max  # 到 [-1, 1]
    # 2. 映射到最近的 NF4 量化点
    W_4bit = find_nearest(W_normalized, NF4_LEVELS)
    # 3. 存 4bit 索引 + 1 个缩放因子 abs_max
    return W_4bit, abs_max

def dequantize_nf4(W_4bit, abs_max):
    W_normalized = NF4_LEVELS[W_4bit]
    return W_normalized * abs_max
```

### 4.2 双量化（Double Quantization）

量化后每个张量存一个 FP32 缩放因子 `abs_max`。看似小，但模型有几千个张量，累积起来不小：

- LLMa 65B 约有 16000 个量化块
- 每块一个 FP32 缩放 = 64KB
- 总计 ~1GB 额外开销

**双量化**：把所有缩放因子本身再量化一次（FP32 → 8bit），再存一个二级缩放。

```
原始：W (16bit) + scale (32bit) → 平均 16+ε bit/参数
单量化：W (4bit) + scale (32bit) → 平均 4+0.5 bit/参数
双量化：W (4bit) + scale (8bit) + scale_scale (32bit) → 平均 4+0.127 bit/参数
```

双量化再省 ~0.4 bit/参数，65B 模型省 ~3GB。看起来不多，但对"单卡塞下"是关键。

### 4.3 分页优化器（Paged Optimizer）

**问题**：训练时优化器状态（Adam 的 $m, v$）在梯度更新瞬间会突然增长，导致显存峰值 OOM，即使平均显存够用。

**解法**：用 NVIDIA Unified Memory 的 paging 机制，显存不够时自动把优化器状态页换到 CPU 内存，需要时再换回来。

```python
# bitsandbytes 的分页 Adam
optimizer = bnb.optim.PagedAdamW8bit(params, lr=2e-4)
# 显存峰值时自动 page out 到 CPU
```

**效果**：消除显存峰值，让"平均显存够"就够，不再被瞬时峰值 OOM。

### 4.4 QLoRA 的完整训练流程

```python
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 1. 4bit 加载基座
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70B",
    load_in_4bit=True,          # NF4 量化
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # 双量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时反量化到 BF16
    device_map="auto"
)

# 2. 准备 LoRA
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# 3. 分页优化器
optimizer = bnb.optim.PagedAdamW8bit(
    model.parameters(), lr=2e-4, weight_decay=0.01
)

# 4. 正常训练循环
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**关键点**：

- 基座 4bit 冻结，前向时反量化到 BF16 计算（`bnb_4bit_compute_dtype`）
- LoRA 参数 BF16，正常训练
- 优化器用 8bit 分页版本，省显存 + 防 OOM

### 4.5 精度损失分析

QLoRA 的核心问题：4bit 量化的精度损失有多大？

**论文实验**（Llama-65B，5 个任务）：

| 方法 | 平均准确率 |
|------|-----------|
| 16bit 全参微调 | 73.2% |
| 16bit LoRA | 73.1% |
| **4bit QLoRA** | **72.9%** |

QLoRA 比 16bit 全参微调只低 0.3%，几乎可忽略。关键在 **NF4 的精度 + LoRA 的 BF16 训练**：

- NF4 针对正态分布优化，量化误差小
- LoRA 在 BF16 训练，梯度更新精度无损
- 基座冻结，量化误差不反向传播放大

### 4.6 QLoRA vs LoRA vs 全参

| 维度 | 全参微调 | LoRA | QLoRA |
|------|---------|------|-------|
| 基座精度 | 16bit | 16bit | 4bit |
| 训练参数 | 100% | 0.1-1% | 0.1-1% |
| 70B 显存 | ~650 GB | ~150 GB | ~48 GB |
| 训练速度 | 基准 | 持平 | 慢 30% |
| 效果 | 上限最高 | 略低 | 略低 |
| 硬件门槛 | 多卡 A100 | 2卡 A100 | 单卡 A100 |

**何时选 QLoRA**：

- 显存塞不下 16bit 基座
- 消费级 / 单卡场景
- 快速实验迭代

**何时不用 QLoRA**：

- 显存充裕（多卡 A100）→ 直接 16bit LoRA 更快
- 极致性能 → 全参微调上限更高
- 量化敏感场景（数值精度要求高）

### 4.7 QLoRA 的部署

训练完的 QLoRA 模型部署有两种选择：

**选项 1：保持 4bit + LoRA**

- 基座 4bit 不动，LoRA 适配器单独存
- 推理时前向计算 $W_{4bit}x + BAx$
- 显存最小，但有 4bit 反量化开销

**选项 2：反量化 + 合并 LoRA**

- 4bit 基座反量化到 16bit
- LoRA 合并到基座
- 推理和普通 16bit 模型一样
- 显存翻 4 倍，但推理快

**实践**：显存敏感用选项 1，延迟敏感用选项 2。

### 4.8 QLoRA 的局限

**局限 1：训练慢 30%**。4bit 反量化有开销，比 16bit LoRA 慢约 30%。

**局限 2：量化误差累积**。虽然 NF4 误差小，但深层网络误差累积，超深模型（100+ 层）可能掉点。

**局限 3：不是所有层都适合 4bit**。embedding 层、output 层量化损失大，通常保持高精度。

**局限 4：合并部署精度损失**。4bit LoRA 合并到 16bit 时，量化误差被"固化"，无法回退。

**局限 5：硬件要求**。需要 bitsandbytes 支持，主要 NVIDIA GPU，AMD/苹果支持不完善。

### 4.9 QLoRA 的演进

- **QLoRA** (2023-05)：原始论文，NF4 + 双量化 + 分页优化器
- **AWQ-LoRA**：AWQ 量化 + LoRA，激活感知，精度更高
- **GPTQ-LoRA**：GPTQ 量化 + LoRA，另一条量化路线
- **HQQ-LoRA**：Half-Quadratic Quantization，无数据量化
- **2024-2025**：3bit QLoRA（GGUF 格式）出现，进一步压显存但精度损失增大

---

## L5 · 沿革与坑

### 5.1 沿革

- **2023-05**：Dettmers 发 QLoRA 论文，首次实现单卡 48GB 微调 65B
- **2023-07**：QLoRA 被 NeurIPS 接收，社区爆发式采用
- **2023 下半年**：HuggingFace transformers + PEFT + bitsandbytes 集成 QLoRA，成为标配
- **2024**：AWQ-LoRA / GPTQ-LoRA 等变体涌现，精度进一步提升
- **2025**：3bit QLoRA 出现，消费级 24GB 显卡微调 30B 成为可能

### 5.2 常见坑

**坑 1：用 INT4 而不是 NF4**。INT4 假设均匀分布，LLM 权重正态分布，INT4 量化损失大。必须用 NF4。

**坑 2：忘了双量化**。不开 `bnb_4bit_use_double_quant=True`，显存多占几 GB，单卡塞不下。

**坑 3：计算精度用 FP32**。`bnb_4bit_compute_dtype=torch.float32` 看似精度高，实际慢且不必要。BF16 够用。

**坑 4：LoRA 学习率用全参的**。QLoRA 本质还是 LoRA，学习率要比全参大 5-10 倍，$1e-4$ 到 $2e-4$。

**坑 5：embedding 层也量化**。embedding 量化损失大，要用 `llm_int8_skip_modules` 跳过。

**坑 6：batch size 太大 OOM**。虽然基座 4bit 省，但激活值随 batch size 线性增长。要调小 batch + 梯度累积。

**坑 7：忘了分页优化器**。不用 `PagedAdamW8bit`，显存峰值 OOM。普通 AdamW 不分页。

**坑 8：合并 4bit 模型精度丢失**。反量化 + 合并 LoRA 时累积误差。合并后要评估对比未合并版本。

**坑 9：评估只看训练 loss**。4bit 量化模型 loss 下降正常，但下游任务可能掉点。要在目标任务上评估。

**坑 10：QLoRA 当万能解**。QLoRA 解决显存问题，不解决效果问题。复杂任务还是要全参微调。

**坑 11：序列长度太长**。长序列激活值爆显存，QLoRA 省的是权重不是激活。要限制 `max_seq_len` 或用 gradient checkpointing。

**坑 12：跨量化格式迁移**。NF4 训的 LoRA 不能直接用到 GPTQ/AWQ 模型上，量化基准不同。要重新训。

### 5.3 面试怎么考

1. **QLoRA 的三大创新？** 答：NF4 量化（正态分布最优）+ 双量化（量化常数再量化）+ 分页优化器（防 OOM）。
2. **NF4 为什么比 INT4 好？** 答：LLM 权重正态分布，NF4 用分位数量化点，信息论最优；INT4 假设均匀分布，尾部浪费中心稀疏。
3. **QLoRA 怎么实现单卡 48GB 微调 70B？** 答：基座 4bit NF4（35GB）+ LoRA BF16（15GB）+ 分页优化器防峰值。
4. **QLoRA 的精度损失？** 答：比 16bit 全参微调低 ~0.3%，几乎可忽略。关键是 NF4 精度高 + LoRA 在 BF16 训练。
5. **QLoRA 什么时候不用？** 答：显存充裕（用 16bit LoRA 更快）、极致性能（全参上限高）、量化敏感场景。

---

## 速记卡

| 组件 | 精度 | 作用 |
|------|------|------|
| 基座 $W$ | 4bit NF4 | 冻结，省显存 |
| LoRA $A, B$ | BF16 | 训练，保精度 |
| 量化常数 | 8bit（双量化） | 省 0.4bit/参数 |
| 优化器 | 8bit 分页 | 防 OOM |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| 量化类型 | NF4 | 精度 |
| 双量化 | True | 省 3GB |
| 计算精度 | BF16 | 速度 vs 精度 |
| LoRA $r$ | 32-128 | 适配能力 |
| 学习率 | $1e-4$ ~ $2e-4$ | 比 16bit LoRA 持平 |

**显存对比**（65B 模型）：

| 方法 | 显存 | 硬件 |
|------|------|------|
| 16bit 全参 | ~650 GB | 8×A100 80G |
| 16bit LoRA | ~150 GB | 2×A100 80G |
| **4bit QLoRA** | **~48 GB** | **单卡 A100 48G** |

**一句话记忆**：QLoRA = 4bit NF4 量化基座（省显存）+ BF16 LoRA 训练（保精度）+ 双量化 + 分页优化器。单卡 48GB 微调 70B，效果逼近 16bit 全参（低 0.3%）。NF4 是关键（正态分布最优 4bit），训练比 16bit LoRA 慢 30%。消费级 GPU 微调大模型的事实标准。

---

> *上一篇：[LoRA 低秩适配](./lora) -- LoRA 解决参数量，QLoRA 解决显存。*
> *下一篇：[Adapter Tuning 适配器微调](./adapter-tuning) -- LoRA 之前的 PEFT 老前辈，串行小模块。*
