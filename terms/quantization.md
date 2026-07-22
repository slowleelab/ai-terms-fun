---
title: 量化（Quantization）
slug: quantization
category: 模型压缩与加速
tags: [量化, INT8, INT4, GPTQ, AWQ, GGUF, LLM 推理]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 量化（Quantization）

> **一句话 TL;DR**：量化是把模型参数从高精度（FP32/FP16）压到低精度（INT8/INT4）的技术。直接收益：显存占用降 4-8 倍，推理速度提升，让 70B 模型能跑在消费级 GPU 甚至 CPU 上。代价：精度损失，需要 PTQ/QAT/GPTQ/AWQ 等方法控制。量化是大模型落地"消费端"的核心使能技术。

---

## L1 · 一句话点破

量化：**把模型的权重和激活从浮点（FP16/FP32）映射到低精度整数（INT8/INT4），用更少比特表示同样的信息。**

```
FP16:   0.12345678  (16 bit)  ->  显存 100%
INT8:   0.123       (8 bit)   ->  显存 50%，速度更快
INT4:   0.1         (4 bit)   ->  显存 25%，速度最快，精度略损
```

LLaMA-70B 在 FP16 需要 ~140GB 显存（2 张 A100 80G）。INT4 量化后只需 ~35GB，单张 A100 即可跑。这是量化最直接的价值。

## L2 · 通俗类比

存图片的几种格式：

- **BMP（无压缩）**：每个像素 24 bit，文件巨大但无损
- **PNG（无损压缩）**：压缩但无损
- **JPG（有损压缩）**：压缩更狠，文件小，略有画质损失但视觉差异小
- **JPG 高压缩**：文件极小，画质明显下降

量化类似：

- **FP32（无压缩）**：每个参数 32 bit，精度最高，显存最大
- **FP16/BF16（半精度）**：16 bit，几乎无损，显存减半，训练默认
- **INT8（量化）**：8 bit，轻微精度损失，显存再减半
- **INT4（激进量化）**：4 bit，明显但可接受的精度损失，显存再减半

选择量化的权衡：**精度 vs 资源**。多数 LLM 推理场景，INT8 几乎无损，INT4 略损但够用。

为什么量化可行？因为模型参数的分布通常集中在小范围，少量比特足以表示主要信息。这是"信息冗余"的利用。

## L3 · 正经定义

**量化（Quantization）**：把浮点数值映射到低精度整数的过程。对权重张量 $W$（FP16），量化到 INT8：

$$
W_{\text{int8}} = \text{round}(W / s)
$$

反量化：$W \approx s \cdot W_{\text{int8}}$，其中 $s$ 是缩放因子（scale）。

对称量化：$s = \max(|W|) / 127$
非对称量化：引入零点 $z$，$W_{\text{int8}} = \text{round}(W / s) + z$

**按量化时机分**：

| 类型 | 全称 | 时机 | 代表方法 |
|------|------|------|---------|
| **PTQ** | Post-Training Quantization | 训练后 | GPTQ, AWQ, GGUF, SmoothQuant |
| **QAT** | Quantization-Aware Training | 训练中 | LLM-QAT, BitNet |

**按量化粒度分**：

- **per-tensor**：整个张量一个 scale
- **per-channel**：每行/列一个 scale
- **per-group**：每 N 个元素一个 scale（GPTQ/AWQ 常用，group=128）

**常见量化方案对比**：

| 方案 | 比特 | 精度损失 | 速度 | 用途 |
|------|------|---------|------|------|
| FP16/BF16 | 16 | 基准 | 基准 | 训练、推理默认 |
| INT8 (W8A8) | 8 | 极小 | 2x | 服务端推理 |
| INT4 (W4A16) | 4 | 小 | 3x | 消费级 GPU |
| INT3 | 3 | 明显 | 4x | 极限压缩 |
| INT2 / 1-bit | 2/1 | 大 | - | 实验 |

**参考资料**：
- [Frantar et al., 2022 - GPTQ](https://arxiv.org/abs/2210.17323)
- [Lin et al., 2023 - AWQ](https://arxiv.org/abs/2306.00978)
- [Dettmers et al., 2022 - LLM.int8()](https://arxiv.org/abs/2208.07339)
- [Wang et al., 2023 - BitNet](https://arxiv.org/abs/2310.11453)
- [Xiao et al., 2022 - SmoothQuant](https://arxiv.org/abs/2211.10438)

## L4 · 原理深挖

### 4.1 为什么量化有效：参数分布的冗余

大模型参数虽多，但单个参数的有效信息量低。实测发现：

- **权重分布近似高斯**：大部分集中在 0 附近，极端值少
- **8 bit 足以覆盖权重范围**：量化误差远小于权重本身的不确定性
- **激活分布有离群点**：少数通道有大值，需特殊处理（见 4.4）

直觉：如果权重 $W \in [-0.5, 0.5]$ 且大部分在 $[-0.1, 0.1]$，用 8 bit（256 级）量化精度 $\sim 0.004$，远小于权重本身的"重要性"。量化几乎无损。

INT4（16 级，精度 $\sim 0.06$）开始有可见损失，但研究表明：LLM 对权重 INT4 量化相对鲁棒，精度损失通常 <2%。

### 4.2 量化的核心挑战：激活的离群点

权重量化相对容易（分布稳定），但**激活量化难**。原因是激活有"离群点"（outliers）：

- 大部分激活值小（<1）
- 少数通道有极大值（>10），但**对注意力计算关键**
- 用统一 scale 量化时，离群点撑大 scale，小值被压成 0，精度损失大

[LLM.int8() (Dettmers et al., 2022)](https://arxiv.org/abs/2208.07339) 的发现：6.7B 以上模型开始出现"系统性离群点"（几个固定通道一直有大值）。这是大模型量化的核心难题。

**解决方案**：

**① 混合精度**（LLM.int8()）

离群点通道用 FP16，其他用 INT8。代价是工程复杂、速度不如纯 INT8。

**② SmoothQuant**（[Xiao et al., 2022](https://arxiv.org/abs/2211.10438)）

把激活的离群值"平滑"到权重上：

$$
Y = XW = (X / s)(sW) = X' W'
$$

选 $s$ 让 $X'$ 的离群点减小，$W'$ 的范围扩大（权重对量化鲁棒）。一举把激活量化难题转化为权重量化。

**③ W4A16 / W8A16**（权重量化、激活不量化）

最简单方案：只量化权重（W4），激活保持 FP16（A16）。避开激活量化难题。

GPTQ、AWQ、GGUF 都是 W4A16 方案。这是消费级 GPU 跑 LLM 的主流。

### 4.3 GPTQ：基于二阶信息的后训练量化

[GPTQ (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323) 是最流行的 INT4 PTQ 方法之一。

核心思想：逐列量化权重，用 Hessian 矩阵的逆补偿量化误差：

```
对每一列 W[:, j]:
    1. 量化 W[:, j] 到 INT4
    2. 计算量化误差 e = W[:, j] - dequantize(W[:, j])
    3. 用 Hessian 逆 H^{-1} 把误差"分摊"到未量化列
    4. 更新 W[:, j+1:] -= e * H^{-1}[:, j]
```

效果：INT4 量化后精度损失极小（<1%），且量化速度快（70B 模型几小时）。

GPTQ 的局限：

- 量化后需特定推理格式（Marlin kernel 等）
- 对某些模型（如 Mistral）效果一般
- 主要服务端 GPU 优化

### 4.4 AWQ：基于激活感知的量化

[AWQ (Lin et al., 2023)](https://arxiv.org/abs/2306.00978) 是另一流行 INT4 方案，思路不同：

核心洞察：**不是所有权重同样重要**。激活值大的通道对应的权重更敏感，应保护。

方法：

1. 用少量校准数据找"重要通道"（激活值大的通道）
2. 对重要通道的权重缩放（让它们在量化范围内更"分散"）
3. 量化

```
W' = W * s  # s 在重要通道大，其他小
quantize(W')
# 反量化时还原
```

效果：与 GPTQ 精度相当或略好，且**对边缘设备友好**（支持 mobile、CPU 推理）。vLLM、llama.cpp 等都支持 AWQ。

### 4.5 GGUF：消费端的量化格式

[GGUF (llama.cpp)](https://github.com/ggerganov/llama.cpp) 是消费端 LLM 的事实标准格式。特点：

- **CPU 推理优化**：能在纯 CPU 上跑 LLM
- **多种量化级别**：Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 等
- **混合精度**：关键层用高精度，其他用低精度
- **元数据内置**：tokenizer、配置都打包在单文件

GGUF 让"在家用电脑/Mac 上跑 LLaMA"成为可能，是开源 LLM 普及的关键。

### 4.6 QAT：训练时就量化

PTQ 是"训完再量化"，QAT 是"训练时模拟量化"：

```
forward:
    W_quant = quantize(W)  # 模拟量化
    W_dequant = dequantize(W_quant)
    output = X @ W_dequant  # 用反量化权重前向
backward:
    用 straight-through estimator 传梯度
```

QAT 让模型在训练时"适应"量化误差，最终量化后精度损失更小。

代表：[BitNet (Wang et al., 2023)](https://arxiv.org/abs/2310.11453) 用 1-bit 权重（-1 或 +1）训练，推理极快。但 QAT 需要从头训练，成本高，目前主要用于研究。

### 4.7 量化的工程权衡

实际选择量化方案的考量：

| 维度 | 考虑 |
|------|------|
| 硬件 | GPU（A100/H100）vs 消费级（4090/Mac）vs CPU |
| 模型规模 | 7B（INT4 单卡）vs 70B（INT4 多卡）vs 405B（必须量化） |
| 精度需求 | 越关键（医疗/法律）越高精度 |
| 速度需求 | 实时对话 vs 批处理 |
| 部署方式 | 服务端（vLLM/TensorRT-LLM）vs 边缘（llama.cpp） |

经验法则：

- **服务端 A100/H100**：FP16 或 INT8 (W8A8)，速度精度都好
- **消费级 4090**：INT4 (W4A16, GPTQ/AWQ)，性价比最高
- **Mac/CPU**：GGUF Q4_K_M，平衡速度精度
- **极低资源**：INT3 或更激进，接受明显精度损失

## L5 · 沿革与坑

### 沿革

- **2015-2018**：CV 领域 INT8 量化成熟（TensorFlow Lite、PyTorch quantization）。
- **2020-2021**：BERT 等小模型量化普及，但 LLM 量化难（参数多、激活离群点）。
- **2022 年 8 月**：[LLM.int8()](https://arxiv.org/abs/2208.07339) 首次让 6.7B+ 模型 INT8 量化近无损，揭示"系统性离群点"。
- **2022 年 10 月**：[SmoothQuant](https://arxiv.org/abs/2211.10438) 把激活难题转为权重难题。
- **2022 年 11 月**：[GPTQ](https://arxiv.org/abs/2210.17323) 提出，成为 INT4 PTQ 主流。
- **2023 年 6 月**：[AWQ](https://arxiv.org/abs/2306.00978) 提出，与 GPTQ 并列主流。
- **2023-2024**：llama.cpp 的 GGUF 格式让消费端 LLM 普及，"在家跑 LLaMA"成为现实。
- **2023-2024**：[BitNet](https://arxiv.org/abs/2310.11453) 等 1-bit 训练时量化研究推进，但未大规模落地。
- **2024-2025**：vLLM、TensorRT-LLM 等推理引擎深度集成量化，INT4/INT8 成为 LLM 服务默认配置。

### 常见误解

- ❌ **误解**：量化一定让模型变笨。
  ✅ **真相**：INT8 量化几乎无损（<1%）；INT4 损失通常 <2%。多数应用场景可接受。质量损失随比特数下降，不是断崖式。

- ❌ **误解**：比特越低越好，能压就压。
  ✅ **真相**：INT3 以下精度损失明显，可能让模型不可用。选择量化级别要平衡资源与精度，不是越低越好。

- ❌ **误解**：GPTQ 和 AWQ 差不多，随便选。
  ✅ **真相**：精度相近，但部署特性不同。GPTQ 更适合服务端 GPU，AWQ 对边缘设备和 CPU 更友好。按部署场景选。

- ❌ **误解**：量化只在推理用，训练不量化。
  ✅ **真相**：训练也用量化（QAT），如 BitNet 用 1-bit 权重训练。训练量化能进一步压缩，但成本高（4.6）。

- ❌ **误解**：FP16 已经够省了，不需要量化。
  ✅ **真相**：70B 模型 FP16 需 140GB，多数硬件跑不动。INT4 后 35GB，单 A100 可跑。量化让"大模型上消费硬件"成为可能。

- ❌ **误解**：量化后模型行为完全不变。
  ✅ **真相**：量化引入误差，模型输出会有细微差异。在敏感任务（医疗、法律）需评估量化对输出的影响。

### 面试怎么考

1. **"什么是量化？为什么有用？"** --把 FP16/FP32 压到 INT8/INT4，显存降 2-4 倍，速度提升。利用参数分布冗余（L1、4.1）。
2. **"PTQ 和 QAT 的区别？"** --PTQ 训完再量化，简单但精度略损；QAT 训练时模拟量化，精度好但需重训（4.6）。
3. **"GPTQ 和 AWQ 的核心思想？"** --GPTQ 用 Hessian 逆补偿逐列量化误差；AWQ 用激活感知保护重要通道（4.3、4.4）。
4. **"为什么激活量化比权重量化难？"** --激活有系统性离群点，统一 scale 量化时小值被压成 0。SmoothQuant、W4A16 等方案解决（4.2）。
5. **"70B 模型怎么在消费级 GPU 上跑？"** --INT4 量化（GPTQ/AWQ），显存从 140GB 降到 35GB，单 A100 或多 4090 可跑。
6. **"GGUF 是什么？为什么流行？"** --llama.cpp 的消费端格式，支持 CPU 推理、多量化级别、单文件。让家用电脑能跑 LLM（4.5）。

## 延伸阅读

- 📄 [Frantar et al., 2022 - GPTQ](https://arxiv.org/abs/2210.17323)
- 📄 [Lin et al., 2023 - AWQ](https://arxiv.org/abs/2306.00978)
- 📄 [Dettmers et al., 2022 - LLM.int8()](https://arxiv.org/abs/2208.07339)
- 📄 [Xiao et al., 2022 - SmoothQuant](https://arxiv.org/abs/2211.10438)
- 📄 [Wang et al., 2023 - BitNet](https://arxiv.org/abs/2310.11453)
- 📝 [llama.cpp - GGUF format](https://github.com/ggerganov/llama.cpp)
- 🚀 进阶专题·高效微调：[QLoRA](./qlora) -- 本词条讲的是推理侧量化，QLoRA 把 4bit NF4 量化用到训练侧，单卡微调 70B

---

> *上一篇：[幻觉 Hallucination](./hallucination) -- 模型为什么一本正经胡说八道。*
> *下一篇：[知识蒸馏](./knowledge-distillation) -- 用大模型教小模型。*
