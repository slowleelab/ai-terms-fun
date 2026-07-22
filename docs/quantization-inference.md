---
title: 量化推理算法 GPTQ / AWQ
slug: quantization-inference
category: 进阶专题
tags: [GPTQ, AWQ, 权重量化, 4bit 推理, 显存压缩]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# 量化推理算法 GPTQ / AWQ

> 五层读懂一个词。这次拆的是：**GPTQ / AWQ**--把 LLM 权重压到 4bit 还不掉精度的两大主力算法。GPTQ 用二阶信息逐层补偿，AWQ 发现 1% 的显著权重决定精度，保护它们就稳了。一张 3090 跑 70B 不再是梦。

---

## L1 · 一句话点破

**GPTQ = 逐层量化 + Hessian 补偿**：按权重重要性顺序量化，用二阶信息（Hessian）补偿误差。**AWQ = 激活感知 + 显著权重保护**：发现只有约 1% 的「显著权重」（大激活对应的）决定精度，全 FP16 保护它们，其余 99% 4bit 量化。两者都把 70B 模型从 140GB 压到 35GB，精度损失 < 1%。

---

## L2 · 通俗类比

LLM 推理的瓶颈是**显存**：70B 模型 FP16 要 140GB，单卡装不下。

**量化**就像把一本精装书变成口袋本：

- 纸张更薄（4bit 比 16bit 省 4 倍）
- 内容基本一样（精度损失小）
- 携带方便（单卡能装下）

**挑战**：粗暴量化会掉精度。直接把权重 round 到 4bit，困惑度暴涨 10 倍。

**GPTQ 的思路**（逐个量化 + 补偿）：

像**搬家具**：

- 一件一件搬（逐个权重量化）
- 每搬一件，记录位置误差（量化误差）
- 后续家具位置微调，补偿前面误差（Hessian 补偿）
- 搬完整体误差最小化

**AWQ 的思路**（保护重要的）：

像**搬古董**：

- 大部分是普通家具（99% 权重），随便打包
- 少量古董（1% 显著权重），单独保险箱
- 普通家具塞 4bit 口袋，古董保留 FP16
- 关键是：先看哪些是古董（用激活判断）

**关键发现**（AWQ 的核心 insight）：

- 不是「权重大」就重要
- 而是「激活大」对应的权重才重要
- 1% 显著权重决定 99% 的精度

**对比**：

| 维度 | RTN（朴素量化） | GPTQ | AWQ |
|------|----------------|------|-----|
| 思路 | 直接 round | 逐层补偿 | 显著权重保护 |
| 精度 | 差（4bit PPL 翻倍） | 好（PPL+1%） | 好（PPL+1%） |
| 量化时间 | 秒级 | 小时级 | 分钟级 |
| 推理速度 | 基线 | 1.5-2x | 1.5-2x |
| 显存 | 4x 压缩 | 4x 压缩 | 4x 压缩 |

**配合 LoRA**（QLoRA）：4bit 量化基础模型 + LoRA 微调，单卡 24GB 微调 70B。

**代价**：

- 量化过程有算力成本（GPTQ 较慢）
- 推理有反量化开销（但被显存节省抵消）
- 极致低 bit（<4bit）精度掉得快
- 不同层敏感度不同，要分组量化

---

## L3 · 正经定义

**GPTQ**（Generalized Post-Training Quantization）：逐层后训练量化算法。对每层权重 $W$，按列顺序量化每行，用 Hessian 矩阵 $H = X X^T$（$X$ 是该层输入）的二阶信息补偿已量化权重对未量化权重的影响。基于 OBQ（Optimal Brain Quantization）框架，加入 Cholesky 重构和惰性 batch 更新，使其能处理百亿参数模型。

**AWQ**（Activation-aware Weight Quantization）：激活感知权重量化。核心观察：**只有约 1% 的显著权重**（大激活通道对应的）决定模型精度。策略：对每通道计算激活幅度 $s$，用缩放因子 $s$ 将显著权重「放大」（保护），其余权重 4bit 量化，推理时再反缩放。

**共性**：

- 都是 **PTQ**（Post-Training Quantization），无需训练 / 微调
- 都是 **W4A16**（权重 4bit，激活 16bit）
- 都能把 70B 压到 35GB（单卡可跑）
- 都有 < 1% 精度损失

**参考资料**：

- 📄 Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023
- 📄 Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, MLSys 2024
- 📄 Frantar & Alistarhp, *Optimal Brain Compression*（OBC/OBQ，GPTQ 理论基础）
- 🔧 AutoGPTQ：https://github.com/PanQiWei/AutoGPTQ
- 🔧 AutoAWQ：https://github.com/casper-hansen/AutoAWQ
- 🔧 bitsandbytes（NF4 / QLoRA）：https://github.com/bitsandbytes-foundation/bitsandbytes

---

## L4 · 原理深挖

### 4.1 量化的基本框架

**W4A16 量化**：权重 4bit，激活 FP16。

**量化函数**（per-channel）：

$$
\hat{W} = \text{dequant}(\text{quant}(W)) = \frac{\text{round}(W / \Delta)}{\Delta}, \quad \Delta = \frac{\max(|W|)}{7}
$$

其中 $\Delta$ 是 scale，4bit 有 16 个值（含符号位时 [-8, 7]）。

**反量化**：推理时 $\hat{W} \cdot X$，其中 $\hat{W}$ 是反量化回 FP16 的权重。

**朴素量化 RTN**（Round-To-Nearest）：

- 直接 round 到 4bit
- 简单但精度差
- 大模型 4bit RTN 困惑度（PPL）翻 2-3 倍

### 4.2 GPTQ：基于二阶信息的逐层补偿

**目标**：量化后 $\hat{W}$ 与原 $W$ 在输出上尽量一致。

**误差度量**（每层）：

$$
E = \| W X - \hat{W} X \|^2_F
$$

其中 $X$ 是该层的输入（校准数据），$H = X X^T$ 是 Hessian。

**OBQ 思想**：按列（权重维度）逐个量化，每量化一个，更新剩余权重以补偿误差。

**更新公式**（量化第 $i$ 列后）：

$$
W_{:, \text{rest}} \leftarrow W_{:, \text{rest}} - \frac{W_{:, i} - \text{quant}(W_{:, i})}{H_{ii}} \cdot H_{i, \text{rest}}
$$

**直观**：

- 量化 $W_{:, i}$ 引入误差 $\delta_i = W_{:, i} - \text{quant}(W_{:, i})$
- 这个误差通过 Hessian 影响其他列
- 用 $H$ 的二阶信息反向补偿，最小化总误差

**GPTQ 的工程优化**：

1. **Cholesky 重构**：避免 $H$ 数值不稳定
2. **惰性 batch 更新**：批量处理列，减少内存访问
3. **分组量化**：group size = 128，平衡精度和效率
4. **校准数据**：用少量（128 条）真实文本

**GPTQ 量化流程**：

```python
for layer in model.layers:
    X = collect_inputs(layer, calibration_data)  # 校准输入
    H = X.T @ X  # Hessian, [in_dim, in_dim]
    H += diag(H) * 0.01  # 防止奇异
    
    for i in range(0, in_dim, group_size):
        # 量化第 i 组
        for j in range(i, min(i + group_size, in_dim)):
            q = quantize(W[:, j])
            err = (W[:, j] - q) / H[j, j]
            W[:, j] = q
            W[:, j+1:] -= err.unsqueeze(1) @ H[j, j+1:].unsqueeze(0)
    
    layer.weight = W
```

### 4.3 AWQ：激活感知的显著权重保护

**核心发现**（AWQ 的 insight）：

- 不是大权重重要，而是**大激活对应的权重**重要
- 1% 显著权重决定 99% 精度

**为什么不是大权重**？

- 大权重可能是冗余的（其他权重能补偿）
- 大激活对应的权重影响大（输出幅度大）

**AWQ 策略**：

1. 用校准数据找每通道的激活幅度 $s_c$
2. 显著通道（$s_c$ 大）的权重放大 $s$ 倍（保护）
3. 量化权重
4. 推理时反缩放（除以 $s$）

**数学形式**：

$$
\hat{W} = \text{quant}(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1}
$$

其中 $s$ 是 per-channel 缩放因子，显著通道 $s > 1$。

**为什么有效**？

- 显著通道权重大 → 量化误差小（相对值）
- 非显著通道权重小 → 量化误差大但激活小 → 影响小
- 整体输出误差最小化

**AWQ 的优势**：

- 量化快（不需逐权重补偿，几分钟）
- 精度好（与 GPTQ 相当）
- 推理快（per-channel scale 简单）

**搜索最优 $s$**：

```python
# AWQ: 网格搜索最优缩放因子
best_s = None
best_loss = float('inf')
for s_ratio in [0.0, 0.1, 0.2, ..., 1.0]:  # 0%-100% 显著通道
    s = compute_scale(activations, s_ratio)
    W_scaled = W * s
    W_quant = quantize(W_scaled)
    W_dequant = W_quant / s
    loss = compute_output_error(W, W_dequant, X)
    if loss < best_loss:
        best_loss = loss
        best_s = s
```

### 4.4 精度对比

**Llama-2-70B 4bit 量化，WikiText2 PPL**（越低越好）：

| 方法 | FP16 基线 | RTN | GPTQ | AWQ |
|------|----------|-----|------|-----|
| PPL | 3.32 | 7.48 | 3.44 | 3.43 |
| 损失 | - | +125% | +3.6% | +3.3% |

**结论**：

- RTN 4bit 灾难性掉精度
- GPTQ / AWQ 4bit 精度损失 < 5%
- AWQ 略优于 GPTQ，但差异小

**下游任务**（MMLU 5-shot）：

| 方法 | FP16 | GPTQ | AWQ |
|------|------|------|-----|
| MMLU | 68.9 | 68.6 | 68.7 |
| 损失 | - | -0.3 | -0.2 |

**下游任务精度损失 < 1%**，基本无损。

### 4.5 推理加速

**为什么量化能加速**？

- 显存带宽是 decode 阶段瓶颈（memory-bound）
- 4bit 权重显存读取量减少 4x
- 反量化开销小（per-channel scale）

**加速实测**（Llama-2-70B，A100）：

| 方法 | 显存 | 速度（token/s） | 加速比 |
|------|------|----------------|--------|
| FP16 | 140 GB | 12 | 1x |
| GPTQ 4bit | 35 GB | 22 | 1.83x |
| AWQ 4bit | 35 GB | 25 | 2.08x |

**AWQ 比 GPTQ 快**：AWQ 的 per-channel scale 实现更高效（更友好的 GPU kernel）。

**显存优势**：

- 70B FP16：140 GB（2×A100 80G 或 4×A100 40G）
- 70B 4bit：35 GB（1×A100 40G 或 2×RTX 3090）
- 单卡 24GB 也能跑 70B（配合 offload）

### 4.6 量化的工程实践

**主流工具链**：

| 工具 | 支持 | 特点 |
|------|------|------|
| AutoGPTQ | GPTQ | GPTQ 官方实现，社区生态好 |
| AutoAWQ | AWQ | AWQ 官方实现，速度快 |
| bitsandbytes | NF4/FP4 | QLoRA 配套，训练友好 |
| llama.cpp | GGUF Q4_K_M | CPU/Mac 推理 |
| TensorRT-LLM | GPTQ/AWQ/INT8 | NVIDIA 极致优化 |

**vLLM 加载量化模型**：

```bash
# GPTQ
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-70B-GPTQ \
    --quantization gptq

# AWQ
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-70B-AWQ \
    --quantization awq
```

**量化自己的模型**：

```python
# GPTQ
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
model = AutoGPTQForCausalLM.from_pretrained(model_id)
model.quantize(calibration_dataset, quantize_config)
model.save_quantized("model-gptq-4bit")

# AWQ
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained(model_id)
model.quantize(calibration_dataset, quant_config={"w_bit": 4, "q_group_size": 128})
model.save_quantized("model-awq-4bit")
```

### 4.7 量化的局限

**局限 1：低 bit 衰减快**。4bit 是甜点，3bit 精度掉得明显，2bit 几乎不可用。

**局限 2：激活量化更难**。W4A16 还行，W4A4 激活量化精度掉得多（激活 outlier 多）。

**局限 3：不同模型敏感度不同**。LLaMA 比 OPT 更难量化，需要不同 group size。

**局限 4：量化开销**。GPTQ 量化 70B 要几小时，AWQ 几分钟。要算 ROI。

**局限 5：硬件支持**。4bit kernel 需要 Ampere+（A100/3090/H100），老卡不支持。

**局限 6：长上下文衰减**。4bit 量化在长上下文（>4k）下精度衰减更明显。

**局限 7：训练时不能用 PTQ**。PTQ 是推理量化，训练用 QLoRA（NF4 + LoRA）。

### 4.8 量化 vs 蒸馏 vs 剪枝

| 方法 | 思路 | 压缩比 | 精度损失 | 训练成本 |
|------|------|--------|---------|---------|
| 量化（GPTQ/AWQ） | 低 bit 表示 | 4-8x | <5% | 无（PTQ） |
| 蒸馏 | 小模型学大模型 | 任意 | 中 | 高（训练） |
| 剪枝 | 去掉不重要参数 | 2-5x | 中 | 中（结构化） |
| QLoRA | 量化 + LoRA | 4x + 微调 | <1% | 低（LoRA） |

**推荐组合**：

- 推理部署：GPTQ / AWQ 4bit
- 微调大模型：QLoRA（NF4 + LoRA）
- 极致压缩：量化 + 蒸馏 + 剪枝

---

## L5 · 沿革与坑

### 5.1 沿革

- **2022-08**：GPTQ 论文（Frantar et al.），把 OBQ 推广到 GPT 级模型
- **2022-11**：SmoothQuant（激活 outlier 处理，W8A8），激活量化基础
- **2023-06**：AWQ 论文（Lin et al.），激活感知的显著权重保护
- **2023-05**：QLoRA（NF4 + LoRA），4bit 量化训练
- **2023 下半年**：AutoGPTQ / AutoAWQ 工具链成熟
- **2024**：vLLM/TGI/TensorRT-LLM 全面支持 GPTQ/AWQ，4bit 推理成主流
- **2024-2025**：更低 bit（W3A16, W2A16）探索、混合精度量化、训练感知量化

### 5.2 常见坑

**坑 1：4bit 量化期望零精度损失**。4bit 有 < 5% PPL 损失，下游任务 < 1%，但不是零。要评估下游任务。

**坑 2：RTN 当 GPTQ/AWQ 用**。RTN 大模型灾难性掉精度。要用 GPTQ/AWQ。

**坑 3：group_size 设错**。太大精度差，太小效率低。128 是甜点。

**坑 4：校准数据不匹配**。用通用文本量化代码模型，下游精度掉。要用领域相关校准数据。

**坑 5：GPTQ 量化 70B 用小 batch**。GPTQ 量化过程显存大，要 batch 量化或 CPU offload。

**坑 6：以为 AWQ 和 GPTQ 差很多**。两者精度差异 < 1%，主要差在量化速度和推理速度。

**坑 7：4bit 量化在长上下文精度崩**。4bit 长上下文（>8k）精度衰减明显。要评估长上下文场景。

**坑 8：硬件不支持 4bit kernel**。Pascal/Turing 架构不支持 4bit 推理 kernel。要 Ampere+。

**坑 9：量化模型加载方式错**。vLLM/Transformers 加载量化模型要指定 quantization 参数，否则当 FP16 加载。

**坑 10：期望量化提升所有场景**。量化提升 memory-bound 场景（decode、大 batch），compute-bound 场景（小 batch prefill）提升小。

**坑 11：忽略 outlier**。激活有 outlier（部分通道异常大），朴素量化被 outlier 拖累。要 SmoothQuant 处理。

**坑 12：2bit / 3bit 期望可用**。低 bit 精度掉得快。2bit 基本不可用，3bit 需要特殊技术（如 AQLM）。

**坑 13：量化模型和原模型行为不同**。量化是近似，输出有微小差异。要在评估集上验证。

**坑 14：PTQ 模型不能继续训练**。PTQ 是推理量化，要训练用 QLoRA（NF4 + LoRA）。

### 5.3 面试怎么考

1. **GPTQ 和 AWQ 的核心区别？** 答：GPTQ 用二阶信息（Hessian）逐层补偿量化误差，AWQ 发现 1% 显著权重决定精度，保护它们。两者精度相近，AWQ 量化更快、推理更快。
2. **为什么 AWQ 不保护大权重而是大激活对应的权重？** 答：大权重可能冗余（其他权重补偿），大激活对应的权重影响输出幅度大。保护大激活对应的权重最小化输出误差。
3. **4bit 量化为什么是甜点？** 答：<4bit 精度掉得快（2bit 几乎不可用），>4bit 压缩比不够。4bit 是精度和压缩的最佳平衡。
4. **GPTQ 的 Hessian 补偿为什么有效？** 答：量化一列权重引入误差，通过 Hessian 二阶信息反向传播到其他列，最小化总输出误差。比朴素 round 误差小得多。
5. **W4A16 vs W4A4？** 答：W4A16（权重 4bit 激活 16bit）精度好，激活有 outlier 量化难。W4A4 激活量化精度掉得多，需要 SmoothQuant 等处理 outlier。
6. **量化 vs 蒸馏 vs 剪枝？** 答：量化低 bit 表示（4-8x，<5% 损失，无训练），蒸馏小模型学大（任意压缩，中损失，高训练），剪枝去掉不重要参数（2-5x，中损失，中训练）。推理部署首选量化。

---

## 速记卡

**量化框架**：

```
RTN:    round(W / Δ) * Δ              # 朴素，精度差
GPTQ:   逐列量化 + Hessian 补偿         # 二阶信息
AWQ:    quant(W * s) / s, s 按激活     # 显著权重保护
QLoRA:  NF4 量化 + LoRA                # 训练友好
```

**GPTQ 补偿公式**：

$$
W_{:, \text{rest}} \leftarrow W_{:, \text{rest}} - \frac{W_{:, i} - \text{quant}(W_{:, i})}{H_{ii}} \cdot H_{i, \text{rest}}
$$

**AWQ 缩放**：

$$
\hat{W} = \text{quant}(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1}
$$

**精度对比（Llama-2-70B, WikiText2 PPL）**：

| 方法 | PPL | 损失 |
|------|-----|------|
| FP16 | 3.32 | - |
| RTN 4bit | 7.48 | +125% |
| GPTQ 4bit | 3.44 | +3.6% |
| AWQ 4bit | 3.43 | +3.3% |

**推理对比（Llama-2-70B, A100）**：

| 方法 | 显存 | 速度 | 加速 |
|------|------|------|------|
| FP16 | 140 GB | 12 tok/s | 1x |
| GPTQ 4bit | 35 GB | 22 tok/s | 1.83x |
| AWQ 4bit | 35 GB | 25 tok/s | 2.08x |

**典型参数**：

| 参数 | 典型值 | 说明 |
|------|--------|------|
| bits | 4 | 甜点 |
| group_size | 128 | 平衡精度和效率 |
| 校准数据 | 128 条 | 领域相关 |
| desc_act (GPTQ) | False | 顺序无激活 |
| 量化时间 | GPTQ 几小时 / AWQ 几分钟 | |

**主流工具**：

| 工具 | 量化方法 | 适用 |
|------|---------|------|
| AutoGPTQ | GPTQ | 通用 |
| AutoAWQ | AWQ | 通用，更快 |
| bitsandbytes | NF4/FP4 | QLoRA 训练 |
| llama.cpp | GGUF Q4_K_M | CPU/Mac |
| TensorRT-LLM | GPTQ/AWQ | NVIDIA 部署 |

**一句话记忆**：GPTQ/AWQ 是 LLM 4bit 量化的两大主力。GPTQ 用 Hessian 二阶信息逐层补偿量化误差，AWQ 发现 1% 显著权重（大激活对应）决定精度、用缩放因子保护。两者都把 70B 从 140GB 压到 35GB，精度损失 <5% PPL、<1% 下游任务，推理 1.5-2x 加速。4bit 是甜点，group_size 128 是经验值。PTQ 推理部署用 GPTQ/AWQ，训练用 QLoRA（NF4 + LoRA）。RTN 朴素量化大模型灾难性掉精度，别用。

---

> *上一篇：[Speculative Decoding 推测解码](./speculative-decoding) -- 延迟优化末篇，量化进一步压缩显存，单卡跑 70B。*
> *下一篇预告：Agent 专题 -- ReAct / Function Calling / Planning / Memory / Multi-Agent，LLM 从"能聊"到"能干活"的关键一跃。*
