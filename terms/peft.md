---
title: PEFT 总览与选型
slug: peft
category: 进阶专题
tags: [PEFT, 参数高效微调, 选型, LoRA, QLoRA, Adapter, Prefix Tuning]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# PEFT 总览与选型

> 五层读懂一个词。这次拆的是：**PEFT（Parameter-Efficient Fine-Tuning）**--把全参 / LoRA / QLoRA / Adapter / Prefix / Prompt 串成一张地图，告诉你什么场景选什么。

---

## L1 · 一句话点破

**PEFT = 冻结大部分预训练参数 + 只训少量增量参数**。通过低秩分解（LoRA）、瓶颈模块（Adapter）、软提示（Prefix/Prompt）等方式，用 0.01%-5% 的参数量逼近全参微调，省显存、省存储、防遗忘。

---

## L2 · 通俗类比

全参微调是"连根拔起重新栽"--每个参数都要重新学，显存爆炸、每个任务存一份完整模型。PEFT 的哲学是：**预训练模型已经学好通用能力，微调只需小幅度"任务特化"，不用动所有参数**。

不同 PEFT 方法的"小幅度"做法不同：

- **LoRA**：在每层旁加一对小矩阵 $BA$，只训 $A, B$，推理可合并
- **QLoRA**：LoRA + 基座 4bit 量化，进一步省显存
- **Adapter**：每层插入瓶颈小模块，有非线性但推理有开销
- **Prefix Tuning**：每层 K/V 前拼可训练向量
- **Prompt Tuning**：只在 embedding 层拼软提示，最省参数

**一张图看懂 PEFT 谱系**：

```
全参微调（100% 参数）
    │
    ├─ Adapter（1-5%，串行，有非线性）
    │
    ├─ LoRA（0.1-1%，并行，可合并）
    │    │
    │    └─ QLoRA（基座 4bit + LoRA）
    │
    ├─ Prefix Tuning（0.1%，每层 K/V 前缀）
    │
    └─ Prompt Tuning（0.01%，仅 embedding 前缀）
```

**核心权衡轴**：

- 参数量：全参 > Adapter > LoRA > Prefix > Prompt
- 表达能力：全参 > Adapter > LoRA ≈ Prefix > Prompt
- 推理开销：Adapter > Prefix > LoRA(零) ≈ Prompt(零)
- 多任务存储：全参(GB) > LoRA(MB) > Adapter(MB) > Prefix(MB) > Prompt(KB)

**选型的本质**：在"参数量 / 效果 / 推理开销 / 存储成本"四维里找最适合你场景的点。

---

## L3 · 正经定义

**PEFT（Parameter-Efficient Fine-Tuning）**：一类微调方法的总称，通过冻结大部分预训练参数、只训练少量新增或选定的参数，实现高效适配。核心目标是用极小参数量（通常 <5%）逼近全参微调效果。

**PEFT 的三大子类**：

1. **增量式（Additive）**：在模型中新增小模块，只训新模块
   - Adapter：串行瓶颈模块
   - Prefix/Prompt Tuning：前缀向量
2. **重参数式（Reparameterization-based）**：用低秩形式重参数化原权重
   - LoRA：$\Delta W = BA$
   - AdaLoRA：自适应秩分配
   - DoRA：方向 + 幅度分解
3. **选择式（Selective）**：只训原模型的部分参数
   - Bias-only tuning：只训 bias
   - BitFit：只训 bias
   - LayerNorm-only：只训 LayerNorm

**工业实践的主流**：LoRA / QLoRA（占 90%+），Adapter（多任务场景），Prefix/Prompt（超大模型多任务），选择式（学术研究多）。

**参考资料**：

- 📄 Lialin et al., *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning*, arXiv:2303.15647, 2023（PEFT 综述）
- 📄 He et al., *Towards a Unified View of Parameter-Efficient Transfer Learning*, ICLR 2022（PEFT 统一框架）
- 🔧 HuggingFace PEFT 库：https://github.com/huggingface/peft
- 📄 Hu et al., *LoRA*, ICLR 2022（LoRA 原论文）

---

## L4 · 原理深挖

### 4.1 PEFT 的统一视角

He et al. (ICLR 2022) 提出统一视角：**所有 PEFT 方法本质都是在"间接"调整模型的有效权重**。

$$
h' = W h + \Delta h
$$

不同方法的 $\Delta h$ 形式不同：

| 方法 | $\Delta h$ 形式 | 等效权重修改 |
|------|----------------|--------------|
| Adapter | $W_{up} \sigma(W_{down} h)$ | 串行插入（非线性） |
| LoRA | $BAh$ | $\Delta W = BA$（线性并行） |
| Prefix | 注意力机制的 K/V 前缀 | 间接影响 attention |
| Prompt | embedding 前缀 | 间接影响输入 |

**统一视角的洞察**：

- LoRA 和 Adapter 都是"加增量"，但 LoRA 是线性可合并，Adapter 有非线性不可合并
- Prefix/Prompt 是"加输入"，不直接改权重，影响最间接
- 表达能力：直接改权重 > 间接改 attention > 间接改输入

### 4.2 各方法的详细对比

| 维度 | 全参 | Adapter | LoRA | QLoRA | Prefix | Prompt |
|------|------|---------|------|-------|--------|--------|
| 训练参数 | 100% | 1-5% | 0.1-1% | 0.1-1% | 0.1% | 0.01% |
| 基座精度 | 16bit | 16bit | 16bit | **4bit** | 16bit | 16bit |
| 非线性 | 有 | 有 | 无 | 无 | 间接 | 间接 |
| 推理开销 | 基准 | **+5-10%** | 零（合并） | 零（合并） | +少量 | 零 |
| 70B 显存 | ~650GB | ~150GB | ~150GB | **~48GB** | ~140GB | ~140GB |
| 多任务存储 | GB | MB | MB | MB | MB | **KB** |
| 效果上限 | 最高 | 高 | 高 | 高（略低） | 中 | 低（大模型接近） |

### 4.3 选型决策树

```
你的场景是什么？
│
├─ 单卡 / 消费级 GPU + 大模型
│  └─ QLoRA（4bit 基座 + LoRA，单卡 48GB 微调 70B）
│
├─ 多卡 + 中大模型 + 单任务
│  └─ LoRA（16bit 基座 + LoRA，推理可合并零开销）
│
├─ 多任务持续学习 + 任务组合
│  └─ AdapterFusion（多 Adapter 融合，任务不干扰）
│
├─ 超大模型（>10B）+ 海量任务 + 黑盒 API
│  └─ Prompt Tuning（KB 级存储，API 可用）
│
├─ 超大模型 + 中量任务
│  └─ Prefix Tuning（每层注入，效果优于 Prompt）
│
├─ 极致性能 + 资源充足
│  └─ 全参微调（上限最高）
│
└─ 学术研究 / 极致省参数
   └─ BitFit / LayerNorm-only（只训 bias 或 LayerNorm）
```

### 4.4 PEFT 的工程实现

**HuggingFace PEFT 库**是事实标准，统一接口：

```python
from peft import LoraConfig, get_peft_model, TaskType

# 1. 定义 PEFT 配置
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

# 2. 包装模型
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# 输出: trainable params: 1,572,864 || all params: 7,000,000,000 || trainable%: 0.0225%

# 3. 正常训练
trainer = Trainer(model=model, ...)
trainer.train()

# 4. 保存（只存 PEFT 参数，几十 MB）
model.save_pretrained("./lora_adapter")

# 5. 加载（基座 + PEFT 适配器）
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# 6. 合并（推理零开销）
model = model.merge_and_unload()
```

**关键点**：

- `get_peft_model` 自动冻结基座，只解冻 PEFT 参数
- 保存只存适配器（几十 MB），不存基座
- `merge_and_unload` 把 LoRA 合并回基座，推理零开销

### 4.5 PEFT 的训练技巧

**学习率**：

- 全参微调：$2e-5$
- LoRA / QLoRA：$1e-4$ 到 $2e-4$（比全参大 5-10 倍）
- Adapter：$1e-4$
- Prefix/Prompt：$1e-3$ 到 $1e-2$（参数少，学习率要大）

**Batch size**：

- PEFT 参数少，小 batch 梯度噪声大
- 建议 batch ≥ 8 或用梯度累积

**Warmup**：

- PEFT 训练初期不稳定，要 warmup（前 10% 步数线性升温）

**Dropout**：

- LoRA / Adapter 加 dropout（0.05-0.1）防过拟合
- Prefix/Prompt 通常不需要

**目标模块选择**（LoRA）：

- 起步：$W_q, W_v$
- 进阶：$W_q, W_k, W_v, W_o$
- 激进：+ FFN $W_{up}, W_{down}$

### 4.6 PEFT 的评估

**评估维度**：

1. **效果**：下游任务准确率 / loss
2. **参数量**：可训参数占比
3. **显存**：训练 / 推理显存
4. **延迟**：推理延迟变化
5. **存储**：多任务存储成本
6. **训练时间**：端到端训练时间

**评估基准**：

- SuperGLUE：NLU 任务
- MMLU：多任务知识
- 自建任务评估集

**评估坑**：

- 只比参数量不比效果：误导
- 只比小模型不比大模型：Prompt Tuning 在大模型上才接近全参
- 只比单任务不比多任务：AdapterFusion 多任务优势没体现

### 4.7 PEFT 的演进趋势

- **2021-2022**：Adapter / Prefix / Prompt / LoRA 百花齐放
- **2023**：LoRA 成为事实标准，QLoRA 进一步省显存
- **2024**：DoRA / AdaLoRA / LoRA+ 等变体涌现
- **2025**：

  - 多 LoRA 管理（LoRA Hub / LoRA Routing）
  - PEFT + RLHF 结合（DPO + LoRA）
  - 在线学习 / 持续学习场景的 PEFT
  - PEFT 与长上下文、多模态结合

### 4.8 PEFT 的开放问题

**问题 1：表达能力上限**。LoRA 等方法的低秩近似在某些任务（领域大跨度）确实不如全参。何时 PEFT 够用，何时必须全参，没有清晰边界。

**问题 2：多 PEFT 组合**。多个 LoRA 如何组合（加法、路由、混合）才有最佳效果，仍是研究热点。

**问题 3：PEFT 与量化的耦合**。QLoRA 证明 4bit + LoRA 可行，但 3bit / 2bit 的精度损失如何补偿，未解决。

**问题 4：持续学习**。PEFT 在新任务上训新适配器，如何避免旧任务遗忘（AdapterFusion 部分解决但不完美）。

**问题 5：可解释性**。LoRA 的 $BA$ 学到什么、Prefix 的软提示对应什么语义，难以解释。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2019**：Adapter 开山，PEFT 概念成型
- **2020**：Prefix/Prompt Tuning 软提示方向兴起
- **2021**：LoRA 发布，PEFT 进入"低秩重参数化"时代
- **2022**：He et al. 统一框架，PEFT 理论成熟
- **2023**：QLoRA 让消费级 GPU 微调 70B 成为可能，HuggingFace PEFT 库普及
- **2024-2025**：LoRA 成事实标准，变体（DoRA / AdaLoRA）持续演进，PEFT + RLHF / 多模态结合

### 5.2 常见坑

**坑 1：盲目追求最少参数**。Prompt Tuning 参数最少但效果最差，要按场景选，不是越省越好。

**坑 2：LoRA $r$ 调太大当全参用**。$r=256$ 失去参数优势，效果还不一定比全参好。$r \in [8, 64]$。

**坑 3：QLoRA 用 INT4 而非 NF4**。INT4 量化损失大，必须用 NF4（针对正态分布优化）。

**坑 4：Adapter 部署不合并抱怨延迟**。Adapter 不能合并，部署前评估延迟影响，延迟敏感用 LoRA。

**坑 5：学习率用全参的**。PEFT 学习率要比全参大 5-10 倍，用全参学习率训不动。

**坑 6：评估只看 loss**。PEFT loss 下降快但下游任务可能差，要看任务指标。

**坑 7：多 LoRA 直接相加**。多个任务的 LoRA 直接相加会互相干扰，要用 LoRA Hub 或路由。

**坑 8：Prompt Tuning 小模型硬上**。<10B 模型 Prompt Tuning 明显落后，要用 LoRA 或 Prefix。

**坑 9：PEFT 当万能解**。复杂任务 / 新语言 / 新领域，PEFT 上限不够，还是要全参。

**坑 10：忘了合并就部署**。LoRA 没合并就部署，每次前向多算 $BAx$，延迟增加。

**坑 11：PEFT + 量化精度叠加**。QLoRA 合并时量化误差和 LoRA 误差叠加，要校准。

**坑 12：PEFT 不评估多任务存储**。多任务场景要比总存储（基座 + N 个适配器），不只比单任务参数。

### 5.3 面试怎么考

1. **PEFT 的三大子类？** 答：增量式（Adapter / Prefix）、重参数式（LoRA）、选择式（BitFit）。
2. **LoRA 为什么成为主流？** 答：参数少（0.1%）、效果接近全参、推理可合并零开销、工程实现简单。
3. **PEFT 怎么选型？** 答：单卡大模型用 QLoRA，多卡单任务用 LoRA，多任务组合用 AdapterFusion，超大模型多任务用 Prompt Tuning。
4. **PEFT 的表达能力上限？** 答：低秩近似在复杂任务（领域大跨度）不如全参，何时够用没有清晰边界。
5. **PEFT 的统一视角？** 答：所有方法本质都是间接调整有效权重，LoRA 线性可合并最强，Adapter 有非线性但不可合并，Prefix/Prompt 间接影响最弱。

---

## 速记卡

| 方法 | 参数量 | 推理开销 | 70B 显存 | 多任务存储 | 适用场景 |
|------|--------|---------|---------|-----------|---------|
| 全参 | 100% | 基准 | ~650GB | GB | 极致性能 |
| Adapter | 1-5% | +5-10% | ~150GB | MB | 多任务组合 |
| LoRA | 0.1-1% | 零（合并） | ~150GB | MB | 通用首选 |
| QLoRA | 0.1-1% | 零（合并） | **~48GB** | MB | 单卡大模型 |
| Prefix | 0.1% | +少量 | ~140GB | MB | 超大模型 |
| Prompt | 0.01% | 零 | ~140GB | **KB** | 黑盒 API + 海量任务 |

**选型决策**：

```
单卡 + 大模型 -> QLoRA
多卡 + 单任务 -> LoRA
多任务组合 -> AdapterFusion
超大模型 + 黑盒 -> Prompt Tuning
极致性能 -> 全参
```

**关键参数**：

| 参数 | LoRA | QLoRA | Adapter | Prefix |
|------|------|-------|---------|--------|
| $r$ / 瓶颈 | 8-64 | 8-64 | 16-64 | - |
| $\alpha$ | $2r$ | $2r$ | - | - |
| 学习率 | $1e-4$ | $1e-4$ | $1e-4$ | $1e-3$ |
| dropout | 0.05 | 0.05 | 0.05 | - |
| 目标层 | $W_q, W_v$ | $W_q, W_v$ | FFN 后 | 每层 K/V |

**一句话记忆**：PEFT = 冻结基座 + 训少量参数（0.01%-5%）逼近全参。LoRA 是事实标准（0.1% 参数 + 推理可合并零开销），QLoRA 解决显存（4bit + LoRA），Adapter 退居多任务组合，Prefix/Prompt 适合超大模型 + 黑盒 API。选型看四维权衡：参数量 / 效果 / 推理开销 / 存储成本。学习率比全参大 5-10 倍是关键。

---

> *上一篇：[Prefix/Prompt Tuning 前缀调优](./prefix-tuning) -- 最极端的 PEFT，只调软提示向量。*
> *下一篇预告：对齐专题 -- SFT / Reward Model / PPO / DPO / KTO，从监督微调到偏好优化的完整对齐链路。*
