---
title: Prefix/Prompt Tuning 前缀调优
slug: prefix-tuning
category: 进阶专题
tags: [Prefix Tuning, Prompt Tuning, PEFT, 软提示, 前缀向量]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# Prefix/Prompt Tuning 前缀调优

> 五层读懂一个词。这次拆的是：**Prefix Tuning 和 Prompt Tuning**--比 LoRA/Adapter 更极端的 PEFT，不改模型任何参数，只在输入前拼一段可训练的"软提示"向量。参数量 0.01%，但效果上限低于 LoRA。

---

## L1 · 一句话点破

**Prefix/Prompt Tuning = 冻结整个模型 + 只训一段拼接在输入前的虚拟 token 向量**。Prefix Tuning 在每层 attention 的 key/value 前加前缀，Prompt Tuning 只在 embedding 层加，参数量极小但表达能力弱。

---

## L2 · 通俗类比

LoRA 和 Adapter 好歹是在模型**内部**动刀（加旁路或插模块），Prefix/Prompt Tuning 更极端：**模型完全不动，只在输入前面拼一段"暗号"**。

想象 LLM 是个只会按固定流程思考的专家。普通 prompt 是你用自然语言跟他说话（"请帮我总结..."）。Prompt Tuning 是在输入前面拼一段**模型听得懂但人看不懂的向量**，这段向量通过训练学到，能引导模型按特定任务输出。

**两个层次**：

- **Prompt Tuning**（更简单）：只在最开始的 embedding 层拼虚拟 token，后续层通过 attention 间接影响
- **Prefix Tuning**（更强）：在每一层 attention 的 Key 和 Value 前面都拼前缀向量，直接影响每层 attention

**和硬 prompt 的区别**：

- 硬 prompt：自然语言文字，离散，不可微，只能人工设计或搜索
- 软 prompt（Prompt/Prefix Tuning）：连续向量，可微，能梯度优化

**三种 PEFT 的极端程度对比**：

| 方法 | 改什么 | 参数量 | 表达能力 |
|------|--------|--------|---------|
| Adapter | 每层插模块 | 1-5% | 强 |
| LoRA | 每层加 $BA$ 旁路 | 0.1-1% | 中 |
| Prompt Tuning | 只加输入前缀 | 0.01% | 弱 |
| Prefix Tuning | 每层 K/V 加前缀 | 0.1% | 中 |

**核心权衡**：参数越少越省资源，但表达能力越弱。Prompt Tuning 在大模型（>10B）上效果接近全参，小模型上明显落后。

---

## L3 · 正经定义

**Prefix Tuning**：Li & Liang (ACL 2021) 提出，冻结模型，在每层 attention 的 Key 和 Value 前面拼一段可训练的前缀向量（长度 $m$），只训前缀。

**Prompt Tuning**：Lester et al. (2021) 提出，Prompt Tuning 的简化版，只在 embedding 层拼虚拟 token，后续层通过前向传播间接影响。

**Prefix Tuning 的前向**：

对每层 $i$，attention 的 Key/Value 变为：

$$
K_i = [P_K^i; W_K h], \quad V_i = [P_V^i; W_V h]
$$

其中 $P_K^i, P_V^i \in \mathbb{R}^{m \times d}$ 是该层的前缀向量，$m$ 是前缀长度，$[\cdot;\cdot]$ 是拼接。前缀只影响 attention 的 key/value，query 不变。

**Prompt Tuning 的前向**：

$$
h = [P_e; E(x)]
$$

$P_e \in \mathbb{R}^{m \times d}$ 是可训练的 embedding 前缀，$E(x)$ 是输入的 embedding。前缀只在 embedding 层拼接，后续层通过 attention 传播。

**参数量**：

- Prefix Tuning：$L \times 2 \times m \times d$（$L$ 层，每层 K/V 各 $m \times d$）
- Prompt Tuning：$m \times d$（只在 embedding 层）

**参考资料**：

- 📄 Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation*, ACL 2021, arXiv:2101.00190
- 📄 Lester et al., *The Power of Scale for Parameter-Efficient Prompt Tuning*, EMNLP 2021, arXiv:2104.08691
- 📄 Liu et al., *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks*, ACL 2022
- 📄 Qin & Eisner, *Learning How to Ask: Querying LMs with Mixtures of Soft Prompts*, NAACL 2021

---

## L4 · 原理深挖

### 4.1 Prompt Tuning：最简形式

Prompt Tuning 的实现极简：

```python
class PromptTuning(nn.Module):
    def __init__(self, d=4096, m=20):
        super().__init__()
        # 可训练的软提示 embedding
        self.prompt_embeds = nn.Parameter(torch.randn(m, d))
    
    def forward(self, input_ids, model):
        # 1. 输入 embedding
        input_embeds = model.embed_tokens(input_ids)  # [batch, seq, d]
        # 2. 拼接前缀
        prompt = self.prompt_embeds.unsqueeze(0).expand(input_embeds.size(0), -1, -1)
        embeds = torch.cat([prompt, input_embeds], dim=1)  # [batch, m+seq, d]
        # 3. 喂给模型（模型冻结）
        return model(inputs_embeds=embeds)
```

**关键点**：

- 软提示是 embedding 层的连续向量，不是 token id
- 模型完全冻结，只有 `prompt_embeds` 有梯度
- 前缀长度 $m$ 通常 20-100

**为什么有效**：

- 大模型有很强的 in-context learning 能力，前面几个 token 的 embedding 能强引导输出
- 软 prompt 优化空间比硬 prompt 大（连续可微 vs 离散不可微）

### 4.2 Prefix Tuning：每层注入

Prefix Tuning 比 Prompt Tuning 更强：在每层 attention 的 K/V 前都注入前缀。

```python
class PrefixTuning(nn.Module):
    def __init__(self, d=4096, m=20, L=32):
        super().__init__()
        # 每层 K/V 各一组前缀
        self.prefix_keys = nn.Parameter(torch.randn(L, m, d))
        self.prefix_values = nn.Parameter(torch.randn(L, m, d))
    
    def inject(self, layer_idx, k, v):
        # k: [batch, heads, seq, d_head]
        # 在 seq 维度前面拼前缀
        pk = self.prefix_keys[layer_idx].expand(k.size(0), k.size(1), -1, -1)
        pv = self.prefix_values[layer_idx].expand(v.size(0), v.size(1), -1, -1)
        k = torch.cat([pk, k], dim=2)  # [batch, heads, m+seq, d_head]
        v = torch.cat([pv, v], dim=2)
        return k, v
```

**为什么比 Prompt Tuning 强**：

- Prompt Tuning 只在 embedding 层注入，后续层靠 attention 间接传播，信号衰减
- Prefix Tuning 每层直接注入 K/V，直接影响每层 attention 计算
- 实验显示 Prefix Tuning 在小模型（<10B）上明显优于 Prompt Tuning

### 4.3 Prefix 的重参数化（Reparameterization）

直接训 $L \times 2 \times m \times d$ 的前缀参数量大且不稳定。Li & Liang 用**重参数化**：

```python
class PrefixReparam(nn.Module):
    def __init__(self, d=4096, m=20, L=32, r=64):
        super().__init__()
        # 用小矩阵生成前缀，类似 LoRA
        self.prefix_mlp = nn.Sequential(
            nn.Linear(d, r),       # 降维
            nn.Tanh(),
            nn.Linear(r, 2 * L * m * d)  # 升维到所有层前缀
        )
    
    def forward(self):
        # 从一个小输入生成所有前缀
        return self.prefix_mlp(self.input)
```

**好处**：

- 参数量从 $2Lmd$ 降到 $dr + 2Lmdr$（中间瓶颈 $r$）
- 训练更稳定（间接优化）
- 推理时可展开成直接前缀，无额外开销

### 4.4 P-Tuning v2：统一框架

Liu et al. (2022) 的 P-Tuning v2 发现：**Prefix Tuning 加上一些工程优化（多任务、深度提示、任务特定），能在大范围模型规模和任务上逼近全参微调**。

**P-Tuning v2 的结论**：

- 小模型（<1B）：Prefix/Prompt Tuning 明显落后全参
- 中模型（1-10B）：差距缩小
- 大模型（>10B）：Prompt Tuning 接近全参

**关键工程优化**：

- 深度提示（每层注入，而非只 embedding 层）
- 多任务学习（共享前缀）
- 任务特定头（前缀 + 任务头）

### 4.5 软 prompt vs 硬 prompt

| 维度 | 硬 prompt | 软 prompt |
|------|-----------|-----------|
| 形式 | 自然语言文字 | 连续向量 |
| 优化 | 人工设计 / 离散搜索 | 梯度下降 |
| 可微 | 否 | 是 |
| 可解释 | 是 | 否 |
| 灵活性 | 低（词汇有限） | 高（连续空间） |
| 训练成本 | 0（直接用） | 需训练数据 |

**软 prompt 的局限**：

- 不可解释（学到的向量人看不懂）
- 泛化性差（一个任务的软 prompt 不能迁移到另一任务）
- 初始化敏感（不同初始化效果差很多）

### 4.6 三种 PEFT 的表达能力对比

| 方法 | 改动位置 | 非线性 | 表达能力 |
|------|---------|--------|---------|
| Prompt Tuning | 仅 embedding | 无（间接） | 最弱 |
| Prefix Tuning | 每层 K/V | 无（间接） | 中 |
| Adapter | 每层串行模块 | 有（ReLU） | 强 |
| LoRA | 每层并行 $BA$ | 无 | 中 |
| 全参微调 | 所有权重 | 有 | 最强 |

**实验对比**（SuperGLUE，T5 模型）：

| 方法 | 1B 模型 | 11B 模型 |
|------|---------|----------|
| 全参微调 | 100% | 100% |
| Adapter | 99% | 99% |
| LoRA | 98% | 99% |
| Prefix Tuning | 95% | 98% |
| Prompt Tuning | 88% | 97% |

**结论**：大模型上 Prompt Tuning 接近全参，小模型上明显落后。

### 4.7 何时用 Prefix/Prompt Tuning

**适合**：

- 超大模型（>10B），Prompt Tuning 接近全参
- 极致省参数场景（多任务共享，每个任务只存几 KB 前缀）
- 多任务推理（不同任务切换不同前缀，零切换开销）
- 黑盒模型（只能 API 调用，不能改内部）

**不适合**：

- 小模型（<10B），效果明显落后
- 复杂任务（需要强表达能力）
- 需要可解释性（软 prompt 不可解释）
- 少样本场景（软 prompt 需要训练数据）

**实践建议**：

- 超大模型 + 多任务 + 黑盒：Prompt Tuning
- 中模型 + 单任务：LoRA 更优
- 小模型：直接全参微调或 LoRA

### 4.8 Prompt Tuning 的工程优势

**多任务存储**：

- 全参微调：每个任务一份完整模型（GB 级）
- LoRA：每个任务一份适配器（MB 级）
- Prompt Tuning：每个任务一份前缀（**KB 级**）

10000 个任务，Prompt Tuning 总存储 = 10000 × 20KB = 200MB。LoRA 要 10000 × 50MB = 500GB。这是 Prompt Tuning 的杀手锏。

**黑盒模型**：

- Prompt Tuning 只需 API 调用（输入 embedding 可控）
- LoRA/Adapter 要改模型内部
- 对 GPT-4 等闭源模型，Prompt Tuning 是唯一可行的"微调"

---

## L5 · 沿革与坑

### 5.1 沿革

- **2020**：软 prompt 概念出现（Liu et al. P-Tuning）
- **2021-01**：Prefix Tuning（Li & Liang）发布
- **2021-04**：Prompt Tuning（Lester et al.）发布，Google 证明大模型上接近全参
- **2022**：P-Tuning v2 统一框架，发现"深度提示"是关键
- **2022-2023**：LoRA 崛起，Prefix/Prompt Tuning 在 LLM 时代退居超大模型多任务场景
- **2024-2025**：Prompt Tuning 在黑盒模型 API 微调场景复活

### 5.2 常见坑

**坑 1：小模型硬上 Prompt Tuning**。<10B 模型 Prompt Tuning 明显落后全参，要用 LoRA 或 Prefix Tuning。

**坑 2：前缀长度 $m$ 太短**。$m=5$ 表达能力不足，$m \in [20, 100]$ 是经验值。

**坑 3：初始化没调**。随机初始化训练不稳定，要用任务相关 token 的 embedding 初始化。

**坑 4：Prompt Tuning 当 LoRA 替代**。Prompt Tuning 表达能力弱，复杂任务效果差。要按场景选。

**坑 5：Prefix Tuning 没用重参数化**。直接训前缀参数量大且不稳定，要用 MLP 重参数化。

**坑 6：软 prompt 期望可解释**。软 prompt 是连续向量，人看不懂，别想解释为什么这么训。

**坑 7：跨任务复用软 prompt**。一个任务的软 prompt 不能迁移到另一任务，每个任务要重训。

**坑 8：评估只看大模型**。大模型 Prompt Tuning 接近全参，但小模型差。要在目标规模上评估。

**坑 9：多任务前缀切换开销没估**。多任务推理时切换前缀看似零开销，但 cache 失效，实际有成本。

**坑 10：忘了前缀占 context**。前缀 $m$ 个 token 占用 context window，长任务要算进去。

### 5.3 面试怎么考

1. **Prefix Tuning 和 Prompt Tuning 的区别？** 答：Prefix 在每层 attention K/V 注入前缀，Prompt 只在 embedding 层拼接。Prefix 表达能力更强。
2. **软 prompt 和硬 prompt 的区别？** 答：软 prompt 是连续向量可微可优化，硬 prompt 是自然语言不可微。软 prompt 灵活但不可解释。
3. **Prompt Tuning 什么时候接近全参微调？** 答：大模型（>10B）+ 简单任务，Prompt Tuning 接近全参。小模型明显落后。
4. **Prefix Tuning 为什么要重参数化？** 答：直接训前缀参数量大且不稳定，用 MLP 重参数化降维训练更稳，推理时展开无开销。
5. **Prompt Tuning 的杀手锏？** 答：多任务存储极省（KB 级 vs LoRA 的 MB 级），且可用于黑盒 API 模型。

---

## 速记卡

| 方法 | 注入位置 | 参数量 | 表达能力 |
|------|---------|--------|---------|
| Prompt Tuning | 仅 embedding | $md$ | 最弱 |
| Prefix Tuning | 每层 K/V | $2Lmd$ | 中 |
| P-Tuning v2 | 每层（深度提示） | $2Lmd$ | 中 |

**对比表**：

| 方法 | 参数量 | 大模型效果 | 小模型效果 | 多任务存储 |
|------|--------|-----------|-----------|-----------|
| 全参 | 100% | 100% | 100% | GB 级 |
| LoRA | 0.1-1% | 99% | 98% | MB 级 |
| Adapter | 1-5% | 99% | 99% | MB 级 |
| Prefix | 0.1% | 98% | 95% | MB 级 |
| Prompt | 0.01% | 97% | 88% | **KB 级** |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| 前缀长度 $m$ | 20-100 | 表达能力 |
| 重参数化维度 $r$ | 64-128 | 训练稳定性 |
| 初始化 | 任务相关 token | 收敛速度 |

**一句话记忆**：Prefix/Prompt Tuning = 冻结模型只训输入前的软提示向量。Prompt Tuning 仅 embedding 层（0.01% 参数），Prefix Tuning 每层 K/V 注入（0.1%）。大模型（>10B）上接近全参，小模型落后。杀手锏是多任务存储 KB 级 + 黑盒 API 模型可用。复杂任务还是 LoRA 更优。

---

> *上一篇：[Adapter Tuning 适配器微调](./adapter-tuning) -- PEFT 老前辈，瓶颈模块。*
> *下一篇：[PEFT 总览与选型](./peft) -- 全参 / LoRA / QLoRA / Adapter / Prefix / Prompt 怎么选。*
