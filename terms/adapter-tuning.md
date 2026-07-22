---
title: Adapter Tuning 适配器微调
slug: adapter-tuning
category: 进阶专题
tags: [Adapter, PEFT, 适配器, Bottleneck, 瓶颈架构]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# Adapter Tuning 适配器微调

> 五层读懂一个词。这次拆的是：**Adapter Tuning**--LoRA 之前的 PEFT 老前辈，在 Transformer 每层插入瓶颈小模块，只训模块参数。虽被 LoRA 超越，但它的设计思想启发了整个 PEFT 家族。

---

## L1 · 一句话点破

**Adapter = 在 Transformer 每层插入瓶颈模块（down-proj + 非线性 + up-proj）**。冻结原模型，只训 Adapter，参数量 1-5%，但推理时多一层计算，不如 LoRA 可合并。

---

## L2 · 通俗类比

全参微调像给整栋楼重新装修，每面墙都要动。Adapter 的思路更轻巧：**楼不动，每层插一个"小插件"**。这个小插件是个瓶颈结构：先把信号压缩到低维（比如 64 维），过个非线性，再还原回原维度。

为什么有效？因为"适应新任务"的信息量本来就不大（和 LoRA 同一个洞察），用一个低维瓶颈就够编码这些信息。原模型的通用能力不动，Adapter 只学"任务特化"的部分。

**Adapter vs LoRA 的关键区别**：

| 维度 | Adapter | LoRA |
|------|---------|------|
| 结构 | 串行插入新模块 | 并行加 $\Delta W = BA$ |
| 推理 | 多算一层，**有额外开销** | 可合并回 $W$，**零开销** |
| 参数量 | 1-5% | 0.1-1% |
| 表达能力 | 强（有非线性） | 弱（纯线性） |
| 历史 | 2019，PEFT 开山 | 2022，PEFT 新王 |

**核心问题**：Adapter 串行插入，推理时多一层前向，延迟增加。LoRA 并行加 $\Delta W$，推理时可合并到 $W$，零额外开销。这就是 LoRA 取代 Adapter 的关键。

**但 Adapter 没死**：它的设计思想（瓶颈结构、模块化插入、冻结基座）启发了整个 PEFT 家族。而且某些场景（多任务持续学习、模块化组合）Adapter 反而更灵活。

---

## L3 · 正经定义

**Adapter Tuning**：Houlsby et al. (ICML 2019) 提出，参数高效微调方法。在 Transformer 每层的 attention 和 FFN 之后插入一个瓶颈模块（bottleneck module），冻结原模型，只训练 Adapter 参数。

**Adapter 结构**：

$$
h' = h + f(h)
$$

其中 $f(h)$ 是 Adapter 函数：

$$
f(h) = W_{up} \cdot \text{ReLU}(W_{down} \cdot h)
$$

- $W_{down} \in \mathbb{R}^{r \times d}$：降维（$d \to r$，$r \ll d$）
- $W_{up} \in \mathbb{R}^{d \times r}$：升维（$r \to d$）
- ReLU 非线性
- 残差连接 $h + f(h)$，保证初始化时 Adapter 近似恒等映射

**参数量**：每个 Adapter $2rd$，整个模型 $L \times 2 \times 2rd$（$L$ 层，每层 2 个 Adapter）。$r=64$ 时约 1-5% 原模型参数。

**两种插入位置**：

- **原始 Adapter**（Houlsby 2019）：每个 attention 和 FFN 之后各插一个（每层 2 个）
- **AdapterFusion / AdaLoRA**：只在 FFN 之后插（每层 1 个），效果相近参数更少

**参考资料**：

- 📄 Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019, arXiv:1902.00751
- 📄 Pfeiffer et al., *AdapterHub: A Framework for Adapting Transformers*, EMNLP 2020
- 📄 Pfeiffer et al., *AdapterFusion: Non-Destructive Task Composition for Transfer Learning*, EACL 2021
- 📄 Rebuffi et al., *Efficient Parametrization of Multi-domain Deep Neural Networks*（Multi-task Adapter 早期工作）

---

## L4 · 原理深挖

### 4.1 Adapter 的瓶颈结构

Adapter 的核心是**瓶颈（bottleneck）**：先把 $d$ 维输入压到 $r$ 维（$r \ll d$），过非线性，再还原回 $d$ 维。

```python
class Adapter(nn.Module):
    def __init__(self, d=768, r=64):
        super().__init__()
        self.down = nn.Linear(d, r)    # 降维
        self.up = nn.Linear(r, d)      # 升维
        self.act = nn.ReLU()
        # 初始化：up 近零，保证初始时 Adapter ≈ 恒等
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, h):
        residual = h
        h = self.down(h)       # d -> r
        h = self.act(h)        # 非线性
        h = self.up(h)         # r -> d
        return residual + h    # 残差
```

**为什么有效**：

- 瓶颈强制信息压缩，类似自编码器，学到低维表示
- 非线性 ReLU 提供表达能力（LoRA 没有非线性）
- 残差连接保证训练初期 Adapter 接近恒等，从预训练状态平滑过渡

### 4.2 Adapter 的插入位置

**Houlsby 版**（每层 2 个 Adapter）：

```
x -> Attention -> Adapter1 -> Add&Norm -> FFN -> Adapter2 -> Add&Norm -> out
```

**Pfeiffer 版**（每层 1 个 Adapter，更精简）：

```
x -> Attention -> Add&Norm -> FFN -> Adapter -> Add&Norm -> out
```

实验显示 Pfeiffer 版（只加 FFN 后）效果接近 Houlsby 版（加两处），但参数少一半。**实践默认用 Pfeiffer 版**。

### 4.3 Adapter vs LoRA 的结构对比

| 维度 | Adapter | LoRA |
|------|---------|------|
| 连接方式 | 串行（在主路径插入） | 并行（$\Delta W = BA$ 旁路） |
| 非线性 | 有（ReLU/GELU） | 无（纯线性 $BA$） |
| 推理开销 | 多一层前向 | 可合并零开销 |
| 参数量 | $2rd$ per layer | $2rd$ per layer（类似） |
| 初始化 | $W_{up}$ 近零 | $B$ 零 |
| 表达能力 | 强（非线性） | 弱（线性） |

**关键差异：串行 vs 并行**。

- Adapter 串行：$h_{out} = \text{Adapter}(\text{Layer}(h))$，推理时必须多算 Adapter
- LoRA 并行：$h_{out} = Wx + BAx = (W + BA)x$，可合并 $W + BA$，推理零开销

这一差异决定了 LoRA 在工业部署上淘汰 Adapter：**生产环境对推理延迟敏感，LoRA 合并后零开销是杀手锏**。

### 4.4 Adapter 的多任务优势

虽然单任务 LoRA 更优，但 Adapter 在多任务场景有独特优势：

**AdapterFusion**（Pfeiffer et al. 2021）：

```
1. 为每个任务训练独立 Adapter
2. 推理时用一个 Fusion 层融合多个 Adapter 的输出
3. 共享基座 + 多个 Adapter + Fusion 组合
```

**优势**：

- 任务间不干扰（每个 Adapter 独立）
- 可组合（新任务复用旧 Adapter）
- 持续学习（新任务加新 Adapter，不影响旧任务）

**对比 LoRA 多任务**：

- LoRA 多任务：每个任务一个 LoRA，切换时要 merge/restore
- Adapter 多任务：多个 Adapter 同时激活，Fusion 层加权组合

**结论**：多任务组合场景，AdapterFusion 比 LoRA 灵活；单任务部署，LoRA 更高效。

### 4.5 Adapter 的训练与推理

**训练**：

```python
# 冻结基座
for param in base_model.parameters():
    param.requires_grad = False

# 只训 Adapter
for adapter in adapters:
    for param in adapter.parameters():
        param.requires_grad = True

# 训练循环
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()  # 梯度只流过 Adapter
    optimizer.step()
```

**推理**：

```python
# 推理时 Adapter 不能合并，必须保留
output = model(input)  # 每层多算一次 Adapter
```

**延迟影响**：

- Adapter 增加每层 2 次小矩阵乘法（$d \times r$ 和 $r \times d$）
- $r=64$、$d=4096$ 时，每层约 $524K$ FLOPs，相对 attention/FFN 的几亿 FLOPs 可忽略
- 但层数多（70B 模型 80 层）累积起来 5-10% 延迟

### 4.6 Adapter 的变体

- **AdapterFusion**：多 Adapter 融合
- **AdapterDrop**：丢弃部分 Adapter 提速
- **AdaLoRA**：Adapter 思想 + LoRA 形式，自适应分配 $r$
- **Compacter**：低秩矩阵 + 参数化，进一步省参数
- **Parallel Adapter**：并行版 Adapter（不像原始串行），减少推理开销

### 4.7 何时还用 Adapter

**适合**：

- 多任务持续学习（AdapterFusion）
- 模块化组合（不同任务不同 Adapter）
- 需要非线性表达能力（复杂任务）
- 历史项目维护（已有 Adapter 基础设施）

**不适合**：

- 单任务生产部署（LoRA 合并零开销更优）
- 延迟敏感场景
- 新项目（直接用 LoRA）

**实践建议**：新项目用 LoRA，多任务组合场景考虑 AdapterFusion，历史 Adapter 项目可继续维护。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2019-02**：Houlsby 发 Adapter 论文，PEFT 开山之作
- **2019-12**：Pfeiffer 简化版（只加 FFN 后），参数减半
- **2020**：AdapterHub 发布，多任务 Adapter 生态
- **2021**：AdapterFusion 多任务融合
- **2022**：LoRA 发布，PEFT 新王，Adapter 退居多任务场景
- **2023-2024**：Adapter 在 LLM 时代式微，但 AdapterFusion 思想影响后续多 LoRA 工作
- **2025**：Adapter 主要保留在多任务持续学习领域

### 5.2 常见坑

**坑 1：推理不合并 Adapter 抱怨延迟**。Adapter 不能像 LoRA 那样合并，部署时多一层计算。延迟敏感场景别用 Adapter。

**坑 2：$r$ 调太大**。Adapter 参数量本来就比 LoRA 大（多非线性层），$r=256$ 参数爆炸。$r \in [16, 64]$ 够用。

**坑 3：忘了残差初始化**。$W_{up}$ 不零初始化，训练初期 Adapter 输出乱跳，破坏预训练表示。

**坑 4：插入位置太多**。Houlsby 版每层 2 个 Adapter，参数多。默认用 Pfeiffer 版（每层 1 个）。

**坑 5：拿 Adapter 和 LoRA 比单任务效果**。单任务 LoRA 更优（合并零开销），比就是输。要比就比多任务组合。

**坑 6：AdapterFusion 权重没调**。Fusion 层的注意力权重初始不均，某些 Adapter 被忽略。要 warmup + 平衡正则。

**坑 7：多 Adapter 同时激活显存爆**。每个 Adapter 一份参数，多任务同时激活累积显存。要限制同时激活数。

**坑 8：持续学习忘了冻结旧 Adapter**。新任务训练时旧 Adapter 没冻结，旧任务能力被破坏。要冻结旧 Adapter 只训新的。

**坑 9：Adapter 和量化不兼容**。4bit 量化基座 + Adapter，反量化开销 + Adapter 开销叠加，比 QLoRA 慢更多。

**坑 10：评估只看参数量**。Adapter 参数量比 LoRA 大，但推理 FLOPs 增加少。要综合看延迟。

### 5.3 面试怎么考

1. **Adapter 的结构？** 答：瓶颈模块，$W_{down}$ 降维 + ReLU + $W_{up}$ 升维 + 残差连接。
2. **Adapter 和 LoRA 的核心区别？** 答：Adapter 串行插入有非线性，推理有开销；LoRA 并行加 $\Delta W$ 纯线性，可合并零开销。
3. **Adapter 为什么被 LoRA 取代？** 答：LoRA 推理可合并零开销，工业部署延迟敏感，Adapter 多一层计算是硬伤。
4. **Adapter 在什么场景还有优势？** 答：多任务持续学习（AdapterFusion），任务间不干扰，可组合。
5. **Adapter 的初始化？** 答：$W_{up}$ 零初始化，保证训练初期 Adapter 近似恒等，从预训练平滑过渡。

---

## 速记卡

| 组件 | 形状 | 作用 |
|------|------|------|
| $W_{down}$ | $r \times d$ | 降维 |
| ReLU | - | 非线性 |
| $W_{up}$ | $d \times r$ | 升维 |
| 残差 | - | 平滑过渡 |

**对比表**：

| 维度 | Adapter | LoRA |
|------|---------|------|
| 连接 | 串行 | 并行 |
| 非线性 | 有 | 无 |
| 推理开销 | 有 | 零（可合并） |
| 参数量 | 1-5% | 0.1-1% |
| 多任务 | 强（Fusion） | 中（切换） |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| $r$ | 16-64 | 表达能力 |
| 插入位置 | FFN 后 | 参数量 |
| 非线性 | ReLU/GELU | 表达能力 |

**一句话记忆**：Adapter = 每层插入瓶颈小模块（down + 非线性 + up + 残差），冻结基座只训 Adapter。PEFT 开山之作（2019），被 LoRA 取代（串行有推理开销），但 AdapterFusion 多任务组合场景仍有优势。$W_{up}$ 零初始化是关键。新项目用 LoRA，多任务持续学习考虑 Adapter。

---

> *上一篇：[QLoRA 量化低秩适配](./qlora) -- 4bit 量化 + LoRA，消费级 GPU 微调。*
> *下一篇：[Prefix/Prompt Tuning 前缀调优](./prefix-tuning) -- 比 Adapter 更极端，只调前缀向量。*
