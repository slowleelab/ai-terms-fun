---
title: KTO / SimPO 变体
slug: kto-simpo
category: 进阶专题
tags: [KTO, SimPO, IPO, ORPO, DPO 变体, 偏好优化, 二元反馈]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# KTO / SimPO 变体

> 五层读懂一个词。这次拆的是：**DPO 之后的一波变体**--KTO 用二元反馈替代偏好对，SimPO 去掉 ref model，IPO 解决过对齐，ORPO 把 SFT 和偏好优化合一。一个比一个工程友好。

---

## L1 · 一句话点破

**DPO 变体的两大方向**：① 放宽数据要求（KTO 用赞/踩替代偏好对，ORPO 不需先 SFT）② 简化工程（SimPO 去掉 ref model，IPO 修过对齐）。核心思想都源自 DPO 的"策略即 RM"，但在数据、模型、损失上各做优化。

---

## L2 · 通俗类比

DPO 把对齐工程简化了 10 倍，但还有几个痛点：

- **痛点 1：偏好对数据贵**。要同一个 prompt 的两个回答做比较，标注成本高
- **痛点 2：要 ref model**。2 个模型还是嫌多，显存还是大
- **痛点 3：过对齐**。chosen 概率降，模型变差
- **痛点 4：要先 SFT 再 DPO**。两阶段，流程长

DPO 变体各自解决一个痛点：

| 变体 | 解决的痛点 | 核心创新 |
|------|-----------|---------|
| **KTO** | 偏好对贵 | 用二元反馈（赞/踩）替代偏好对 |
| **SimPO** | ref model 占显存 | 去掉 ref，用长度归一化 |
| **IPO** | 过对齐 | 修改损失，避免 chosen 概率降 |
| **ORPO** | 两阶段流程 | SFT + 偏好优化合一 |

**KTO 的洞察**：现实中的反馈多是二元的（点赞/踩、5 星/1 星），不是成对比较。KTO 用前景理论（prospect theory）建模二元反馈，不需要偏好对。

**SimPO 的洞察**：DPO 的 ref model 用来算 $\log \frac{\pi}{\pi_{ref}}$，但可以用长度归一化的 $\log \pi$ 替代，省掉 ref model。1 个模型就能训。

**IPO 的洞察**：DPO 的过对齐源于 Bradley-Terry 损失。IPO 用恒等映射替代 sigmoid，避免过对齐。

**ORPO 的洞察**：SFT 阶段已经在学"怎么回答"，可以同时学"哪个回答更好"。把 SFT loss 和偏好 loss 合一，一步到位。

**这些变体的共同特点**：都是监督学习（不是 RL）、都比 PPO 简单、都建立在 DPO 的"策略即 RM"思想上。工业实践根据数据类型和工程约束选用。

---

## L3 · 正经定义

**KTO（Kahneman-Tversky Optimization）**：Ethayarajh et al. (ICML 2024) 提出，用前景理论建模二元反馈（赞/踩），不需要偏好对。损失函数对正例和负例分别处理，引入损失厌恶参数 $\lambda$。

**SimPO（Simple Preference Optimization）**：Meng et al. (2024) 提出，去掉 DPO 的 ref model，用长度归一化的对数概率替代对数比，简化工程。

**IPO（Identity Preference Optimization）**：Azar et al. (2023) 提出，用平方损失替代 Bradley-Terry 的 sigmoid 损失，解决 DPO 过对齐。

**ORPO（Odds Ratio Preference Optimization）**：Hong et al. (2024) 提出，把 SFT loss 和偏好 loss 合一，不需要先 SFT 再对齐。

**损失函数对比**：

**DPO**：
$$
\mathcal{L}_{DPO} = -\log \sigma(\beta[\Delta \log \pi_c - \Delta \log \pi_r])
$$

**KTO**（二元反馈）：
$$
\mathcal{L}_{KTO} = \mathbb{E}_{y \sim \text{good}}[\sigma(\beta \Delta \log \pi_y - z)] + \lambda \mathbb{E}_{y \sim \text{bad}}[\sigma(z - \beta \Delta \log \pi_y)]
$$

**SimPO**（无 ref）：
$$
\mathcal{L}_{SimPO} = -\log \sigma\left(\frac{\beta}{|y_c|} \log \pi(y_c) - \frac{\beta}{|y_r|} \log \pi(y_r) - \gamma\right)
$$

**IPO**（平方损失）：
$$
\mathcal{L}_{IPO} = (\Delta \log \pi_c - \Delta \log \pi_r - \frac{1}{2\beta})^2
$$

**参考资料**：

- 📄 Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, ICML 2024, arXiv:2402.01306
- 📄 Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*, 2024, arXiv:2405.14734
- 📄 Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences*, AISTATS 2024（IPO）
- 📄 Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model*, 2024, arXiv:2403.07691

---

## L4 · 原理深挖

### 4.1 KTO：二元反馈 + 前景理论

**DPO 的数据痛点**：偏好对要同一 prompt 的两个回答比较，标注贵且难。现实中更多是二元反馈（这个回答赞/踩）。

**KTO 的思路**：用前景理论建模二元反馈。前景理论的核心是**损失厌恶**：人对损失的敏感度大于收益。

**KTO 损失**：

对正例（赞）：
$$
L_{good} = -\sigma(\beta \Delta \log \pi_y - z_0)
$$

对负例（踩）：
$$
L_{bad} = -\lambda \sigma(z_0 - \beta \Delta \log \pi_y)
$$

其中：

- $\Delta \log \pi_y = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$（和 DPO 一样的对数比）
- $z_0 = \beta \text{KL}(\pi_\theta \| \pi_{ref})$（KL 项）
- $\lambda$ 是损失厌恶参数（通常 1.0-1.5）

**直觉**：

- 正例：增大 $\Delta \log \pi_y$（让好回答概率升）
- 负例：减小 $\Delta \log \pi_y$（让坏回答概率降），且权重 $\lambda$ 更大（损失厌恶）

**KTO 的优势**：

- 数据便宜（二元反馈 vs 偏好对）
- 数据量大（现实中二元反馈远多于偏好对）
- 效果接近 DPO（论文实验）

**KTO 的代价**：

- 信息量少（二元 vs 偏好对）
- 对标注质量敏感（赞/踩标准要明确）
- $\lambda$ 调参（损失厌恶强度）

### 4.2 SimPO：去掉 ref model

**DPO 的工程痛点**：要 ref model，2 个模型显存大。

**SimPO 的洞察**：DPO 的 $\log \frac{\pi}{\pi_{ref}}$ 是为了 KL 约束。但可以用**长度归一化的 $\log \pi$** 替代，省掉 ref model。

**SimPO 损失**：

$$
\mathcal{L}_{SimPO} = -\log \sigma\left(\frac{\beta}{|y_c|} \log \pi(y_c|x) - \frac{\beta}{|y_r|} \log \pi(y_r|x) - \gamma\right)
$$

**关键改动**：

1. 去掉 $\log \pi_{ref}$，只用 $\log \pi$
2. 长度归一化：$\frac{1}{|y|} \log \pi(y)$，解决长回答 log prob 低的问题
3. 引入 $\gamma$ margin，确保 chosen 和 rejected 有足够差距

**SimPO 的优势**：

- 1 个模型（policy only），显存减半
- 长度归一化，解决 DPO 偏好短回答的问题
- 工程最简

**SimPO 的代价**：

- 无 KL 约束，可能偏离 SFT 太远
- $\gamma$ 调参（margin 大小）
- 理论基础弱（不如 DPO 严谨）

### 4.3 IPO：解决过对齐

**DPO 的过对齐问题**：Bradley-Terry 损失让 chosen 和 rejected 概率都降，只是 rejected 降更多。

**IPO 的思路**：用平方损失替代 sigmoid，避免过对齐。

**IPO 损失**：

$$
\mathcal{L}_{IPO} = \left(\log \frac{\pi(y_c)}{\pi_{ref}(y_c)} - \log \frac{\pi(y_r)}{\pi_{ref}(y_r)} - \frac{1}{2\beta}\right)^2
$$

**直觉**：让对数比差等于 $\frac{1}{2\beta}$（一个常数），而不是越大越好。这避免了 DPO 的"无限推大差距"导致的过对齐。

**IPO 的优势**：

- 解决过对齐
- 损失有界（平方损失）

**IPO 的代价**：

- 收敛慢（平方损失梯度小）
- $\beta$ 调参更敏感

### 4.4 ORPO：SFT + 偏好合一

**DPO 的流程痛点**：要先 SFT 再 DPO，两阶段，流程长。

**ORPO 的思路**：SFT 阶段同时在学"怎么回答"，可以同时学"哪个回答更好"。把 SFT loss 和偏好 loss 合一。

**ORPO 损失**：

$$
\mathcal{L}_{ORPO} = \mathcal{L}_{SFT}(y_c) + \lambda \mathcal{L}_{OR}(y_c, y_r)
$$

其中 $\mathcal{L}_{OR}$ 是 odds ratio 损失：

$$
\mathcal{L}_{OR} = -\log \sigma\left(\log \frac{\text{odds}(y_c)}{\text{odds}(y_r)}\right)
$$

$\text{odds}(y) = \frac{\pi(y)}{1 - \pi(y)}$ 是 odds。

**ORPO 的优势**：

- 一步到位（SFT + 偏好）
- 不需要 ref model
- 不需要预训练 SFT 模型

**ORPO 的代价**：

- 联合训练调参复杂
- SFT 和偏好的权衡（$\lambda$）难调

### 4.5 各方法对比

| 方法 | 模型数 | 数据 | ref | 特点 |
|------|--------|------|-----|------|
| PPO | 4 | 偏好对 | ✅ | RL，效果最好 |
| DPO | 2 | 偏好对 | ✅ | 监督，工程简单 |
| KTO | 2 | 二元反馈 | ✅ | 数据便宜 |
| SimPO | 1 | 偏好对 | ❌ | 显存最小 |
| IPO | 2 | 偏好对 | ✅ | 解决过对齐 |
| ORPO | 1 | 偏好对 | ❌ | SFT + 偏好合一 |

### 4.6 选型决策

```
你的数据是什么？
│
├─ 偏好对（chosen vs rejected）
│  ├─ 显存极紧 -> SimPO（1 模型）
│  ├─ 过对齐严重 -> IPO
│  ├─ 想一步到位（不先 SFT）-> ORPO
│  └─ 通用默认 -> DPO
│
├─ 二元反馈（赞/踩）
│  └─ KTO
│
├─ 在线采样 + 追求极致
│  └─ PPO
│
└─ 不确定
   └─ DPO（最稳）
```

### 4.7 工程实践

**主流选择**（2024-2025）：

- **开源社区**：DPO 为主，SimPO / ORPO 崛起
- **大厂**：PPO 为主，DPO 用于快速实验
- **数据受限场景**：KTO（二元反馈多）
- **显存极紧**：SimPO（1 模型）

**组合使用**：

- SFT -> DPO：经典两阶段
- SFT -> DPO -> KTO：DPO 后用 KTO 微调
- ORPO：一步到位
- SimPO：直接从 SFT 模型训

### 4.8 评估方法

**偏好准确率**：在 held-out 偏好对上，模型是否给 chosen 更高分数。

**生成质量**：

- 人工评估（黄金标准）
- 自动评估（用 GPT-4 当裁判，AlpacaEval）
- Benchmark（MMLU、MT-Bench）

**对齐指标**：

- 有用性（helpfulness）
- 无害性（harmlessness）
- 诚实性（honesty）

**评估坑**：

- 只看偏好准确率：过对齐的模型准确率高但生成差
- 只看 benchmark：对齐效果在 benchmark 上体现不出来
- 不看人工评估：对齐最终要服务人类偏好

---

## L5 · 沿革与坑

### 5.1 沿革

- **2023-05**：DPO 提出，开启偏好优化简化浪潮
- **2023-10**：IPO 提出，解决过对齐
- **2024-02**：KTO 提出，二元反馈替代偏好对
- **2024-03**：ORPO 提出，SFT + 偏好合一
- **2024-05**：SimPO 提出，去掉 ref model
- **2024-2025**：DPO 变体百花齐放，工业实践按场景选用

### 5.2 常见坑

**坑 1：KTO 二元反馈标准不明确**。赞/踩标准模糊，标注一致性差。要明确标注指南。

**坑 2：SimPO 无 KL 约束崩盘**。去掉 ref 后无约束，策略偏离太远。要监控生成质量。

**坑 3：IPO 收敛慢**。平方损失梯度小，训练慢。要更多 epoch 或调学习率。

**坑 4：ORPO 调参复杂**。SFT 和偏好 loss 权重 $\lambda$ 难调。要在验证集上搜索。

**坑 5：盲目追新**。新变体不一定比 DPO 好，要在自己场景验证。

**坑 6：忽略数据质量**。变体再好，数据差也白搭。数据质量 > 方法选择。

**坑 7：忘了 SFT 基础**。除 ORPO 外都要先 SFT。SFT 差后续全差。

**坑 8：评估只看偏好准确率**。过对齐的模型准确率高但生成差。要看生成质量。

**坑 9：KTO $\lambda$ 设错**。损失厌恶 $\lambda$ 太大对齐弱，太小对负例不敏感。1.0-1.5 起步。

**坑 10：SimPO $\gamma$ 设错**。margin $\gamma$ 太大过对齐，太小分不开。0.5-2.0 起步。

**坑 11：组合方法不验证**。SFT -> DPO -> KTO 这种组合不验证就上，可能互相干扰。要逐步验证。

**坑 12：以为变体能替代 PPO**。变体在多数场景够用，但追求极致效果还是 PPO。

### 5.3 面试怎么考

1. **KTO 和 DPO 的核心区别？** 答：KTO 用二元反馈（赞/踩）+ 前景理论，DPO 用偏好对 + Bradley-Terry。KTO 数据便宜。
2. **SimPO 怎么去掉 ref model？** 答：用长度归一化的 $\log \pi$ 替代 $\log \frac{\pi}{\pi_{ref}}$，省掉 ref，但加 margin $\gamma$ 保差距。
3. **IPO 解决什么问题？** 答：DPO 的过对齐。用平方损失替代 sigmoid，让对数比差等于常数而非无限推大。
4. **ORPO 的创新？** 答：SFT loss + 偏好 loss 合一，不需要先 SFT 再对齐，一步到位。
5. **这些变体的共同特点？** 答：都是监督学习（非 RL）、都基于 DPO 的"策略即 RM"思想、都比 PPO 简单。

---

## 速记卡

| 方法 | 模型数 | 数据 | ref | 解决痛点 |
|------|--------|------|-----|---------|
| DPO | 2 | 偏好对 | ✅ | 基线 |
| KTO | 2 | 二元反馈 | ✅ | 偏好对贵 |
| SimPO | 1 | 偏好对 | ❌ | ref 占显存 |
| IPO | 2 | 偏好对 | ✅ | 过对齐 |
| ORPO | 1 | 偏好对 | ❌ | 两阶段流程 |

**选型决策**：

```
偏好对 + 显存紧 -> SimPO
偏好对 + 过对齐 -> IPO
偏好对 + 一步到位 -> ORPO
偏好对 + 通用 -> DPO
二元反馈 -> KTO
追求极致 -> PPO
```

**关键参数**：

| 方法 | 关键参数 | 典型值 |
|------|---------|--------|
| DPO | $\beta$ | 0.1 |
| KTO | $\lambda$（损失厌恶） | 1.0-1.5 |
| SimPO | $\gamma$（margin） | 0.5-2.0 |
| IPO | $\beta$ | 0.1 |
| ORPO | $\lambda$（SFT 权重） | 0.5-1.0 |

**一句话记忆**：DPO 变体两大方向--放宽数据（KTO 二元反馈 / ORPO 不需先 SFT）+ 简化工程（SimPO 去 ref / IPO 修过对齐）。都基于 DPO 的"策略即 RM"思想，都是监督学习。选型看数据类型和工程约束：二元反馈用 KTO，显存紧用 SimPO，过对齐用 IPO，一步到位用 ORPO，通用用 DPO，极致用 PPO。

---

> *上一篇：[DPO 直接偏好优化](./dpo) -- DPO 变体都建立在 DPO 的"策略即 RM"思想上。*
> *下一篇预告：推理工程专题 -- KV-Cache / PagedAttention / Continuous Batching / 推测解码，LLM 部署的加速秘籍。*
