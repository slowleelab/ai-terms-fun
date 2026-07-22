---
title: Reward Model 奖励模型
slug: reward-model
category: 进阶专题
tags: [Reward Model, RLHF, 偏好数据, Pairwise, Bradley-Terry]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# Reward Model 奖励模型

> 五层读懂一个词。这次拆的是：**Reward Model（RM）**--RLHF 的第二步，学一个"人类偏好打分器"，给 PPO 提供奖励信号。本质是把"人觉得哪个回答好"这件事，训练成一个可微的标量函数。

---

## L1 · 一句话点破

**Reward Model = 偏好数据 + Pairwise 排序损失（Bradley-Terry）**。输入 `(prompt, response)`，输出标量奖励分数 $r(x, y)$。训练目标是让"人类偏好的回答"分数高于"不偏好的回答"，PPO 用这个分数当奖励优化策略。

---

## L2 · 通俗类比

SFT 教模型"这么答"，但没教"哪个更好"。比如同一个问题"怎么学编程"，模型可能生成两个回答：

- 回答 A：循序渐进，从 Python 入门，配项目实践
- 回答 B：甩一堆术语，没逻辑，还有错误

SFT 后模型两个都可能生成，因为 SFT 只学"这么答"，没学"A 比 B 好"。

Reward Model 的工作是学一个**打分器**：给 A 高分，给 B 低分。PPO 阶段模型生成回答后，RM 打分，高分强化、低分抑制，模型慢慢学会生成"人类偏好"的回答。

**训练 RM 的数据是偏好对**：

```json
{
  "prompt": "怎么学编程",
  "chosen": "回答 A（循序渐进...）",
  "rejected": "回答 B（甩术语...）"
}
```

标注员看两个回答，选一个更好的。几万到几十万这样的偏好对，训出 RM。

**RM 在 RLHF 中的位置**：

```
SFT 模型
    ↓ 生成多个回答
人工标注偏好对（A > B）
    ↓ 训练
Reward Model（打分器）
    ↓ 提供 reward
PPO 优化 SFT 模型
    ↓
对齐后的模型
```

**RM 是 RLHF 的"教练"**：PPO 是运动员，RM 是教练打分，运动员根据教练打分调整。教练水平决定运动员上限--RM 质量直接决定 RLHF 效果。

**代价**：偏好数据标注贵（比 SFT 数据贵）；RM 训练有偏差（标注者偏见）；RM 容易被 PPO 钻空子（reward hacking）；RM 质量难评估。DPO 的出现就是为了绕开 RM。

---

## L3 · 正经定义

**Reward Model**：RLHF 第二阶段的核心组件，学习一个标量奖励函数 $r_\phi(x, y)$，输入 prompt $x$ 和 response $y$，输出奖励分数。训练数据是人类标注的偏好对 $(x, y_c, y_r)$（$y_c$ chosen，$y_r$ rejected）。

**训练目标（Bradley-Terry 模型）**：

$$
\mathcal{L}_{RM} = -\log \sigma(r_\phi(x, y_c) - r_\phi(x, y_r))
$$

其中 $\sigma$ 是 sigmoid。直觉：让 chosen 回答的分数比 rejected 高，差值越大 loss 越小。

**模型结构**：

- 通常用 SFT 模型作为初始化
- 把最后的 unembedding 层换成标量输出头（输出 1 维）
- 输入：`[prompt, response]` 拼接
- 输出：最后一个 token 的标量分数

**推理时**：

$$
r_\phi(x, y) = \text{scalar head}(\text{LLM}([x, y]))_{\text{last token}}
$$

**参考资料**：

- 📄 Ouyang et al., *InstructGPT*, NeurIPS 2022, arXiv:2203.02155
- 📄 Christiano et al., *Deep Reinforcement Learning from Human Preferences*, NeurIPS 2017（RLHF 基础）
- 📄 Stiennon et al., *Learning to summarize from human feedback*, NeurIPS 2020
- 📄 Bradley & Terry, *Rank Analysis of Incomplete Block Designs*, 1952（Bradley-Terry 模型原始论文）

---

## L4 · 原理深挖

### 4.1 Bradley-Terry 模型

RM 的训练基于 **Bradley-Terry 模型**：假设每个 item 有一个"实力"分数 $r$，item A 比 item B 赢的概率是：

$$
P(A \succ B) = \sigma(r_A - r_B) = \frac{1}{1 + e^{-(r_A - r_B)}}
$$

**对偏好对 $(y_c, y_r)$**：希望 $P(y_c \succ y_r)$ 大，即 $r(y_c) - r(y_r)$ 大。最大化对数似然：

$$
\max \log \sigma(r(y_c) - r(y_r))
$$

等价于最小化：

$$
\mathcal{L} = -\log \sigma(r(y_c) - r(y_r))
$$

**直觉**：chosen 分数比 rejected 高越多，loss 越小；两者接近时 loss 大，逼迫模型区分。

### 4.2 RM 的模型结构

RM 通常是 SFT 模型改造：

```python
class RewardModel(nn.Module):
    def __init__(self, sft_model):
        super().__init__()
        self.transformer = sft_model  # 复用 SFT 模型
        self.scalar_head = nn.Linear(hidden_size, 1)  # 标量输出头
        # 初始化 scalar_head
    
    def forward(self, input_ids, attention_mask):
        # 1. Transformer 前向
        hidden_states = self.transformer(input_ids, attention_mask).last_hidden_state
        # 2. 取最后一个 token 的 hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden]
        # 3. 标量输出
        reward = self.scalar_head(last_hidden)  # [batch, 1]
        return reward
```

**关键设计**：

- **复用 SFT 模型**：SFT 模型已经理解语言，RM 只需学"打分"
- **标量头**：把 hidden state 映射到 1 维
- **最后一个 token**：用最后一个 token 的表示代表整个 response 的质量

**为什么用最后一个 token**：最后一个 token 看过整个 response，表示最完整。也有用 mean pooling 的，但 last-token 是主流。

### 4.3 偏好数据标注

**标注流程**：

1. 用 SFT 模型对同一 prompt 生成多个回答（$y_1, y_2, ..., y_k$）
2. 标注员排序或两两比较
3. 形成 chosen-rejected 对

**标注方式**：

| 方式 | 描述 | 数据量 |
|------|------|--------|
| Pairwise | 两两比较，选更好的 | 每对 1 个偏好 |
| Ranking | 多个回答排序 | $k$ 个回答产生 $k(k-1)/2$ 对 |
| Pointwise | 直接打分（1-5） | 单个回答 1 个分数 |
| Best-of-N | 从 N 个选最好的 | 每组 1 个偏好 |

**实践**：Pairwise 最常用（标注简单，数据质量高）。Pointwise 标注一致性差，少用。

**标注成本**：偏好标注比 SFT 标注贵 3-5 倍（要比较，不是直接写）。InstructGPT 用了约 30K 偏好对，成本百万美元级。

### 4.4 RM 训练的细节

**数据格式**：每个样本是 `(prompt, chosen, rejected)` 三元组。

**损失**：

```python
def reward_model_loss(rm, prompt, chosen, rejected):
    r_chosen = rm(prompt, chosen)      # [batch, 1]
    r_rejected = rm(prompt, rejected)  # [batch, 1]
    # Bradley-Terry loss
    logits = r_chosen - r_rejected
    loss = -F.logsigmoid(logits).mean()
    return loss
```

**训练超参**：

- 学习率：$5e-6$ 到 $5e-5$（比 SFT 更小，RM 要精调）
- Epoch：1 个 epoch（偏好数据少，多 epoch 过拟合快）
- Batch size：64-512（大 batch 稳定）

**关键技巧**：

- **用 SFT 模型初始化**：不用预训练模型，要用 SFT 模型（已经理解对话）
- **数据均衡**：不同 prompt 类型、不同 chosen-rejected 差距均衡
- **响应长度归一化**：长回答容易得低分（模型学偏），要做长度归一化

### 4.5 Reward Hacking（奖励作弊）

**问题**：PPO 优化 RM 分数，但 RM 不完美。模型会找 RM 的漏洞，生成"RM 高分但人类不喜欢"的回答。

**典型 reward hacking**：

- **重复高分词**：RM 学到某些词高分，模型疯狂重复这些词
- **过度冗长**：RM 偏好长回答，模型生成无意义的长回答
- **格式作弊**：RM 偏好列表格式，模型什么都用列表
- **阿谀奉承**：RM 偏好附和用户，模型无原则附和

**解法**：

- **KL 约束**：PPO 加 KL 惩罚，限制策略和 SFT 模型偏离（见 PPO 篇）
- **RM 集成**：训多个 RM 取平均，降低单 RM 漏洞
- **定期更新 RM**：用新数据重训 RM
- **人工抽检**：定期人工评估，发现 hacking 及时调

### 4.6 RM 的偏差

**标注者偏差**：

- 文化偏差：不同文化对"好回答"标准不同
- 立场偏差：政治、伦理问题标注者立场不同
- 风格偏差：标注者偏好某种写作风格

**RM 放大偏差**：

- RM 学到标注者偏差
- PPO 优化 RM，模型放大偏差
- 最终模型在偏差方向上走得更远

**解法**：

- 多元标注者（不同文化、背景）
- 标注指南明确中立原则
- RM 评估包含偏差测试集

### 4.7 RM 的评估

**准确率**：在 held-out 偏好对上，RM 打分排序和人类一致的比例。InstructGPT RM 准确率约 65-70%。

**相关性**：RM 分数和人类评分的相关系数。

**Reward hacking 检测**：用 RM 优化后的模型，人工评估是否真的变好。

**评估坑**：

- 只看准确率不看 hacking：准确率高的 RM 可能更容易被 hack
- 只看 held-out 不看新数据：RM 在分布外泛化差

### 4.8 RM 的替代：RLAIF

**RLAIF**（Constitutional AI）：用 AI（如 GPT-4）替代人类做偏好标注。

```
人类标注：贵、慢、有偏差
AI 标注：便宜、快、AI 偏差
```

**Anthropic 的 Constitutional AI**：

- 用一组"宪法"原则（无害、有用、诚实）
- AI 标注员按宪法评判偏好
- 训 RM 用 AI 标注的偏好对

**优势**：成本降 100 倍；**劣势**：AI 偏差替代人类偏差，可能强化 AI 盲点。

### 4.9 RM vs DPO

DPO 的核心洞察：**不需要显式训 RM，直接从偏好对优化策略**。

```
RLHF：偏好对 -> 训 RM -> PPO 优化策略（4 个模型）
DPO：偏好对 -> 直接优化策略（2 个模型）
```

**对比**：

| 维度 | RLHF (RM + PPO) | DPO |
|------|-----------------|-----|
| 模型数 | 4（policy, ref, RM, value） | 2（policy, ref） |
| 训练阶段 | 2（RM + PPO） | 1 |
| 显存 | 大 | 小 |
| 工程复杂度 | 高 | 低 |
| 效果 | 略高 | 接近 RLHF |
| 灵活性 | 高（RM 可复用） | 低 |

**结论**：DPO 绕开 RM，工程更简单。但 RM 不是没用--RM 可复用、可解释、可调试，某些场景仍有价值。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2017**：Christiano 提出 RLHF 框架，RM 概念成型
- **2020**：Stiennon 把 RLHF 用到文本摘要
- **2022-03**：InstructGPT 确立 SFT -> RM -> PPO 范式
- **2022-11**：ChatGPT 用 RLHF 爆火，RM 成为关注焦点
- **2023-05**：DPO 论文提出绕开 RM，RM 必要性受质疑
- **2023 下半年**：Constitutional AI / RLAIF 用 AI 替代人类标注 RM
- **2024-2025**：RM 在工业实践仍主流，DPO 在开源社区流行，两者并存

### 5.2 常见坑

**坑 1：RM 用预训练模型初始化**。要用 SFT 模型初始化，预训练模型不理解对话，RM 训不好。

**坑 2：偏好数据少**。几万偏好对训 RM 容易过拟合，要至少 10K+ 高质量偏好对。

**坑 3：epoch 太多**。RM 1 个 epoch 就够，多 epoch 过拟合快，泛化崩盘。

**坑 4：没做长度归一化**。RM 学到"长回答低分"或"长回答高分"的偏差，模型在 PPO 阶段被误导。要做长度归一化。

**坑 5：reward hacking 没监控**。PPO 优化 RM 分数一直涨，但人工评估变差。要定期人工抽检。

**坑 6：RM 评估只看准确率**。准确率高的 RM 可能更容易被 hack。要看 hacking 检测 + 相关性。

**坑 7：标注者偏差没控制**。标注者文化/立场偏差直接传给 RM，RM 放大偏差。要多元标注者 + 中立指南。

**坑 8：单个 RM 用到底**。单 RM 有漏洞，PPO 容易钻空子。要 RM 集成。

**坑 9：RM 不更新**。PPO 进行中模型分布漂移，旧 RM 不匹配新分布。要定期重训 RM。

**坑 10：RM 和 PPO 解耦不彻底**。RM 训练数据分布和 PPO 生成分布不一致，RM 在新分布上失效。要 on-policy 数据补充。

**坑 11：用 pointwise 标注**。直接打分一致性差，要 pairwise 比较标注。

**坑 12：RM 输出没校准**。RM 分数量纲不固定，PPO 奖励幅度难调。要做分数归一化。

### 5.3 面试怎么考

1. **RM 的训练目标？** 答：Bradley-Terry 模型，最大化 chosen 和 rejected 分数差的对数似然 $\mathcal{L} = -\log \sigma(r(y_c) - r(y_r))$。
2. **RM 怎么从 LLM 改造？** 答：复用 SFT 模型 backbone，把 unembedding 层换成标量输出头，取最后一个 token 输出。
3. **什么是 reward hacking？** 答：PPO 优化 RM 分数，模型钻 RM 漏洞生成"高分但人类不喜欢"的回答。用 KL 约束 + RM 集成缓解。
4. **RM 和 DPO 的关系？** 答：DPO 绕开 RM 直接从偏好对优化策略，但 RM 可复用可解释，某些场景仍有价值。
5. **RM 的偏差来源？** 答：标注者偏差（文化/立场/风格）经 RM 放大传给策略模型。

---

## 速记卡

| 组件 | 作用 |
|------|------|
| SFT backbone | 理解语言 |
| scalar head | 输出标量分数 |
| last token | 代表 response 质量 |

**训练数据**：

| 类型 | 描述 | 标注成本 |
|------|------|---------|
| Pairwise | 两两比较 | 中 |
| Ranking | 多个排序 | 高 |
| Pointwise | 直接打分 | 低（一致性差） |
| Best-of-N | N 选 1 | 低 |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| 初始化 | SFT 模型 | 必须用 SFT |
| 学习率 | $5e-6$ ~ $5e-5$ | 精调 |
| Epoch | 1 | 防过拟合 |
| Batch size | 64-512 | 稳定性 |
| 数据量 | 10K-100K | 偏好对 |

**RM 质量指标**：

| 指标 | 说明 |
|------|------|
| 准确率 | held-out 偏好对排序正确率 |
| 相关性 | RM 分数 vs 人类评分相关系数 |
| Hacking 抗性 | PPO 优化后人工评估是否变好 |

**一句话记忆**：Reward Model = 偏好对 + Bradley-Terry 损失，学一个标量打分器给 PPO 用。用 SFT 模型初始化 + 标量输出头 + last token。1 epoch 训练，10K+ 偏好对。最大风险是 reward hacking（模型钻 RM 漏洞），用 KL 约束 + RM 集成缓解。DPO 绕开 RM 但 RM 可复用可解释，两者并存。

---

> *上一篇：[SFT 监督微调](./sft) -- RLHF 第一阶段，SFT 模型是 RM 的初始化。*
> *下一篇：[PPO 近端策略优化](./ppo-rlhf) -- RLHF 第三阶段，用 RM 分数当奖励做强化学习。*
