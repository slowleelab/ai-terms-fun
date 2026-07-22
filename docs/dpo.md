---
title: DPO 直接偏好优化
slug: dpo
category: 进阶专题
tags: [DPO, 直接偏好优化, RLHF 替代, 偏好对, 监督学习]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# DPO 直接偏好优化

> 五层读懂一个词。这次拆的是：**DPO（Direct Preference Optimization）**--绕开 RM 和 PPO，把偏好优化变成一个二分类问题。2 个模型替代 4 个，监督学习替代强化学习，效果接近 PPO 但工程简单 10 倍。

---

## L1 · 一句话点破

**DPO = 偏好对 + Bradley-Terry 损失 + 隐式 RM**。数学上证明：不需要显式训 RM，策略模型本身就是一个隐式 RM。直接用偏好对做二分类（chosen vs rejected），2 个模型（policy + ref）替代 PPO 的 4 个，监督学习替代强化学习。

---

## L2 · 通俗类比

PPO 对齐像开工厂：先训一个质检员（RM），再让工人（policy）根据质检员打分改进，还要请一个老师傅（value model）估基准。4 个人协同，流程复杂，出错率高。

DPO 的洞察：**质检员其实可以不要**。数学推导发现，最优策略 $\pi^*$ 和参考策略 $\pi_{ref}$ 的对数比，本身就是一个奖励函数：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \text{const}
$$

也就是说，**策略模型自带 RM 功能**。直接用偏好对训练策略，让 chosen 的 $\log \frac{\pi^*}{\pi_{ref}}$ 比 rejected 大，就等价于 RLHF。

**DPO 的训练**：

- 输入：偏好对 $(x, y_c, y_r)$
- 目标：让 $\log \frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} > \log \frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)}$
- 损失：Bradley-Terry 形式，$\sigma(\beta[\Delta \log \pi_c - \Delta \log \pi_r])$

**对比 PPO**：

| 维度 | PPO | DPO |
|------|-----|-----|
| 模型数 | 4（policy, ref, RM, value） | 2（policy, ref） |
| 训练范式 | 强化学习 | 监督学习 |
| 显存（70B） | ~1.5TB | ~300GB |
| 稳定性 | 差（超参敏感） | 好（监督学习） |
| 工程 | 复杂 | 简单 |
| 效果 | 略高 | 接近 PPO |
| 在线性 | 在线采样 | 离线数据 |

**DPO 的代价**：

- 离线数据，不能在线采样（效果略低于 PPO）
- 没有 value model，优势估计粗
- 对 SFT 初始化敏感（SFT 差则 DPO 差）
- 容易"过对齐"（chosen 概率降，rejected 概率降更多）

**但 DPO 的工程优势太大了**：2 模型替代 4 模型，监督学习替代强化学习，开源社区几乎全转向 DPO。

---

## L3 · 正经定义

**DPO（Direct Preference Optimization）**：Rafailov et al. (NeurIPS 2023) 提出，通过数学推导证明 RLHF 的最优策略隐含一个奖励函数，从而绕开显式 RM 和 PPO，直接用偏好对监督训练策略模型。

**核心推导**：

RLHF 的目标是在 KL 约束下最大化奖励：

$$
\max_\pi \mathbb{E}[r(x, y)] - \beta \text{KL}(\pi \| \pi_{ref})
$$

这个优化问题有闭式解：

$$
\pi^*(y|x) = \pi_{ref}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right) / Z(x)
$$

反解出 $r$：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

代入 Bradley-Terry 损失（$Z(x)$ 在 chosen vs rejected 比较中消掉）：

$$
\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)}\right)
$$

**直觉**：让 chosen 的对数概率比（相对 ref）大于 rejected 的对数概率比。

**参考资料**：

- 📄 Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023, arXiv:2305.18290
- 🔧 TRL 库 DPO 实现：https://huggingface.co/docs/trl/dpo_trainer
- 📄 von Werra et al., *DPO Trainer Walkthrough*（HuggingFace 工程实践）
- 📄 Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, ICML 2024（DPO 后续）

---

## L4 · 原理深挖

### 4.1 从 RLHF 到 DPO 的推导

**Step 1: RLHF 目标**

RLHF 在 KL 约束下最大化奖励：

$$
\max_\pi \mathbb{E}_{y \sim \pi}[r(x, y)] - \beta \text{KL}(\pi(\cdot|x) \| \pi_{ref}(\cdot|x))
$$

**Step 2: 闭式解**

这是约束优化，用变分法求解，最优策略：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right)
$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp(r(x,y)/\beta)$ 是归一化常数。

**Step 3: 反解奖励**

从 $\pi^*$ 表达式反解 $r$：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

**Step 4: 代入 Bradley-Terry**

RM 用 Bradley-Terry 训练：

$$
\mathcal{L}_{RM} = -\log \sigma(r(x, y_c) - r(x, y_r))
$$

代入反解的 $r$，$Z(x)$ 在 $r(y_c) - r(y_r)$ 中消掉：

$$
\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)}\right)
$$

**关键洞察**：$Z(x)$ 消掉是因为 chosen 和 rejected 共享同一 prompt，归一化常数相同。这就是 DPO 不需要显式 RM 的原因。

### 4.2 DPO 的实现

```python
def dpo_loss(policy_model, ref_model, chosen_ids, rejected_ids, prompt_ids, beta=0.1):
    # 1. 计算 policy 的 log prob
    chosen_logp = log_prob(policy_model, prompt_ids, chosen_ids)
    rejected_logp = log_prob(policy_model, prompt_ids, rejected_ids)
    
    # 2. 计算 ref 的 log prob（无梯度）
    with torch.no_grad():
        ref_chosen_logp = log_prob(ref_model, prompt_ids, chosen_ids)
        ref_rejected_logp = log_prob(ref_model, prompt_ids, rejected_ids)
    
    # 3. 对数比
    chosen_ratio = chosen_logp - ref_chosen_logp      # log(π/π_ref) for chosen
    rejected_ratio = rejected_logp - ref_rejected_logp
    
    # 4. DPO loss
    logits = beta * (chosen_ratio - rejected_ratio)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

**关键点**：

- 只在 response token 上算 log prob（和 SFT 一样 mask prompt）
- ref_model 冻结，无梯度
- $\beta$ 控制 KL 约束强度

### 4.3 DPO 的超参

| 超参 | 典型值 | 影响 |
|------|--------|------|
| $\beta$ | 0.1-0.5 | KL 约束强度 |
| 学习率 | $5e-7$ ~ $5e-6$ | 比 SFT 小 |
| Epoch | 1-3 | 防过对齐 |
| batch size | 64-256 | 稳定性 |

**$\beta$ 的作用**：

- $\beta$ 大：KL 约束强，策略偏离 ref 小，对齐弱
- $\beta$ 小：KL 约束弱，策略偏离 ref 大，对齐强但可能崩
- 经验：$\beta = 0.1$ 是默认，追求强对齐降到 0.05

**学习率**：DPO 学习率比 SFT 还小（$5e-7$ 到 $5e-6$），因为 DPO 是精调 SFT 模型，改动要小。

### 4.4 DPO 的"过对齐"问题

**现象**：DPO 训练后，chosen 的概率可能也下降（只是 rejected 下降更多）。

**原因**：DPO 优化的是 $\log \frac{\pi(y_c)}{\pi_{ref}(y_c)} - \log \frac{\pi(y_r)}{\pi_{ref}(y_r)}$，只要差值变大就行。模型可能让两边都降，只要 rejected 降更多。

**后果**：chosen 概率降，模型整体生成质量下降。

**解法**：

- **加 SFT loss**：DPO + SFT 联合训练，保住 chosen 概率
- **IPO / SimPO**：DPO 变体，修改损失避免过对齐
- **监控 chosen 概率**：chosen 概率降太多就早停

```python
# DPO + SFT 联合损失
loss = dpo_loss + sft_loss_weight * sft_loss(policy_model, chosen_ids)
```

### 4.5 DPO 的数据要求

**偏好对质量**：

- chosen 和 rejected 差距适中：差距太大容易学，没挑战；差距太小难学
- 多样性：覆盖不同任务、不同长度、不同差距
- 正确性：chosen 确实比 rejected 好

**数据量**：

- 5K-50K 偏好对够用
- 比 PPO 的偏好数据要求低（离线数据可复用）
- 比 SFT 数据要求略高（要偏好标注）

**数据来源**：

- 人工标注（最贵最好）
- 用 SFT 模型生成多个回答，人工排序
- 用强模型（GPT-4）做偏好判断（RLAIF 思路）

### 4.6 DPO vs PPO 深度对比

| 维度 | PPO | DPO |
|------|-----|-----|
| 模型数 | 4 | 2 |
| 训练范式 | 强化学习（on-policy） | 监督学习（off-policy） |
| 数据 | 在线采样 | 离线偏好对 |
| 显存 | ~1.5TB (70B) | ~300GB (70B) |
| 稳定性 | 差 | 好 |
| 超参敏感 | 高 | 中 |
| 效果 | 略高 | 接近 PPO |
| 工程复杂度 | 极高 | 低 |
| RM 显式 | 是 | 否（隐式） |
| 在线学习 | 支持 | 不支持 |

**PPO 效果略高的原因**：

- on-policy 采样，策略和 RM 交互更充分
- value model 估计优势更精细
- 在线数据适应策略变化

**DPO 工程优势**：

- 2 模型替代 4 模型
- 监督学习稳定
- 离线数据可复用
- 调参简单

**实践**：算力充足 + 追求极致 -> PPO；算力有限 + 快速迭代 -> DPO。开源社区 90% 用 DPO。

### 4.7 DPO 的变体

DPO 之后涌现大量变体，解决 DPO 的各种问题：

- **IPO**（Identity Preference Optimization）：修改损失避免过对齐
- **SimPO**：去掉 ref model，用长度归一化（下一篇详谈）
- **KTO**：用二元反馈（赞/踩）替代偏好对（下一篇详谈）
- **ORPO**：SFT + 偏好优化合一，不需先 SFT
- **CPO**：Contrastive Preference Optimization，对比学习思路
- **Iterative DPO**：迭代 DPO，模拟在线学习

### 4.8 DPO 的局限

**局限 1：离线数据**。不能在线采样，效果略低于 PPO。

**局限 2：过对齐**。chosen 概率可能降，要加 SFT loss 或监控。

**局限 3：对 SFT 敏感**。SFT 模型差，DPO 效果差。DPO 是精调，不是从零学。

**局限 4：无 value model**。优势估计粗，不能精细控制。

**局限 5：偏好对必需**。和 KTO 不同，DPO 必须有偏好对，不能只用二元反馈。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2023-05**：Rafailov 发 DPO 论文，证明"策略模型即 RM"
- **2023-07**：DPO 被 NeurIPS 接收，开源社区爆发式采用
- **2023 下半年**：Zephyr / Llama-2-Chat 等用 DPO 对齐，效果接近 PPO
- **2024-02**：KTO 提出二元反馈替代偏好对
- **2024-05**：SimPO 提出，去掉 ref model
- **2024-2025**：DPO 成为开源 LLM 对齐事实标准，PPO 退居大厂

### 5.2 常见坑

**坑 1：学习率太大**。DPO 学习率要比 SFT 还小，$5e-7$ 到 $5e-6$。用 $1e-5$ 直接崩。

**坑 2：$\beta$ 设错**。$\beta$ 太小过对齐崩盘，太大对齐弱。0.1 起步。

**坑 3：过对齐没监控**。chosen 概率降太多没发现，模型整体变差。要监控 chosen/rejected 概率。

**坑 4：不加 SFT loss**。纯 DPO 容易过对齐，要加 SFT loss 联合训练。

**坑 5：偏好对质量差**。chosen 和 rejected 差距不合理（太大或太小），DPO 学不到东西。

**坑 6：SFT 模型差**。DPO 是精调 SFT，SFT 差 DPO 也差。要先做好 SFT。

**坑 7：epoch 太多**。3+ epoch 过对齐，1-2 epoch 够。

**坑 8：ref model 用错**。ref 必须是 SFT 模型，不是预训练模型。

**坑 9：DPO 当万能解**。DPO 在多数场景够用，但追求极致效果还是 PPO。

**坑 10：忘了 length normalization**。长回答 log prob 低，DPO 偏好短回答。要做长度归一化。

**坑 11：偏好数据只来自一个模型**。用 SFT 模型生成所有偏好对，多样性差。要多模型多温度采样。

**坑 12：评估只看偏好准确率**。DPO 偏好准确率高但生成质量差（过对齐）。要看生成质量。

### 5.3 面试怎么考

1. **DPO 怎么绕开 RM？** 答：数学推导证明最优策略隐含奖励 $r = \beta \log \frac{\pi^*}{\pi_{ref}} + \text{const}$，代入 Bradley-Terry 损失，$Z(x)$ 在偏好对比较中消掉，不需要显式 RM。
2. **DPO 的损失函数？** 答：$\mathcal{L} = -\log \sigma(\beta[\log \frac{\pi(y_c)}{\pi_{ref}(y_c)} - \log \frac{\pi(y_r)}{\pi_{ref}(y_r)}])$。
3. **DPO vs PPO 的核心区别？** 答：DPO 2 模型监督学习离线数据，PPO 4 模型强化学习在线采样。DPO 工程简单效果略低，PPO 效果好但复杂。
4. **DPO 的过对齐问题？** 答：chosen 和 rejected 概率都降，只是 rejected 降更多。用 SFT loss 联合训练或监控 chosen 概率缓解。
5. **DPO 的 $\beta$ 作用？** 答：KL 约束强度，$\beta$ 大约束强对齐弱，$\beta$ 小约束弱对齐强。默认 0.1。

---

## 速记卡

| 组件 | 作用 | 训练 |
|------|------|------|
| Policy $\pi_\theta$ | 当前策略 | ✅ |
| Reference $\pi_{ref}$ | SFT 基线 | ❌ |

**损失函数**：

$$
\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)}\right)
$$

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| $\beta$ | 0.1 | KL 约束 |
| 学习率 | $5e-7$ ~ $5e-6$ | 稳定性 |
| Epoch | 1-2 | 防过对齐 |
| SFT loss 权重 | 0.1-0.5 | 防过对齐 |

**对比表**：

| 维度 | PPO | DPO |
|------|-----|-----|
| 模型数 | 4 | 2 |
| 训练 | RL | 监督 |
| 显存 | ~1.5TB | ~300GB |
| 稳定性 | 差 | 好 |
| 效果 | 略高 | 接近 |

**一句话记忆**：DPO = 偏好对 + Bradley-Terry 损失 + 隐式 RM。数学证明策略模型自带 RM 功能，绕开显式 RM 和 PPO。2 模型监督学习替代 4 模型强化学习，显存降 5 倍。$\beta=0.1$ 控 KL，学习率 $5e-7$，1-2 epoch。最大坑是过对齐（chosen 概率降），加 SFT loss 缓解。开源社区对齐事实标准。

---

> *上一篇：[PPO 近端策略优化](./ppo-rlhf) -- DPO 绕开 PPO 的复杂工程，效果接近。*
> *下一篇：[KTO / SimPO 变体](./kto-simpo) -- DPO 之后：KTO 用二元反馈，SimPO 去掉 ref。*
