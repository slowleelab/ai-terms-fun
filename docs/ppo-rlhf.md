---
title: PPO 近端策略优化
slug: ppo-rlhf
category: 进阶专题
tags: [PPO, RLHF, 强化学习, KL 约束, GAE, 策略梯度]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# PPO 近端策略优化

> 五层读懂一个词。这次拆的是：**PPO（Proximal Policy Optimization）**--RLHF 的第三阶段，用 RM 的分数当奖励，强化学习优化 SFT 模型。4 个模型同框训练，工程最复杂但效果最好。

---

## L1 · 一句话点破

**PPO = 策略梯度 + Clip 信任域 + KL 约束 + GAE 优势估计**。让 SFT 模型生成回答，RM 打分当奖励，策略梯度优化模型，但用 clip 和 KL 约束防止偏离太远。RLHF 里效果最好但工程最复杂的方法。

---

## L2 · 通俗类比

SFT + RM 之后，我们有了：

- 一个会对话的模型（SFT 模型，叫 $\pi_{SFT}$）
- 一个会打分的教练（RM）

PPO 的工作是让模型**根据教练打分自我改进**：

1. 模型生成回答 $y$
2. RM 打分 $r(x, y)$
3. 如果分数高，强化这种生成（增加概率）
4. 如果分数低，抑制这种生成（降低概率）
5. 但不能改太猛（否则变成"只会刷分"的废柴，即 reward hacking）

**关键约束**：第 5 点是 PPO 的精髓。纯策略梯度会让模型疯狂优化 RM 分数，钻 RM 漏洞。PPO 用两个机制防止：

- **Clip**：每次更新幅度限制在 $[1-\epsilon, 1+\epsilon]$ 范围内
- **KL 约束**：模型不能偏离 SFT 模型太远（KL 散度惩罚）

**PPO 的 4 个模型**（显存爆炸的根源）：

| 模型 | 作用 | 是否训练 |
|------|------|---------|
| Policy Model $\pi_\theta$ | 当前策略，生成回答 | ✅ 训练 |
| Reference Model $\pi_{ref}$ | SFT 模型，计算 KL | ❌ 冻结 |
| Reward Model | 打分 | ❌ 冻结 |
| Value Model $V_\phi$ | 估计状态价值，算优势 | ✅ 训练 |

4 个大模型同框，70B 模型 PPO 显存要 8×A100，工程门槛极高。这也是 DPO 出现的动力--PPO 太重了。

**PPO 的代价**：4 模型同框显存爆炸；训练不稳定（超参敏感）；工程复杂；reward hacking 风险。但效果是所有对齐方法里最好的，GPT-4 / Claude 都用 PPO。

---

## L3 · 正经定义

**PPO（Proximal Policy Optimization）**：Schulman et al. (2017) 提出的策略梯度算法，通过 clip 机制限制策略更新幅度，平衡样本效率和稳定性。在 RLHF 中，PPO 用 RM 分数作为奖励，优化 SFT 模型策略，同时用 KL 约束防止偏离 SFT 太远。

**PPO-RLHF 目标函数**：

$$
\mathcal{L}_{PPO} = \mathbb{E}\left[ \min\left( \frac{\pi_\theta(y|x)}{\pi_{old}(y|x)} \hat{A}, \text{clip}\left(\frac{\pi_\theta(y|x)}{\pi_{old}(y|x)}, 1-\epsilon, 1+\epsilon\right) \hat{A} \right) \right] - \beta \text{KL}(\pi_\theta \| \pi_{ref})
$$

其中：

- $\frac{\pi_\theta}{\pi_{old}}$：重要性采样比率（新策略 vs 旧策略的概率比）
- $\hat{A}$：优势函数估计（GAE）
- $\epsilon$：clip 范围（通常 0.1-0.2）
- $\beta$：KL 惩罚系数
- $\pi_{ref}$：SFT 参考模型

**奖励设计**：

$$
R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

RM 分数减去 KL 惩罚，KL 项防止偏离 SFT 太远。

**参考资料**：

- 📄 Schulman et al., *Proximal Policy Optimization Algorithms*, 2017, arXiv:1707.06347
- 📄 Ouyang et al., *InstructGPT*, NeurIPS 2022（PPO 应用到 LLM）
- 📄 Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (GAE), ICLR 2016
- 🔧 TRL 库（HuggingFace PPO 实现）：https://github.com/huggingface/trl

---

## L4 · 原理深挖

### 4.1 策略梯度基础

RL 的核心是**策略梯度**：最大化期望奖励。

$$
J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(x, y)]
$$

梯度：

$$
\nabla J = \mathbb{E}\left[ \nabla \log \pi_\theta(y|x) \cdot A(x, y) \right]
$$

其中 $A(x, y)$ 是优势函数，衡量"这个回答比平均好多少"。

**直觉**：如果 $A > 0$（比平均好），梯度方向增加 $\pi_\theta(y|x)$（多生成这种）；如果 $A < 0$（比平均差），减少 $\pi_\theta(y|x)$。

**问题**：纯策略梯度不稳定，单次更新可能让策略崩坏。PPO 的改进就是限制更新幅度。

### 4.2 PPO 的 Clip 机制

PPO 用**重要性采样比率**衡量策略变化：

$$
r_t(\theta) = \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{old}(y_t | x, y_{<t})}
$$

- $r_t = 1$：策略没变
- $r_t > 1$：新策略更倾向生成这个 token
- $r_t < 1$：新策略更不倾向

**Clip 目标**：

$$
L^{CLIP} = \mathbb{E}\left[ \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

**两种情况**：

- $\hat{A}_t > 0$（好 token）：鼓励 $r_t$ 增大，但 clip 在 $1+\epsilon$，防止过度强化
- $\hat{A}_t < 0$（差 token）：鼓励 $r_t$ 减小，但 clip 在 $1-\epsilon$，防止过度抑制

**直觉**：clip 创建"信任域"，每次更新幅度限制，防止策略崩坏。$\epsilon = 0.2$ 意味着每次策略变化最多 20%。

### 4.3 GAE 优势估计

优势函数 $A(x, y)$ 衡量"这个回答比平均好多少"。直接用奖励 $R$ 减去 baseline $V(x)$：

$$
A(x, y) = R(x, y) - V_\phi(x)
$$

$V_\phi$ 是 value model，估计"这个 prompt 的平均奖励"。

**GAE（Generalized Advantage Estimation）**：考虑多步奖励，平衡偏差和方差：

$$
\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差，$\gamma$ 是折扣因子，$\lambda$ 是 GAE 参数。

**在 RLHF 里**：response 是 token 序列，每个 token 都有优势估计。GAE 平滑多步奖励信号。

### 4.4 KL 约束：防 reward hacking

**为什么需要 KL 约束**：

PPO 优化 RM 分数，但 RM 有漏洞。模型会钻空子生成"RM 高分但人类不喜欢"的回答。

**KL 惩罚**：

$$
R_{total}(x, y) = r_\phi(x, y) - \beta \text{KL}(\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x))
$$

KL 项惩罚策略偏离 SFT 模型。$\beta$ 越大，约束越强。

**直觉**：SFT 模型是"正常对话"的基线，PPO 优化时不能离它太远。离太远意味着模型在"钻空子"。

**实践**：

- $\beta$ 通常 0.01-0.5
- $\beta$ 太大：约束强，学不动
- $\beta$ 太小：约束弱，reward hacking
- 自适应 $\beta$：根据 KL 动态调整

### 4.5 PPO-RLHF 完整算法

```python
def ppo_rlhf(policy_model, ref_model, reward_model, value_model, 
             prompts, tokenizer, epochs=200, clip_eps=0.2, beta=0.1):
    optimizer = AdamW(policy_model.parameters(), lr=1e-6)
    
    for epoch in range(epochs):
        # 1. 采样：policy 生成回答
        responses = policy_model.generate(prompts)
        
        # 2. 打分：RM 打分
        rewards = reward_model(prompts, responses)
        
        # 3. KL 惩罚：policy vs ref
        log_ratio = log_prob(policy_model, responses) - log_prob(ref_model, responses)
        kl_penalty = beta * log_ratio.mean()
        rewards = rewards - kl_penalty
        
        # 4. 优势估计：GAE
        values = value_model(prompts, responses)
        advantages = gae(rewards, values, gamma=1.0, lam=0.95)
        
        # 5. PPO 更新（多 epoch）
        old_log_probs = log_prob(policy_model, responses).detach()
        for ppo_epoch in range(4):  # 通常 4 个 PPO epoch
            new_log_probs = log_prob(policy_model, responses)
            ratio = (new_log_probs - old_log_probs).exp()
            
            # clip 目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # value loss
            value_loss = mse(value_model(prompts, responses), rewards)
            
            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### 4.6 PPO 的 4 个模型与显存

**4 个模型**：

| 模型 | 大小 | 训练 | 显存（70B） |
|------|------|------|-----------|
| Policy $\pi_\theta$ | 70B | ✅ | 140GB（权重） + 梯度 + Adam = ~600GB |
| Reference $\pi_{ref}$ | 70B | ❌ | 140GB |
| Reward Model | 70B | ❌ | 140GB |
| Value Model $V_\phi$ | 70B | ✅ | ~600GB |
| **合计** | | | **~1.5TB** |

**显存优化**：

- Reference / RM / Value 用 LoRA 或量化
- Policy 用全参（要训）
- Gradient checkpointing
- 实际 8×A100 80G 可跑 70B PPO

### 4.7 PPO 的超参与稳定性

PPO 以**超参敏感**著称，调不好就崩：

| 超参 | 典型值 | 影响 |
|------|--------|------|
| 学习率 | $1e-6$ ~ $1e-5$ | 太大崩，太小学不动 |
| clip $\epsilon$ | 0.1-0.2 | 信任域大小 |
| KL $\beta$ | 0.01-0.5 | 防 hacking |
| PPO epoch | 4 | 每批数据重用次数 |
| GAE $\lambda$ | 0.95 | 优势估计偏差方差权衡 |
| batch size | 512+ | 稳定性 |
| temperature | 0.7-1.0 | 生成多样性 |

**稳定性技巧**：

- Warmup：前 10% 步数线性升温
- 梯度裁剪：max_grad_norm = 1.0
- 监控 KL：KL 超阈值就停（早停）
- 监控 reward：reward 突涨可能是 hacking

### 4.8 PPO vs DPO vs 其他

| 维度 | PPO | DPO | KTO |
|------|-----|-----|-----|
| 模型数 | 4 | 2 | 2 |
| 数据 | 偏好对 | 偏好对 | 二元反馈 |
| 训练 | 强化学习 | 监督学习 | 监督学习 |
| 显存 | ~1.5TB (70B) | ~300GB | ~300GB |
| 稳定性 | 差 | 好 | 好 |
| 效果 | 最好 | 接近 PPO | 接近 DPO |
| 工程复杂度 | 极高 | 低 | 低 |
| 灵活性 | 高（RM 可复用） | 中 | 中 |

**何时用 PPO**：

- 追求极致效果（GPT-4 / Claude 级别）
- 有充足算力（多卡 A100）
- 有经验丰富的 RL 工程师
- 在线学习场景（持续收集偏好）

**何时不用 PPO**：

- 算力有限 -> DPO
- 工程能力不足 -> DPO
- 快速迭代 -> DPO
- 数据是二元反馈（赞/踩）-> KTO

---

## L5 · 沿革与坑

### 5.1 沿革

- **2017**：Schulman 提出 PPO，用于游戏 RL
- **2017**：Christiano 提出 RLHF 框架
- **2022-03**：InstructGPT 把 PPO 用到 LLM 对齐
- **2022-11**：ChatGPT 用 PPO 爆火
- **2023-05**：DPO 提出绕开 PPO，简化对齐
- **2023 下半年**：开源社区转向 DPO，但大厂仍用 PPO
- **2024-2025**：PPO 在大厂仍主流，DPO/KTO 在开源流行，GRPO 等新变体出现

### 5.2 常见坑

**坑 1：学习率太大崩盘**。PPO 学习率要比 SFT 小 10 倍，$1e-6$ 到 $1e-5$。用 $1e-4$ 直接崩。

**坑 2：clip $\epsilon$ 太大**。$\epsilon=0.5$ 信任域太大，策略崩坏。$0.1-0.2$ 是安全值。

**坑 3：KL $\beta$ 设错**。$\beta$ 太小 reward hacking，太大学不动。要从 0.1 起步，监控 KL 动态调。

**坑 4：PPO epoch 太多**。每批数据重用 10+ epoch，策略过拟合到这批数据。4 epoch 够。

**坑 5：忘了监控 reward hacking**。reward 一直涨但人工评估变差。要定期人工抽检。

**坑 6：4 模型显存没估**。70B PPO 要 1.5TB 显存，8×A100 起步。算力不够别硬上。

**坑 7：value model 训不好**。value loss 不降，优势估计不准，策略更新方向错。要监控 value loss。

**坑 8：生成温度设错**。温度太低多样性差，温度太高生成噪声。0.7-1.0 是经验值。

**坑 9：KL 超阈值不早停**。KL 持续涨，策略偏离 SFT 太远，模型崩坏。要设 KL 阈值早停。

**坑 10：batch size 太小**。PPO 对 batch size 敏感，<128 不稳定。要 512+。

**坑 11：reward 不归一化**。RM 分数量纲不固定，advantage 估计偏。要做 reward 归一化。

**坑 12：PPO 当万能解**。PPO 效果好但工程重，DPO 在多数场景够用。不是非要 PPO。

### 5.3 面试怎么考

1. **PPO 的 clip 机制作用？** 答：限制策略更新幅度在 $[1-\epsilon, 1+\epsilon]$ 信任域内，防止单次更新崩坏策略。
2. **PPO-RLHF 为什么需要 KL 约束？** 答：防止策略偏离 SFT 太远导致 reward hacking，KL 惩罚让模型在"优化 RM 分数"和"保持正常对话"间平衡。
3. **PPO 的 4 个模型分别是什么？** 答：Policy（训练）、Reference（SFT 冻结，算 KL）、Reward Model（冻结，打分）、Value Model（训练，算优势）。
4. **PPO 为什么比 DPO 效果好？** 答：PPO 在线采样（on-policy），策略和 RM 交互更充分；有 value model 估计优势；KL 约束更精细。DPO 是离线监督学习，表达力弱。
5. **PPO 的工程难点？** 答：4 模型显存爆炸、超参敏感、训练不稳定、reward hacking 监控。

---

## 速记卡

| 模型 | 作用 | 训练 |
|------|------|------|
| Policy $\pi_\theta$ | 生成回答 | ✅ |
| Reference $\pi_{ref}$ | KL 约束基线 | ❌ |
| Reward Model | 打分 | ❌ |
| Value $V_\phi$ | 优势估计 | ✅ |

**目标函数**：

$$
L^{CLIP} = \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t) - \beta \text{KL}
$$

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| 学习率 | $1e-6$ ~ $1e-5$ | 稳定性 |
| clip $\epsilon$ | 0.1-0.2 | 信任域 |
| KL $\beta$ | 0.01-0.5 | 防 hacking |
| PPO epoch | 4 | 数据重用 |
| GAE $\lambda$ | 0.95 | 优势估计 |
| batch size | 512+ | 稳定性 |

**显存对比**（70B）：

| 方法 | 模型数 | 显存 |
|------|--------|------|
| PPO | 4 | ~1.5TB |
| DPO | 2 | ~300GB |
| KTO | 2 | ~300GB |

**一句话记忆**：PPO = 策略梯度 + clip 信任域 + KL 约束 + GAE 优势，RLHF 第三阶段，4 模型同框显存爆炸但效果最好。clip 限制单次更新幅度，KL 防 reward hacking，GAE 估优势。学习率 $1e-6$、clip 0.2、KL $\beta$ 0.1、4 epoch 是经验默认。大厂用 PPO 追求极致效果，开源社区转向 DPO 省工程。

---

> *上一篇：[Reward Model 奖励模型](./reward-model) -- RM 给 PPO 提供奖励信号。*
> *下一篇：[DPO 直接偏好优化](./dpo) -- 绕开 RM 和 PPO，直接从偏好对优化策略。*
