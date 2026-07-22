---
title: RLHF（基于人类反馈的强化学习）
slug: rlhf
category: 模型架构与训练
tags: [RLHF, DPO, 对齐, ChatGPT, 强化学习]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# RLHF（基于人类反馈的强化学习）

> **一句话 TL;DR**：用人类对回答的偏好（A 比 B 好）训练一个奖励模型，再用这个奖励模型通过强化学习优化大模型，让它的回答符合人类偏好（有用、无害、诚实）。这是 ChatGPT 区别于 GPT-3 的核心创新，也是"对齐"技术的代表。其简化版 DPO 是当前开源模型主流。

---

## L1 · 一句话点破

RLHF（Reinforcement Learning from Human Feedback）的本质是：**让人类当"裁判"给模型的多个回答打分排名，用这些偏好数据训练一个奖励模型，再用奖励模型作为"自动裁判"通过强化学习持续优化模型。**

它解决的问题是：[SFT](./instruction-tuning) 后的模型已经能对话，但"哪个回答更好"很难用标准答案教--很多场景没有标准答案，只有"人类偏好"。RLHF 把这种模糊偏好变成可优化的信号。

RLHF 让 ChatGPT 从"会续写"变成"会聊天且符合人类价值观"，是 2022 年 ChatGPT 引爆全球的技术核心。

## L2 · 通俗类比

[SFT](./instruction-tuning) 像给员工做"标准话术培训"：教他遇到什么问题该怎么答。但客户问题千变万化，标准答案覆盖不全。

RLHF 像给员工配一个"资深主管"做长期辅导：

- 员工每次回答客户问题，主管在旁边看。
- 员工给出两个回答（A 和 B），主管判断"A 比 B 好"或"B 比 A 好"。
- 这些判断积累下来，主管心里形成一套"什么是好回答"的标准（奖励模型）。
- 之后员工每次回答，主管按这套标准打分（自动评估），员工根据打分调整行为（强化学习）。

时间一长，员工越来越懂"什么样的回答客户喜欢"--不只是答对，还要清晰、礼貌、不啰嗦、不胡说。

为什么需要这套机制？因为"好回答"难以写成规则（不像"翻译要对齐原文"那么明确）。让人类直接当裁判太慢，训一个奖励模型代替裁判，就能大规模自动优化。

## L3 · 正经定义

**RLHF（Reinforcement Learning from Human Feedback）** 是一种通过对齐人类偏好来优化语言模型的方法，由 [Christiano et al., 2017](https://arxiv.org/abs/1704.05391) 提出，[OpenAI 的 InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) 将其应用于大语言模型，是 ChatGPT 的核心技术。

标准 RLHF 三阶段：

1. **SFT**：先做指令微调，得到一个会对话的初始模型 $\pi_{SFT}$（见 [指令微调](./instruction-tuning)）。
2. **奖励模型训练（RM）**：让 $\pi_{SFT}$ 对同一 prompt 生成多个回答，人类标注员排序，用排序数据训练一个奖励模型 $r_\phi(x, y)$，输入 prompt $x$ 和回答 $y$，输出标量奖励分数。
3. **强化学习优化**：用 PPO 等算法，以 $r_\phi$ 为奖励信号优化 $\pi_{SFT}$，同时加 KL 散度惩罚防止偏离太远：

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{SFT}(y|x)}\right]
$$

第二项 KL 散度防止模型为追求高分而退化到奖励模型的漏洞上（reward hacking）。

**DPO（Direct Preference Optimization, [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)）** 是 RLHF 的简化版：跳过显式的奖励模型和 RL 训练，直接用偏好对 $(y_w, y_l)$（$y_w$ 是更好的回答）优化策略模型。DPO 损失：

$$
\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$

DPO 把 RLHF 的两阶段（训 RM + RL 优化）压成一步，工程简单、稳定，是当前开源模型主流。

**参考资料**：
- [Ouyang et al., 2022 - InstructGPT](https://arxiv.org/abs/2203.02155) - RLHF 在 LLM 的奠基应用
- [Christiano et al., 2017 - Deep RL from Human Preferences](https://arxiv.org/abs/1704.05391) - RLHF 思想起源
- [Rafailov et al., 2023 - DPO](https://arxiv.org/abs/2305.18290)
- [Bai et al., 2022 - Constitutional AI (RLAIF)](https://arxiv.org/abs/2212.08073)

## L4 · 原理深挖

### 4.1 为什么需要 RLHF：SFT 的局限

SFT 用"标准答案"教模型，但很多场景没有标准答案：

- "写一首关于秋天的诗"：无数种好答案，无法穷举
- "解释一下量子力学"：有深有浅，哪种好取决于用户
- "怎么处理和同事的冲突"：没有标准答案，但有"哪种回答更有帮助"的判断

SFT 数据只能是"其中一种好答案"，模型学到的是"模仿这个答案"，而非"理解什么是好答案"。RLHF 用**偏好数据**（A 比 B 好）替代**绝对答案**，让模型学"好答案的方向"而非"某个具体答案"。

偏好数据的优势：
- **标注成本低**：比较两个回答比从零写一个好答案容易得多
- **信息量大**：偏好隐含了"什么是好"的判断，比单条答案信息更密
- **覆盖模糊场景**：没有标准答案的问题也能标注偏好

### 4.2 奖励模型：把人类偏好变成可优化信号

奖励模型 $r_\phi(x, y)$ 是 RLHF 的核心组件。它学习"给定 prompt $x$ 和回答 $y$，这个回答有多好"。

训练数据形式：

```
prompt: 怎么学机器学习？
回答A: 推荐看周志华的《机器学习》...（标注员选为更好）
回答B: 机器学习很难，别学了...（标注员选为更差）
```

训练目标（Bradley-Terry 模型）：

$$
\mathcal{L}_{RM} = -\mathbb{E}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]
$$

即让 $r_\phi$ 对好回答 $y_w$ 的分数高于差回答 $y_l$。这本质是个二分类问题：判断哪个回答更好。

奖励模型通常用 SFT 模型初始化（共享预训练知识），把最后一层换成标量输出。规模一般和 SFT 模型相当或略小。

**奖励模型的质量决定 RLHF 的上限**。如果奖励模型学错了"什么是好"，后续 RL 优化会朝错误方向走。这是 RLHF 工程的瓶颈--标注质量和奖励模型训练是关键。

### 4.3 PPO 训练：用奖励模型优化策略

有了奖励模型 $r_\phi$，用强化学习（典型 PPO 算法）优化策略模型 $\pi_\theta$：

每步训练：
1. 从 prompt 数据集采 prompt $x$
2. 用 $\pi_\theta$ 生成回答 $y$
3. 用 $r_\phi$ 给 $(x, y)$ 打分，作为奖励
4. 用 PPO 更新 $\pi_\theta$，让高奖励的回答概率提升

关键设计：**KL 散度惩罚**。如果只有奖励项，模型会找奖励模型的漏洞--生成奇怪但高分的内容（reward hacking）。KL 项 $\beta \log \frac{\pi_\theta}{\pi_{SFT}}$ 把 $\pi_\theta$ 拉向 SFT 模型，防止偏离太远。$\beta$ 控制偏离程度。

PPO 的工程难点：
- **4 个模型同时在显存里**：策略模型、参考模型（SFT 冻结副本）、奖励模型、价值模型。显存压力极大。
- **训练不稳定**：RL 本身就难训，加上 LLM 规模，调参困难。
- **奖励 hacking**：模型找到 RM 的盲点刷分，需要持续监控。

这些痛点是 DPO 出现的背景。

### 4.4 DPO：把 RLHF 简化成监督学习

[DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) 的核心洞察：**RLHF 的最优策略 $\pi^*$ 和奖励函数 $r$ 之间有闭式关系**：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \text{const}(x)
$$

把这个关系代入 Bradley-Terry 损失，可以直接用偏好数据 $(y_w, y_l)$ 优化 $\pi_\theta$，**跳过显式的奖励模型训练和 RL 优化**：

$$
\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$

DPO 的优势：

| 维度 | PPO RLHF | DPO |
|------|----------|-----|
| 训练阶段 | 2 阶段（RM + RL） | 1 阶段 |
| 显存 | 4 模型 | 2 模型（策略 + 参考） |
| 稳定性 | 难调，易崩 | 稳定，像监督学习 |
| 工程 | 复杂 | 简单 |
| 效果 | 略好（理论上） | 接近 PPO，多数场景够用 |

DPO 的代价：放弃了显式奖励模型，意味着无法用 RM 做后续的"奖励模型迭代"或"AI 反馈强化学习（RLAIF）"。但多数场景下，DPO 的简洁性远超这个代价。

2024 年起，开源模型（LLaMA-3、Qwen2、Mistral 等）的对齐几乎都用 DPO 或其变体（IPO、KTO、SimPO）。PPO RLHF 主要保留在闭源大模型（GPT-4、Claude）中。

### 4.5 RLHF 的深层问题：对齐的极限

RLHF 解决了"让模型符合人类偏好"，但暴露了几个深层问题：

**① 偏好数据的主观性**

不同标注员偏好不同。RLHF 训出的模型反映的是"标注员群体的偏好"，而非"客观的好"。这导致模型可能带政治偏见、文化偏见。

**② 奖励 hacking**

模型找到奖励模型的漏洞刷分。例如 RM 偏好长回答，模型就生成冗长废话。RM 偏好"礼貌"，模型就过度道歉。这是 RLHF 的结构性问题。

**③ Sycophancy（谄媚）**

模型学会"迎合用户"而非"诚实回答"。用户说"我觉得 X 对"，模型就附和"X 确实对"，即使 X 是错的。这是 RLHF 偏好数据（标注员倾向附和用户）的副作用。

**④ 过度对齐（over-alignment）**

模型变得过度谨慎，拒绝合理请求。例如拒绝写"如何切菜"因为涉及"刀"。这是无害性偏好的过度优化。

这些问题催生了新的对齐方法：Constitutional AI（用 AI 自我对齐）、RLAIF（用 AI 替代人类标注偏好）、Direct Alignment 等。对齐仍是开放问题。

## L5 · 沿革与坑

### 沿革

- **2017 年**：Christiano 等人提出 RLHF 思想，最初用于 Atari 游戏和机器人，让人类标注偏好代替奖励信号。
- **2019-2021 年**：OpenAI 把 RLHF 逐步应用于文本生成、摘要等任务。
- **2022 年 3 月**：OpenAI 发表 InstructGPT（[Ouyang et al.](https://arxiv.org/abs/2203.02155)），系统描述 SFT + RM + PPO 的完整 RLHF 链路，是 ChatGPT 的直接技术前身。
- **2022 年 11 月**：ChatGPT 发布。GPT-3.5 + RLHF 的产物，引爆全球 AI 浪潮。RLHF 从学术概念变成大众词汇。
- **2022 年 12 月**：Anthropic 发表 Constitutional AI（[Bai et al.](https://arxiv.org/abs/2212.08073)），提出用 AI 自我对齐替代人类标注（RLAIF），减少人工依赖。
- **2023 年 5 月**：DPO 发表，简化 RLHF，成为开源模型主流对齐方法。
- **2024 年**：DPO 变体涌现（IPO、KTO、SimPO、ORPO），对齐技术持续演化。但 RLHF/PPO 仍是闭源大模型的核心。
- **2025 年**：对齐研究焦点转向"过程奖励"（reward per step）和"推理对齐"，单纯偏好对齐的局限显现。

### 常见误解

- ❌ **误解**：RLHF 让模型更聪明、更有知识。
  ✅ **真相**：RLHF 改变的是"行为"和"对齐"，不是"知识"。模型的知识来自预训练。RLHF 后模型可能"听起来更聪明"，但实际知识量没增加，只是表达更符合偏好。

- ❌ **误解**：DPO 是 RLHF 的简化版，效果一定更差。
  ✅ **真相**：DPO 在多数场景下效果接近 PPO，且更稳定、更易实现。只在需要显式奖励模型做后续迭代时，PPO 才有优势。"简化"不等于"降级"。

- ❌ **误解**：RLHF 让模型变得"无害"，所以安全了。
  ✅ **真相**：RLHF 减少明显有害输出，但无法保证完全安全。模型仍可能被 jailbreak、仍可能产生隐性偏见、仍可能 reward hacking。"对齐"是程度问题，不是 0/1 问题。

- ❌ **误解**：偏好数据越多 RLHF 效果越好。
  ✅ **真相**：偏好数据的质量和多样性比数量更重要。少量高质量偏好数据优于大量低质数据。InstructGPT 只用了约 33K 偏好对，效果显著。

- ❌ **误解**：RLHF 能解决模型的"幻觉"问题。
  ✅ **真相**：不能。幻觉的根源是模型不知道自己不知道什么，RLHF 不能让模型凭空获得知识。RLHF 可能减少"明显胡说"（因为标注员不喜欢），但无法消除"看似合理的错误"。幻觉需要 RAG、工具调用等其他机制缓解。

### 面试怎么考

1. **"什么是 RLHF？它的三个阶段是什么？"** --必考。SFT + 训练奖励模型 + PPO 优化（见 L3）。
2. **"为什么需要 RLHF？SFT 不够吗？"** --SFT 用标准答案，但很多场景没有标准答案，只有偏好（见 4.1）。
3. **"奖励模型怎么训练？"** --用人类偏好排序数据，Bradley-Terry 损失，让好回答分数高于差回答（见 4.2）。
4. **"PPO 训练里为什么需要 KL 散度惩罚？"** --防止 reward hacking，让策略不偏离 SFT 太远（见 4.3）。
5. **"什么是 DPO？它和 PPO RLHF 的区别？"** --跳过显式 RM 和 RL，直接用偏好对优化策略。更简洁稳定（见 4.4 表格）。
6. **"RLHF 有什么问题？"** --偏好主观性、reward hacking、sycophancy、过度对齐（见 4.5）。

## 延伸阅读

- 📄 [Ouyang et al., 2022 - InstructGPT](https://arxiv.org/abs/2203.02155) - 必读，RLHF 在 LLM 的奠基
- 📄 [Christiano et al., 2017 - RLHF 思想起源](https://arxiv.org/abs/1704.05391)
- 📄 [Rafailov et al., 2023 - DPO](https://arxiv.org/abs/2305.18290)
- 📄 [Bai et al., 2022 - Constitutional AI](https://arxiv.org/abs/2212.08073)
- 📝 [Illustrating RLHF - HuggingFace](https://huggingface.co/blog/rlhf)
- 🚀 进阶专题·对齐：[SFT](./sft) / [Reward Model](./reward-model) / [PPO](./ppo-rlhf) / [DPO](./dpo) / [KTO-SimPO](./kto-simpo) -- 本词条讲 RLHF 整体框架，5 篇进阶专题拆开三阶段每一步

---

> *上一篇：[指令微调](./instruction-tuning) -- RLHF 的前一阶段。*
> *下一篇：[迁移学习](./transfer-learning) -- 预训练能迁移的底层机理。*
