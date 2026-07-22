---
title: SFT 监督微调
slug: sft
category: 进阶专题
tags: [SFT, 监督微调, 指令数据, 对齐, ChatML]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# SFT 监督微调

> 五层读懂一个词。这次拆的是：**SFT（Supervised Fine-Tuning）**--对齐的起点，用人工标注的指令-回答对，教预训练模型学会"听人话、说人话"。RLHF 三阶段里的第一阶段，也是所有对齐方法的必经之路。

---

## L1 · 一句话点破

**SFT = 指令-回答对数据 + 监督交叉熵训练**。把"只会续写"的基座模型，变成"听指令、按格式、有礼貌"的对话模型。本质是 next-token prediction，但只在回答部分算 loss。

---

## L2 · 通俗类比

预训练模型是个读遍全网的书呆子，能续写但不会"对话"。你问它"今天天气怎么样"，它可能续写成"今天天气怎么样？这是一个好问题。让我们从气象学说起..."--自言自语，不直接回答。

SFT 的工作是教它**对话礼仪**：

- **听指令**：用户说"总结这段话"，就总结，不跑题
- **按格式**：问"写诗"就给诗，问"列要点"就给列表
- **有礼貌**：不输出有害内容，不胡说八道

教法很朴素：准备几万到几十万条**指令-回答对**，告诉模型"这种问题应该这么答"，用监督学习（交叉熵）训练。

**SFT 数据示例**：

```json
{
  "instruction": "用一句话解释什么是 RAG",
  "output": "RAG 是检索增强生成，让 LLM 先检索相关文档再生成答案。"
}
{
  "instruction": "把这句话翻译成英文：今天天气很好",
  "output": "The weather is nice today."
}
```

**SFT 在对齐链路中的位置**：

```
预训练（next-token prediction）
    ↓
SFT（监督微调，学会对话）        ← 本篇
    ↓
Reward Model（学人类偏好）
    ↓
PPO / DPO（强化学习 / 偏好优化）
    ↓
对齐后的对话模型
```

**SFT 是所有对齐方法的起点**：

- RLHF 的第一阶段就是 SFT
- DPO 也要先做 SFT 再做偏好优化
- 即使不做 RL，SFT 单独也能产出可用的对话模型（早期的 Alpaca、Vicuna 就是纯 SFT）

**代价**：需要高质量指令数据；数据偏差直接传给模型；过拟合会失去通用能力；但相对 RLHF，SFT 工程简单、数据要求明确、训练稳定，是性价比最高的对齐手段。

---

## L3 · 正经定义

**SFT（Supervised Fine-Tuning）**：在预训练语言模型基础上，用人工标注的指令-回答对（instruction-response pairs）做监督微调，使模型学会遵循指令、按对话格式输出。训练目标是对回答部分做 next-token prediction（交叉熵损失），指令部分不算 loss。

**数据格式**（ChatML 模板为例）：

```
<|im_start|>user
用一句话解释什么是 RAG<|im_end|>
<|im_start|>assistant
RAG 是检索增强生成，让 LLM 先检索相关文档再生成答案。<|im_end|>
```

**损失函数**：

$$
\mathcal{L}_{SFT} = -\sum_{t=1}^{T} \mathbb{1}_{t \in \text{response}} \log p_\theta(y_t \mid y_{<t}, x)
$$

其中 $x$ 是指令，$y$ 是回答，$\mathbb{1}_{t \in \text{response}}$ 是指示函数，只在回答 token 上算 loss（指令 token 的 loss 被 mask 掉）。

**关键设计**：

- **Loss masking**：只对 assistant 部分算 loss，user 部分不算
- **多轮对话**：每个 assistant 回答都算 loss，user 轮次不算
- **模板一致性**：训练用什么模板，推理就用什么模板

**参考资料**：

- 📄 Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), NeurIPS 2022, arXiv:2203.02155
- 📄 Taori et al., *Stanford Alpaca*, 2023（SFT 实践范例）
- 📄 Chiang et al., *Vicuna*, 2023
- 📄 Wang et al., *Self-Instruct: Aligning Language Models with Self-Generated Instructions*, ACL 2023
- 📄 Zhou et al., *LIMA: Less Is More for Alignment*, NeurIPS 2023

---

## L4 · 原理深挖

### 4.1 从预训练到 SFT：任务转变

预训练目标是**无监督续写**：

$$
\mathcal{L}_{pretrain} = -\sum_t \log p(x_t \mid x_{<t})
$$

所有 token 都算 loss，模型学到的是"统计续写"。

SFT 目标是**指令条件生成**：

$$
\mathcal{L}_{SFT} = -\sum_{t \in response} \log p(y_t \mid y_{<t}, x)
$$

只在回答 token 算 loss，模型学到的是"听指令后回答"。

**关键转变**：

- 输入从"纯文本"变成"指令 + 回答"的结构化对话
- Loss 从"全 token"变成"仅回答 token"
- 目标从"续写"变成"对齐指令"

### 4.2 Loss Masking 的实现

```python
def sft_loss(logits, labels, mask):
    """
    logits: [batch, seq, vocab]
    labels: [batch, seq]
    mask: [batch, seq]  # 1 for response tokens, 0 for instruction/padding
    """
    # 1. shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = mask[:, 1:]
    
    # 2. cross entropy
    loss_fct = CrossEntropyLoss(reduction='none')
    losses = loss_fct(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1)
    )
    
    # 3. 只在 response token 上算 loss
    losses = losses * shift_mask.reshape(-1)
    
    # 4. 平均
    return losses.sum() / shift_mask.sum()
```

**为什么 mask 掉指令部分**：

- 指令是给定的，不是模型要生成的，算 loss 无意义
- 如果指令也算 loss，模型会学着"生成指令"，偏离对话目标
- 只算回答 loss，模型专注学"怎么回答"

### 4.3 指令数据构造

SFT 的核心是**高质量指令数据**。数据质量 > 数据数量（LIMA 证明 1000 条高质量数据就能对齐）。

**数据来源**：

1. **人工标注**：最贵但质量最高（InstructGPT、ChatGPT 用这种方法）
2. **开源数据集**：Alpaca、ShareGPT、FLAN、Dolly 等
3. **Self-Instruct**：用 LLM 自动生成指令（种子指令 -> LLM 扩展 -> 过滤）
4. **蒸馏**：用 GPT-4 等强模型生成回答（Alpaca 就是 LLaMA + GPT-4 蒸馏）
5. **真实用户对话**：ShareGPT 抓取真实 ChatGPT 对话

**数据类型分布**（典型 SFT 数据集）：

| 类型 | 比例 | 例子 |
|------|------|------|
| 开放问答 | 25% | "为什么天是蓝的" |
| 写作 | 20% | "写一首关于秋天的诗" |
| 翻译 | 10% | "翻译这句话" |
| 代码 | 10% | "写一个快排" |
| 数学 | 10% | "解这个方程" |
| 角色扮演 | 10% | "扮演导游介绍北京" |
| 其他 | 15% | 总结、改写、分类等 |

**数据质量要求**：

- 指令清晰、无歧义
- 回答正确、有依据
- 多样性（覆盖不同任务、不同难度、不同长度）
- 无有害内容

### 4.4 SFT 的训练细节

**学习率**：比预训练小一个数量级，通常 $1e-5$ 到 $5e-5$。预训练 $1e-4$，SFT 不能用那么大，否则破坏预训练知识。

**Epoch 数**：通常 2-3 个 epoch。太多会过拟合（模型记住训练数据），太少学不充分。

**Batch size**：大 batch（64-128）更稳定。配合 gradient accumulation 在显存受限时模拟大 batch。

**学习率调度**：cosine decay，warmup 前 3% 步数。

**LoRA SFT**：

- 全参 SFT：所有参数更新，效果上限高，但显存大
- LoRA SFT：只训 LoRA 参数，显存小，效果略低但接近
- 实践：70B 以下用 LoRA SFT，追求极致效果用全参

**伪代码**：

```python
def sft_train(model, dataset, epochs=3, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    
    for epoch in range(epochs):
        for batch in dataset:
            # 1. 前向
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']  # -100 for masked tokens
            )
            loss = outputs.loss  # HuggingFace 自动 mask -100
            # 2. 反向
            loss.backward()
            # 3. 更新
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

### 4.5 多轮对话的 SFT

多轮对话数据：

```
user: 帮我写一首诗
assistant: <诗1>
user: 换成五言绝句
assistant: <诗2>
```

**Loss 计算**：每个 assistant 回答都算 loss，user 部分不算。

```python
# 多轮对话的 mask 构造
input_ids = [user1, assistant1, user2, assistant2]
mask =      [0,     0,         0,     1]      # 只在 assistant 部分为 1
```

**关键**：多轮对话要让模型看到完整上下文，不能只训最后一轮。

### 4.6 SFT 的过拟合与灾难性遗忘

**过拟合**：

- 训练 epoch 太多，模型记住训练数据，泛化变差
- 表现：训练 loss 持续降，但验证集效果变差
- 解法：early stopping、监控验证集、限制 epoch 数

**灾难性遗忘**：

- SFT 数据分布和预训练不同，模型忘了预训练知识
- 表现：SFT 后模型在通用 benchmark（MMLU）上掉点
- 解法：

  - 小学习率（$1e-5$ 而非 $1e-4$）
  - 混合预训练数据（SFT 数据 + 少量预训练数据）
  - LoRA（冻结基座，天然防遗忘）

### 4.7 SFT 的数据量 vs 质量

**LIMA 的发现**（Zhou et al. 2023）：

- 1000 条**高质量**人工数据，SFT 后效果接近 GPT-4
- 关键是质量，不是数量

**对比实验**：

| 数据量 | 质量 | 效果 |
|--------|------|------|
| 1000 条 | 极高（LIMA） | 接近 GPT-4 |
| 52K 条 | 中等（Alpaca，GPT-4 蒸馏） | 可用，但不如 LIMA |
| 100K+ 条 | 参差（ShareGPT） | 看清洗质量 |

**实践建议**：

- 小而精（1K-10K 高质量）> 大而杂（100K+ 参差）
- 数据清洗比数据收集重要
- 人工写 > GPT-4 蒸馏 > 自动生成

### 4.8 SFT vs RLHF vs DPO

| 维度 | SFT | RLHF (RM + PPO) | DPO |
|------|-----|-----------------|-----|
| 数据 | 指令-回答 | 指令-回答 + 偏好对 | 指令-回答 + 偏好对 |
| 目标 | 模仿回答 | 最大化奖励 | 偏好对分类 |
| 训练 | 监督学习 | 强化学习 | 监督学习 |
| 复杂度 | 低 | 高（4 个模型） | 中（2 个模型） |
| 效果 | 基础对齐 | 更好（人类偏好） | 接近 RLHF |
| 依赖 | 无 | 先要 SFT | 先要 SFT |

**关键**：SFT 是 RLHF 和 DPO 的前置步骤。不做 SFT 直接 RLHF/DPO，效果会差很多。

### 4.9 SFT 的局限

**局限 1：上限受数据限制**。SFT 只能学到数据里的回答模式，无法超越标注者水平。

**局限 2：偏好对齐弱**。SFT 教的是"这么答"，不是"这个比那个好"。细粒度偏好需要 RLHF/DPO。

**局限 3：数据偏差**。标注者的偏差（文化、立场）直接传给模型。

**局限 4：泛化有限**。SFT 数据覆盖的任务做得好，没覆盖的差。

**局限 5：有害内容**。SFT 数据如果有害内容清洗不彻底，模型会学有害行为。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2022-03**：InstructGPT 论文确立 SFT -> RM -> PPO 三阶段
- **2023-03**：Alpaca 证明 GPT-4 蒸馏 + SFT 能让 LLaMA 接近 ChatGPT
- **2023-03**：Vicuna 用 ShareGPT 真实对话 SFT，效果更好
- **2023-05**：LIMA 证明 1000 条高质量数据就够对齐
- **2023 下半年**：SFT 成为开源 LLM 对齐标配
- **2024-2025**：SFT 数据工程化，出现数据筛选、清洗、合成的系统方法论

### 5.2 常见坑

**坑 1：数据没清洗**。ShareGPT 等开源数据含大量噪声、有害内容、GPT 风格化套话。要做严格清洗。

**坑 2：loss 没 mask**。指令部分也算 loss，模型学成"生成指令"。要用 `-100` 或 mask 掉指令 token。

**坑 3：学习率太大**。用预训练的学习率（$1e-4$），破坏预训练知识，MMLU 暴跌。SFT 用 $1e-5$ 到 $5e-5$。

**坑 4：epoch 太多**。训 10+ epoch 过拟合，模型记住训练数据，泛化崩盘。2-3 epoch 够。

**坑 5：数据量追求大**。10 万条低质量数据不如 1 万条高质量。LIMA 证明质量 >> 数量。

**坑 6：模板不一致**。训练用 ChatML，推理用 Llama Chat，模板不匹配效果差。训练和推理模板必须一致。

**坑 7：多轮对话只训最后一轮**。多轮数据只对最后一轮 assistant 算 loss，浪费中间轮次的监督信号。每轮 assistant 都要算 loss。

**坑 8：忘了防遗忘**。SFT 后通用能力掉点。要混合预训练数据或用 LoRA。

**坑 9：数据类型单一**。全是问答类，模型不会写代码/数学。数据要覆盖多种任务类型。

**坑 10：评估只看 loss**。loss 降不代表对话能力好。要用人工评估 + benchmark。

**坑 11：GPT-4 蒸馏数据风格化**。蒸馏数据全是 GPT-4 的"我很高兴帮你..."风格，模型学成 GPT-4 复读机。要多样化回答风格。

**坑 12：SFT 后不做 RLHF/DPO**。SFT 是基础对齐，细粒度偏好还要 RLHF/DPO。纯 SFT 模型在偏好对齐上弱。

### 5.3 面试怎么考

1. **SFT 的损失函数？** 答：对回答部分做 next-token prediction（交叉熵），指令部分 mask 掉不算 loss。
2. **SFT 在 RLHF 中的位置？** 答：第一阶段，先 SFT 让模型学会对话，再 RM + PPO 做偏好对齐。
3. **SFT 学习率为什么比预训练小？** 答：防止破坏预训练知识（灾难性遗忘），预训练 $1e-4$，SFT $1e-5$。
4. **LIMA 的发现？** 答：1000 条高质量数据 SFT 就能接近 GPT-4，质量 >> 数量。
5. **SFT 的局限？** 答：上限受数据限制、偏好对齐弱、数据偏差传递、泛化有限。

---

## 速记卡

| 阶段 | 数据 | 目标 | Loss |
|------|------|------|------|
| 预训练 | 无监督文本 | 续写 | 全 token 交叉熵 |
| **SFT** | 指令-回答对 | 对话 | 仅回答 token 交叉熵 |
| RLHF | + 偏好对 | 偏好对齐 | PPO 奖励 |
| DPO | + 偏好对 | 偏好对齐 | 偏好对分类 |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| 学习率 | $1e-5$ ~ $5e-5$ | 防遗忘 |
| Epoch | 2-3 | 欠拟合 vs 过拟合 |
| Batch size | 64-128 | 稳定性 |
| 数据量 | 1K-100K | 质量 > 数量 |
| 模板 | ChatML / Llama Chat | 训练推理一致 |

**数据质量阶梯**：

| 数据 | 质量 | 代表 |
|------|------|------|
| 人工标注 | 最高 | InstructGPT |
| 1K 精选 | 极高 | LIMA |
| GPT-4 蒸馏 | 中高 | Alpaca |
| 真实对话 | 看清洗 | Vicuna/ShareGPT |
| 自动生成 | 参差 | Self-Instruct |

**一句话记忆**：SFT = 指令-回答对 + 仅回答 token 算交叉熵。对齐起点，把续写模型变对话模型。Loss masking 只算回答部分，学习率比预训练小一个数量级防遗忘，2-3 epoch 够。质量 >> 数量（LIMA 1000 条接近 GPT-4）。是 RLHF/DPO 的前置步骤，不做 SFT 直接 RLHF 效果差。

---

> *上一篇：[PEFT 总览与选型](./peft) -- SFT 常配 LoRA 用，PEFT 选型是前置知识。*
> *下一篇：[Reward Model 奖励模型](./reward-model) -- SFT 之后，RLHF 的第二步，学人类偏好。*
