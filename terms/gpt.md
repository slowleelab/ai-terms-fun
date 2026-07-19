---
title: GPT / LLaMA（仅解码器）
slug: gpt
category: 模型架构与训练
tags: [GPT, LLaMA, 解码器, 大模型, 自回归]
author: ai-terms-fun
created: 2026-07-19
updated: 2026-07-19
---

# GPT / LLaMA（仅解码器）

> **一句话 TL;DR**：只用 [Transformer](./transformer) 的解码器，靠"预测下一个 token"这一个目标训练，就能在足够大时涌现出通用能力。GPT 系列证明了这条路，LLaMA 系列把它开源化、工程化，成为当今所有主流大模型（Claude、Qwen、Mistral、DeepSeek）的共同架构。

---

## L1 · 一句话点破

GPT/LLaMA 的本质是：**纯解码器 Transformer，训练目标只有一个--给定前文预测下一个 token。** 看似简单到无聊的任务，在足够大的数据和参数下，涌现出了理解、推理、对话、写代码等通用能力。

它和 [BERT](./bert) 的根本区别：BERT 是"完形填空"（理解），GPT 是"接龙"（生成）。前者把模型锁在理解任务上，后者因为生成目标的统一性，在 scaling law 下成为大模型时代的主流。

## L2 · 通俗类比

想象训练一个"接龙大师"：

- 你给他海量文本，让他反复做"给定前半句，猜下一个字"的练习。这个任务简单到小孩都能玩，不需要任何人工标注。
- 一开始他只会接简单的字（"今天天气" -> "好"）。
- 练到一定程度，他开始能接复杂的逻辑（"如果 A 大于 B，且 B 大于 C，那么 A" -> "大于"）。
- 练到海量规模，他不仅会接龙，还能听懂你的问题、按格式回答、写代码、做翻译--因为你给他的问题本身就是"接龙的题目"，他只要接着写下去就行。

关键洞察：**所有任务都可以被表达成"续写"**。

- 翻译 = "把下面英文翻成中文：..." 然后续写中文
- 问答 = "问：... 答：" 然后续写答案
- 写代码 = "实现这个功能：" 然后续写代码

这就是 GPT 的统一性优势：一个训练目标（预测下一个 token），搞定所有任务。代价是早期需要 prompt engineering 把任务包装成续写，后来演化成指令微调 + RLHF 让模型学会"被问问题时怎么回答"。

## L3 · 正经定义

**GPT（Generative Pre-trained Transformer）** 是 OpenAI 自 2018 年起发布的生成式预训练语言模型系列（[Radford et al. 2018](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)），采用纯 Transformer 解码器架构。**LLaMA** 是 Meta 于 2023 年发布的开源大模型系列（[Touvron et al. 2023](https://arxiv.org/abs/2302.13971)），架构与 GPT 同属仅解码器，但在工程上做了多项优化。

两者的核心架构要素：

- **仅解码器 Transformer 堆叠**：去掉原版的编码器和交叉注意力，只保留掩码自注意力 + FFN。
- **自回归语言模型预训练**：目标函数是负对数似然 $\mathcal{L} = -\sum_t \log P(x_t | x_{<t})$，即"预测下一个 token"。
- **因果掩码**：自注意力中未来位置被 $-\infty$ 屏蔽，保证生成时只能看前文。

训练流程（现代版）：

1. **预训练**：海量无标注文本，自回归目标，占训练计算量的 99%。
2. **指令微调（SFT）**：少量高质量的"指令-回答"对，让模型学会听指令。
3. **RLHF / DPO**：用人类反馈做强化学习，让回答更符合偏好（详见 [RLHF](./rlhf)）。

关键规模数据：GPT-3（2020）175B 参数，LLaMA-2 70B，LLaMA-3 405B，GPT-4 参数量未公开但据传超 1T。

**参考资料**：
- [Radford et al., 2018 - GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Brown et al., 2020 - GPT-3](https://arxiv.org/abs/2005.14165)
- [Touvron et al., 2023 - LLaMA](https://arxiv.org/abs/2302.13971)
- [Touvron et al., 2023 - LLaMA-2](https://arxiv.org/abs/2307.09288)

## L4 · 原理深挖

### 4.1 "预测下一个 token"为什么这么强大

这个目标看似简单，但有几个被低估的特性：

**① 任务统一性**

任何 NLP 任务都能表达成续写。这消除了"每个任务一个架构、一个目标"的工程负担。BERT 要为分类、NER、QA 设计不同头，GPT 全用同一个头（输出层 + softmax）。

**② 数据无限性**

互联网上有海量文本，都是天然的"前文-下一个 token"对。无需任何标注，预训练数据几乎是免费的（相对而言）。这让 scaling law 成为可能--数据不是瓶颈（至少早期不是）。

**③ 压缩即智能**

预测下一个 token 本质是在压缩文本：模型必须理解文本的规律才能预测准。[Ilya Sutskever](https://www.youtube.com/watch?v=Ft4ZjA2UNPE) 多次强调，足够好的压缩等价于理解。一个能完美预测下一个 token 的模型，必然理解了语法、语义、世界知识、推理逻辑。

**④ 涌现能力**

在某个规模（约 10B 参数）之上，模型开始展现出训练数据里没有显式教的能力：少样本学习、思维链推理、指令遵循。这些能力的涌现机制至今仍是研究热点，但它们的载体就是这个看似简单的目标。

### 4.2 LLaMA 相比 GPT 的工程改进

LLaMA 把 GPT 的架构开源化，并在多个细节上优化，成为开源大模型的事实基座：

| 改进点 | 原版 GPT | LLaMA | 作用 |
|--------|----------|-------|------|
| 位置编码 | 可学习绝对位置 | RoPE | 支持长上下文外推 |
| 归一化 | LayerNorm (Post-LN) | RMSNorm (Pre-LN) | 训练更稳定，计算更省 |
| 激活函数 | GeLU | SwiGLU | 效果略好，是 LLaMA-2 起的标准 |
| 注意力 | 标准 MHA | GQA（LLaMA-2 70B 起） | 推理 KV cache 更省 |
| 词表 | BPE 50K | SentencePiece 32K-128K | 多语言支持更好 |

其中 SwiGLU 和 RMSNorm 是现代大模型的标志性改动：

**SwiGLU**（[Shazeer 2020](https://arxiv.org/abs/2002.05202)）：把 FFN 的 ReLU/GeLU 激活换成带门控的 GLU 变体：

$$
\text{SwiGLU}(x) = \text{Swish}(x W_1) \otimes (x W_2)
$$

效果略好于 GeLU，代价是多一个投影矩阵（参数增加 50%）。LLaMA、PaLM、Qwen 等都用它。

**RMSNorm**（[Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)）：LayerNorm 的简化版，去掉减均值那步，只做缩放归一化。计算量少 10-20%，效果相当。所有现代大模型都用它替代 LayerNorm。

### 4.3 自回归生成的代价

"预测下一个 token"的统一性是优势，但也带来代价：

**① 推理效率低**

自回归生成必须逐 token 串行：生成第 $t$ 个 token 要等前 $t-1$ 个都生成完。无法像 BERT 那样一次处理整个序列。这是为什么推理引擎（vLLM、TensorRT-LLM）和 KV cache 优化如此重要（见 [推理引擎](./inference-engine)）。

**② 训练-推理不对称**

训练时整个序列并行计算（teacher forcing），推理时逐 token 生成。这种不对称让推理优化复杂化。

**③ 错误累积**

生成时一旦某个 token 错了，后续 token 基于错误前文继续生成，错误会累积。这是 [幻觉](./hallucination) 的机制之一。

**④ 单向理解**

处理输入 prompt 时也是从左到右看，不如 BERT 的双向看得全。大模型靠规模弥补了这个差距，但理论上对纯理解任务，BERT 路线更高效。

### 4.4 从 GPT-3 到现代大模型的演化

GPT-3（2020）证明了"大模型 + 预测下一个 token"能涌现出 in-context learning，但原始 GPT-3 其实很难用--输出格式乱、不遵循指令、容易胡说。

现代大模型的飞跃来自训练流程的完善，而非架构革命：

- **指令微调（SFT）**：用高质量"指令-回答"数据，让模型学会"被问问题该怎么答"。GPT-3.5 起的标准操作。
- **RLHF**：用人类偏好数据做强化学习，对齐人类价值观。ChatGPT 的核心创新。
- **DPO**（[Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)）：RLHF 的简化版，无需训练 reward model，直接用偏好对优化。当前开源模型主流。
- **工具使用 / Agent**：让模型学会调用外部工具（搜索、代码执行），突破纯文本生成的限制。

架构本身自 2020 年来没有革命性变化，但训练数据质量、训练流程、对齐技术的进步让同样架构的能力天差地别。

## L5 · 沿革与坑

### 沿革

- **2018 年 6 月**：OpenAI 发表 GPT-1，117M 参数。和 BERT 几乎同时发布，用仅解码器 + 自回归预训练。反响不如 BERT。
- **2019 年 2 月**：GPT-2，1.5B 参数。展示了零样本生成的惊人能力，OpenAI 因"太危险"分阶段发布，引发 AI 安全讨论。
- **2020 年 5 月**：GPT-3，175B 参数。展示了 few-shot in-context learning，证明 scaling 能涌现新能力。这是大模型时代的真正起点。
- **2022 年 11 月**：ChatGPT 发布。GPT-3.5 + RLHF，对话能力引爆全球。大模型从学术圈走向大众。
- **2023 年 2 月**：Meta 发布 LLaMA（7B/13B/33B/65B），开源。性能接近 GPT-3，但参数小得多，可在单卡运行。引爆开源大模型生态。
- **2023 年 7 月**：LLaMA-2，商用许可。引入 GQA，效果大幅提升。
- **2024 年**：LLaMA-3（8B/70B/405B）、Qwen2、Mistral、DeepSeek-V3 等开源模型百花齐放。架构趋同（仅解码器 + RoPE + SwiGLU + RMSNorm + GQA），差异主要在数据和训练。
- **2024-2025 年**：MoE（混合专家）成为前沿方向，DeepSeek-V3、Mixtral 等用稀疏激活在同等效果下大幅降低推理成本。

### 常见误解

- ❌ **误解**：GPT 比架构比 BERT 先进，所以赢了。
  ✅ **真相**：架构没有先进落后之分。GPT 胜出是因为"统一训练目标 + scaling law 友好 + in-context learning 涌现"的工程组合优势。在纯理解任务上，BERT 路线同等规模下其实更高效。

- ❌ **误解**：GPT 的"预测下一个 token"只是统计模式匹配，没有真正理解。
  ✅ **真相**：这是长期争议。但 Ilya 等人的观点有道理：能完美预测下一个 token 的模型，必然压缩了文本中的规律，包括语法、语义、世界知识、推理逻辑。"压缩即理解"不是修辞，是有信息论支撑的论断。模型是否"真正"理解取决于对"理解"的定义，但它的能力超越了简单模式匹配。

- ❌ **误解**：LLaMA 是 GPT 的开源复刻。
  ✅ **真相**：LLaMA 架构和 GPT 同属仅解码器，但工程细节（RoPE、RMSNorm、SwiGLU、GQA）都和原版 GPT 不同。LLaMA 是独立优化的架构，不是复刻。

- ❌ **误解**：参数量越大模型越强。
  ✅ **真相**：scaling law 说的是"算力、数据、参数三者协同增长"。只堆参数不堆数据，效果会饱和。Chinchilla 法则（[Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)）表明，给定算力预算，数据量和参数量应大致 1:1 增长（每个参数约 20 个 token）。LLaMA 就是数据驱动的典型--比 GPT-3 参数少很多，但用更多数据训练，效果更好。

- ❌ **误解**：现代大模型的进步主要靠架构创新。
  ✅ **真相**：架构自 2020 年来变化不大（仅解码器 + RoPE + SwiGLU + RMSNorm + GQA 已经是标准）。真正的进步来自：训练数据质量、数据配比、对齐技术（SFT/RLHF/DPO）、训练流程优化。架构是地基，但地基之上盖什么楼，数据和对齐决定。

### 面试怎么考

1. **"GPT 的训练目标是什么？为什么这么简单却这么强大？"** --预测下一个 token。任务统一性 + 数据无限性 + 压缩即智能 + 涌现能力（见 4.1）。
2. **"GPT 和 BERT 的核心区别？"** --架构（解码器 vs 编码器）+ 目标（自回归 vs MLM）+ 方向（单向 vs 双向）+ 适合任务（生成 vs 理解）。
3. **"LLaMA 相比 GPT 做了哪些改进？"** --RoPE、RMSNorm、SwiGLU、GQA（见 4.2 表格）。
4. **"为什么大模型都用仅解码器，而不是编码器-解码器？"** --见 [编码器-解码器](./encoder-decoder) L4.4，统一训练目标 + scaling law + in-context learning + 工程简洁。
5. **"什么是 Chinchilla 法则？它对训练意味着什么？"** --算力预算下数据和参数应 1:1 增长（每参数约 20 token）。意味着堆参数不堆数据是低效的。
6. **"自回归生成的代价是什么？"** --推理串行低效 + 训练推理不对称 + 错误累积 + 单向理解（见 4.3）。

## 延伸阅读

- 📄 [Brown et al., 2020 - GPT-3](https://arxiv.org/abs/2005.14165)
- 📄 [Touvron et al., 2023 - LLaMA](https://arxiv.org/abs/2302.13971)
- 📄 [Touvron et al., 2023 - LLaMA-2](https://arxiv.org/abs/2307.09288)
- 📄 [Kaplan et al., 2020 - Scaling Laws](https://arxiv.org/abs/2001.08361)
- 📄 [Hoffmann et al., 2022 - Chinchilla](https://arxiv.org/abs/2203.15556)
- 📄 [Shazeer, 2020 - GLU Variants](https://arxiv.org/abs/2002.05202) - SwiGLU 来源
- 📝 [LLaMA 解读 - 大模型架构演化](https://github.com/meta-llama/llama)

---

> *上一篇：[BERT](./bert) -- 仅编码器路线。*
> *下一篇：[传统模型：CNN / RNN / LSTM](./cnn-rnn-lstm) -- Transformer 之前的世界。*
