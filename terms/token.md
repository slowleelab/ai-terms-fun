---
title: Token（词元）
slug: token
category: 数据表示与编码
tags: [Token, 词元, 词表, Embedding, 上下文窗口]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# Token（词元）

> **一句话 TL;DR**：Token 是大模型处理文本的最小单元--介于"字"和"词"之间的"子词"。模型不直接读字符，而是把文本切成 token 序列，每个 token 对应一个 ID 和一个 [embedding](./embedding)。理解 token 是理解大模型"看到什么"的基础。

---

## L1 · 一句话点破

Token：**大模型处理文本的最小单元。一个 token 通常对应一个子词（如 "play"、"ing"、","），有时是单字，有时是整词。**

```
"playing" -> ["play", "ing"]   # 2 个 token
"我喜欢" -> ["我", "喜欢"]      # 中文，2 个 token（好 tokenizer）
"unhappiness" -> ["un", "happiness"]  # 罕见词拆子词
```

每个 token 在词表里有唯一 ID。模型实际处理的是 ID 序列，每个 ID 查表得到一个 [embedding](./embedding) 向量。

Token 是文本和模型之间的"原子"。所有大模型操作（生成、推理、计算 [上下文窗口](./context-window)）都以 token 为单位。

## L2 · 通俗类比

读一本外语书：

- **字本位读法**：一个字一个字读。慢，但每个字都认识
- **词本位读法**：一个词一个词读。快，但罕见词不认识
- **token 读法**：常见词整体读，罕见词拆成熟悉的部分读。如"unhappiness"读成"un + happiness"

大模型的 token 就是"子词"。它是"字"和"词"之间的折中：

- 不像字那么碎（每个字一个 token，序列太长）
- 不像词那么粗（每个词一个 token，词表爆炸）
- 高频部分整体，低频部分拆解

直觉：token 是"模型觉得自然的文本单元"。这个"自然"由 [tokenizer](./tokenizer) 训练时学出来，基于语料统计。

Token 的几个关键属性：

- **长度不固定**：英文 1 token ≈ 0.75 词，中文 1 token ≈ 0.5-2 字（看 tokenizer）
- **依赖 tokenizer**：同一文本，不同 tokenizer 切出不同 token 序列
- **决定上下文**：模型上下文窗口以 token 计（如 32K token）

## L3 · 正经定义

**Token**：大模型处理文本的最小单元，由 [tokenizer](./tokenizer) 把文本切分得到。

每个 token 在词表 $V$ 中有唯一 ID $v \in \{1, 2, ..., |V|\}$。模型输入是 token ID 序列 $[v_1, v_2, ..., v_T]$，每个 ID 通过 embedding 矩阵 $E \in \mathbb{R}^{|V| \times d}$ 查表得到向量：

$$
e_t = E[v_t]
$$

模型实际处理的是向量序列 $[e_1, e_2, ..., e_T]$。

**Token 的层级**：

| 层级 | 单元 | 例子 |
|------|------|------|
| 字符 | 单个字 | 'p', 'l', 'a', 'y' |
| **子词（token）** | 部分词 | "play", "ing" |
| 词 | 完整词 | "playing" |
| 句 | 句子 | "I am playing." |

**Token 的统计**：

- 英文：1 token ≈ 4 字符 ≈ 0.75 词
- 中文：1 token ≈ 0.5-2 字（看 tokenizer）
- 代码：1 token ≈ 3-5 字符
- 1 页英文 ≈ 500 词 ≈ 700 token

**参考资料**：
- [Sennrich et al., 2015 - BPE](https://arxiv.org/abs/1508.07909)
- [OpenAI - Tokenizer](https://platform.openai.com/tokenizer)
- [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course/chapter6)

## L4 · 原理深挖

### 4.1 Token 是如何被模型使用的

完整流程：

```
1. 输入文本: "I love LLMs"
2. Tokenize: ["I", " love", " L", "LMs"]
3. 查 ID: [40, 3012, 443, 7654]
4. 查 Embedding: [E[40], E[3012], E[443], E[7654]]  # 每个是 d 维向量
5. 模型前向: 处理向量序列
6. 输出 logits: 在词表上的概率分布
7. 采样下一个 token ID
8. 拼到序列末尾，回到 5
```

Token 是模型世界的"原子"。模型不理解"字符"，只理解 token ID 及其 embedding。

### 4.2 Token ID 和 Embedding 的关系

每个 token ID 对应 embedding 矩阵的一行：

$$
E = \begin{bmatrix} \text{--- } e_1 \text{ ---} \\ \text{--- } e_2 \text{ ---} \\ \vdots \\ \text{--- } e_{|V|} \text{ ---} \end{bmatrix}
$$

- $E$ 是模型参数，训练时学习
- 每行 $e_i$ 是 token $i$ 的"语义向量"
- 相似 token 的 embedding 在向量空间中接近

Embedding 矩阵规模：$|V| \times d$。如 LLaMA-70B 词表 32K、维度 8192，embedding 矩阵 32K × 8192 = ~1 亿参数。LLaMA-3 词表 128K，embedding 矩阵更大。

见 [Embedding](./embedding) 词条详述。

### 4.3 Token 长度的实际影响

Token 长度直接影响：

**① 上下文窗口利用率**

模型 [上下文窗口](./context-window) 是 token 数（如 32K）。同样内容，token 多的占用多。

中文在 LLaMA-2（tokenizer 差）："我爱机器学习" 可能 10+ token，32K 上下文只能装几千中文字。LLaMA-3（优化中文）："我爱机器学习" 约 4 token，32K 装更多。

**② 推理/训练成本**

成本按 token 计费（API）或按 token 计算量（自部署）。token 多 = 成本高。

中文场景用优化 tokenizer 的模型（如 Qwen2），成本可降数倍。

**③ 序列长度限制**

模型最大序列长度 = 上下文窗口 = token 数。长文档处理受限于 tokenizer 效率。

### 4.4 Token 的特殊类型

**① 特殊 token**

模型有专门的特殊 token：

- `<bos>` / `<s>`: 序列开始
- `<eos>` / `</s>`: 序列结束
- `<pad>`: 填充
- `<unk>`: 未知 token（子词方法后罕见）

这些 token 也有 embedding，参与计算。

**② 数字 token**

数字 tokenization 影响数学能力。如：

- `"12345"` 整体 1 token：模型把 12345 当作"一个整体"
- `"12" + "345"` 2 token：模型看到部分数字
- `"1" + "2" + "3" + "4" + "5"` 5 token：最碎

不同 tokenizer 对数字处理不同。GPT-4 优化了数字 tokenization（每 3 位一组），数学能力强。

**③ 代码 token**

代码 tokenization 影响代码能力。如缩进、标点是否单独 token：

```
"  if x:" 好的 tokenizer: ["  ", "if", " x", ":"]
差的 tokenizer: [" ", " ", "i", "f", " x", ":"]
```

代码 tokenizer 好的模型，代码能力通常更强。

### 4.5 Token 数的计算

实务中估算 token 数的简单规则：

| 语言 | 规则 |
|------|------|
| 英文 | 1 token ≈ 4 字符 ≈ 0.75 词 |
| 中文（好 tokenizer） | 1 token ≈ 1-2 字 |
| 中文（差 tokenizer） | 1 token ≈ 0.5 字（即 1 字 ≈ 2 token） |
| 代码 | 1 token ≈ 3-5 字符 |

经验估算：

- 1 页英文 ≈ 500 词 ≈ 700 token
- 1 页中文 ≈ 800 字 ≈ 500-1500 token（看 tokenizer）
- 1 小时会议录音转写 ≈ 8K-15K token

实际 token 数用 tokenizer.encode() 精确计算。

### 4.6 Token 与定价

API 定价以 token 为单位：

- GPT-4 (2024): 输入 $30 / 1M token，输出 $60 / 1M token
- Claude 3.5 Sonnet: 输入 $3 / 1M token，输出 $15 / 1M token
- LLaMA 3.1 70B (Together AI): 输入 $0.88 / 1M token

理解 token 数 = 理解成本。优化 prompt（用更少 token 表达同样意思）= 降本。

中文场景用 tokenizer 好的模型，相同内容 token 少，成本自然低。

### 4.7 Token 的边界与争议

**① Token 不是语义单元**

token 是统计单元，不一定是语义单元。如 "New" + " York" 是 2 token，但语义上是 1 实体。模型需要学会跨 token 理解语义。

**② Token 边界影响模型行为**

某些任务对 token 边界敏感。如"找所有以 'ing' 结尾的词"--token 边界不同，结果不同。

**③ Tokenization 的不连续性**

微小文本变化可能导致 token 数大变。如加一个空格、改一个标点，token 序列可能完全不同。这让 tokenization 对抗攻击成为可能（如 prompt injection 的变种）。

## L5 · 沿革与坑

### 沿革

- **1990s-2010s**：NLP 主流是 word-level 或 char-level，受 OOV 困扰。
- **2015-2016**：[BPE](https://arxiv.org/abs/1508.07909) 引入子词，token 概念成型。
- **2018-2020**：BERT、GPT-2 等用子词 token，词表 30K-50K。
- **2020-2023**：GPT-3/4 等大模型，token 成为定价和上下文的基本单位。tiktoken 开源。
- **2024**：LLaMA-3 词表扩到 128K，中文 token 优化。Qwen2 等中文模型 token 效率显著提升。
- **2025**：多模态 token（图像 token、音频 token）成为新方向。Token 概念从文本扩展到通用模态。

### 常见误解

- ❌ **误解**：token = 词。
  ✅ **真相**：token 是子词，介于字和词之间。"playing" 可能是 2 token（"play" + "ing"）。token 不等于词（L1、L3）。

- ❌ **误解**：所有模型 token 切分一样。
  ✅ **真相**：每个模型有自己的 tokenizer，token 切分不同。同样文本，GPT-4 和 LLaMA-3 的 token 数可能差 2 倍（4.3）。

- ❌ **误解**：1 个汉字 = 1 个 token。
  ✅ **真相**：不一定。差 tokenizer（如 LLaMA-2）1 字可能 2-3 token；好 tokenizer（如 Qwen2）1-2 字 1 token（4.3）。

- ❌ **误解**：token 数和字符数成正比。
  ✅ **真相**：大致相关但不严格。微小文本变化可能让 token 数大变（4.7）。

- ❌ **误解**：英文和中文 token 效率一样。
  ✅ **真相**：英文通常更高效（1 词约 1-2 token）。中文在差 tokenizer 上低效，新模型才接近（4.3）。

- ❌ **误解**：token 是模型的"理解单元"。
  ✅ **真相**：token 是统计单元，不是语义单元。模型需要学会跨 token 组合语义（4.7）。

### 面试怎么考

1. **"什么是 token？"** --大模型处理文本的最小单元，介于字和词之间的子词。模型实际处理 token ID 序列（L1、L3）。
2. **"token 和词的区别？"** --token 是子词，不等于词。"playing" 可能 2 token。token 由 tokenizer 决定（L1）。
3. **"token 数怎么估算？"** --英文 1 token ≈ 4 字符 ≈ 0.75 词；中文看 tokenizer，1 字 0.5-2 token（4.5）。
4. **"为什么中文 token 效率重要？"** --影响上下文利用率、推理成本、序列长度。LLaMA-3、Qwen2 优化中文 token（4.3）。
5. **"token 和 embedding 的关系？"** --每个 token ID 对应 embedding 矩阵的一行。模型通过查表得到 token 的向量表示（4.2）。

## 延伸阅读

- 📄 [Sennrich et al., 2015 - BPE](https://arxiv.org/abs/1508.07909)
- 📝 [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- 📝 [HuggingFace NLP Course - Tokenizers](https://huggingface.co/learn/nlp-course/chapter6)
- 📝 [tiktoken](https://github.com/openai/tiktoken)

---

> *上一篇：[分词器 Tokenizer](./tokenizer) -- 文本怎么变成模型能吃的数字。*
> *下一篇：[分块 Chunking](./chunking) -- 长文本怎么切成段。*
