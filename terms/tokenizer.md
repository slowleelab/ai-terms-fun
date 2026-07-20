---
title: Tokenizer（分词器）
slug: tokenizer
category: 数据表示与编码
tags: [Tokenizer, BPE, WordPiece, SentencePiece, Tiktoken, 子词]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# Tokenizer（分词器）

> **一句话 TL;DR**：Tokenizer 是把文本切成模型能处理的最小单元（token）的工具。它是大模型与人类语言之间的"翻译层"--文本进，token 序列出；token 序列进，文本出。主流方法是 BPE（GPT 系列、LLaMA）和 WordPiece（BERT）。Tokenizer 的设计直接影响模型对中文、代码、罕见词的处理能力，是常被低估却极其关键的一环。

---

## L1 · 一句话点破

Tokenizer：**把字符串切分成 token 序列，每个 token 对应词表中的一个 ID。**

```
文本: "我喜欢大模型"
Tokenize: ["我", "喜欢", "大", "模型"]  # 中文按字/词
IDs: [251, 8927, 3456, 10245]
```

```
文本: "I love LLMs"
Tokenize: ["I", " love", " L", "LMs"]  # 英文按子词
IDs: [40, 3012, 443, 7654]
```

模型不直接处理文本，只处理 token ID 序列。Tokenizer 是文本和模型之间的唯一接口。

为什么需要 tokenize？因为模型只能处理数字（向量）。要把"文本"变成"数字序列"，需要一个确定性的映射。Tokenizer 就是这个映射。

## L2 · 通俗类比

把一本书读给一个只会读"词卡"的孩子：

- **字本位**：每个汉字一张卡。"我喜欢大模型" = "我/喜/欢/大/模/型" 6 张卡。
- **词本位**：常用词一张卡。"我喜欢大模型" = "我/喜欢/大/模型" 4 张卡。
- **子词本位**：常用词一张卡，罕见词拆成更小的"子词"。"unhappiness" = "un + happiness" 或 "un + happy + ness"。

每种方式权衡：

- **字本位**：词表小（中文几千字），但每个字信息量低，序列长
- **词本位**：序列短，但词表巨大（每个词一张卡），罕见词处理不了
- **子词本位**：折中。常用词整体一张卡，罕见词拆成常见子词

大模型几乎都用**子词本位**（subword），兼顾效率和泛化。具体算法有 BPE、WordPiece、Unigram 等。

为什么 tokenize 这么重要？因为它决定：

- **模型看到的"基本单元"**：同一文本，不同 tokenizer 切出不同序列
- **序列长度**：影响 [上下文窗口](./context-window) 利用率
- **多语言能力**：中文 tokenizer 不好的模型，中文又慢又差
- **代码能力**：代码 tokenizer 不好的模型，代码处理差

## L3 · 正经定义

**Tokenizer**：文本与 token ID 序列之间的双向映射函数。

$$
\text{encode}: \text{字符串} \to \text{token ID 序列}
$$
$$
\text{decode}: \text{token ID 序列} \to \text{字符串}
$$

**主流算法**：

| 算法 | 思路 | 代表 |
|------|------|------|
| **BPE** | 从字符开始，合并最高频对 | GPT-2/3/4, LLaMA |
| **WordPiece** | 类似 BPE，但用似然选合并 | BERT, DistilBERT |
| **Unigram** | 从大词表反向删减 | T5, mBART, XLNet |
| **SentencePiece** | 不依赖空格的框架（含 BPE/Unigram） | LLaMA, T5 |
| **Byte-level** | 以字节为原子，覆盖所有字符 | GPT-2/3/4 (tiktoken) |

**关键概念**：

- **词表（vocabulary）**：所有 token 的集合，大小通常 30K-200K
- **特殊 token**：`<pad>`, `<bos>`, `<eos>`, `<unk>` 等
- **OOV（Out-of-Vocabulary）**：罕见词的处理，子词方法基本解决

**主流模型的 tokenizer**：

| 模型 | 算法 | 词表大小 | 中文效率 |
|------|------|---------|---------|
| GPT-2 | BPE | 50K | 差（1 字多 token） |
| GPT-4 | BPE (tiktoken) | 100K | 中等 |
| LLaMA | SentencePiece BPE | 32K | 差 |
| LLaMA-3 | tiktoken-like | 128K | 好 |
| Qwen2 | BPE | 152K | 极好（中文优化） |
| BERT | WordPiece | 30K | 中等 |

**参考资料**：
- [Sennrich et al., 2015 - BPE](https://arxiv.org/abs/1508.07909) - BPE 用于 NMT
- [Schuster & Nakajima, 2012 - WordPiece](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)
- [Kudo & Richardson, 2018 - SentencePiece](https://arxiv.org/abs/1804.10959)
- [OpenAI - tiktoken](https://github.com/openai/tiktoken)
- [Mielke et al., 2021 - Word Segmentation Survey](https://aclanthology.org/2021.acl-tutorials.3/)

## L4 · 原理深挖

### 4.1 BPE：从字符到子词的贪心合并

Byte-Pair Encoding（BPE）是最流行的子词算法。

**训练过程**：

```
1. 初始化：每个字符是一个 token
   "low" -> ['l', 'o', 'w']
   "lower" -> ['l', 'o', 'w', 'e', 'r']

2. 统计相邻 token 对的频率，合并最高频对
   "l" + "o" 出现 5 次 -> 合并为 "lo"
   现在："low" -> ['lo', 'w']

3. 重复合并，直到达到目标词表大小
```

每步合并最高频的相邻对，逐渐构建子词词表。结果：

- 高频词整体成为 token（如 "the", "ing"）
- 罕见词被拆成常见子词（如 "unhappiness" -> "un" + "happiness"）

BPE 的优势：

- **无 OOV**：任何词都能被拆成已知子词（最差拆成字符）
- **词表可控**：合并次数决定词表大小
- **跨语言**：在多语言数据上训练，自动学多语言子词

### 4.2 WordPiece：BPE 的似然变体

WordPiece（[Schuster & Nakajima, 2012](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)）和 BPE 思路类似，但合并准则不同：

- BPE：合并频率最高的对
- WordPiece：合并使语言模型似然提升最大的对

$$
\text{score}(x, y) = \frac{P(xy)}{P(x) P(y)}
$$

选 score 最大的对合并。这种"似然准则"让 WordPiece 倾向合并"语义相关"的子词，而非纯频率高的。

BERT 系列用 WordPiece。

### 4.3 Unigram：反向删减

Unigram Language Model（[Kudo, 2018](https://arxiv.org/abs/1804.10959)）思路相反：

1. 初始化：超大候选词表（如百万级）
2. 用 EM 算法估计每个子词的概率
3. 删除使总似然损失最小的子词
4. 重复直到目标词表大小

Unigram 的优势：每个词有多种分词方式，按概率选。如 "unhappiness" 可分为 "un + happiness" 或 "unh + appiness"，按概率选最优。

T5、mBART 用 Unigram。

### 4.4 SentencePiece：跨语言友好

[SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1804.10959) 不是一个算法，而是一个框架，特点：

- **不依赖空格分词**：对中文、日文等无空格语言友好
- **直接处理 raw text**：不需预分词
- **支持 BPE 和 Unigram**：两种算法都可用
- **可逆**：decode 一定能还原原文本

为什么需要不依赖空格？英语用空格分词天然，但中文、日文、泰文没有空格。传统做法是先用分词工具（如 jieba）分词，再 tokenize。SentencePiece 直接在 raw text 上学，避免分词工具的偏见。

LLaMA、T5 等用 SentencePiece。

### 4.5 Byte-level BPE：终极覆盖

GPT-2/3/4 用的 BPE 有个特殊设计：**以字节为原子**，而非字符。

为什么？字符层面有"未登录字符"问题（如某些 Unicode 字符不在词表里）。字节层面覆盖所有可能输入（任何字符串都是字节序列），无 OOV。

代价：词表略大（需要更多 token 覆盖字节组合），但彻底解决编码问题。

[tiktoken](https://github.com/openai/tiktoken) 是 OpenAI 的 byte-level BPE 实现，GPT-4 用。LLaMA-3 也转向类似设计，词表扩到 128K。

### 4.6 中文 tokenizer 的痛点

中文 tokenizer 是大模型的痛点：

**① 中文效率低**

英文 "I love you" 通常 3 token。中文"我爱你"在 GPT-2 是 6+ token（每个字 2-3 token，因为字节级 BPE）。这导致：

- 中文 [上下文窗口](./context-window) 利用率低（同样长度装更少内容）
- 中文推理慢（token 多，前向多）
- 中文训练成本高

**② 中文优化**

LLaMA-3、Qwen2 等新模型大幅优化中文 tokenizer：

- 词表扩到 128K-152K
- 加入中文常用字、词作为单独 token
- 中文 1-2 字符 1 token，效率接近英文

Qwen2 在中文上效率比 LLaMA-2 高 3-5 倍，是中文场景首选之一。

**③ 分词歧义**

中文分词本身有歧义（"结婚的和尚未结婚的"）。子词方法用 BPE 合并，避免传统分词歧义，但可能产生"奇怪"的子词。

### 4.7 Tokenizer 对模型的影响

Tokenizer 不只是预处理工具，它直接定义模型看到的世界：

**① 影响能力**

- 数字 tokenization：`"12345"` 可能是 1 token（"12345"）或多个（"12" + "345"），影响数学能力
- 代码 tokenization：缩进、标点是否单独 token，影响代码能力
- 罕见词：能否整体 token，影响专有名词处理

**② 影响效率**

- 序列长度直接决定推理/训练成本
- 中文 tokenizer 差的模型，相同内容 token 多，成本高数倍

**③ 影响一致性**

- Tokenizer 改了，模型必须重训（token embedding 变了）
- 这就是 LLaMA-2 到 LLaMA-3 换 tokenizer 是大改动

Tokenizer 的选择是模型设计的基础决策，影响深远。

## L5 · 沿革与坑

### 沿革

- **1990s-2010s**：NLP 主流是 word-level 或 char-level tokenize，受 OOV 困扰。
- **2015-2016**：[Sennrich et al. - BPE for NMT](https://arxiv.org/abs/1508.07909) 引入子词，解决罕见词翻译。
- **2016**：Google 提出 WordPiece，用于神经机器翻译。
- **2018**：[SentencePiece](https://arxiv.org/abs/1804.10959) 发布，跨语言友好。BERT 用 WordPiece，GPT 用 BPE。
- **2019**：GPT-2 用 byte-level BPE，彻底解决 OOV。
- **2020-2022**：GPT-3、LLaMA 等大模型普遍用 BPE/SentencePiece。中文 tokenizer 痛点显现。
- **2023**：OpenAI tiktoken 开源。LLaMA-2 中文效率低被诟病。
- **2024**：LLaMA-3 词表扩到 128K，中文优化。Qwen2 等中文模型 tokenizer 优化显著。
- **2025**：多模态 tokenizer（图文混合 token）成为新方向。

### 常见误解

- ❌ **误解**：tokenizer 就是分词工具（如 jieba）。
  ✅ **真相**：分词工具是传统 NLP 概念（中文切词）。大模型的 tokenizer 是子词切分，目标不同（建词表+切分），算法不同（BPE 等）。两者不应混淆（4.4）。

- ❌ **误解**：所有模型用同一个 tokenizer。
  ✅ **真相**：每个模型有自己的 tokenizer，词表、算法都不同。换 tokenizer 必须重训模型（4.7）。

- ❌ **误解**：英文和中文 tokenization 效率差不多。
  ✅ **真相**：差很多。早期模型（GPT-2、LLaMA）中文效率低数倍。新模型（LLaMA-3、Qwen2）才接近（4.6）。

- ❌ **误解**：BPE 是 GPT 发明的。
  ✅ **真相**：BPE 是 1994 年的压缩算法，[Sennrich et al. 2015](https://arxiv.org/abs/1508.07909) 把它引入 NLP。GPT-2 用的是 byte-level BPE 变体（4.5）。

- ❌ **误解**：词表越大越好。
  ✅ **真相**：词表大覆盖广，但 embedding 参数多、训练慢。需平衡。GPT-4 的 100K、LLaMA-3 的 128K 是当前合理范围。

- ❌ **误解**：tokenizer 改进对模型影响小。
  ✅ **真相**：影响巨大。LLaMA-2 到 LLaMA-3 换 tokenizer（词表 32K->128K，中文优化）是性能提升的关键因素之一。tokenizer 定义模型看到的世界（4.7）。

### 面试怎么考

1. **"什么是 tokenizer？为什么需要？"** --文本和 token ID 序列的双向映射。模型只处理数字，tokenizer 是文本-模型的接口（L1、L3）。
2. **"BPE 算法怎么工作？"** --从字符开始，迭代合并最高频相邻对，构建子词词表。无 OOV（4.1）。
3. **"BPE 和 WordPiece 的区别？"** --BPE 按频率合并，WordPiece 按似然提升合并（4.1、4.2）。
4. **"为什么需要 byte-level BPE？"** --以字节为原子，覆盖所有可能字符，彻底解决 OOV。GPT-2/3/4 用（4.5）。
5. **"中文 tokenizer 有什么痛点？"** --效率低（同内容 token 多）、分词歧义。LLaMA-3、Qwen2 等优化（4.6）。
6. **"为什么换 tokenizer 要重训模型？"** --token embedding 变了，模型必须重新学。LLaMA-2 到 LLaMA-3 换 tokenizer 是大改动（4.7）。

## 延伸阅读

- 📄 [Sennrich et al., 2015 - BPE for NMT](https://arxiv.org/abs/1508.07909)
- 📄 [Kudo & Richardson, 2018 - SentencePiece](https://arxiv.org/abs/1804.10959)
- 📝 [HuggingFace NLP Course - Tokenizers](https://huggingface.co/learn/nlp-course/chapter6)
- 📝 [OpenAI tiktoken](https://github.com/openai/tiktoken)
- 📝 [Mielke et al., 2021 - Word Segmentation Survey](https://aclanthology.org/2021.acl-tutorials.3/)

---

> *上一篇：[推理引擎](./inference-engine) -- 生产部署的加速器。*
> *下一篇：[Token](./token) -- 大模型的最小信息单元。*
