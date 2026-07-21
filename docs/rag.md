---
title: RAG（检索增强生成）
slug: rag
category: 评估与应用
tags: [LLM, 检索, 向量数据库, RAG]
author: ai-terms-fun
created: 2026-07-19
updated: 2026-07-19
---

# RAG（检索增强生成）

> **一句话 TL;DR**：考试不让背书，但允许开卷--RAG 就是给大模型发了张准考证和一本翻到目标页的参考书。

---

## L1 · 一句话点破

RAG = **Retrieval-Augmented Generation** = "检索增强生成"。

一句话本质：**生成回答前，先从外部知识库检索相关片段，塞进 prompt 让模型基于上下文作答。**

是当前 LLM 应用层最被滥用、也最被低估的一个词。说它被滥用，是因为"我做了个 RAG"往往约等于"我调了 LangChain 的 `RetrievalQA.from_chain_type()`"；说它被低估，是因为真把它做好的人，能拉开同行一个身位。

## L2 · 生活类比

想象你雇了一个**记忆力奇差、但理解力惊人**的实习生来回答客户问题。

- 如果不给他资料，他会凭印象瞎编（这就是 LLM 的 hallucination）。
- 如果你把公司所有文档提前塞进他脑子里，培训成本高，而且文档一更新就得重训。
- **RAG 的做法**：给他一张能查公司内部系统的工牌，每次客户提问，他先去查相关文件，读完再回答。

关键词是"**每次**"和"**相关**"--这决定了 RAG 的成败不在生成模型有多强，而在你能不能把"相关文件"这步做对。这步做烂了，等于给实习生发了一堆乱序的碎纸，他读得越认真，编得越自信。

## L3 · 正经定义

**RAG（Retrieval-Augmented Generation）** 是一种将外部知识检索与生成模型结合的方法：在生成回答前，先从一个外部知识库中检索与查询相关的文档片段，再将这些片段作为上下文拼入 prompt，由生成模型基于上下文产出最终回答。

一个标准 RAG 系统包含三个组件：

1. **索引（Indexing）**：把知识库文档切分成 chunk，每个 chunk 用 embedding 模型编码成向量，存入向量数据库。
2. **检索（Retrieval）**：用户 query 同样编码成向量，在向量库中做最近邻搜索，取 top-k 相关 chunk。
3. **生成（Generation）**：把 top-k chunk 作为 context 拼进 prompt，喂给 LLM 生成回答。

该范式由 Lewis 等人在 2020 年的论文 [《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》](https://arxiv.org/abs/2005.11401) 中正式提出，原本用于解决参数化模型（如 T5）的知识固化问题。2023 年后随 ChatGPT 的普及，成为企业落地 LLM 的默认架构。

**参考资料**：
- [Lewis et al., 2020 - RAG 原始论文](https://arxiv.org/abs/2005.11401)
- [Gao et al., 2023 - Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)

## L4 · 原理深挖

### 4.1 为什么是 RAG，而不是微调？

知识更新问题有两种解法：

| 方案 | 更新成本 | 知识溯源 | 适合场景 |
|------|----------|----------|----------|
| 微调（fine-tune） | 高（重训 + 评估） | 不可溯源（融进了权重） | 风格、格式、能力塑形 |
| RAG | 低（删改向量库即可） | 可溯源（能指给用户看来源） | 事实性、时效性知识 |

**结论**：要的是"模型说话像本公司客服"-> 微调；要的是"模型知道公司今天的库存"-> RAG。两者不互斥，生产系统经常组合使用。

### 4.2 检索的本质：向量相似度

把文本编码成稠密向量 $v \in \mathbb{R}^d$（典型 $d=768$ 或 $1536$），用余弦相似度衡量相关性：

$$
\text{sim}(q, d) = \frac{v_q \cdot v_d}{\|v_q\| \, \|v_d\|}
$$

为什么是余弦而不是欧氏距离？因为文本向量的"模长"往往编码了信息量/长度等 nuisance 变量，而"方向"才编码语义--我们关心的是"在说同一件事"，不是"说了多重的同一件事"。

### 4.3 naive RAG 的三个经典翻车点

这是 90% 的"我做了 RAG 但答得不好"的根因：

**① Chunk 切得不对**

按固定字数切（比如每 500 字一刀），会把一个完整的论证腰斩。比如把"我们公司不退款，除非商品在 7 天内损坏"切成了"我们公司不退款"和"除非商品在 7 天内损坏"两段，检索到前者就完蛋。

> 解法：按语义边界切（标题、段落、句号），或用递归切分器（LangChain 的 `RecursiveCharacterTextSplitter`）。

**② Embedding 和 query 的分布不一致**

文档是陈述句（"退款政策是……"），query 是疑问句（"能退款吗？"）。朴素 embedding 下，这两句话的向量不一定近。

> 解法：query rewriting，把用户问题改写成更适合检索的形式（如 HyDE：先让 LLM 假设一个答案，用假答案去检索）。

**③ Top-k 不等于"够用"**

取 top-5，但相关的就 1 条，剩下 4 条是噪声，反而干扰生成。

> 解法：检索后加 reranker（如 bge-reranker、Cohere rerank），用 cross-encoder 精排；或加相关性阈值过滤。

### 4.4 进阶：从 naive RAG 到 modular RAG

Gao 等人 2023 年的综述把 RAG 演进分成三代：

- **Naive RAG**：Index → Retrieve → Generate，一条线走到底。
- **Advanced RAG**：在检索前后加优化（query rewriting、reranking、chunk 优化）。
- **Modular RAG**：检索变成可插拔模块，支持多轮检索、检索-生成交替（如 Flan-T5 + REPLUG）、自反思（如 Self-RAG）。

记住一句话：**RAG 的工程难点，90% 在检索，10% 在生成。** 大模型已经够强了，你的瓶颈永远在"喂给它的东西对不对"。

### 4.5 最小可运行 Demo

参见 [`demos/rag/`](../demos/rag/) -- 用 ~60 行 Python + sentence-transformers + FAISS，从零跑通一个能用的 RAG，不依赖任何云服务。

## L5 · 沿革与坑

### 沿革

- **2020 年**，Facebook AI Research（FAIR）的 Lewis 等人发表原始 RAG 论文。当时的主角还不是 LLM，而是 T5、BART 这类 seq2seq 模型，目标是做开放域问答。当时几乎没人关心。
- **2022 年底 ChatGPT 发布**，所有人突然发现"模型会一本正经胡说八道"，RAG 作为最直接的"外挂知识"方案被翻出来，瞬间起飞。
- 一篇 2020 年的论文，2023 年才被大面积引用--学术圈也有"守得云开见月明"。

### 常见误解

- ❌ **误解**：RAG = 接个向量数据库。
  ✅ **真相**：向量库只是存储层。一个能用的 RAG 系统，检索前的 chunking、embedding 选型、query rewriting，检索后的 reranking、context 压缩，每一步都是工程坑。向量库是最不重要的那一环。

- ❌ **误解**：RAG 能让模型"学会"新知识。
  ✅ **真相**：RAG 只是把知识**临时塞进 context**，模型权重没变。关掉检索，模型还是什么都不知道。把"学知识"和"查资料"混为一谈，是新手最大的概念错误。

- ❌ **误解**：RAG 一定比微调便宜。
  ✅ **真相**：小规模、低频更新时确实便宜；但文档量大到几千万 chunk、且每次检索都要做 rerank 时，硬件成本会反超。

### 面试怎么考

1. **"RAG 和微调什么时候用哪个？"** ——考你对两者定位的理解（见 4.1）。标准答案：知识用 RAG，能力/风格用微调。
2. **"你的 RAG 答得不准，怎么排查？"** ——考工程经验。标准回答顺序：先看检索召回了什么（这步 80% 的问题在这里），再看 context 是否过长被截断，最后看生成模型是否按要求引用。**永远先查检索，再查生成。**
3. **"为什么用余弦相似度而不是欧氏距离？"** ——考 embedding 的基础理解（见 4.2）。
4. **"HyDE 是什么，解决什么问题？"** ——考你是否跟得上进阶方案（见 4.3②）。

## 延伸阅读

- 📄 [Lewis et al., 2020 - RAG 原始论文](https://arxiv.org/abs/2005.11401)
- 📄 [Gao et al., 2023 - RAG 综述](https://arxiv.org/abs/2312.10997)（强烈推荐，把 RAG 的演进讲透了）
- 📄 [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511) - 让模型自己决定要不要检索（详见 [Self-RAG 词条](./self-rag)）
- 🔧 [LlamaIndex 文档](https://docs.llamaindex.ai/) - 工程实践参考
- 📊 [RAG评测：RAGAS](https://docs.ragas.io/) - 怎么客观评估一个 RAG 系统
- 🚀 进阶专题：[GraphRAG](./graphrag) / [Self-RAG](./self-rag) / [Corrective RAG](./corrective-rag) / [Agentic RAG](./agentic-rag) / [Multi-modal RAG](./multimodal-rag) - RAG 范式的演进与变体

---

> *下一篇预告：Embedding -- 为什么"国王 - 男人 + 女人 ≈ 女王"这个例子能上每一本 NLP 教科书，以及它其实骗了你。*
