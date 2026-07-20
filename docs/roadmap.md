---
title: 内容路线图
description: AI 黑话翻译器的完整概念树与编写进度。
---

# 🗺️ 内容路线图

这是本项目的完整内容地图。每个 AI 名词按它在知识体系中的位置归类，你既可以从任意节点切入阅读，也可以从最底层一路读到顶层，建立完整认知。

**图例**：✅ 已完成　⬜ 待编写

当前进度：**45 / 58**。目标每周更新 2~3 个。想认领某个 ⬜ 词条？读 贡献指南 后开 Issue 说明即可。

---

## 🏗️ 模型架构与训练

### 基础架构

- ✅ [Transformer](./transformer) -- 现代大模型的底座
- ⬜ 自注意力 Self-Attention -- Transformer 的核心机制
- ✅ [多头注意力 Multi-Head Attention](./multi-head-attention) -- 为什么"多头"比"单头"强
- ✅ [编码器-解码器 Encoder-Decoder](./encoder-decoder) -- 三种架构的分工
- ✅ [BERT（仅编码器）](./bert) -- 理解型任务的代表
- ✅ [GPT / LLaMA（仅解码器）](./gpt) -- 生成型任务的代表
- ✅ [传统模型：CNN / RNN / LSTM](./cnn-rnn-lstm) -- Transformer 之前的世界
- ✅ [模型组件：参数 / 层 / 激活函数](./model-components) -- ReLU、GeLU 为什么长那样

### 训练范式

- ✅ [预训练 Pre-training](./pre-training) -- 先在海量数据上"读万卷书"
- ✅ [微调 Fine-tuning](./fine-tuning) -- 再针对任务"练一套拳"
- ✅ [指令微调 Instruction Tuning](./instruction-tuning) -- 让模型学会听人话
- ✅ [RLHF](./rlhf) -- 用人类反馈做强化学习
- ✅ [迁移学习 Transfer Learning](./transfer-learning) -- 为什么预训练能迁移
- ✅ [损失函数 Loss Function](./loss-function) -- 模型怎么知道自己错没错
- ✅ [优化器（Adam / AdamW）](./optimizer) -- 怎么沿着梯度下山
- ✅ [过拟合 & 正则化](./overfitting) -- Dropout / 权重衰减为什么有效

---

## ⚡ 推理与生成

- ✅ [自回归生成 Autoregressive](./autoregressive) -- 一个 token 一个 token 往外蹦
- ⬜ 解码策略
  - ✅ [贪婪解码](./greedy-decoding)
  - ✅ [束搜索 Beam Search](./beam-search)
  - ✅ [Top-k 采样](./top-k-sampling)
  - ✅ [Top-p 采样](./top-p-sampling)
- ✅ [温度 Temperature](./temperature) -- 一个参数怎么调节"创造力"
- ✅ [幻觉 Hallucination](./hallucination) -- 模型为什么一本正经胡说八道

---

## 🗜️ 模型压缩与加速

- ✅ [量化（INT8 / INT4）](./quantization) -- 用更少比特存权重
- ✅ [知识蒸馏 Knowledge Distillation](./knowledge-distillation) -- 大模型教小模型
- ✅ [剪枝 Pruning](./pruning) -- 删掉没用的连接
- ✅ [推理引擎（vLLM / TensorRT-LLM）](./inference-engine) -- 生产部署的加速器

---

## 🔤 数据表示与编码

### 文本预处理

- ✅ [分词器 Tokenizer](./tokenizer) -- BPE / WordPiece / SentencePiece
- ✅ [Token 词元](./token) -- 模型的最小处理单位
- ✅ [分块 Chunking](./chunking) -- 长文档怎么切

### 向量表示

- ✅ [Embedding - 嵌入](./embedding) -- 把语义塞进高维坐标系
- ✅ [高维向量](./high-dim-vector) -- 维度灾难与维度的几何意义
- ✅ [稠密向量 vs 稀疏向量](./dense-sparse-vector)
- ✅ [位置编码 Positional Encoding](./positional-encoding) -- Transformer 怎么知道顺序
- ✅ [多模态 Embedding（CLIP）](./clip) -- 图文对齐到同一空间

### 上下文管理

- ✅ [上下文窗口 Context Window](./context-window) -- 模型的"短期记忆"边界

---

## 🔍 检索与索引

### 关键词检索

- ✅ [倒排索引 Inverted Index](./inverted-index)
- ✅ [TF-IDF](./tf-idf)
- ✅ [BM25](./bm25) -- 关键词检索的工业标准

### 向量检索

- ✅ [KNN / ANN](./knn-ann) -- 精确 vs 近似最近邻
- ✅ [索引算法：HNSW / IVF / PQ / LSH](./ann-algorithms)
- ✅ [算法库：Faiss / ScaNN / Annoy](./ann-libraries)
- ✅ [向量数据库](./vector-database) -- Milvus / Pinecone / Weaviate / Qdrant / Chroma

### 神经检索模型

- ✅ [双塔模型 Two-Tower](./two-tower) -- 稠密召回主力
- ⬜ 交叉编码器 Cross-encoder -- 精排主力
- ⬜ ColBERT 迟交互 -- 召回与精排的折中

### 排序与融合

- ⬜ 召回 vs 重排序 -- 多阶段排序的分工
- ⬜ Top-K
- ⬜ 混合搜索 Hybrid Search -- 关键词 + 向量
- ⬜ RRF 倒数排名融合
- ⬜ 加权重排 Weighted Fusion
- ⬜ 学习排序 LTR

---

## 📊 评估与应用

### 评估指标

- ⬜ Hit Rate
- ⬜ Recall@K / Precision@K
- ⬜ MRR 平均倒数排名
- ⬜ NDCG 归一化折损累计增益

### 应用框架

- ✅ [RAG - 检索增强生成](./rag) -- 给大模型发一本翻到目标页的参考书
- ⬜ 知识库 Knowledge Base

---

## 阅读建议

这棵树本身也暗示了学习路径：

1. **入门**：先读 [Embedding](./embedding) 和 [RAG](./rag)，理解当下最火的应用栈是怎么搭起来的。
2. **打地基**：回头补 Transformer / 自注意力 / 预训练，搞清楚生成模型从哪来。
3. **做工程**：再看 量化 / 推理引擎 / 向量数据库，知道怎么把模型真正部署起来。

每个词条都有 L1~L5 五层，按需取用即可。
