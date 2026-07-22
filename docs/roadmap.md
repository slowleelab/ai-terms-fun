---
title: 内容路线图
description: AI 黑话翻译器的完整概念树与编写进度。
---

# 🗺️ 内容路线图

这是本项目的完整内容地图。每个 AI 名词按它在知识体系中的位置归类，你既可以从任意节点切入阅读，也可以从最底层一路读到顶层，建立完整认知。

**图例**：✅ 已完成　⬜ 待编写

当前进度：**58 / 58**　进阶 RAG：**5 / 5**　高效微调：**5 / 5**　对齐：**5 / 5**　推理工程：**5 / 5**　Agent：**5 / 5**。目标每周更新 2~3 个。想认领某个 ⬜ 词条？读 贡献指南 后开 Issue 说明即可。

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
- ✅ [交叉编码器 Cross-encoder](./cross-encoder) -- 精排主力
- ✅ [ColBERT 迟交互](./colbert) -- 召回与精排的折中

### 排序与融合

- ✅ [召回 vs 重排序](./recall-rerank) -- 多阶段排序的分工
- ✅ [Top-K 检索](./top-k)
- ✅ [混合搜索 Hybrid Search](./hybrid-search) -- 关键词 + 向量
- ✅ [RRF 倒数排名融合](./rrf)
- ✅ [加权重排 Weighted Fusion](./weighted-fusion)
- ✅ [学习排序 LTR](./ltr)

---

## 📊 评估与应用

### 评估指标

- ✅ [Hit Rate](./hit-rate)
- ✅ [Recall@K / Precision@K](./recall-precision-at-k)
- ✅ [MRR](./mrr) 平均倒数排名
- ✅ [NDCG](./ndcg) 归一化折损累计增益

### 应用框架

- ✅ [RAG - 检索增强生成](./rag) -- 给大模型发一本翻到目标页的参考书
- ✅ [知识库 Knowledge Base](./knowledge-base) -- RAG 的知识存储与管理

---

## 🚀 进阶专题

进阶专题是基础 58 词条之外的增量系列，按主题分批扩展。每篇都假设你已读过相关基础词条。

### 进阶 RAG

聚焦 RAG 范式的演进与变体。假设已读 [RAG](./rag) 和 [知识库](./knowledge-base)。

- ✅ [GraphRAG 图谱增强检索](./graphrag) -- 知识图谱 + 社区聚类，专治全局性问题
- ✅ [Self-RAG 自反思检索](./self-rag) -- 反思 token 让 LLM 自决检索与评价
- ✅ [Corrective RAG 检索纠错](./corrective-rag) -- 即插即用的检索质检 + web 兜底
- ✅ [Agentic RAG 智能体检索](./agentic-rag) -- 多轮自主推理循环，专治复杂问题
- ✅ [Multi-modal RAG 多模态检索](./multimodal-rag) -- 跨模态对齐，图文混排检索

### 高效微调 PEFT

聚焦参数高效微调方法。假设已读 [微调 Fine-tuning](./fine-tuning) 和 [量化](./quantization)。

- ✅ [LoRA 低秩适配](./lora) -- 冻结基座 + 训低秩 $BA$，事实标准
- ✅ [QLoRA 量化低秩适配](./qlora) -- 4bit NF4 + LoRA，单卡微调 70B
- ✅ [Adapter Tuning 适配器微调](./adapter-tuning) -- PEFT 老前辈，瓶颈模块
- ✅ [Prefix/Prompt Tuning 前缀调优](./prefix-tuning) -- 只调软提示向量，最省参数
- ✅ [PEFT 总览与选型](./peft) -- 全参 / LoRA / QLoRA / Adapter / Prefix / Prompt 怎么选

### 对齐 Alignment

聚焦从监督微调到偏好优化的完整对齐链路。假设已读 [RLHF](./rlhf) 和 [指令微调](./instruction-tuning)。

- ✅ [SFT 监督微调](./sft) -- 对齐起点，指令-回答对 + 仅回答 loss
- ✅ [Reward Model 奖励模型](./reward-model) -- 偏好对 + Bradley-Terry，RLHF 的打分器
- ✅ [PPO 近端策略优化](./ppo-rlhf) -- RLHF 第三阶段，4 模型同框，效果最好
- ✅ [DPO 直接偏好优化](./dpo) -- 绕开 RM 和 PPO，2 模型监督学习
- ✅ [KTO / SimPO 变体](./kto-simpo) -- DPO 之后：二元反馈 / 去 ref / 修过对齐

### 推理工程

聚焦 LLM 部署侧的加速技术。假设已读 [量化](./quantization) 和 [推理优化](./inference-engine)。

- ✅ [KV-Cache 键值缓存](./kv-cache) -- 避免重复算 K/V，自回归推理的基础优化
- ✅ [PagedAttention 分页注意力](./paged-attention) -- 按需分页 KV-Cache，vLLM 的内存管理革命
- ✅ [Continuous Batching 连续批处理](./continuous-batching) -- iteration 级动态拼 batch，GPU 利用率 30%→90%
- ✅ [Speculative Decoding 推测解码](./speculative-decoding) -- 小模型草拟、大模型并行批改，无损降延迟 2-3x
- ✅ [量化推理算法 GPTQ/AWQ](./quantization-inference) -- 4bit 权重量化，70B 单卡可跑，精度损失 <1%

### Agent

聚焦 LLM Agent 系统的核心范式。假设已读 [Agentic RAG](./agentic-rag) 和 [推理引擎](./inference-engine)。

- ✅ [ReAct 推理与行动](./react) -- Agent 奠基范式，Thought-Action-Observation 交错循环
- ✅ [Function Calling 函数调用](./function-calling) -- 结构化工具调用，ReAct 的工程化升级
- ✅ [Planning 任务规划](./planning) -- Plan-and-Solve / ToT / LATS / Re-plan，先规划再执行
- ✅ [Memory 记忆机制](./memory) -- 短期+长期+反思，MemGPT 式分层管理
- ✅ [Multi-Agent 多智能体](./multi-agent) -- 角色分工协作，AutoGen/MetaGPT/CAMEL

---

## 阅读建议

这棵树本身也暗示了学习路径：

1. **入门**：先读 [Embedding](./embedding) 和 [RAG](./rag)，理解当下最火的应用栈是怎么搭起来的。
2. **打地基**：回头补 Transformer / 自注意力 / 预训练，搞清楚生成模型从哪来。
3. **做工程**：再看 量化 / 推理引擎 / 向量数据库，知道怎么把模型真正部署起来。

每个词条都有 L1~L5 五层，按需取用即可。
