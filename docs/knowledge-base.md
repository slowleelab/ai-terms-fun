---
title: 知识库 Knowledge Base
slug: knowledge-base
category: 评估与应用
tags: [知识库, RAG, 文档管理, chunking, 索引, 知识管理]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# 知识库 Knowledge Base

> 五层读懂一个词。这次拆的是：**知识库（Knowledge Base）**--RAG 的知识存储与管理，从原始文档到可检索 chunk 的工程化处理。

---

## L1 · 一句话点破

**知识库 = 原始文档 + 切分（chunking）+ 索引（vector + inverted）+ 元数据 + 更新机制**。RAG 系统的知识基础设施，决定 LLM 能"知道"什么。

---

## L2 · 通俗类比

LLM 像一个博学但有截止日期的专家，训练数据外的知识不知道。RAG 给专家配一个"随身图书馆"：

- **原始文档**：图书馆的藏书（PDF、网页、数据库）
- **切分**：把书拆成"小卡片"（chunk），每卡 200~500 字，方便检索
- **索引**：给每张卡片建"目录"（向量索引 + 关键词索引）
- **元数据**：卡片标签（来源、时间、作者、分类）
- **更新机制**：新书入库、旧书下架的流程

用户问问题时：

1. 在卡片库中检索最相关的几张卡（召回）
2. 把卡片交给专家（LLM 上下文）
3. 专家基于卡片回答

**知识库质量决定 RAG 上限**：

- 卡片切得不好（太长、太短、语义截断）-> 检索不准
- 索引建得不好（向量差、关键词漏）-> 召回不全
- 元数据不全 -> 过滤查询失效
- 更新不及时 -> 信息过时

工程上，知识库是 RAG 系统最容易被低估、但影响最大的部分。

---

## L3 · 正经定义

**知识库 (Knowledge Base, KB)**：RAG 系统中存储、组织、检索知识的子系统，核心组件：

**1. 文档接入（Ingestion）**

- 数据源：PDF、Word、HTML、Markdown、数据库、API
- 解析：文本提取（PyPDF、Unstructured）、表格/图像处理
- 清洗：去噪、去重、格式化

**2. 文档切分（Chunking）**

- 把长文档切成可检索的 chunk
- 策略：固定大小、滑动窗口、语义切分、结构化切分
- 详见 chunking 词条

**3. 索引构建（Indexing）**

- **向量索引**：chunk embedding + ANN（HNSW/IVF-PQ）
- **倒排索引**：BM25 / TF-IDF 关键词索引
- **元数据索引**：B-tree / bitmap 索引

**4. 元数据管理（Metadata）**

- 来源、时间、作者、分类、权限
- 用于过滤查询、个性化、审计

**5. 检索接口（Retrieval）**

- 向量检索（语义）
- 关键词检索（BM25）
- 混合检索 + 融合
- 过滤查询（metadata）

**6. 更新机制（Update）**

- 增量更新：新文档加入
- 删除：旧文档下架
- 重建：索引退化后重建

**7. 评估与监控**

- 检索质量（Hit Rate@K、Recall@K）
- 索引健康（recall 退化、内存占用）
- 更新延迟

**典型架构**：

```
原始文档
  ↓ 解析 + 清洗
结构化文本
  ↓ 切分
Chunks
  ↓ Embedding + 索引
知识库（向量索引 + 倒排索引 + 元数据）
  ↓ 检索
Top-K Chunks
  ↓ 喂给 LLM
生成回答
```

**伪代码**：

```python
class KnowledgeBase:
    def __init__(self, vector_db, embedding_model, bm25_index):
        self.vector_db = vector_db  # Milvus / Qdrant
        self.embedder = embedding_model  # BGE / E5
        self.bm25 = bm25_index  # Elasticsearch

    def ingest(self, documents):
        """文档入库"""
        chunks = []
        for doc in documents:
            # 1. 切分
            doc_chunks = self.chunk(doc.text, chunk_size=512, overlap=50)
            # 2. Embedding
            embeddings = self.embedder.encode([c.text for c in doc_chunks])
            # 3. 存入向量库
            self.vector_db.upsert([
                {
                    'id': c.id,
                    'vector': emb,
                    'payload': {
                        'text': c.text,
                        'source': doc.source,
                        'timestamp': doc.timestamp,
                        'metadata': doc.metadata,
                    }
                }
                for c, emb in zip(doc_chunks, embeddings)
            ])
            # 4. 存入 BM25 索引
            self.bm25.index([
                {'id': c.id, 'text': c.text, 'metadata': doc.metadata}
                for c in doc_chunks
            ])

    def retrieve(self, query, top_k=5, filter=None):
        """混合检索"""
        # 向量检索
        query_vec = self.embedder.encode(query)
        dense_results = self.vector_db.search(
            query_vec, top_k=top_k * 2, filter=filter
        )
        # BM25 检索
        sparse_results = self.bm25.search(query, top_k=top_k * 2, filter=filter)
        # RRF 融合
        fused = rrf_fuse([dense_results, sparse_results], top_k=top_k)
        return fused

    def chunk(self, text, chunk_size=512, overlap=50):
        """切分（简化版，实际用语义切分）"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(Chunk(text=text[start:end]))
            start = end - overlap
        return chunks
```

---

## L4 · 原理深挖

### 4.1 切分策略的影响

切分是知识库质量的决定因素：

**固定大小切分**：

- 简单，按字符数切
- 问题：可能切断句子、段落，语义不完整

**滑动窗口**：

- 固定大小 + overlap（如 512 tokens，overlap 50）
- 优势：边界 chunk 仍含上下文
- 问题：仍可能切断语义

**语义切分**：

- 按段落、句子、标题切
- 优势：语义完整
- 问题：chunk 长度不均

**结构化切分**：

- 按 Markdown 标题、HTML 标签、PDF 章节切
- 优势：保留文档结构
- 问题：依赖文档格式

**递归切分**：

- 先按大结构（章节），再按小结构（段落），最后按句子
- LangChain 的 `RecursiveCharacterTextSplitter` 是典型实现

**chunk_size 的选择**：

| chunk_size | 优势 | 劣势 | 适用 |
|------------|------|------|------|
| 128~256 | 召回高（粒度细） | 上下文不足 | 精确片段检索 |
| 512 | 平衡 | - | 通用 |
| 1024+ | 上下文完整 | 召回低（粒度粗） | 长文档摘要 |

**实践**：chunk_size = 256~512 tokens + overlap 50~100，是多数 RAG 场景的甜蜜点。

### 4.2 Embedding 模型的选择

embedding 质量决定向量检索上限：

**通用模型**：

- **BGE-large / BGE-M3**（智源）：中文最强之一
- **E5 / GTE**：多语言
- **sentence-BERT**：经典
- **OpenAI text-embedding-3**：商业 API

**选择考量**：

- **语言**：中文优先 BGE，英文 E5 / OpenAI
- **维度**：768 / 1024 / 1536，维度高精度高但存储大
- **速度**：CPU 推理 vs GPU 推理
- **多向量**：BGE-M3 支持稀疏 + 稠密 + 多向量

**实践**：

- 中文 RAG：BGE-M3 或 BGE-large
- 英文 RAG：E5-large 或 OpenAI embedding
- 多语言：BGE-M3 或 multilingual-E5

### 4.3 元数据的设计

元数据决定过滤查询能力：

**典型元数据字段**：

```json
{
  "source": "company_wiki",
  "title": "RLHF 技术文档",
  "author": "张三",
  "timestamp": "2026-07-21",
  "category": "AI/训练",
  "tags": ["RLHF", "PPO", "强化学习"],
  "permissions": ["team_a", "team_b"],
  "version": "v2.1",
  "language": "zh"
}
```

**元数据索引**：

- B-tree：范围查询（如 timestamp > 2026-01-01）
- Bitmap：分类查询（如 category = "AI/训练"）
- 倒排：标签查询（如 tags contains "RLHF"）

**过滤查询的实践**：

- 权限过滤：用户只能看自己有权限的 chunk
- 时间过滤：只检索最近更新的 chunk
- 分类过滤：限定领域（如只搜法律文档）

### 4.4 索引的维护

**增量更新**：

- 新文档：embedding + 入库
- 删除：标记删除 + 索引重建
- 更新：删除 + 重新插入

**索引退化**：

- HNSW 增删后图结构退化
- recall 悄悄下降
- 定期重建索引

**版本管理**：

- embedding 模型升级后，旧向量不兼容
- 新旧索引共存过渡
- 灰度切换

**监控指标**：

- recall@K（定期用评估集测）
- 查询延迟
- 内存占用
- 索引大小
- 更新延迟

### 4.5 知识库的评估

**检索质量**：

- Hit Rate@K：top-k 是否含相关 chunk
- Recall@K：召回了多少相关 chunk
- MRR@K：第一个相关 chunk 位置
- NDCG@K：综合排序质量

**端到端**：

- 答案准确率：LLM 基于检索 chunk 答对比例
- 答案忠实度：答案是否基于 chunk（不幻觉）
- 答案相关性：答案是否切题

**评估框架**：

- RAGAS：context_recall / context_precision / answer_relevancy / faithfulness
- TruLens：context_relevance / groundedness / answer_relevance
- LangSmith：自定义评估器

### 4.6 知识库的常见架构

**单机架构**（小规模）：

- 文档存储：本地文件系统
- 向量库：FAISS / Chroma
- 关键词索引：Whoosh / SQLite FTS
- 适用：$N < 10^6$ chunks

**分布式架构**（中大规模）：

- 文档存储：S3 / MinIO
- 向量库：Milvus / Qdrant 集群
- 关键词索引：Elasticsearch
- 适用：$N \in [10^6, 10^9]$

**云原生架构**（超大规模）：

- 存储计算分离
- 自动扩缩容
- 多租户隔离
- 适用：$N > 10^9$ 或多租户

### 4.7 知识库 vs 传统数据库

| 维度 | 传统数据库 | 知识库 |
|------|------------|--------|
| 数据模型 | 结构化行/列 | 非结构化文本 + 向量 |
| 查询 | SQL 精确匹配 | 语义 + 关键词检索 |
| 索引 | B-tree / Hash | HNSW + 倒排 |
| 一致性 | 强 ACID | 弱（最终一致） |
| 更新 | 行级 | chunk 级 |
| 适用 | 事务数据 | 知识检索 |

**融合趋势**：

- PostgreSQL + pgvector：关系数据库加向量
- Elasticsearch：搜索 + kNN
- 向量数据库加 SQL 接口

### 4.8 知识库的工程挑战

**挑战 1：文档解析**。

- PDF 表格、图像、公式解析难
- 多格式（PDF/Word/HTML/Markdown）统一处理
- OCR 错误传播

**挑战 2：切分优化**。

- 不同文档类型最优切分策略不同
- 长 chunk vs 短 chunk trade-off
- 语义完整 vs 检索粒度

**挑战 3：增量更新**。

- 文档变更检测（hash 比较）
- 部分 chunk 更新 vs 全量重建
- 索引一致性

**挑战 4：多模态**。

- 图像、表格、公式的向量化
- 多模态 embedding（CLIP 等）
- 跨模态检索

**挑战 5：权限控制**。

- 用户级权限过滤
- 不泄露无权限内容
- 性能（过滤查询慢）

**挑战 6：版本管理**。

- 文档版本（v1, v2）
- embedding 模型版本
- 索引版本
- 灰度切换

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **1980s**：专家系统时代的知识库（规则库）
- **1990s**：语义网、知识图谱兴起
- **2000s**：企业内容管理（ECM）、Wiki 系统普及
- **2010s**：向量检索兴起，但知识库概念仍偏传统
- **2020**：RAG 概念提出，知识库成为 LLM 应用的核心组件
- **2022**：LangChain、LlamaIndex 等 RAG 框架普及知识库工程
- **2023**：向量数据库爆发，知识库成为独立产品类别
- **2024**：多模态知识库、GraphRAG（知识图谱 + RAG）兴起

### 5.2 工程常见坑

**坑 1：切分太粗或太细**。

chunk_size=1024 召回率低（粒度粗），chunk_size=64 上下文不足。建议 256~512 tokens + overlap。

**坑 2：忘了语义切分**。

纯字符切分切断句子，检索不准。用语义切分（按段落/句子）或递归切分。

**坑 3：embedding 模型不归一化**。

embedding 输出不归一化，IP 距离不等价于余弦。统一 L2 normalize。

**坑 4：只建向量索引不建关键词索引**。

向量检索漏精确匹配（专有名词、代码）。要建 BM25 倒排索引，混合检索。

**坑 5：元数据不全**。

只存 text 不存 metadata，无法过滤查询。要存 source、timestamp、category、permissions 等。

**坑 6：忘了增量更新**。

文档更新后索引未更新，检索到旧内容。要建立文档变更检测 + 自动更新流程。

**坑 7：索引退化不监控**。

HNSW 增删后 recall 悄悄下降。定期用评估集测 recall，退化时重建。

**坑 8：embedding 模型升级没版本管理**。

换模型后旧向量不兼容，直接覆盖导致检索失效。新旧索引共存过渡。

**坑 9：文档解析不彻底**。

PDF 表格、图像、公式解析丢失信息。要用 Unstructured、PyMuPDF 等专业解析工具。

**坑 10：评估只看检索指标**。

Hit Rate@5 高但答案准确率低，可能是 LLM 问题或 chunk 上下文不足。要联合评估检索 + 生成。

**坑 11：知识库太大 LLM 看不完**。

top-k=20 chunk × 500 tokens = 10000 tokens，LLM context 紧张。要控制 K 和 chunk_size。

**坑 12：忘了去重**。

同一内容不同来源入库多次，检索 top-k 全是重复。要按内容 hash 去重。

### 5.3 知识库的选型

**自建 vs 托管**：

- 自建：灵活、可控、省钱（大规模时），但运维重
- 托管：省心、快，但贵、灵活性低

**主流方案**：

| 方案 | 类型 | 适用 |
|------|------|------|
| LangChain + Chroma | 自建（轻量） | 原型、小规模 |
| LlamaIndex + Qdrant | 自建（中量） | 中规模生产 |
| Milvus + Elasticsearch | 自建（重量） | 大规模生产 |
| Pinecone + Cohere | 托管 | 不想运维 |
| Databricks Vector Search | 云原生 | 已有 Databricks |
| Azure AI Search | 云原生 | 企业级 |

### 5.4 知识库的进阶方向

**GraphRAG**：

- 知识图谱 + 向量检索
- 实体关系增强检索
- 多跳推理

**Multi-modal RAG**：

- 文本 + 图像 + 表格统一向量空间
- CLIP 等多模态 embedding
- 跨模态检索

**Agentic RAG**：

- LLM Agent 自主决定检索策略
- 多次检索、工具调用
- 复杂问题分解

**Self-RAG**：

- LLM 自评检索质量
- 不够相关时重新检索
- 动态调整 K

**Long-context RAG**：

- 128K+ context LLM
- 减少检索需求
- 但 lost in middle 问题

### 5.5 知识库的评估闭环

```
1. 准备评估集（query + 金标准 chunk + 金标准答案）
2. 知识库检索 top-k
3. 评估检索指标（Hit Rate@K、Recall@K、NDCG@K）
4. LLM 生成答案
5. 评估答案指标（准确率、忠实度、相关性）
6. 分析瓶颈：
   - 检索低 -> 优化切分、embedding、混合检索
   - 答案低 -> 优化 LLM prompt、context 排序
7. 迭代优化
```

**关键监控指标**：

- 检索：Hit Rate@5、Recall@5、NDCG@5
- 生成：答案准确率、忠实度
- 系统：查询延迟、索引大小、更新延迟

---

## 速记卡

| 组件 | 作用 |
|------|------|
| 文档接入 | 解析 + 清洗原始文档 |
| 切分 | 拆成可检索 chunk |
| 索引 | 向量 + 倒排 + 元数据 |
| 元数据 | 过滤、个性化、权限 |
| 检索 | 混合检索 + 融合 |
| 更新 | 增量 + 重建 |
| 评估 | 检索 + 生成指标 |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| chunk_size | 256~512 tokens | 召回率 vs 上下文 |
| overlap | 50~100 tokens | 边界连续性 |
| embedding 维度 | 768 / 1024 / 1536 | 精度 vs 存储 |
| top_k | 3~10 | 召回 vs context cost |

**主流栈**：

| 规模 | 向量库 | 关键词索引 | 框架 |
|------|--------|------------|------|
| 原型 | Chroma | Whoosh | LangChain |
| 中量 | Qdrant | Elasticsearch | LlamaIndex |
| 大量 | Milvus | Elasticsearch | 自研 |
| 托管 | Pinecone | Cohere | - |

**一句话记忆**：知识库 = 文档接入 + 切分 + 索引（向量 + 倒排 + 元数据）+ 检索 + 更新 + 评估。切分质量和 embedding 质量决定上限，混合检索（BM25 + Dense）是标配，元数据支持过滤查询。RAG 系统最容易被低估但影响最大的部分。

---

> *上一篇：[NDCG 归一化折损累计增益](./ndcg) -- 综合排序质量指标。*
> *下一篇：[内容路线图](./roadmap) -- 58 个 AI 黑话词条全部完成，回到路线图查看完整进度。*
