---
title: ANN 库：FAISS / Milvus / Qdrant / Weaviate
slug: ann-libraries
category: 检索与索引
tags: [向量检索, ANN, FAISS, Milvus, Qdrant, Weaviate, 向量数据库]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# ANN 库：FAISS / Milvus / Qdrant / Weaviate

> 五层读懂一个词。这次拆的是：**主流 ANN 库的工程实现差异**--从 FAISS 的算法库到 Milvus/Qdrant/Weaviate 的向量数据库。

---

## L1 · 一句话点破

**FAISS 是算法库（嵌入应用，无服务），Milvus/Qdrant/Weaviate/Pinecone 是向量数据库（独立服务，带 CRUD、过滤、分布式、混合检索）**。前者是积木，后者是房子。

---

## L2 · 通俗类比

- **FAISS**：像一个高级工具箱，里面有电钻、扳手、螺丝刀（各种 ANN 算法）。你得自己组装家具（管理向量生命周期、并发、持久化）。灵活但要写代码。
- **Milvus**：像一个全功能家具店，你要什么直接点单（CRUD API、过滤、分布式）。贵（部署重）但省心。
- **Qdrant**：像一个精品家具店，单机性能极强，Rust 写的没 GC 抖动，过滤查询优化好。
- **Weaviate**：像一个智能家居套装，自带混合检索、模块化 embedding 接入，开箱即用。
- **Pinecone**：像云家具租赁服务，全托管，你只管用，底层不用管。贵但零运维。

选哪个？取决于团队规模、运维能力、数据量、延迟要求。

---

## L3 · 正经定义

### FAISS (Facebook AI Similarity Search)

**Facebook AI Research 2017 开源**。C++ 核心带 Python 绑定，是 ANN 算法的事实标准库。

**特点**：

- **算法库**，非服务：嵌入 Python/C++ 应用，无独立进程
- **支持算法**：暴力（IndexFlat）、IVF、IVF-PQ、HNSW、HNSW-IVF、PQ、SQ、LSH 等几乎所有主流算法
- **GPU 加速**：FAISS-GPU 支持多 GPU 并行，单机千万级向量毫秒级
- **无 CRUD**：向量加进去后删改困难（部分索引支持 `remove_ids`，但效率低）
- **无服务**：需自己包装成服务（gRPC/HTTP），管理并发、持久化

**典型用法**：

```python
import faiss
import numpy as np

# 构建 HNSW 索引
dim = 768
index = faiss.IndexHNSWFlat(dim, 32)  # M=32
index.hnsw.efConstruction = 200

# 训练（HNSW 不需要训练，IVF/PQ 需要）
vectors = np.random.rand(100000, dim).astype('float32')
faiss.normalize_L2(vectors)  # 归一化
index.add(vectors)

# 查询
query = np.random.rand(1, dim).astype('float32')
faiss.normalize_L2(query)
index.hnsw.efSearch = 100
D, I = index.search(query, 10)  # 返回 top-10 的距离和 ID
```

**适用**：

- 算法研究、性能调优
- 嵌入式部署，单机小规模
- 需要极致性能控制
- 已有应用框架，向量只是其中一个组件

### Milvus

**Zilliz 2019 开源**。云原生分布式向量数据库，存储计算分离架构。

**特点**：

- **分布式**：支持十亿级向量，水平扩展
- **存储计算分离**：存储用 S3/MinIO，计算节点无状态
- **多索引**：支持 HNSW、IVF-Flat、IVF-PQ、DiskANN、SCANN 等
- **CRUD**：完整增删改查
- **过滤查询**：标量字段过滤 + 向量检索混合
- **混合检索**：2.4+ 支持稀疏 + 稠密向量混合
- **生态**：Attu（GUI）、Milvus Backup、Spark Connector

**架构**：

- Proxy：API 网关
- Query Node：查询节点（无状态）
- Data Node：数据写入节点
- Index Node：建索引节点
- Etcd：元数据
- S3/MinIO：对象存储
- Pulsar/Kafka：消息队列

**适用**：

- 大规模（> 亿级）向量
- 需要分布式部署
- 团队有运维能力
- 已有云原生基础设施

### Qdrant

**Qdrant 2021 开源**。Rust 写的向量搜索引擎，单机性能强。

**特点**：

- **Rust 实现**：无 GC 抖动，延迟稳定
- **单机优先**：单机性能极强，分布式通过 raft 同步
- **过滤查询优化**：payload 索引 + 向量索引联合，过滤性能业界领先
- **CRUD**：完整支持
- **混合检索**：稀疏 + 稠密向量（1.10+）
- **API**：HTTP/JSON，gRPC

**典型用法**：

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

client = QdrantClient(host="localhost", port=6333)

# 创建集合
client.create_collection(
    collection_name="my_docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# 插入
client.upsert(
    collection_name="my_docs",
    points=[
        PointStruct(id=1, vector=[0.1, ...], payload={"category": "news"}),
        PointStruct(id=2, vector=[0.2, ...], payload={"category": "tech"}),
    ],
)

# 带过滤的查询
results = client.search(
    collection_name="my_docs",
    query_vector=[0.15, ...],
    query_filter=Filter(
        must=[{"key": "category", "match": {"value": "tech"}}]
    ),
    limit=10,
)
```

**适用**：

- 单机或小集群（< 千万级）
- 低延迟敏感场景
- 过滤查询复杂
- 偏好 Rust 生态

### Weaviate

**SeMI Technologies 2019 开源**。Go 写的向量搜索引擎，模块化设计。

**特点**：

- **模块化 embedding**：内置对接 OpenAI、Cohere、HuggingFace、Sentence-Transformers 等，输入文本自动 embedding
- **混合检索**：BM25 + 向量内置融合（alpha 参数调权重）
- **GraphQL API**：查询用 GraphQL，灵活但学习曲线
- **CRUD**：完整支持
- **多模态**：支持文本、图像、PDF 等

**适用**：

- 需要"开箱即用"的语义搜索
- 不想自己管理 embedding 模型
- 需要混合检索且偏好内置方案

### Pinecone

**Pinecone 2019 商业产品**。全托管云向量数据库。

**特点**：

- **全托管**：用户不接触底层，API 即用
- **自动分片**：根据数据量自动扩展
- **混合检索**：稀疏 + 稠密
- **Serverless**：按用量计费
- **企业级**：SSO、审计日志、合规

**适用**：

- 不想运维
- 团队小但数据大
- 预算充足
- 需要 SLA 保证

### 其他值得注意的

- **Vespa**：Yahoo 开源，老牌搜索引擎 + 向量检索，功能强大但复杂
- **Elasticsearch**：8.0+ 内置 kNN（基于 HNSW/Lucene），适合已有 ES 栈的团队
- **OpenSearch**：ES fork，同样支持 kNN
- **pgvector**：PostgreSQL 扩展，把向量检索搬进关系数据库
- **Chroma**：轻量级，Python 优先，适合原型和教学
- **LanceDB**：基于 Lance 列式存储，嵌入式向量库，适合本地分析

---

## L4 · 原理深挖

### 4.1 FAISS 的索引家族

FAISS 的核心是 `Index` 抽象，通过组合形成不同索引：

**基础索引**：

- `IndexFlat`：暴力，无压缩，100% 召回
- `IndexLSH`：LSH 哈希
- `IndexPQ`：纯 PQ 压缩
- `IndexScalarQuantizer`：标量量化（8-bit / 4-bit / FP16）
- `IndexHNSWFlat`：HNSW + 原始向量
- `IndexHNSWPQ` / `IndexHNSWSQ`：HNSW + 量化

**复合索引**：

- `IndexIVFFlat`：IVF + 簇内精确
- `IndexIVFPQ`：IVF + PQ（最常用大规模索引）
- `IndexIVFScalarQuantizer`：IVF + 标量量化
- `IndexIVFPQFastScan`：IVF-PQ + SIMD 优化（4-bit PQ + AVX512）
- `IndexHNSWIVF`：HNSW 找簇心 + IVF

**选型经验**：

| 场景 | 推荐索引 |
|------|----------|
| 小规模（< 10^5）+ 高召回 | `IndexFlatL2` 或 `IndexHNSWFlat` |
| 中规模（10^5~10^7）+ 速度优先 | `IndexHNSWFlat` |
| 大规模（10^7~10^9）+ 内存敏感 | `IndexIVFPQ` |
| 极大规模（10^9+）+ 磁盘 | DiskANN（FAISS 1.7+） |
| GPU 加速 | `index_cpu_to_gpu` 任意索引 |

### 4.2 向量数据库 vs 算法库

| 维度 | 算法库（FAISS） | 向量数据库（Milvus/Qdrant/Weaviate） |
|------|-----------------|--------------------------------------|
| 部署 | 嵌入应用 | 独立服务 |
| CRUD | 弱 | 强 |
| 持久化 | 自己管理 | 内置 |
| 过滤查询 | 无 | 有（payload 过滤） |
| 分布式 | 无 | 有 |
| 混合检索 | 需自己拼 | 内置（部分） |
| 性能调优 | 直接 | 通过配置 |
| 运维成本 | 低 | 高（除 Pinecone 等托管） |
| 学习曲线 | 陡（C++/算法） | 中（API/SQL-like） |

**何时用算法库**：

- 已有应用，向量只是组件
- 需要极致性能调优
- 单机部署
- 团队有能力包装服务层

**何时用向量数据库**：

- 需要独立服务
- 多应用共享向量数据
- 需要 CRUD 和持久化
- 大规模分布式

### 4.3 过滤查询的实现

向量数据库的核心差异化能力之一是**带 metadata 过滤的向量检索**。三种实现策略：

**Pre-filter**：先按 metadata 过滤出子集，再在子集上 ANN。问题：子集小或分布偏斜时 ANN 索引失效。

**Post-filter**：ANN 找大候选集，再按 metadata 过滤。问题：候选集大部分被过滤掉，top-k 不足。

**Hybrid / In-filter**：ANN 检索时动态检查过滤条件，丢弃不满足的候选。这是 Qdrant、Milvus 主流实现。

**Qdrant 的优化**：payload 字段建索引（如 B-tree、bitmap），ANN 检索时如果候选不满足过滤条件直接跳过。优化得好可以做到接近无过滤的速度。

**Elasticsearch / OpenSearch**：用 Lucene 的 HNSW 实现 + Lucene 原生的过滤机制，性能中等。

### 4.4 混合检索（Hybrid Search）

现代向量数据库开始内置混合检索（稀疏 + 稠密）：

**Weaviate**：BM25 + 向量，alpha 参数调权重（0=纯 BM25，1=纯向量）。

**Milvus 2.4+**：支持稀疏向量字段（如 BM25、SPLADE）+ 稠密向量字段，查询时分别检索后融合（RRF 或加权）。

**Qdrant 1.10+**：稀疏 + 稠密，查询时融合。

**Pinecone**：稀疏 + 稠密，内置融合。

**融合方法**：

- **RRF (Reciprocal Rank Fusion)**：$\text{score}(d) = \sum_i \frac{1}{k + \text{rank}_i(d)}$，$k$ 常用 60
- **加权融合**：$\text{score}(d) = \alpha \cdot s_{\text{dense}}(d) + (1-\alpha) \cdot s_{\text{sparse}}(d)$，需归一化

详见后续 hybrid-search、rrf 词条。

### 4.5 分布式架构

**Milvus（云原生）**：

- 存储计算分离：存储用 S3/MinIO，计算节点无状态
- 查询节点分片：每个 query node 处理部分数据
- 索引节点独立：建索引不阻塞查询
- 适合云上弹性扩展

**Qdrant（单机优先）**：

- 单机性能优先，Rust 写的内存管理好
- 分布式通过 raft 同步，sharding 需手动配置
- 适合中小规模或对延迟敏感的场景

**Pinecone（全托管）**：

- 用户不可见底层架构
- 自动分片、自动复制、自动扩缩容
- 适合不想运维的团队

**Elasticsearch（关系数据库基因）**：

- shard + replica 机制
- 与现有 ES 栈集成方便
- 向量检索性能不如专业向量库

### 4.6 量化与压缩

向量数据库普遍支持量化以节省内存：

**Qdrant**：支持 Scalar Quantization（int8）、Product Quantization、Binary Quantization。

**Milvus**：支持 SQ、PQ、DiskANN（磁盘）。

**Weaviate**：支持 PQ 压缩。

**Pinecone**：内置自动量化，用户不可见。

量化让 10 亿级向量可装单机内存，代价是召回略降（通常 recall@10 从 0.99 降到 0.95）。

### 4.7 评测与选型

**性能评测**：

- QPS / 延迟 / 召回：每库都提供 benchmark，但口径不一
- ann-benchmarks 只评算法库（FAISS、ScaNN、HNSWlib 等），不评向量数据库
- 向量数据库选型需自己测，参考 [VectorDBBench](https://github.com/zilliztech/VectorDBBench)

**功能对比**：

| 库 | CRUD | 过滤 | 混合检索 | 分布式 | 量化 | 多模态 |
|----|------|------|----------|--------|------|--------|
| FAISS | 弱 | 无 | 无 | 无 | 强 | 无 |
| Milvus | 强 | 强 | 有 | 强 | 强 | 弱 |
| Qdrant | 强 | 极强 | 有 | 中 | 强 | 无 |
| Weaviate | 强 | 强 | 内置 | 中 | 中 | 强 |
| Pinecone | 强 | 强 | 有 | 自动 | 自动 | 无 |
| pgvector | 强（SQL） | 强（SQL） | 通过插件 | 通过 PG | 中 | 无 |
| Chroma | 中 | 中 | 无 | 无 | 弱 | 无 |

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **2017**：FAISS 开源（Facebook AI Research），ANN 算法库霸主诞生
- **2019**：Milvus 1.0（Zilliz）；Weaviate 开源；Pinecone 商业化
- **2020**：Milvus 2.0 重写为云原生架构；FAISS 1.6 加入 HNSW
- **2021**：Qdrant 开源（Rust）；Vespa 加入 ANN
- **2022**：Elasticsearch 8.0 内置 kNN；pgvector 崛起
- **2023**：向量数据库大爆发（受 ChatGPT / RAG 推动）；Milvus 2.4 支持稀疏向量；Qdrant 1.7 稀疏向量；Pinecone Serverless
- **2024**：FAISS 1.8 DiskANN；混合检索成标配；向量数据库进入企业级

### 5.2 选型常见坑

**坑 1：用 FAISS 当数据库**。

FAISS 没有 CRUD、持久化、并发控制，硬要当数据库用，得自己包装一大堆代码。除非你真的只需要只读检索，否则用向量数据库省心。

**坑 2：用 Milvus 但数据量小**。

Milvus 部署重（etcd、MinIO、Pulsar、多个节点），$N < 100$ 万时杀鸡用牛刀。这时 Qdrant 单机或 pgvector 更合适。

**坑 3：相信厂商 benchmark**。

每家都声称自己最快，但测试条件不同（数据集、参数、硬件、是否过滤）。务必自己用 VectorDBBench 在自己场景下测。

**坑 4：忘了归一化向量**。

很多库默认 IP（内积）距离，如果你的 embedding 没归一化，结果会偏向长向量。统一 L2 normalize 后再写入。

**坑 5：过滤查询性能差**。

某些库的过滤是 post-filter（先 ANN 再过滤），过滤比例高时 top-k 大量不足。要用支持 in-filter 的库（Qdrant、Milvus）。

**坑 6：增量更新导致索引退化**。

HNSW 删节点后图结构退化，新插入节点邻居选择基于现有图。定期重建索引，或用支持高效增删的库。

**坑 7：内存估算不足**。

向量数据库内存占用 = 原始向量 + 索引结构 + 元数据 + 缓存。生产环境要留 30~50% 余量。$N = 10^7, D = 768$，原始向量 28 GB，加索引和缓存要 50 GB+。

**坑 8：分布式部署忘了副本**。

单副本（replica=1）节点故障即数据不可用。生产至少 2 副本。

**坑 9：向量版本不兼容**。

embedding 模型升级（如 bge-large -> bge-m3）后向量维度/语义变了，旧索引不能直接用。要做版本管理，新旧索引共存过渡。

**坑 10：用 pgvector 装大规模**。

pgvector 适合 $N < 10^7$ 且已有 PostgreSQL 的场景。更大规模性能不如专业向量库（HNSW 实现没那么精调）。

### 5.3 选型决策树

```
已有 PostgreSQL 且 N < 10^7?
  └─ pgvector（零运维成本）

只需算法嵌入应用?
  └─ FAISS

N < 10^8 且想要单机 + 低延迟?
  └─ Qdrant（Rust 性能稳定）
      或 Chroma（原型/教学）

N > 10^8 且需分布式?
  └─ Milvus（云原生）
      或 Pinecone（全托管）

需要内置混合检索 + embedding?
  └─ Weaviate

已有 Elasticsearch 栈?
  └─ Elasticsearch 8+ kNN

不想运维?
  └─ Pinecone 或托管 Milvus（Zilliz Cloud）
```

### 5.4 新趋势

**Serverless 向量库**：Pinecone Serverless、Milvus Lite，按用量计费，零运维。

**磁盘向量检索**：DiskANN 让 10 亿级向量单机服务，减少分布式复杂度。

**原生混合检索**：稀疏 + 稠密在同一引擎内融合，省去外部融合层。

**多模态向量**：文本、图像、音频向量同库管理，跨模态检索。

**GPU 向量检索**：FAISS-GPU、cuVS 用 GPU 加速大规模检索。

**AI 原生数据库**：把 LLM、embedding、向量检索深度整合，如 LanceDB、TurboPipes。

---

## 速记卡

| 库 | 类型 | 语言 | 部署 | 适用规模 | 特色 |
|----|------|------|------|----------|------|
| FAISS | 算法库 | C++/Python | 嵌入 | 任意 | 算法全、GPU 加速 |
| Milvus | 向量数据库 | Go | 分布式 | 10^9+ | 云原生、可扩展 |
| Qdrant | 向量数据库 | Rust | 单机/小集群 | 10^7 | 过滤强、延迟稳 |
| Weaviate | 向量数据库 | Go | 单机/集群 | 10^8 | 内置混合检索 |
| Pinecone | 托管服务 | - | 云 | 任意 | 全托管、零运维 |
| pgvector | PostgreSQL 扩展 | C | 单机 | 10^7 | SQL 生态 |
| Chroma | 轻量库 | Python | 嵌入 | 10^6 | 原型/教学 |

**一句话记忆**：FAISS 是算法库（积木），Milvus/Qdrant/Weaviate/Pinecone 是向量数据库（房子）。$N < 10^7$ 优先 Qdrant 或 pgvector，$N > 10^8$ 用 Milvus 或 Pinecone，不想运维用 Pinecone 或托管服务。

---

> *上一篇：[索引算法 HNSW / IVF / PQ / LSH](./ann-algorithms) -- ANN 算法原理对比。*
> *下一篇：[向量数据库](./vector-database) -- 向量数据库的工程设计与选型。*
