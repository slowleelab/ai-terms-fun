---
title: 向量数据库
slug: vector-database
category: 检索与索引
tags: [向量检索, 向量数据库, Milvus, Pinecone, Qdrant, pgvector, 工程设计]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 向量数据库

> 五层读懂一个词。这次拆的是：**向量数据库**--从"存向量"到"管理向量生命周期"的工程设计。

---

## L1 · 一句话点破

**向量数据库 = 向量索引 + 存储 + CRUD + 过滤 + 分布式 + 混合检索 + 多租户**。它把 FAISS 这种算法库变成可独立部署、可水平扩展、可长期运维的基础设施。

---

## L2 · 通俗类比

FAISS 是算法库（电钻），向量数据库是家具工厂（电钻 + 仓库 + 工人 + 物流 + CRM）。

你开店卖家具（做 AI 应用），有几种选：

- **自己买电钻做**（用 FAISS）：灵活但要自己解决存储、并发、持久化、扩容
- **租工厂车间**（用 Milvus 自部署）：基础设施齐了，但你要雇人（运维）管
- **用精品小作坊**（用 Qdrant 单机）：单机性能好，适合中小规模
- **托管工厂**（用 Pinecone / Zilliz Cloud）：连工人都不用雇，按用量付钱
- **自家后院搭车间**（用 pgvector）：复用已有的 PostgreSQL，省事但规模有限

向量数据库的本质：**把向量当作一等公民管理，像关系数据库管理行一样**。

---

## L3 · 正经定义

**向量数据库 (Vector Database)**：专门为高维向量检索设计的数据库系统，核心能力包括：

1. **向量存储**：原始向量 + 元数据（payload / attributes）
2. **ANN 索引**：HNSW / IVF-PQ / DiskANN 等加速结构
3. **CRUD**：增删改查向量及其元数据
4. **过滤检索**：metadata 过滤 + 向量近邻
5. **混合检索**：稀疏（关键词）+ 稠密（向量）融合
6. **分布式**：分片（sharding）+ 副本（replication）
7. **持久化**：向量与索引落盘
8. **事务 / 一致性**：写入后立即可查 / 副本间一致
9. **多租户**：集合（collection）隔离

**核心数据模型**：

```
Collection
  └─ Shard (分片，按 hash 或范围)
      └─ Segment (段，数据组织单元)
          ├─ Vector Index (HNSW/IVF/...)
          ├─ Payload Index (B-tree/bitmap/inverted)
          ├─ Raw Vector Storage
          └─ Payload Storage (metadata)
```

**核心查询模式**：

```text
1. 纯向量检索：search(query_vec, top_k)
2. 过滤检索：search(query_vec, filter={"category": "tech"}, top_k)
3. 混合检索：search(query_vec, query_sparse, fusion="rrf", top_k)
4. 范围检索：search(query_vec, radius=0.5)  # 距离阈值内的全部
```

**主流向量数据库**：

| 数据库 | 类型 | 部署 | 特色 |
|--------|------|------|------|
| Milvus | 开源 | 分布式 | 云原生、十亿级 |
| Qdrant | 开源 | 单机/小集群 | Rust、过滤强 |
| Weaviate | 开源 | 单机/集群 | 内置混合检索 |
| Pinecone | 托管 | 云 | 全托管、零运维 |
| Chroma | 开源 | 嵌入 | 原型、轻量 |
| pgvector | 开源扩展 | PostgreSQL | SQL 生态 |
| LanceDB | 开源 | 嵌入 | 列式存储、分析 |
| Elasticsearch | 开源 | 分布式 | 搜索引擎 + kNN |
| Vespa | 开源 | 分布式 | 老牌搜索 + 向量 |

---

## L4 · 原理深挖

### 4.1 向量数据库的架构分层

通用向量数据库架构（以 Milvus 为典型）：

**接入层（Access Layer）**：

- Proxy：API 网关，接收请求，路由到对应节点
- 负载均衡、认证、限流

**协调层（Coordinator）**：

- RootCoord：元数据管理（collection schema、shard 分配）
- QueryCoord：查询节点调度、负载均衡
- DataCoord：数据节点调度、segment 管理
- IndexCoord：索引节点调度

**工作层（Worker）**：

- QueryNode：查询执行，从 segment 加载向量+索引
- DataNode：写入 segment，刷盘
- IndexNode：建索引（CPU 密集）

**存储层（Storage）**：

- 元数据：etcd（强一致）
- 对象存储：S3/MinIO（向量、索引、segment）
- 消息队列：Pulsar/Kafka（写入流）

**为什么这么分层**：存储计算分离，工作节点无状态可随意扩缩容，存储用云原生对象存储无限扩展。代价：架构复杂、依赖多、运维重。

### 4.2 分片与副本

**分片（Sharding）**：

- **哈希分片**：`shard_id = hash(vec_id) % num_shards`，均匀分布但跨 shard 查询
- **范围分片**：按 ID 范围，但向量查询通常不按 ID 范围，不实用
- **聚类分片**：按向量聚类分片，相似向量同 shard，查询只查相关 shard（省查询但维护成本高）

实际多数用哈希分片，查询时所有 shard 并行再 merge。

**副本（Replication）**：

- 每个 shard 有 N 个副本（如 replica=2）
- 写入：同步 / 异步复制到副本
- 查询：副本都可服务，提高吞吐和可用性
- 一致性：强一致（同步写）vs 最终一致（异步写），权衡可用性

### 4.3 Segment：数据组织单元

向量数据库通常用 segment 作为数据组织单元（类似 Lucene 的 segment）：

**Segment 特性**：

- 不可变（immutable）：写入后不修改
- 大小有限制（如 1GB ~ 10GB）
- 段内建索引（HNSW/IVF-PQ 等）
- 多个 segment 可合并（compaction）

**写入流程**：

1. 新向量写入 growing segment（内存中）
2. growing segment 满后 flush 为 sealed segment（落盘）
3. sealed segment 建索引
4. 多个 small segment 定期合并为大 segment

**删除处理**：segment 不可变，删除标记到 tombstone，查询时过滤，compaction 时真正删除。

**更新处理**：删除 + 插入（旧 ID 标记删除，新向量新 ID 或同 ID 插入）。

### 4.4 过滤检索的实现

**挑战**：metadata 过滤 + 向量 ANN 的组合。如"在我订阅的作者中找最相似的文章 top-10"。

**三种实现策略**：

**Pre-filter**：先按 metadata 过滤出子集，再在子集上 ANN。

```python
subset = filter(metadata, predicate)  # 子集
results = ann_search(query_vec, subset, top_k)
```

问题：子集小或分布偏斜时 ANN 索引失效（HNSW 图结构假设全量数据）。

**Post-filter**：ANN 找大候选集（如 10×top_k），再按 metadata 过滤。

```python
candidates = ann_search(query_vec, top_k * 10)
results = filter(candidates, predicate)[:top_k]
```

问题：候选集大部分被过滤掉，top_k 不足。过滤比例高时（如 99% 被过滤）几乎不可用。

**In-filter / Hybrid**：ANN 检索时动态检查过滤条件。

```python
# HNSW 图遍历时，邻居不满足过滤条件则跳过
def filtered_hnsw_search(query, predicate, ef, top_k):
    visited = set()
    candidates = [(dist(query, entry), entry)]
    while candidates:
        d, c = pop_nearest(candidates)
        if satisfies(c, predicate):
            results.add(c)
        for n in neighbors(c):
            if n not in visited and satisfies(n, predicate):
                visited.add(n)
                push(candidates, (dist(query, n), n))
    return results[:top_k]
```

Qdrant、Milvus 用 in-filter。优化好的话，过滤性能接近无过滤。

**Qdrant 的优化**：

- payload 字段建索引（B-tree、bitmap、hash）
- 预估过滤比例，自适应选择 pre/in/post
- 高过滤比例（> 95%）用 pre-filter + 暴力扫描
- 低过滤比例用 in-filter

### 4.5 混合检索

现代向量数据库开始内置混合检索（稀疏 + 稠密）：

**架构**：

- 稀疏向量字段（如 BM25、SPLADE 编码）
- 稠密向量字段（如 sentence-BERT、BGE 编码）
- 两个字段独立索引
- 查询时分别检索后融合

**融合方法**：

- **RRF**：$\text{score}(d) = \sum_i \frac{1}{k + \text{rank}_i(d)}$，$k = 60$ 经验
- **加权**：$\text{score}(d) = \alpha \cdot \text{normalize}(s_{\text{dense}}) + (1-\alpha) \cdot \text{normalize}(s_{\text{sparse}})$
- **学习排序**：训练 LTR 模型融合

详见后续 hybrid-search、rrf 词条。

### 4.6 持久化与索引重建

**持久化层级**：

1. **内存**：热数据 + 索引（HNSW 图、IVF 簇心）
2. **SSD**：原始向量、segment 数据
3. **对象存储**：冷数据、备份

**索引重建**：

- 增删频繁后 HNSW 图退化（边指向已删除节点）
- 定期重建索引恢复质量
- 重建时新旧索引共存，零停机切换

**Milvus 的 segcore**：C++ 实现的 segment 引擎，支持向量化执行、SIMD 优化。

### 4.7 量化与压缩

向量数据库普遍支持量化节省内存：

**Scalar Quantization (SQ)**：float32 -> int8，4 倍压缩，距离误差小。

**Product Quantization (PQ)**：分段聚类，几十倍压缩，距离误差中等。

**Binary Quantization**：float -> 1 bit（sign），32 倍压缩，距离误差大，需 rerank。

**实践**：

- 内存够用：不量化或 SQ
- 内存紧：PQ（recall@10 通常 0.90~0.95）
- 极致压缩：Binary + rerank

### 4.8 多租户与隔离

**单 collection 多租户**：所有租户数据同 collection，payload 标记租户 ID，查询时过滤。简单但隔离弱。

**多 collection**：每租户一 collection，强隔离但管理复杂、资源浪费。

**分片隔离**：租户数据固定到特定 shard，物理隔离。Milvus、Pinecone 支持。

**实践**：

- 租户少（< 100）：多 collection
- 租户多但数据少：单 collection + payload 过滤
- 租户多且数据大：分片隔离

### 4.9 一致性模型

向量数据库的一致性通常弱于关系数据库：

- **强一致**：写入立即可查（同步刷盘 + 副本同步），代价是延迟高
- **最终一致**：写入异步同步到副本，短暂不一致，但吞吐高
- **会话一致**：同一客户端读写有序

Milvus 提供 4 级一致性：Strong / Bounded Staleness / Session / Eventually。默认 Bounded Staleness（容忍秒级延迟换吞吐）。

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **2017**：FAISS 开源，但只是算法库，不是数据库
- **2019**：Milvus 1.0、Pinecone、Weaviate 同年出现，向量数据库概念成形
- **2020**：Milvus 2.0 重写为云原生架构（存储计算分离）
- **2021**：Qdrant 开源（Rust），Chroma 发布
- **2022**：pgvector 兴起；Elasticsearch 8.0 内置 kNN；向量数据库进入主流
- **2023**：受 ChatGPT / RAG 推动爆发；Milvus 2.4 稀疏向量；Pinecone Serverless；资本大量涌入
- **2024**：混合检索成标配；DiskANN 集成；企业级特性（SSO、审计、合规）成熟

### 5.2 选型与使用常见坑

**坑 1：用 FAISS 当数据库**。

FAISS 没有 CRUD、持久化、并发控制。硬要当数据库用得自己包装服务层。除非只读检索，否则用向量数据库。

**坑 2：数据量小却上 Milvus**。

Milvus 部署重（etcd、MinIO、Pulsar 等），$N < 100$ 万时杀鸡用牛刀。用 Qdrant 单机或 pgvector。

**坑 3：过滤查询性能差**。

某些库默认 post-filter，过滤比例高时 top_k 不足。要选支持 in-filter 的库（Qdrant、Milvus）。

**坑 4：忘了归一化**。

embedding 写入前必须 L2 normalize，否则 IP 距离不等价于余弦，结果偏向长向量。

**坑 5：内存估算不足**。

向量数据库内存 = 原始向量 + 索引 + 元数据 + 缓存。$N = 10^7, D = 768$ 时原始向量 28 GB，加索引缓存至少 50 GB。生产要留 30~50% 余量。

**坑 6：副本数 = 1**。

单副本节点故障即不可用。生产至少 2 副本。

**坑 7：增量更新导致索引退化**。

HNSW 删节点后图结构退化。定期重建索引，或用支持高效增删的库。

**坑 8：embedding 模型升级后向量不兼容**。

换模型后维度/语义变了，旧索引不能用。要做版本管理，新旧索引共存过渡。

**坑 9：跨 collection 查询不必要**。

把本该同 collection 的数据拆成多 collection，导致查询要跨 collection 合并。设计时想清楚数据模型。

**坑 10：忘了监控索引质量**。

recall 不在监控里，悄悄掉到 0.7 也不知道。监控 recall@k（用评估集定期测）、查询延迟、内存占用、副本状态。

**坑 11：用 pgvector 装大规模**。

pgvector 适合 $N < 10^7$ 且已有 PostgreSQL。更大规模 HNSW 实现不如专业向量库精调。

**坑 12：忽略成本**。

Pinecone 等托管服务按用量计费，大规模时月费可能数千到数万美元。自部署虽省订阅费但要算运维人力。

### 5.3 向量数据库 vs 关系数据库

| 维度 | 关系数据库 | 向量数据库 |
|------|------------|------------|
| 数据模型 | 行/列 | 向量 + payload |
| 查询 | SQL | ANN + 过滤 |
| 索引 | B-tree/Hash | HNSW/IVF/PQ |
| 距离 | 无 | L2/IP/Cosine |
| ACID | 强 | 弱 |
| 规模 | TB 级 | 十亿级向量 |
| 适用 | 结构化数据 | 高维向量检索 |

**融合趋势**：

- PostgreSQL + pgvector：关系数据库加向量扩展
- Elasticsearch：搜索引擎加 kNN
- 向量数据库加 SQL：Milvus 2.4+ 支持 SQL-like 查询

### 5.4 选型决策树

```
已有 PostgreSQL 且 N < 10^7?
  └─ pgvector（零额外运维）

只需算法嵌入应用?
  └─ FAISS（自己包装服务）

N < 10^8 且想要单机低延迟?
  └─ Qdrant
      或 Chroma（原型）

N > 10^8 且需分布式?
  └─ Milvus（云原生）
      或 Pinecone（全托管）

需要内置混合检索 + embedding?
  └─ Weaviate

已有 Elasticsearch 栈?
  └─ Elasticsearch 8+ kNN

不想运维 + 预算充足?
  └─ Pinecone 或 Zilliz Cloud
```

### 5.5 新趋势

**Serverless 向量库**：Pinecone Serverless、Milvus Lite，按用量计费零运维。

**磁盘向量检索**：DiskANN 让 10 亿级向量单机服务，减少分布式复杂度。

**原生混合检索**：稀疏 + 稠密同引擎融合，省外部融合层。

**AI 原生数据库**：把 LLM、embedding、向量检索深度整合（如 LanceDB、TurboPipes）。

**多模态向量**：文本、图像、音频向量同库管理，跨模态检索。

**图数据库融合**：Neo4j、Memgraph 加向量检索，支持"图 + 向量"复合查询。

---

## 速记卡

| 维度 | 关键能力 |
|------|----------|
| 存储 | 向量 + payload + 索引 + 元数据 |
| 索引 | HNSW / IVF-PQ / DiskANN |
| 查询 | ANN + 过滤 + 混合检索 + 范围 |
| 分布式 | 分片 + 副本 + 存储计算分离 |
| 一致性 | 强一致 / 最终一致 / 会话一致 |
| 量化 | SQ / PQ / Binary |
| 多租户 | 单 collection / 多 collection / 分片隔离 |

**主流选型**：

| 场景 | 推荐 |
|------|------|
| $N < 10^7$ + 已有 PG | pgvector |
| $N < 10^8$ + 单机 | Qdrant |
| $N > 10^8$ + 分布式 | Milvus |
| 不想运维 | Pinecone / Zilliz Cloud |
| 内置混合检索 | Weaviate |
| 原型 / 教学 | Chroma |

**一句话记忆**：向量数据库 = 向量索引 + 存储 + CRUD + 过滤 + 分布式 + 混合检索。$N < 10^7$ 用 pgvector，$N < 10^8$ 用 Qdrant，更大用 Milvus 或托管服务。

---

> *上一篇：[ANN 库：FAISS / Milvus / Qdrant / Weaviate](./ann-libraries) -- 算法库与向量数据库的工程对比。*
> *下一篇：[双塔模型 Two-Tower](./two-tower) -- 稠密召回的主力架构。*
