---
title: 混合搜索 Hybrid Search
slug: hybrid-search
category: 检索与召回
tags: [混合检索, BM25, Dense Retrieval, 融合, RAG, 神经检索]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# 混合搜索 Hybrid Search

> 五层读懂一个词。这次拆的是：**混合搜索（Hybrid Search）**--BM25 字面召回 + Dense 语义召回的融合，现代 RAG 和搜索系统的标配。

---

## L1 · 一句话点破

**BM25 擅长精确匹配和长尾词，Dense Retrieval 擅长语义匹配，两者融合互补**。混合搜索就是两路并行召回，再用 RRF 或加权融合合并结果。这是现代检索系统的默认配置。

---

## L2 · 通俗类比

找一本"讲猫的书"：

- **BM25**（字面匹配）：找书名、目录、正文中含"猫"字的书。强项是精确命中"猫"，但"小猫咪""feline"等同义但不同字的书漏掉。
- **Dense Retrieval**（语义匹配）：理解"猫"的语义，能找到讲"小猫咪""feline""宠物猫"的书，哪怕没"猫"字。强项是语义泛化，但"猫"字精确匹配不如 BM25。

**混合搜索**：两路并行，把两路结果融合。一本讲"小猫咪"的书，BM25 可能漏（没"猫"字），但 Dense 召回（语义近），融合后进入候选。

反过来，"iPhone 15 Pro" 这种精确型号：

- BM25：精确命中"15 Pro"
- Dense：可能因为语义近召回"14 Pro"或"Pro Max"

混合后 BM25 路保精确，Dense 路保语义，互补。

工程实现：两路独立检索 top-k，融合算法合并（RRF 或加权）。简单但有效，几乎所有现代搜索系统标配。

---

## L3 · 正经定义

**混合搜索 (Hybrid Search)**：多路检索并行，融合后返回 top-k。最常见的是稀疏（BM25）+ 稠密（Dense Retrieval）两路。

**架构**：

```
Query
  ├── BM25 召回 → top-k_sparse
  ├── Dense 召回 → top-k_dense
  └── (可选) ColBERT 召回 → top-k_colbert
       ↓
  融合（RRF / 加权 / LTR）
       ↓
  Top-K 最终候选
       ↓
  (可选) Cross-encoder 精排
```

**融合方法**：

**1. RRF (Reciprocal Rank Fusion)**：

$$
\text{score}_{\text{RRF}}(d) = \sum_{i \in \text{channels}} \frac{1}{k + \text{rank}_i(d)}
$$

$k$ 常用 60。无需归一化，对分数尺度不敏感，鲁棒。

**2. 加权融合 (Weighted Fusion)**：

$$
\text{score}_{\text{weighted}}(d) = \sum_{i} w_i \cdot \text{normalize}(s_i(d))
$$

需归一化各路分数，调权重 $w_i$。

**3. 学习排序 (LTR)**：

训练模型融合多路分数，考虑更多特征。详见 ltr 词条。

**伪代码**：

```python
def hybrid_search(query, bm25_index, dense_index, top_k=10, k_rrf=60):
    # 多路召回
    bm25_results = bm25_index.search(query, top_k=top_k * 5)
    dense_results = dense_index.search(query_vec, top_k=top_k * 5)

    # RRF 融合
    scores = defaultdict(float)
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] += 1 / (k_rrf + rank + 1)
    for rank, (doc_id, _) in enumerate(dense_results):
        scores[doc_id] += 1 / (k_rrf + rank + 1)

    # 排序取 top-k
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top_k]
```

**RAG 中的典型用法**：

```python
def rag_hybrid_retrieve(query, k=5):
    # 1. 多路召回
    bm25_results = bm25_search(query, top_k=20)
    dense_results = dense_search(query, top_k=20)

    # 2. RRF 融合
    fused = rrf_fuse([bm25_results, dense_results])

    # 3. (可选) Cross-encoder 精排
    reranked = cross_encoder.rerank(query, fused[:20])

    # 4. 取 top-k 喂 LLM
    return reranked[:k]
```

---

## L4 · 原理深挖

### 4.1 为什么 BM25 和 Dense 互补

**BM25 强项**：

- **精确匹配**：专有名词、ID、代码、产品型号
- **长尾词**：训练时没见过的词，BM25 不受影响
- **可解释**：分数可追溯到具体词命中
- **零样本**：无训练数据即可用
- **低延迟**：CPU 毫秒级

**Dense Retrieval 强项**：

- **语义匹配**：同义词、近义词、跨语言
- **泛化能力**：未见过的表达也能匹配
- **隐式理解**：意图、上下文
- **多模态**：文本、图像、音频统一向量空间

**互补场景**：

| 查询类型 | BM25 | Dense | 谁强 |
|----------|------|-------|------|
| "iPhone 15 Pro" | 精确命中 | 可能误召 14 Pro | BM25 |
| "想买手机" | 漏"智能电话" | 召回"智能电话" | Dense |
| "RLHF 是什么" | 精确命中 RLHF | 同义召回"基于人类反馈的强化学习" | 互补 |
| "GPT-4 的训练成本" | 精确命中 GPT-4 | 语义召回"大模型训练费用" | 互补 |
| 错别字查询 | 漏召回 | 可能容错召回 | Dense |

工程结论：**单路都有盲区，混合后覆盖率显著提升**。业界实测，混合检索的 recall@10 比 BM25 单路高 10~20%，比 Dense 单路高 5~10%。

### 4.2 RRF 为什么鲁棒

RRF 的设计哲学：**只看排名，不看分数**。

**优势**：

- **尺度无关**：BM25 分数 [0, 50+]，Dense 分数 [0, 1]，直接加权需归一化。RRF 只用排名，天然免归一化。
- **异常值鲁棒**：某路模型某次给极高分数，不影响 RRF（排名不变）。
- **简单**：只需调 $k$ 一个参数，$k=60$ 几乎通用。

**RRF 的直觉**：

- 排名第 1 的文档贡献 $1/61 \approx 0.0164$
- 排名第 10 的文档贡献 $1/70 \approx 0.0143$
- 排名第 100 的文档贡献 $1/160 \approx 0.0063$

排名越高贡献越大，但递减平缓，不会让 top-1 主导。

**$k$ 的选择**：

- $k$ 小：top 排名贡献大，分布尖锐（更看重 top）
- $k$ 大：分布平缓（更平等对待所有排名）
- $k = 60$ 是经验值，源自 [Cormack et al. 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

### 4.3 加权融合的归一化

加权融合 $\sum_i w_i \cdot \text{normalize}(s_i(d))$ 需归一化各路分数：

**Min-Max 归一化**：

$$
\text{normalize}(s) = \frac{s - \min}{\max - \min}
$$

**Z-score 归一化**：

$$
\text{normalize}(s) = \frac{s - \mu}{\sigma}
$$

**Softmax 归一化**：

$$
\text{normalize}(s_i) = \frac{\exp(s_i / \tau)}{\sum_j \exp(s_j / \tau)}
$$

**实践**：Min-Max 最常用。归一化后各路分数尺度一致，可加权。

**权重 $w_i$ 的调优**：

- $w_{\text{BM25}} + w_{\text{Dense}} = 1$
- 默认 0.5 / 0.5
- 精确匹配场景调高 BM25 权重（如代码、产品型号）
- 语义匹配场景调高 Dense 权重（如问答、长尾查询）

**权重学习的困难**：

- 不同查询最优权重不同
- 需评估集调参
- 可用 LTR 学习动态权重

### 4.4 混合检索的工程实现

**方案 1：独立系统 + 外部融合**：

- BM25 用 Elasticsearch / Lucene
- Dense 用 FAISS / Milvus
- 应用层读两路结果，融合

优点：灵活，各路独立优化。缺点：维护两套系统，延迟两倍。

**方案 2：统一向量数据库**：

- Weaviate、Qdrant、Milvus 等内置混合检索
- 一个 API 调用完成两路 + 融合

优点：简单，延迟低。缺点：灵活性受限。

**方案 3：Elasticsearch 8+ kNN**：

- ES 同时支持 BM25 和 kNN 向量检索
- 用 `bool query` 组合

```json
{
  "query": {
    "bool": {
      "should": [
        { "match": { "text": "query text" } },
        { "knn": { "query_vector": [...], "field": "embedding", "num_candidates": 100 } }
      ]
    }
  }
}
```

优点：已有 ES 栈无缝集成。缺点：ES kNN 性能不如专业向量库。

### 4.5 多路混合：不止 BM25 + Dense

现代系统可能多路混合：

**典型多路**：

- BM25（字面）
- Dense Retrieval（语义）
- ColBERT（late interaction，精度高）
- 用户行为（点击、收藏）
- 业务规则（最新、热门）

**多路融合的挑战**：

- 各路分数尺度差异大
- 各路 recall 不同
- 计算资源消耗大

**实践**：3~4 路是上限。再多则复杂度上升收益递减。BM25 + Dense 是核心，ColBERT 是可选补充。

### 4.6 混合检索 vs 稀疏神经检索

另一种"混合"思路：**用神经模型生成稀疏向量**，在同一索引中融合字面和语义。

**SPLADE (Formal et al. 2021)**：

- 用 BERT 生成稀疏向量（词项 + 权重）
- 仍用倒排索引检索
- 自带语义扩展（如"猫"扩展到"宠物、feline"）

**对比**：

| 方法 | 索引 | 语义 | 精确匹配 |
|------|------|------|----------|
| BM25 | 倒排 | 弱 | 强 |
| SPLADE | 倒排 | 中 | 强 |
| Dense | 向量 | 强 | 中 |
| BM25 + Dense | 双索引 | 强 | 强 |

SPLADE 在某些数据集上接近混合检索效果，且只需一套倒排索引。但生态不如混合检索成熟。

### 4.7 RAG 中的混合检索

RAG 是混合检索的最大应用场景：

**典型 RAG 检索流程**：

```
用户查询
  ↓
1. BM25 召回 top-20（精确匹配 chunks）
2. Dense 召回 top-20（语义匹配 chunks）
  ↓
3. RRF 融合 → top-20 候选
  ↓
4. (可选) Cross-encoder 精排 → top-5
  ↓
5. 喂给 LLM 生成回答
```

**为什么 RAG 必须混合**：

- 知识库中可能有专有名词、代码、表格，Dense 容易漏
- 用户查询可能用同义词，BM25 漏
- 混合后覆盖率显著提升

**RAG 混合检索的实测**：

- 纯 BM25：答案准确率 60~70%
- 纯 Dense：答案准确率 70~80%
- 混合：答案准确率 80~90%

提升 10~20 个百分点，是 RAG 系统的"免费午餐"。

### 4.8 混合检索的评估

**评估指标**：

- Recall@K（top-K 含相关文档比例）
- MRR@K（第一个相关文档排名）
- NDCG@K（排序质量）

**评估方法**：

1. 准备评估集（query + 相关文档标注）
2. 分别评估 BM25、Dense、Hybrid
3. 对比 Recall@10、NDCG@10

**典型结果**：

| 方法 | Recall@10 | NDCG@10 |
|------|-----------|---------|
| BM25 | 0.65 | 0.55 |
| Dense | 0.75 | 0.65 |
| Hybrid (RRF) | 0.85 | 0.75 |
| Hybrid + Cross-encoder | 0.90 | 0.85 |

Hybrid 比单路提升 10~20%，加 Cross-encoder 再提升 5~10%。

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **1990s**：搜索引擎以 BM25 单路为主
- **2010s**：Dense Retrieval（DSSM 等）兴起，但性能不如 BM25
- **2020**：DPR 论文证明 Dense 在 NQ 数据集上超越 BM25，但两者互补
- **2021**：业界开始普及混合检索（BM25 + Dense）
- **2022**：Weaviate、Qdrant 等内置混合检索
- **2023**：RAG 爆发，混合检索成为标配
- **2024**：稀疏神经检索（SPLADE、BGE-M3 稀疏）作为混合检索的替代方案兴起

### 5.2 工程常见坑

**坑 1：两路 K 不一致**。

BM25 召回 top-100，Dense 召回 top-20，融合时 Dense 路被淹没。两路 K 要相近（如都 100 或都 50）。

**坑 2：加权融合没归一化**。

BM25 分数 [0, 50+]，Dense 分数 [0, 1]，直接加权 BM25 主导。必须归一化或用 RRF。

**坑 3：用 RRF 时 $k$ 设错**。

$k=1$ 时排名差异剧烈，$k=1000$ 时几乎平等。$k=60$ 是经验值，多数场景适用。

**坑 4：忘了查询向量的归一化**。

Dense 路的 embedding 必须 L2 normalize，否则 IP 不等价于余弦，结果偏。

**坑 5：维护两套索引同步问题**。

BM25 索引（Elasticsearch）和 Dense 索引（Milvus）独立维护，文档增删时可能不同步。要么用统一系统（Weaviate 等），要么严格同步更新流程。

**坑 6：评估只看融合后**。

只看 Hybrid 的 Recall，不知道是 BM25 还是 Dense 贡献。要分别评估各路 + 融合，定位瓶颈。

**坑 7：用通用 embedding 模型在垂直领域**。

通用 BGE / E5 在法律、医疗、代码等领域可能不如 BM25。垂直领域要微调或用领域 embedding。

**坑 8：忘了 Cross-encoder 精排**。

混合检索召回好但排序精度有限，加 Cross-encoder 精排能再提升 5~10%。

**坑 9：权重不调**。

默认 0.5/0.5 不一定最优。精确匹配场景调高 BM25（如 0.7/0.3），语义场景调高 Dense（如 0.3/0.7）。用评估集调参。

**坑 10：忽略延迟**。

两路并行检索，延迟 = max(两路延迟)，不是相加。但如果两路串行（先 BM25 再 Dense），延迟叠加。务必并行。

### 5.3 主流混合检索系统

**Weaviate**：

- 内置 BM25 + Dense 混合
- `alpha` 参数调权重（0=纯 BM25，1=纯 Dense）
- 单 API 调用完成

**Qdrant**：

- 1.10+ 支持稀疏 + 稠密向量
- 同一 collection 存两种向量
- 查询时融合

**Milvus 2.4+**：

- 支持稀疏向量字段（如 BM25、SPLADE）
- 与稠密向量混合检索
- 内置 RRF 融合

**Elasticsearch 8+**：

- BM25 + kNN 向量检索
- 用 `bool query` 组合
- 适合已有 ES 栈

**Pinecone**：

- 稀疏 + 稠密向量
- 内置融合

### 5.4 混合检索的变体

**稀疏神经检索**（SPLADE、BGE-M3 稀疏）：

- 用神经模型生成稀疏向量
- 同一倒排索引中融合字面和语义
- 单索引，比双索引简单

**多向量检索**（ColBERT）：

- per-token 向量 + late interaction
- 精度高于 Dense 单向量
- 可作为混合检索的一路

**多模态混合**：

- 文本 BM25 + 文本 Dense + 图像 Dense
- 跨模态检索
- CLIP 等模型支持

### 5.5 何时不用混合检索

虽然混合检索是主流，但这些场景可能不需要：

1. **纯语义问答**：查询和文档都是自然语言，语义匹配为主，Dense 单路够（如 FAQ 问答）
2. **精确 ID 检索**：用户查 ID、产品型号，BM25 单路够
3. **极低延迟场景**：两路延迟叠加，< 10ms 场景可能用单路
4. **小规模库**：< 1000 文档，单路 brute force 即可
5. **资源受限**：维护两套索引成本高，单路够用

但 90% 的现代搜索和 RAG 场景，混合检索都是更优选择。

---

## 速记卡

| 维度 | 混合搜索 |
|------|----------|
| 架构 | 多路召回 + 融合 |
| 主流组合 | BM25 + Dense Retrieval |
| 融合方法 | RRF / 加权 / LTR |
| RRF 公式 | $\sum_i 1 / (k + \text{rank}_i)$，$k=60$ |
| 优势 | 互补精确匹配与语义匹配 |
| 典型提升 | Recall@10 比 BM25 +10~20%，比 Dense +5~10% |
| 必备组件 | RAG 系统标配 |

**核心融合方法**：

| 方法 | 公式 | 优势 | 劣势 |
|------|------|------|------|
| RRF | $\sum 1/(k + \text{rank})$ | 无需归一化、鲁棒 | 不利用分数差异 |
| 加权 | $\sum w_i \cdot \text{norm}(s_i)$ | 利用分数、可调权 | 需归一化、调参 |
| LTR | 学习模型融合 | 最优 | 需训练数据、复杂 |

**一句话记忆**：混合搜索 = BM25（字面）+ Dense（语义）两路召回 + RRF/加权融合。BM25 强精确和长尾，Dense 强语义和泛化，互补提升 Recall@10 约 10~20%。RAG 系统标配，主流向量数据库内置。

---

> *上一篇：[Top-K 检索](./top-k) -- K 的选择与权衡。*
> *下一篇：[RRF 倒数排名融合](./rrf) -- 最常用的混合检索融合算法。*
