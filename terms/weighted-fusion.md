---
title: 加权重排 Weighted Fusion
slug: weighted-fusion
category: 检索与召回
tags: [融合算法, 加权融合, 混合检索, 归一化, 检索]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# 加权重排 Weighted Fusion

> 五层读懂一个词。这次拆的是：**Weighted Fusion（加权融合）**--RRF 的替代方案，利用分数差异，需归一化和调参。

---

## L1 · 一句话点破

**加权融合把各路检索的分数归一化后按权重线性组合，$\sum w_i \cdot \text{norm}(s_i)$**。比 RRF 更精细（利用分数差异），但需归一化和调权重，鲁棒性不如 RRF。

---

## L2 · 通俗类比

继续用美食评论家的比喻：

- **RRF**：只看两份榜单的排名，不看星级/分数
- **加权融合**：看星级/分数，但要把米其林 1~3 星和大众点评 0~5 分归一化到同一尺度，再加权

具体步骤：

1. **归一化**：米其林 3 星 -> 1.0，2 星 -> 0.67，1 星 -> 0.33；大众点评 5 分 -> 1.0，4 分 -> 0.8，...
2. **加权**：米其林权重 0.6（更权威），大众点评 0.4（更接地气）
3. **组合**：餐厅 A 米其林 3 星 + 大众 4 分 -> $0.6 \times 1.0 + 0.4 \times 0.8 = 0.92$

加权融合利用了分数差异（3 星比 1 星差距大），但需要归一化（不然尺度不一）和调权重（米其林权重大还是大众点评权重大）。

什么时候用加权融合而不是 RRF？

- 各路分数尺度一致或可归一化
- 有评估集调权重
- 想利用分数差异
- 某路明显更强（如 Dense >> BM25）

---

## L3 · 正经定义

**加权融合 (Weighted Fusion / Weighted Score Fusion)**：对多路检索结果，每路返回 $(d, s_i(d))$，文档 $d$ 的融合分数为：

$$
\text{score}_{\text{weighted}}(d) = \sum_{i=1}^{M} w_i \cdot \text{normalize}(s_i(d))
$$

其中：

- $M$：检索路数
- $s_i(d)$：第 $i$ 路对文档 $d$ 的原始分数
- $\text{normalize}(\cdot)$：归一化函数
- $w_i$：第 $i$ 路的权重，$\sum w_i = 1$

**归一化方法**：

**1. Min-Max 归一化**：

$$
\text{norm}_{\text{mm}}(s) = \frac{s - \min_{d \in C_i} s(d)}{\max_{d \in C_i} s(d) - \min_{d \in C_i} s(d)}
$$

$C_i$ 是第 $i$ 路的候选集。归一化到 [0, 1]。

**2. Z-score 归一化**：

$$
\text{norm}_{\text{z}}(s) = \frac{s - \mu_i}{\sigma_i}
$$

$\mu_i, \sigma_i$ 是第 $i$ 路分数的均值和标准差。归一化到均值为 0、方差为 1，但范围不固定。

**3. Softmax 归一化**：

$$
\text{norm}_{\text{sm}}(s_i(d)) = \frac{\exp(s_i(d) / \tau)}{\sum_{d' \in C_i} \exp(s_i(d') / \tau)}
$$

$\tau$ 是温度。归一化后所有文档分数和为 1，分布形状可调。

**4. Max 归一化**：

$$
\text{norm}_{\text{max}}(s) = \frac{s}{\max_{d \in C_i} s(d)}
$$

最简单，归一化到 [0, 1]，但 top 分数总为 1，丢失 top 间的相对差异。

**伪代码**：

```python
def weighted_fuse(scored_lists, weights, method='minmax'):
    """
    scored_lists: list of [(doc_id, score), ...] 每路
    weights: list of floats, 权重
    """
    assert abs(sum(weights) - 1.0) < 1e-6, "权重和必须为 1"

    # 1. 归一化每路分数
    normed_lists = []
    for scored in scored_lists:
        scores = [s for _, s in scored]
        if method == 'minmax':
            lo, hi = min(scores), max(scores)
            normed = [(d, (s - lo) / (hi - lo) if hi > lo else 1.0) for d, s in scored]
        elif method == 'max':
            mx = max(scores)
            normed = [(d, s / mx) if mx > 0 else (d, 0) for d, s in scored]
        normed_lists.append(normed)

    # 2. 加权累加
    fused = defaultdict(float)
    for w, normed in zip(weights, normed_lists):
        for doc_id, s in normed:
            fused[doc_id] += w * s

    return sorted(fused.items(), key=lambda x: -x[1])

# 示例
bm25_results = [("doc1", 12.5), ("doc3", 10.2), ("doc2", 8.7)]
dense_results = [("doc2", 0.95), ("doc1", 0.92), ("doc4", 0.88)]
weights = [0.4, 0.6]  # BM25 0.4, Dense 0.6

fused = weighted_fuse([bm25_results, dense_results], weights, method='minmax')
# BM25 归一化: doc1=1.0, doc3=0.5, doc2=0
# Dense 归一化: doc2=1.0, doc1=0.67, doc4=0
# 加权: doc1=0.4*1.0+0.6*0.67=0.80, doc2=0.4*0+0.6*1.0=0.60, doc3=0.4*0.5+0=0.20
# 结果: [("doc1", 0.80), ("doc2", 0.60), ("doc3", 0.20)]
```

---

## L4 · 原理深挖

### 4.1 为什么需要归一化

各路检索的分数尺度差异巨大：

- **BM25**：[0, 50+]，与文档长度、查询长度相关
- **Dense (cosine)**：[-1, 1] 或 [0, 1]（归一化后）
- **Dense (IP)**：[0, +∞)，未归一化时与向量模长相关
- **Cross-encoder**：[0, 1]（sigmoid 后）或任意值（logits）
- **ColBERT (max-sim)**：[0, L_q]，与查询长度相关

直接加权会让尺度大的路主导。归一化让各路分数到同一尺度 [0, 1]，加权才有意义。

**归一化的难点**：

- 不同查询的 BM25 分数范围差异大（短查询 vs 长查询）
- 不同查询的 Dense 分数分布不同（明确意图 vs 模糊意图）
- 归一化丢失绝对分数信息（如 BM25 高分意味着强字面匹配）

### 4.2 归一化方法的对比

**Min-Max**：

- 优点：归一化到 [0, 1]，直观
- 缺点：对异常值敏感（一个极高分数压扁其他）
- 适用：分数分布稳定

**Z-score**：

- 优点：考虑分布形状，鲁棒
- 缺点：范围不固定，可能负值
- 适用：分数分布有偏

**Softmax**：

- 优点：分布形状可调（温度 $\tau$）
- 缺点：计算复杂，需调 $\tau$
- 适用：需要强调 top 排名

**Max**：

- 优点：最简单
- 缺点：top 分数总为 1，丢失 top 间差异
- 适用：粗略归一化

**实践**：Min-Max 最常用。若分数分布有异常值，用 Z-score。Softmax 在需要突出 top 时用。

### 4.3 权重的调优

权重 $w_i$ 决定各路的相对重要性：

**默认值**：$w_i = 1/M$（等权）

**调优方法**：

**1. 网格搜索**：

```python
best_score, best_w = 0, None
for w_bm25 in np.arange(0, 1.01, 0.1):
    w_dense = 1 - w_bm25
    score = evaluate([w_bm25, w_dense])  # 评估 NDCG@10
    if score > best_score:
        best_score, best_w = score, (w_bm25, w_dense)
```

**2. 贝叶斯优化**：用 Optuna 等工具，更高效。

**3. 学习排序 (LTR)**：把权重作为模型参数学习。

**经验值**：

- 通用场景：BM25 0.3~0.5，Dense 0.5~0.7
- 精确匹配场景（代码、产品）：BM25 0.6~0.8
- 语义匹配场景（问答、长尾）：Dense 0.7~0.9
- 多路（BM25 + Dense + ColBERT）：权重按精度排序，ColBERT 通常最高

### 4.4 加权融合 vs RRF

| 维度 | RRF | 加权融合 |
|------|-----|----------|
| 利用的信息 | 仅排名 | 排名 + 分数 |
| 归一化需求 | 无 | 必须 |
| 参数 | $k$（默认 60） | 权重 $w_i$ + 归一化方法 |
| 鲁棒性 | 强 | 中（异常值敏感） |
| 调参成本 | 几乎无 | 需评估集 |
| 适用场景 | 尺度不一致、快速原型 | 尺度一致、有评估集 |
| 精度上限 | 中 | 高（调好参时） |

**实践选择**：

- 默认 RRF，简单鲁棒
- 有评估集 + 各路分数尺度可控时，试加权融合
- 多路（> 3）时，加权融合可能更精细
- 极致精度时升级 LTR

### 4.5 加权融合的变体

**1. Normalized Weighted Fusion**：

归一化后加权，再加一次归一化（让最终分数在 [0, 1]）：

$$
\text{score}(d) = \frac{\sum_i w_i \cdot \text{norm}(s_i(d))}{\sum_i w_i}
$$

**2. Geometric Mean Fusion**：

几何平均而非线性加权：

$$
\text{score}(d) = \prod_i \text{norm}(s_i(d))^{w_i}
$$

对"所有路都给高分"的文档更友好（任一路低分会大幅拉低总分）。

**3. RRF with Weights (Weighted RRF)**：

$$
\text{score}(d) = \sum_i \frac{w_i}{k + \text{rank}_i(d)}
$$

RRF 的加权变体，兼具两者优点。

**4. CombSUM / CombMNZ**：

经典信息检索融合方法：

- **CombSUM**：$\sum_i \text{norm}(s_i(d))$
- **CombMNZ**：$\text{CombSUM}(d) \times |\{i : d \in C_i\}|$（被多路召回的文档加权）

### 4.6 加权融合的工程实现

**Elasticsearch 8+**：

```json
{
  "query": {
    "bool": {
      "should": [
        { "match": { "text": "query" } }
      ]
    }
  },
  "knn": {
    "field": "embedding",
    "query_vector": [...],
    "num_candidates": 100,
    "boost": 0.6  // Dense 权重
  },
  "query": {
    "boost": 0.4  // BM25 权重
  }
}
```

ES 的 `boost` 实质是加权融合，但归一化方式是 ES 内部处理。

**Weaviate**：

```python
result = client.query.get("Document", ["text"]) \
    .with_hybrid(query="query", alpha=0.5) \
    .do()
```

`alpha` 调 BM25/Dense 权重，0=纯 BM25，1=纯 Dense。内部是加权融合。

**Qdrant 1.10+**：

```python
result = client.search(
    collection_name="docs",
    query=query_dense_vec,
    sparse_query=query_sparse_vec,
    fusion=Fusion.DBSF,  # Distance Based Score Fusion (加权变体)
    limit=10,
)
```

DBSF 是 Qdrant 的加权融合实现。

**Milvus 2.4+**：

```python
result = client.hybrid_search(
    reqs=[dense_req, sparse_req],
    fusion_type=WeightedRanker(0.4, 0.6),  # 加权融合
    limit=10,
)
```

### 4.7 加权融合的局限

**局限 1：归一化丢失信息**。

归一化让各路分数到 [0, 1]，但丢失了绝对分数信息。BM25 分数 30（强匹配）和 5（弱匹配）归一化后可能都是 1.0（如果 30 是 top 分数）。

**局限 2：异常值敏感**。

某路某次给极高分数，归一化后其他文档分数被压扁。Min-Max 尤其敏感。

**局限 3：权重难调**。

权重 $w_i$ 需评估集调参，不同查询最优权重不同。全局权重是折中，不最优。

**局限 4：不学习查询特征**。

固定权重不区分查询类型。精确匹配查询应调高 BM25 权重，语义查询应调高 Dense 权重。LTR 能学习动态权重。

**局限 5：归一化方法选择主观**。

Min-Max、Z-score、Softmax 各有 bias，选择影响结果。

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **1990s**：信息检索早期就有加权融合（CombSUM、CombMNZ 等）
- **2000s**：Meta-search 引擎用加权融合合并多搜索结果
- **2020**：Dense Retrieval 兴起，BM25 + Dense 加权融合普及
- **2022**：Elasticsearch、Weaviate、Qdrant 内置加权融合
- **2023**：RAG 系统常用加权融合（alpha 参数）
- **2024**：与 RRF 并列为混合检索两大融合方法

### 5.2 使用常见坑

**坑 1：忘了归一化**。

直接加权 BM25（[0, 50+]）和 Dense（[0, 1]），BM25 主导。必须归一化或用 RRF。

**坑 2：归一化用错方法**。

Min-Max 对异常值敏感，Z-score 范围不固定。要根据分数分布选。

**坑 3：权重不调**。

默认 0.5/0.5 不一定最优。用评估集调参，精确匹配场景调高 BM25，语义场景调高 Dense。

**坑 4：各路 K 不一致**。

BM25 召回 top-50，Dense 召回 top-200，归一化时 min/max 基于不同候选集，结果偏。各路 K 要相近。

**坑 5：权重全局固定**。

不同查询最优权重不同。固定权重是折中。可学习动态权重（LTR）或按查询类型分类后用不同权重。

**坑 6：评估只看融合后**。

只看加权融合的 NDCG，不知道哪路贡献。要分别评估各路 + 融合，定位瓶颈。

**坑 7：用加权融合但路数多**。

5 路以上加权融合调参复杂，且各路分数分布差异大时归一化困难。多路用 RRF 更鲁棒。

**坑 8：归一化基于全库而非候选集**。

归一化的 min/max 应基于当前查询的候选集，不是全库。全库归一化让 top 分数偏低。

**坑 9：忘了负分数处理**。

Z-score 归一化后可能有负值，加权后可能负分。要加偏移或用其他归一化。

**坑 10：加权融合后不精排**。

加权融合召回好但排序精度有限，加 Cross-encoder 精排能再提升 5~10%。

### 5.3 加权融合 vs RRF 决策

**用加权融合**：

- 各路分数尺度一致（如都归一化余弦）
- 有评估集调权重
- 想利用分数差异
- 某路明显更强（如 Dense >> BM25）
- 2~3 路

**用 RRF**：

- 各路分数尺度不一致
- 无评估集调参
- 快速原型
- 多路（3~5）
- 鲁棒性优先

**用 LTR**：

- 有大量训练数据
- 需要加入更多特征
- 追求极致精度
- 工程能力强

### 5.4 主流系统的加权融合支持

| 系统 | 加权融合 | 实现 |
|------|----------|------|
| Elasticsearch 8+ | 有 | `boost` 参数 |
| Weaviate | 有 | `alpha` 参数（0=BM25, 1=Dense） |
| Qdrant 1.10+ | 有 | `Fusion.DBSF` |
| Milvus 2.4+ | 有 | `WeightedRanker` |
| Pinecone | 有 | 内置 alpha 参数 |

主流向量数据库都支持加权融合，API 各异但原理相同。

### 5.5 加权融合的现代演进

**学习权重 (Learned Weights)**：

用神经网络学习每路的权重，甚至每查询动态权重。比固定权重更精细。

**Contextual Fusion**：

根据查询上下文（如查询类型、用户历史）动态选择融合方法和权重。

**Neural Fusion**：

用 cross-encoder 直接融合多路结果，精度最高但成本高。

**Sparse-Dense Native Fusion**：

BGE-M3 等模型同时输出稀疏 + 稠密向量，训练时已联合优化，融合效果更好。

---

## 速记卡

| 维度 | 加权融合 |
|------|----------|
| 公式 | $\sum_i w_i \cdot \text{norm}(s_i)$ |
| 关键步骤 | 归一化 + 加权 |
| 归一化方法 | Min-Max / Z-score / Softmax / Max |
| 优势 | 利用分数差异、可调权 |
| 劣势 | 需归一化、调参、异常值敏感 |
| 适用 | 尺度一致、有评估集 |
| 替代 | RRF（更鲁棒）、LTR（更精细） |

**核心公式**：

$$
\text{score}(d) = \sum_{i=1}^{M} w_i \cdot \text{normalize}(s_i(d)), \quad \sum w_i = 1
$$

**归一化方法选择**：

| 方法 | 公式 | 适用 |
|------|------|------|
| Min-Max | $(s - \min) / (\max - \min)$ | 通用 |
| Z-score | $(s - \mu) / \sigma$ | 异常值多 |
| Softmax | $\exp(s/\tau) / \sum \exp$ | 强调 top |
| Max | $s / \max$ | 粗略 |

**一句话记忆**：加权融合 = 归一化各路分数到 [0, 1] + 按权重线性组合。比 RRF 精细（利用分数差异），但需归一化和调权重。各路尺度一致 + 有评估集时用加权融合，否则用 RRF。

---

> *上一篇：[RRF 倒数排名融合](./rrf) -- 最常用的鲁棒融合算法。*
> *下一篇：[学习排序 LTR](./ltr) -- 融合方法的终极形态，用机器学习排序。*
