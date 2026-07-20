---
title: RRF 倒数排名融合
slug: rrf
category: 检索与召回
tags: [融合算法, RRF, 混合检索, 排序融合, 检索]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# RRF 倒数排名融合

> 五层读懂一个词。这次拆的是：**RRF (Reciprocal Rank Fusion)**--最常用的多路检索融合算法，简单鲁棒，无需归一化。

---

## L1 · 一句话点破

**RRF 把每路检索的文档排名取倒数累加，第 1 名贡献 $1/(k+1)$，第 $r$ 名贡献 $1/(k+r)$，按累加分数排序**。只看排名不看分数，天然避免各路分数尺度不一致问题，一个参数 $k=60$ 几乎通用。

---

## L2 · 通俗类比

两个美食评论家推荐餐厅：

- 评论家 A（米其林标准）：第 1 名"小馆 A"，第 2 名"小馆 B"，...
- 评论家 B（大众点评标准）：第 1 名"小馆 C"，第 2 名"小馆 A"，...

怎么融合两份榜单？

**加权融合的问题**：米其林分数 1~3 星，大众点评分数 0~5 分，尺度不同，加权谁主谁次难定。

**RRF 的做法**：只看排名，不看分数。

- "小馆 A"：A 第 1 + B 第 2 = $1/61 + 1/62 \approx 0.0164 + 0.0161 = 0.0325$
- "小馆 B"：A 第 2 + B 不在前 = $1/62 + 0 \approx 0.0161$
- "小馆 C"：A 不在前 + B 第 1 = $0 + 1/61 \approx 0.0164$

"小馆 A" 在两份榜单都靠前，累加分数最高，融合后第 1。

**直觉**：一个东西被多路都排在前面，它就是真的好。RRF 是这种直觉的最简实现。

---

## L3 · 正经定义

**RRF (Reciprocal Rank Fusion)**：对多路检索结果，每路返回 ranked list，文档 $d$ 的 RRF 分数为：

$$
\text{RRF}(d) = \sum_{i=1}^{M} \frac{1}{k + \text{rank}_i(d)}
$$

其中：

- $M$：检索路数（如 BM25 + Dense 两路，$M=2$）
- $\text{rank}_i(d)$：文档 $d$ 在第 $i$ 路的排名（1-indexed）
- $k$：平滑常数，控制 top 排名与长尾的权重比，常用 60

**如果文档 $d$ 不在某路的 top-N**：$\text{rank}_i(d) = \infty$，贡献为 0。

**伪代码**：

```python
from collections import defaultdict

def rrf_fuse(ranked_lists, k=60, top_k=10):
    """
    ranked_lists: list of lists, 每路是 [(doc_id, score), ...] 已按分数降序
    返回融合后的 top_k
    """
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top_k]

# 示例
bm25_results = [("doc1", 12.5), ("doc3", 10.2), ("doc2", 8.7)]
dense_results = [("doc2", 0.95), ("doc1", 0.92), ("doc4", 0.88)]

fused = rrf_fuse([bm25_results, dense_results], k=60, top_k=3)
# doc1: 1/61 + 1/62 = 0.0325
# doc2: 1/63 + 1/61 = 0.0323
# doc3: 1/62 + 0   = 0.0161
# doc4: 0    + 1/63 = 0.0159
# 结果: [("doc1", 0.0325), ("doc2", 0.0323), ("doc3", 0.0161)]
```

**关键性质**：

- **尺度无关**：只用排名，不用分数，无需归一化
- **平滑**：$k$ 防止 top-1 主导（如 $k=0$ 时第 1 名贡献 $1/1 = 1$，远超第 2 名 $1/2$）
- **长尾友好**：第 100 名仍有贡献 $1/160 \approx 0.006$
- **简单**：一个参数 $k$，几乎不需调

---

## L4 · 原理深挖

### 4.1 为什么 RRF 比加权融合鲁棒

**加权融合的痛点**：

- BM25 分数范围 [0, 50+]
- Dense 分数范围 [0, 1]
- 直接加权 BM25 主导，需归一化
- 归一化方法（min-max、z-score、softmax）各有 bias
- 异常值（如某次 BM25 给极高分数）影响大

**RRF 的鲁棒性**：

- 只用排名，分数尺度不影响
- 异常值不影响（再高分数也只是第 1 名）
- 无需归一化，实现简单

**实测**：RRF 在多数场景下与精心调参的加权融合持平甚至更好，且无需调参。这是 RRF 成为默认选择的原因。

### 4.2 $k$ 参数的意义

$k$ 控制 top 排名与长尾的权重比：

**$k=0$**：

- 第 1 名贡献 $1/1 = 1$
- 第 2 名贡献 $1/2 = 0.5$
- 第 10 名贡献 $1/10 = 0.1$
- 分布极陡，top-1 主导

**$k=60$**（默认）：

- 第 1 名贡献 $1/61 \approx 0.0164$
- 第 2 名贡献 $1/62 \approx 0.0161$
- 第 10 名贡献 $1/70 \approx 0.0143$
- 第 100 名贡献 $1/160 \approx 0.0063$
- 分布平缓，top 排名略重但不主导

**$k=\infty$**：

- 所有排名贡献近似相等
- 退化为"被几路召回"的计数

**$k$ 的选择经验**：

- $k=60$：经验默认，源自 [Cormack et al. 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)，多数场景适用
- $k=30$：更看重 top 排名
- $k=100$：更平等对待
- 实践：调 $k$ 的影响通常 < 1%，用 60 即可

### 4.3 RRF 的理论解释

RRF 可以从贝叶斯角度解释。假设：

- 每路检索是对文档相关性的独立观察
- 排名为 $r$ 的文档相关性后验概率 $\propto 1/(k+r)$

这等价于一个"排名先验"，假设排名越前越可能相关，但避免 top-1 主导。

**与 Condorcet 投票的关系**：RRF 类似多投票者按排名投票，被多路排在前的文档"票数"高。区别是 RRF 用倒数加权（连续），Condorcet 用二元投票。

**与 Borda 计数的关系**：Borda 计数给第 $r$ 名 $N-r$ 分，RRF 给 $1/(k+r)$ 分。RRF 的倒数形式让 top 排名差距更显著（第 1 vs 第 2 差大），长尾贡献小但非零。

### 4.4 RRF vs 加权融合 vs LTR

| 方法 | 公式 | 优势 | 劣势 |
|------|------|------|------|
| RRF | $\sum 1/(k + \text{rank})$ | 无需归一化、鲁棒、简单 | 不利用分数差异 |
| 加权融合 | $\sum w_i \cdot \text{norm}(s_i)$ | 利用分数、可调权 | 需归一化、调参 |
| LTR | 学习模型融合 | 最优、可加特征 | 需训练数据、复杂 |

**何时用 RRF**：

- 多路分数尺度差异大
- 无评估集调参
- 快速原型
- 多路 recall 接近时（不需精细调权）

**何时用加权融合**：

- 各路分数已归一化
- 有评估集调权重
- 某路明显更强（如 Dense >> BM25）
- 需要利用分数差异

**何时用 LTR**：

- 有大量训练数据
- 需要加入更多特征（点击率、用户行为等）
- 追求极致精度
- 工程能力强

**实践**：从 RRF 起步，评估不足时切换到加权融合，数据充足时升级到 LTR。

### 4.5 RRF 的变体

**Weighted RRF**：给每路不同权重

$$
\text{WRRF}(d) = \sum_{i=1}^{M} \frac{w_i}{k + \text{rank}_i(d)}
$$

$w_i$ 反映路 $i$ 的可信度。比 RRF 灵活，比加权融合鲁棒。

**RRF with Cutoff**：只融合每路 top-N，忽略长尾

```python
def rrf_fuse_with_cutoff(ranked_lists, k=60, cutoff=100, top_k=10):
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked[:cutoff], start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
```

避免长尾文档（排名 1000+）仍贡献分数，节省计算。

**Normalized RRF**：归一化 RRF 分数到 [0, 1]

$$
\text{NRRF}(d) = \frac{\text{RRF}(d)}{\max_{d'} \text{RRF}(d')}
$$

便于跨查询比较和阈值过滤。

### 4.6 RRF 在主流系统中的实现

**Elasticsearch 8+**：

```json
{
  "query": {
    "match": { "text": "query" }
  },
  "knn": {
    "field": "embedding",
    "query_vector": [...],
    "num_candidates": 100
  },
  "rank": {
    "rrf": { "window_size": 100, "rank_constant": 60 }
  }
}
```

`window_size` 是每路参与融合的文档数，`rank_constant` 是 $k$。

**Weaviate**：

```python
result = client.query.get("Document", ["text"]) \
    .with_hybrid(query="query", alpha=0.5) \
    .with_limit(10) \
    .do()
```

`alpha` 调 BM25/Dense 权重，内部融合算法可配（默认加权，可切 RRF）。

**Qdrant**：

```python
result = client.search(
    collection_name="docs",
    query=query_dense_vec,
    sparse_query=query_sparse_vec,
    fusion=Fusion.RRF,  # 或 Fusion.DBSF
    limit=10,
)
```

1.10+ 支持稀疏 + 稠密 + RRF 融合。

**Milvus 2.4+**：

```python
result = client.hybrid_search(
    reqs=[dense_req, sparse_req],
    fusion_type=RRFRanker(k=60),
    limit=10,
)
```

内置 RRF 和加权融合。

### 4.7 RRF 的局限

**局限 1：不利用分数差异**。

两路都把文档 A 排第 1，但一路给 0.99（极相关），一路给 0.51（勉强相关）。RRF 视为同等贡献，丢失分数信息。

**局限 2：长尾文档仍贡献分数**。

排名 1000 的文档贡献 $1/1060 \approx 0.0009$，虽小但累加后可能影响。需配 cutoff 限制。

**局限 3：对路数敏感**。

路数多时长尾文档被多路召回的概率高，RRF 分数被稀释。3~4 路是甜蜜点，更多路效果递减。

**局限 4：不区分相关性强度**。

RRF 假设每路的排名等价，但实际某路可能更可信。需用 Weighted RRF。

**局限 5：不学习**。

RRF 是固定公式，不学习查询特征、文档特征。LTR 能学但需训练数据。

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **2009**：Cormack, Clarke, Buett 等在 SIGIR 论文 "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" 正式提出 RRF
- **2009~2018**：RRF 在 TREC 评测中表现优异，但工业采用有限（BM25 单路为主）
- **2020**：Dense Retrieval 兴起，BM25 + Dense 两路需求催生 RRF 工业化
- **2022**：Elasticsearch 8.0 集成 RRF；Weaviate、Qdrant 内置混合检索
- **2023**：RAG 爆发，RRF 成为混合检索默认融合算法
- **2024**：主流向量数据库全部支持 RRF；RRF 成为混合检索事实标准

### 5.2 使用常见坑

**坑 1：忘了设 cutoff**。

每路 top-10000 都参与融合，长尾文档贡献累加，可能让 top 排名文档被稀释。建议 cutoff = 100~500。

**坑 2：路数太多**。

5 路以上 RRF 效果递减，且计算成本上升。3~4 路是甜蜜点。

**坑 3：各路 K 不一致**。

BM25 召回 top-50，Dense 召回 top-200，融合时 Dense 路长尾文档贡献累加。各路 K 要相近。

**坑 4：$k$ 调错**。

$k=0$ 让 top-1 主导，$k=1000$ 让所有排名近似相等。默认 60 几乎通用，非必要不调。

**坑 5：评估只看融合后**。

只看 RRF 的 NDCG@10，不知道是哪路贡献。要分别评估各路 + RRF，定位瓶颈。

**坑 6：用 RRF 但各路分数尺度本来一致**。

如果各路分数尺度一致（如都用归一化余弦），加权融合可能优于 RRF（利用分数差异）。RRF 适合尺度不一致场景。

**坑 7：RRF 后不精排**。

RRF 召回好但排序精度有限，加 Cross-encoder 精排能再提升 5~10%。

**坑 8：稀疏 + 稠密向量用 RRF 时分数尺度问题**。

SPLADE 等稀疏神经检索的分数尺度与 Dense 不同，RRF 是合适选择。但若两路都归一化得好，加权融合也可。

**坑 9：忘了排名 1-indexed**。

实现时排名从 0 还是 1 开始影响分数。Cormack 原论文是 1-indexed，$k=60$ 时第 1 名贡献 $1/61$。

**坑 10：RRF 用在单路**。

RRF 是多路融合算法，单路检索用 RRF 没意义（退化为 $1/(k+\text{rank})$，等价于按原排名）。

### 5.3 RRF 的适用场景

**适合用**：

- BM25 + Dense 混合检索
- 多路召回融合（3~4 路）
- 分数尺度不一致
- 无评估集调参
- 快速原型

**不适合**：

- 单路检索
- 路数 > 5
- 各路分数尺度一致 + 有评估集（加权融合更好）
- 需要利用分数差异
- 需要加入更多特征（用 LTR）

### 5.4 RRF 与其他融合方法对比

| 场景 | RRF | 加权融合 | LTR |
|------|-----|----------|-----|
| BM25 + Dense（尺度不一致） | 推荐 | 需归一化 | 过度 |
| 多路（3~4 路） | 推荐 | 可调 | 适合 |
| 有大量训练数据 | 一般 | 一般 | 推荐 |
| 快速原型 | 推荐 | 需调参 | 不适合 |
| 极致精度 | 不够 | 不够 | 推荐 |
| 跨查询可比分数 | 不行 | 可行（归一化） | 可行 |

### 5.5 RRF 的现代演进

**Learned RRF**：用神经网络学习 $k$ 参数（甚至每路不同 $k$），小幅提升但复杂。

**Neural Fusion**：用 cross-encoder 直接融合多路结果，精度最高但成本高。

**Adaptive Fusion**：根据查询类型动态选择融合方法（如简单查询用 RRF，复杂查询用 LTR）。

**Sparse-Dense Native Fusion**：BGE-M3 等模型同时输出稀疏 + 稠密向量，训练时已联合优化，融合效果更好。

---

## 速记卡

| 维度 | RRF |
|------|-----|
| 公式 | $\sum_i 1/(k + \text{rank}_i)$ |
| 默认 $k$ | 60 |
| 优势 | 无需归一化、鲁棒、简单 |
| 劣势 | 不利用分数差异 |
| 适用 | 多路融合、尺度不一致 |
| 不适用 | 单路、需利用分数 |
| 替代 | 加权融合、LTR |

**核心公式**：

$$
\text{RRF}(d) = \sum_{i=1}^{M} \frac{1}{k + \text{rank}_i(d)}, \quad k=60
$$

**典型数值**（$k=60$）：

| 排名 | 单路贡献 |
|------|----------|
| 1 | 0.0164 |
| 10 | 0.0143 |
| 100 | 0.0063 |
| 1000 | 0.0009 |

**一句话记忆**：RRF = $\sum 1/(k + \text{rank})$，只看排名不看分数，无需归一化，$k=60$ 几乎通用。简单鲁棒，是混合检索融合的事实标准。

---

> *上一篇：[混合搜索 Hybrid Search](./hybrid-search) -- 多路召回融合的工程架构。*
> *下一篇：[加权重排 Weighted Fusion](./weighted-fusion) -- RRF 的替代方案，利用分数差异。*
