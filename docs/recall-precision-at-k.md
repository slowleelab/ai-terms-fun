---
title: Recall / Precision@K
slug: recall-precision-at-k
category: 评估与应用
tags: [评估指标, Recall, Precision, 检索, 召回率, 精确率]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# Recall / Precision@K

> 五层读懂一个词。这次拆的是：**Recall@K 和 Precision@K**--检索评估的两大基础指标，互补衡量召回率和精确率。

---

## L1 · 一句话点破

**Recall@K = top-K 中相关文档占所有相关文档的比例（召回了多少），Precision@K = top-K 中相关文档占 top-K 的比例（前排有多准）**。前者关心不漏，后者关心不混，是检索评估的双子指标。

---

## L2 · 通俗类比

继续用考试比喻：

- **Recall@K**：这题有 10 个正确答案，你前 K 个答案里包含了几个？
  - K=5，你答对 4 个 -> Recall@5 = 4/10 = 40%
  - 关心"漏了多少"

- **Precision@K**：你前 K 个答案里，有几个是对的？
  - K=5，你答对 4 个 -> Precision@5 = 4/5 = 80%
  - 关心"答错了多少"

**两者的张力**：

- 想提高 Recall：多答（K 大），但可能多答错的（Precision 降）
- 想提高 Precision：少答精答（K 小），但可能漏（Recall 降）

**极端例子**：

- K=全库，全答：Recall=100%，Precision 极低（混入大量不相关）
- K=1，只答最确定的：Precision 高，Recall 低（漏大量相关）

检索系统的目标：**在可接受的 K 下，Recall 和 Precision 都高**。但通常 trade-off，需根据场景权衡。

---

## L3 · 正经定义

**Recall@K (召回率)**：top-K 中相关文档占所有相关文档的比例。

$$
\text{Recall@K} = \frac{|\text{top-}K \cap \text{relevant}|}{|\text{relevant}|}
$$

**Precision@K (精确率)**：top-K 中相关文档占 top-K 的比例。

$$
\text{Precision@K} = \frac{|\text{top-}K \cap \text{relevant}|}{K}
$$

**多查询平均**：

$$
\text{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{top-}K(q) \cap \text{rel}(q)|}{|\text{rel}(q)|}
$$

$$
\text{Precision@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{top-}K(q) \cap \text{rel}(q)|}{K}
$$

**性质**：

- 范围 [0, 1]，越高越好
- Recall@K 随 K 增大单调不减（top-K 越大，召回越多）
- Precision@K 不一定随 K 单调（top-K 越大，可能混入不相关，Precision 降）
- 当 $|\text{rel}| = K$ 且全部命中时，Recall@K = Precision@K = 1

**伪代码**：

```python
def recall_precision_at_k(results, relevant_docs, k):
    """
    results: dict, query_id -> list of retrieved doc_ids (ranked)
    relevant_docs: dict, query_id -> set of relevant doc_ids
    """
    recalls, precisions = [], []
    for q_id, retrieved in results.items():
        top_k = retrieved[:k]
        rel = relevant_docs[q_id]
        hits = len(set(top_k) & rel)
        recall = hits / len(rel) if rel else 1.0
        precision = hits / k
        recalls.append(recall)
        precisions.append(precision)
    return sum(recalls) / len(recalls), sum(precisions) / len(precisions)

# 示例
results = {
    "q1": ["d1", "d2", "d3", "d4", "d5"],  # top-5
    "q2": ["d6", "d7", "d8", "d9", "d10"],
}
relevant = {
    "q1": {"d2", "d7", "d11"},  # 3 个相关
    "q2": {"d1", "d11"},         # 2 个相关
}
recall, precision = recall_precision_at_k(results, relevant, k=5)
# q1: hits=1 (d2), recall=1/3, precision=1/5
# q2: hits=0, recall=0, precision=0
# 平均: recall=1/6, precision=1/10
```

---

## L4 · 原理深挖

### 4.1 Recall 和 Precision 的关系

**数学关系**：

$$
\text{Recall@K} = \text{Precision@K} \cdot \frac{K}{|\text{rel}|}
$$

- 当 $K = |\text{rel}|$：Recall@K = Precision@K
- 当 $K < |\text{rel}|$：Recall@K < Precision@K（K 小时 precision 高但 recall 低）
- 当 $K > |\text{rel}|$：Recall@K > Precision@K（K 大时 recall 高但 precision 低）

**Trade-off 曲线**：

横轴 K，纵轴指标。Recall@K 随 K 单调上升，Precision@K 通常先升后降。

**F1@K**：调和平均

$$
\text{F1@K} = 2 \cdot \frac{\text{Precision@K} \cdot \text{Recall@K}}{\text{Precision@K} + \text{Recall@K}}
$$

综合 Precision 和 Recall，但检索评估中不常用（K 固定时 F1 不直观）。

### 4.2 为什么需要两个指标

**单看 Recall 的问题**：

- Recall@100 = 100% 容易达到（top-100 包含所有相关）
- 但 top-100 中可能 99 个不相关，用户看不过来

**单看 Precision 的问题**：

- Precision@1 = 100% 容易（top-1 命中）
- 但可能漏掉大量相关文档

**两者结合**：

- Recall@K：保证不漏（召回阶段重点）
- Precision@K：保证前排准（精排阶段重点）

**典型组合**：

- 召回阶段评估：Recall@1000（候选集含相关文档比例）
- 精排阶段评估：Precision@10（top-10 中相关比例）

### 4.3 K 的选择与场景

**召回阶段（K 大）**：

- K=500~1000
- 评估 Recall@K（不漏好货）
- Precision@K 此时低（top-1000 混入大量不相关），不关注

**精排阶段（K 小）**：

- K=5~20
- 评估 Precision@K（top-k 都是相关的）
- Recall@K 此时低（top-10 不可能含所有相关），不关注

**展示阶段（K=10）**：

- K=10
- 同时看 Recall@10 和 Precision@10
- 用户看 top-10，既要准又要全

**RAG 场景**：

- K=3~10
- Recall@K 关注（top-k 含答案 chunk）
- Precision@K 关注（top-k 不混入噪声 chunk，避免 LLM 受干扰）

### 4.4 Recall/Precision 曲线与 AP

不同 K 下的 Recall-Precision 曲线揭示系统特性：

- **曲线下面积（AUPRC）**：综合评估
- **Average Precision (AP)**：$\frac{1}{|\text{rel}|} \sum_{k} \text{Precision@k} \cdot \mathbb{1}[\text{doc at k is rel}]$
- **mAP (Mean AP)**：多查询 AP 平均

AP 同时考虑 Recall 和 Precision 在不同 K 下的表现，是更综合的指标。

### 4.5 当相关文档数量为 0 或 K

**边界情况 1：$|\text{rel}| = 0$**（无相关文档）

- Recall@K 定义模糊，通常记为 1.0（无漏召）或 0.0
- Precision@K = 0（top-K 全不相关）

**边界情况 2：$|\text{rel}| < K$**

- Recall@K 上限为 1.0（top-K 包含所有相关）
- Precision@K 上限为 $|\text{rel}|/K$（不可能全相关）

**边界情况 3：$|\text{rel}| > K$**

- Recall@K 上限为 $K/|\text{rel}|$（不可能全召回）
- Precision@K 上限为 1.0（top-K 全相关）

**实践**：评估时报告 $|\text{rel}|$ 分布，避免边界情况误导。

### 4.6 Recall@K vs Hit Rate@K

| 指标 | 公式 | 关心 |
|------|------|------|
| Hit Rate@K | $\mathbb{1}[\text{top-}K \cap \text{rel} \ne \emptyset]$ | 有无命中 |
| Recall@K | $|\text{top-}K \cap \text{rel}| / |\text{rel}|$ | 命中比例 |

**当 $|\text{rel}| = 1$**：Hit Rate@K = Recall@K（每查询 1 个相关，命中即 100% recall）

**当 $|\text{rel}| > 1$**：Recall@K 更细（区分命中 1 个 vs 命中多个）

**RAG 场景**：每查询通常标 1 个金标准 chunk，Hit Rate@K = Recall@K。

### 4.7 Recall/Precision 在多阶段排序中的应用

```
全库 (10^6)
  ↓ 召回（双塔 + ANN + BM25）
top-1000
  ↓ 评估 Recall@1000（应 > 95%）
top-100
  ↓ 粗排
top-100
  ↓ 评估 Recall@100（应 > 90%）、Precision@100
top-10
  ↓ 精排
top-10
  ↓ 评估 Precision@10（应 > 80%）、Recall@10
最终展示
```

每阶段评估不同 K 的 Recall 和 Precision，定位瓶颈。

### 4.8 工程上的评估流程

**评估集准备**：

1. 采样 500~1000 查询（代表生产分布）
2. 每查询标注 3~10 个相关文档（金标准）
3. 可选：标注分级相关性（高度相关/相关/一般）

**评估流程**：

```python
def evaluate_retriever(retriever, eval_set, k_values=[1, 5, 10, 100, 1000]):
    """
    eval_set: list of (query, relevant_docs)
    """
    results = {}
    for k in k_values:
        recalls, precisions = [], []
        for query, rel in eval_set:
            retrieved = retriever.search(query, top_k=max(k_values))
            for k_val in [k]:
                top_k = retrieved[:k_val]
                hits = len(set(top_k) & rel)
                recalls.append(hits / len(rel))
                precisions.append(hits / k_val)
        results[f'Recall@{k}'] = sum(recalls) / len(recalls)
        results[f'Precision@{k}'] = sum(precisions) / len(precisions)
    return results
```

**典型结果**：

| K | Recall@K | Precision@K |
|---|----------|-------------|
| 1 | 0.45 | 0.45 |
| 5 | 0.65 | 0.30 |
| 10 | 0.75 | 0.20 |
| 100 | 0.90 | 0.05 |
| 1000 | 0.98 | 0.01 |

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **1950s**：信息检索早期就定义 Recall 和 Precision
- **1990s**：TREC 评测用 Recall/Precision 曲线评估搜索系统
- **2000s**：Recall@K 和 Precision@K 成为标准指标
- **2020**：DPR 等论文用 Recall@K 评估稠密检索
- **2023+**：RAG 评估用 Recall@K（多相关文档）或 Hit Rate@K（单相关）

### 5.2 使用常见坑

**坑 1：金标准标注不全**。

只标 3 个相关文档，实际有 10 个。Recall@K 低估。建议每查询标 5~10 个。

**坑 2：K 选错**。

召回阶段评估 Precision@10（无意义，召回不求精确），精排阶段评估 Recall@1000（无意义，精排不求全）。每阶段评估对应 K。

**坑 3：跨查询平均方式错**。

macro-average（每查询独立算再平均）vs micro-average（全查询合并算）。检索评估用 macro-average。

**坑 4：忽略相关文档数量分布**。

有些查询 1 个相关，有些 20 个相关。直接平均 Recall 偏向多相关查询。可按 $|\text{rel}|$ 分桶评估。

**坑 5：评估集太小**。

100 query 的 Recall 方差大。建议至少 500 query。

**坑 6：用 Recall@K 比较不同 K**。

Recall@10 = 0.7 和 Recall@100 = 0.9 不可直接比（K 不同）。同 K 下比。

**坑 7：金标准与检索粒度不一致**。

金标准是段落，检索 chunk 是句子。统一粒度。

**坑 8：忘了边界情况**。

$|\text{rel}| = 0$ 或 $|\text{rel}| > K$ 时指标含义变化。报告 $|\text{rel}|$ 分布。

**坑 9：只看 Recall 不看 Precision**。

Recall@1000 = 95% 但 Precision@1000 = 1%，用户看不过来。要结合看。

**坑 10：评估集不代表生产**。

评估 query 太简单，生产 query 更难。要从生产采样。

### 5.3 Recall vs Precision 的场景权衡

**高 Recall 优先**：

- 召回阶段（不漏好货）
- 法律、医疗检索（漏相关文档代价大）
- RAG 检索（top-k 含答案 chunk）

**高 Precision 优先**：

- 精排阶段（top-k 都是相关的）
- 移动搜索（屏幕小，只看 top-3）
- 问答系统（top-1 要准）

**平衡**：

- 网页搜索（top-10 既准又全）
- 电商搜索（top-20 覆盖用户意图）

### 5.4 Recall/Precision 的替代指标

**F1@K**：调和平均，综合两者

**AP / mAP**：不同 K 下 Precision 平均，更综合

**NDCG@K**：考虑排序位置和分级相关性

**MRR@K**：第一个相关文档的倒数排名

**选择建议**：

- 简单评估：Recall@K + Precision@K
- 综合评估：mAP 或 NDCG@K
- RAG 单答案：Hit Rate@K（等价 Recall@K）
- 排序质量：MRR@K 或 NDCG@K

### 5.5 RAG 中的 Recall/Precision

RAG 评估中：

- **Recall@K**：top-k 含多少金标准 chunk（多金标准时用）
- **Precision@K**：top-k 中相关 chunk 比例（避免噪声）
- **Hit Rate@K**：top-k 是否含金标准（单金标准时用，等价 Recall@K）

**RAG 特殊考量**：

- chunk 大小影响 Recall（小 chunk 召回高但可能截断信息）
- chunk 重叠影响 Recall（重叠 chunk 提升召回但可能重复）
- LLM context 限制 K（K 大但 LLM 看不完）

**RAG 评估框架**：

- RAGAS：`context_recall`（类似 Recall@K）、`context_precision`（类似 Precision@K）
- TruLens：`context_relevance`（综合 Recall/Precision）

---

## 速记卡

| 维度 | Recall@K | Precision@K |
|------|----------|-------------|
| 公式 | $|\text{top-}K \cap \text{rel}| / |\text{rel}|$ | $|\text{top-}K \cap \text{rel}| / K$ |
| 关心 | 不漏 | 不混 |
| 随 K 变化 | 单调上升 | 通常先升后降 |
| 召回阶段重点 | 是 | 否 |
| 精排阶段重点 | 否 | 是 |

**关系**：

$$
\text{Recall@K} = \text{Precision@K} \cdot \frac{K}{|\text{rel}|}
$$

**典型 K**：

| 阶段 | K | 关注 |
|------|---|------|
| 召回 | 500~1000 | Recall@K |
| 粗排 | 100~500 | Recall@K + Precision@K |
| 精排 | 10~100 | Precision@K |
| RAG | 3~10 | Recall@K + Precision@K |

**一句话记忆**：Recall@K = 召回了多少相关（不漏），Precision@K = 前排有多准（不混）。召回阶段看 Recall@1000，精排阶段看 Precision@10，RAG 看 Recall@5 + Precision@5。两者互补，缺一不可。

---

> *上一篇：[Hit Rate](./hit-rate) -- RAG 评估的事实标准。*
> *下一篇：[MRR 平均倒数排名](./mrr) -- 关心第一个相关文档位置的指标。*
