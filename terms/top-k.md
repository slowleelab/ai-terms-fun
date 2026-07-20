---
title: Top-K 检索
slug: top-k
category: 检索与召回
tags: [检索, top-k, 召回, RAG, 评估指标]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# Top-K 检索

> 五层读懂一个词。这次拆的是：**Top-K 检索**--检索系统中"取前 K 个"的语义、工程含义与调参权衡。

---

## L1 · 一句话点破

**Top-K 是检索系统返回前 K 个最相关结果的策略，K 的选择是 recall、延迟、下游 cost 三者的权衡**。召回阶段 K 大（保 recall），精排 / RAG 上下文阶段 K 小（控延迟和 token cost）。

---

## L2 · 通俗类比

图书馆员给你找书：

- **K=10**：只给最像的 10 本，精准但可能漏掉相关的（recall 低）
- **K=1000**：给 1000 本候选，全但你看不完（信息过载）
- **K=100**：折中，候选足够多又不至于淹没

不同阶段 K 不同：

- **召回阶段 K=1000**：先把可能相关的全捞出来，宁可多不可漏
- **精排阶段 K=100**：粗筛后留 100 本
- **最终展示 K=10**：精排后只展示 10 本
- **RAG 喂 LLM K=5**：只把最相关的 5 段塞进 LLM 上下文，受 token 限制

K 不是拍脑袋，是 recall、延迟、cost 的三角权衡。

---

## L3 · 正经定义

**Top-K 检索**：给定查询 $q$ 和文档库 $V$，返回与 $q$ 相关性分数最高的 K 个文档：

$$
\text{TopK}_K(q, V) = \text{top-}K \text{ of } \{(d, \text{score}(q, d)) \mid d \in V\}
$$

**K 的工程含义**：

| 阶段 | 典型 K | 目标 |
|------|--------|------|
| 召回（双塔/ANN） | 100~1000 | 高 recall，宁多勿漏 |
| 粗排 | 100~500 | 中等过滤 |
| 精排（Cross-encoder） | 10~100 | 高 precision |
| 重排（多样性） | 10~20 | 多样性 + 业务 |
| RAG 上下文 | 3~10 | 受 LLM token 限制 |
| 搜索结果展示 | 10~20 | 用户体验 |

**关键指标**：

- **Recall@K**：top-K 中含真实相关文档的比例
- **Precision@K**：top-K 中相关文档的比例
- **MRR@K**：第一个相关文档的倒数排名
- **NDCG@K**：考虑排序位置的归一化指标

**K 与指标的关系**：

- K 大 -> Recall 高、Precision 低（top-K 中混入更多不相关）
- K 小 -> Recall 低、Precision 高（只保留最高分）
- 召回阶段看 Recall@K，精排阶段看 Precision@K / NDCG@K

**伪代码**：

```python
def top_k_retrieval(query, index, k, score_fn):
    """返回 top-k 文档"""
    # 1. 召回阶段：ANN 找候选（k_recall >> k）
    candidates = index.search(query, top_k=k * 10)

    # 2. 精排阶段：cross-encoder 重排
    scored = [(c, score_fn(query, c)) for c in candidates]
    scored.sort(key=lambda x: -x[1])

    # 3. 取 top-k
    return [c for c, _ in scored[:k]]
```

**RAG 中的 top-k**：

```python
def rag_retrieve(query, vector_db, k=5):
    """RAG 检索 top-k chunks 喂给 LLM"""
    results = vector_db.search(query, top_k=k)
    context = "\n\n".join([r.text for r in results])
    prompt = f"基于以下资料回答：\n{context}\n\n问题：{query}"
    return llm.complete(prompt)
```

---

## L4 · 原理深挖

### 4.1 为什么 K 是关键调参

K 的影响贯穿整个系统：

**对 recall 的影响**：

- 假设真实相关文档有 20 个，分布在分数排名 1, 3, 5, ..., 39
- K=10：top-10 中只含 5 个相关，Recall@10 = 5/20 = 25%
- K=20：top-20 中含 10 个相关，Recall@20 = 50%
- K=40：top-40 中含 20 个相关，Recall@40 = 100%

K 越大 recall 越高，但 K 大也意味着下游 cost 高。

**对延迟的影响**：

- 召回 K=1000：ANN 毫秒级，影响小
- 精排 K=100：Cross-encoder 100 次前向，百毫秒级
- 精排 K=1000：Cross-encoder 1000 次前向，秒级，不可行

**对 RAG cost 的影响**：

- top-k=5：5 chunks × 500 tokens = 2500 tokens 上下文
- top-k=20：20 chunks × 500 tokens = 10000 tokens 上下文
- LLM 推理 cost 与 token 数成正比，top-k 大直接涨 cost

### 4.2 召回阶段的 K：宁多勿漏

召回阶段的核心目标是**高 recall**，K 通常较大（100~1000）。

**为什么 K 大**：

- 召回模型（双塔）精度有限，真正相关的文档可能排到 top-500
- K 小则漏召回，精排再准也救不回
- 召回延迟与 K 关系不大（ANN 复杂度 $O(\log N)$，多取 top-100 还是 top-1000 差异小）

**K 的经验值**：

- 小规模库（< 10^5）：K = 100~500
- 中规模库（10^5~10^7）：K = 500~1000
- 大规模库（> 10^7）：K = 1000~5000

**多路召回的 K**：每路独立召回 K，融合后取 top-K_total。如双塔 K=500 + BM25 K=500 + ColBERT K=500，融合后取 top-1000 给精排。

### 4.3 精排阶段的 K：求 precision

精排阶段的核心目标是**高 precision**，K 通常较小（10~100）。

**为什么 K 小**：

- 精排模型（Cross-encoder）每对前向，K 大延迟大
- 用户只看 top 结果，top-10 精度比 top-100 精度更重要
- 精排 K 受延迟约束（< 500ms）

**K 的经验值**：

- 精排 K = 50~200（受延迟约束）
- 重排 K = 10~20（用户体验）
- 最终展示 K = 10（搜索结果首页）

### 4.4 RAG 中的 K：受 token 限制

RAG 把 top-k chunks 塞进 LLM 上下文，K 受 token 限制：

**约束**：

- LLM context window（如 4K、8K、32K、128K）
- 每个 chunk 大小（如 256、512、1024 tokens）
- K × chunk_size < context_window - query_tokens - response_tokens

**K 的经验值**：

- 短 chunk（256 tokens）+ 8K context：K ≈ 20
- 中 chunk（512 tokens）+ 32K context：K ≈ 50
- 长 chunk（1024 tokens）+ 128K context：K ≈ 100

**K 与 RAG 性能**：

- K 太小：信息不足，LLM 答不全
- K 太大：上下文稀释，LLM 难聚焦（lost in the middle），cost 涨
- 经验：K = 3~10 是多数 RAG 场景的甜蜜点

### 4.5 K 与 chunk 大小的耦合

RAG 中 K 和 chunk_size 是耦合的：

**短 chunk + 大 K**：

- 每个聚焦但可能截断信息
- 召回率高（粒度细）
- 上下文分散（LLM 需要整合多段）

**长 chunk + 小 K**：

- 每个完整但可能含冗余
- 召回率低（粒度粗，相关 chunk 可能漏）
- 上下文集中（LLM 易聚焦）

**经验**：

- chunk_size = 256~512 tokens，K = 5~10：通用平衡
- chunk_size = 1024+，K = 3~5：长文档摘要场景
- chunk_size = 128，K = 20+：精确片段检索场景

### 4.6 动态 K：按需调整

固定 K 不一定最优，可动态调整：

**按查询难度**：

- 简单查询（如"RLHF 是什么"）：K=3 够
- 复杂查询（如"对比 RLHF 和 DPO 的优缺点"）：K=10+

**按相关性分数**：

- top 分数高且断崖式下降：K 小（前几个就够）
- top 分数都接近：K 大（需要更多候选区分）

**按 LLM 自评**：

- 让 LLM 先看 top-3，不够再要更多
- 迭代式 RAG（如 FLARE）

### 4.7 K 与评估指标的关系

不同 K 对应不同评估指标：

**Recall@K**：

$$
\text{Recall@K} = \frac{|\text{top-K} \cap \text{relevant}|}{|\text{relevant}|}
$$

**Precision@K**：

$$
\text{Precision@K} = \frac{|\text{top-K} \cap \text{relevant}|}{K}
$$

**MRR@K**（Mean Reciprocal Rank）：

$$
\text{MRR@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q^*} \cdot \mathbb{1}[\text{rank}_q^* \le K]
$$

$\text{rank}_q^*$ 是第一个相关文档的排名。

**NDCG@K**（Normalized Discounted Cumulative Gain）：

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}, \quad \text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

详见后续 hit-rate、recall-precision-at-k、mrr、ndcg 词条。

**K 的选择与指标**：

- 召回阶段评估：Recall@1000
- 精排阶段评估：NDCG@10 / MRR@10
- RAG 评估：Recall@5（top-5 是否含答案）

### 4.8 工程上的 K 调优

**调优流程**：

1. 准备评估集（query + 相关文档标注）
2. 固定召回模型，扫 K = [10, 50, 100, 500, 1000]，测 Recall@K
3. 选 Recall 满足要求（如 > 95%）的最小 K，作为召回 K
4. 固定精排模型，扫 K = [5, 10, 20, 50, 100]，测 NDCG@K
5. 选 NDCG 满足要求且延迟可接受的最大 K，作为精排 K
6. RAG 中扫 K = [3, 5, 10, 20]，测答案准确率 + cost

**常见调优错误**：

- 直接照搬其他系统的 K（每系统数据分布不同）
- 只看 recall 不看延迟（K 大延迟大）
- 只看一个 K 值（应该扫多个找最优）
- 忽略 chunk_size 与 K 的耦合

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **1990s**：搜索引擎返回 top-10 结果，K=10 几乎是用户认知默认
- **2000s**：Learning to Rank 时代，K=10 和 K=100 是评估标准（NDCG@10）
- **2010s**：深度召回兴起，K 在召回阶段从 10 扩到 1000
- **2020**：DPR 论文用 Recall@1000 / Recall@100 评估稠密召回
- **2022+**：RAG 兴起，K 受 LLM context 限制，K=3~10 成主流
- **2023+**：长 context LLM（128K）出现，K 上限提升，但 lost in the middle 问题显现

### 5.2 调参常见坑

**坑 1：召回 K 太小**。

K=50 时召回率不足，精排再准也救不回。建议 K=500~1000，给精排足够候选。

**坑 2：精排 K 太大**。

K=1000 时 Cross-encoder 延迟秒级，不可行。加粗排预筛，或减 K 到 100~200。

**坑 3：RAG K 与 chunk_size 不匹配**。

chunk_size=1024 + K=20 = 20480 tokens，可能超出 LLM context。算 K × chunk_size + query + response < context_window。

**坑 4：K 大就一定好**。

RAG 中 K 大反而可能降低性能（lost in the middle，LLM 难聚焦）。K=5 可能比 K=20 答得更好。务必实测。

**坑 5：固定 K 不调**。

不同查询难度不同，固定 K 不一定最优。简单查询 K=3 够，复杂查询 K=10+。可做动态 K。

**坑 6：评估只看一个 K**。

只看 Recall@10 不够，要看 Recall@K 曲线（K=10, 100, 1000）才能判断召回质量。

**坑 7：忽略 K 对 cost 的影响**。

RAG 中 K 大直接涨 LLM 推理 cost。生产环境要算 cost / query。

**坑 8：多路召回 K 不均衡**。

双塔 K=1000 + BM25 K=100，BM25 路召回率不足。多路 K 要相近，融合后才有效。

**坑 9：K 与重排脱节**。

精排 K=100，但重排只取 K=5，中间 95 个候选白算。重排 K 要与精排 K 协调。

**坑 10：用 top-1 评估**。

只看 top-1 准不准，忽略排序质量。top-1 评估方差大，应看 MRR@K 或 NDCG@K。

### 5.3 不同场景的 K 推荐

| 场景 | 召回 K | 精排 K | 展示 K |
|------|--------|--------|--------|
| 网页搜索 | 1000 | 100 | 10 |
| 电商搜索 | 500 | 50 | 20 |
| 企业文档搜索 | 500 | 50 | 10 |
| 客服问答 | 100 | 20 | 3 |
| RAG（8K context） | 100 | 20 | 5 |
| RAG（32K context） | 200 | 50 | 10 |
| RAG（128K context） | 500 | 100 | 20 |
| 推荐召回 | 1000 | 200 | 50 |

### 5.4 RAG 中 K 的最新研究

**Lost in the Middle (Liu et al. 2023)**：

- LLM 对长上下文中间的信息利用不足
- top-k 中相关 chunk 放中间，LLM 可能忽略
- 建议：把最重要的 chunk 放首尾

**迭代式 RAG (FLARE, Jiang et al. 2023)**：

- LLM 边生成边判断是否需要更多检索
- 动态 K，避免一次塞太多

**Self-RAG (Asai et al. 2023)**：

- LLM 自己评估检索质量
- 不够相关时重新检索
- 动态调整 K

**多跳 RAG**：

- 复杂问题需要多次检索
- 每次 K 小（如 3），多跳累积

### 5.5 K 与上下文工程

现代 RAG 把 K 选择纳入"上下文工程"：

- **chunk_size 选择**：决定单 chunk 信息密度
- **K 选择**：决定上下文总量
- **chunk 排序**：重要 chunk 放首尾（避 lost in middle）
- **chunk 去重**：相似 chunk 合并，避免冗余
- **chunk 压缩**：长 chunk 摘要后塞入

这些决策共同影响 RAG 性能，K 是其中一环。

---

## 速记卡

| 阶段 | 典型 K | 主要约束 |
|------|--------|----------|
| 召回 | 100~1000 | recall 高 |
| 粗排 | 100~500 | 中等过滤 |
| 精排 | 10~100 | precision 高 + 延迟 |
| 重排 | 10~20 | 多样性 |
| RAG 上下文 | 3~10 | LLM token 限制 |
| 展示 | 10~20 | 用户体验 |

**K 与指标**：

- Recall@K：top-K 中相关文档比例
- Precision@K：top-K 中相关文档密度
- MRR@K：第一个相关文档的倒数排名
- NDCG@K：考虑排序位置的归一化指标

**K 调优铁律**：

1. 召回 K 宁多勿少（保 recall）
2. 精排 K 受延迟约束（控 precision）
3. RAG K 受 token 限制（控 cost）
4. K 与 chunk_size 耦合（联合调）
5. RAG 中 K 大不一定好（lost in middle）

**一句话记忆**：Top-K 是检索系统返回前 K 个的策略，K 是 recall、延迟、cost 的三角权衡。召回 K 大（保 recall），精排 K 小（求 precision），RAG K 受 token 限制（控 cost），展示 K 由用户体验定。

---

> *上一篇：[召回与精排](./recall-rerank) -- 两阶段检索架构。*
> *下一篇：[混合搜索 Hybrid Search](./hybrid-search) -- 关键词 + 向量的多路召回融合。*
