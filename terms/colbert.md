---
title: ColBERT
slug: colbert
category: 检索与召回
tags: [神经检索, ColBERT, late interaction, 多向量, 精排, 召回]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# ColBERT

> 五层读懂一个词。这次拆的是：**ColBERT**--late interaction 模型，介于双塔和 Cross-encoder 之间的精度速度折中。

---

## L1 · 一句话点破

**ColBERT 保留文档每个 token 的向量（而非压成单一向量），查询时 token 级 max-sim 累加得分**。文档侧可预计算，查询侧有 token 级交互，精度高于双塔、速度远快于 Cross-encoder。

---

## L2 · 通俗类比

双塔模型像"两张身份证对比"：query 压成一张"身份证"，doc 压成一张，看两证像不像。信息被压缩损失大。

Cross-encoder 像"面对面逐句深聊"：query 和 doc 拼一起逐词交叉对比，精度最高但每对都要重聊。

ColBERT 像"两张详细简历逐条对比"：

- query 不压缩，每个词都有一张"小卡片"（per-token 向量）
- doc 也不压缩，每个词都有一张"小卡片"
- 对比时，query 每张卡片去 doc 的所有卡片里找最像的一张（max-sim），累加所有 query 卡片的最高匹配分

这种"延迟交互"（late interaction）让 query 和 doc 在最后一步才交互，但交互粒度是 token 级而非整体级。文档卡片可离线预计算建索引，查询时只算 query 卡片与候选 doc 卡片的 max-sim。

精度比双塔高（保留了 token 级信息），速度比 Cross-encoder 快（doc 卡片预计算）。代价：索引比双塔大 N 倍（每 token 一向量而非每 doc 一向量）。

---

## L3 · 正经定义

**ColBERT (Contextualized Late Interaction over BERT)**：Khattab & Zaharia 2020 提出。查询和文档分别过 BERT，每个 token 输出一个向量（不 pooling），最后做 late interaction。

**架构**：

1. **Query Encoder**：query 过 BERT，每个 token 输出 $D$ 维向量。query 长度 $L_q$，输出矩阵 $\vec{Q} \in \mathbb{R}^{L_q \times D}$。
2. **Document Encoder**：doc 过 BERT，每个 token 输出 $D$ 维向量。doc 长度 $L_d$，输出矩阵 $\vec{D} \in \mathbb{R}^{L_d \times D}$。
3. **Late Interaction**：对 query 每个 token 向量，找 doc 中最相似的 token 向量（max-sim），累加：

$$
\text{score}(q, d) = \sum_{i=1}^{L_q} \max_{j=1}^{L_d} \vec{Q}_i \cdot \vec{D}_j
$$

**关键性质**：

- 文档侧 per-token 向量**可预计算**，存索引
- 查询侧 per-token 向量在线编码
- max-sim 是 token 级交互，比双塔点积精，比 Cross-encoder 全 attention 简单

**伪代码**：

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel

class ColBERT:
    def __init__(self, model_name='bert-base-uncased', dim=128):
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = torch.nn.Linear(768, dim)  # ColBERT 用 128 维降维

    def encode(self, ids, mask):
        """返回 per-token 向量，已归一化"""
        out = self.encoder(ids, mask)
        vecs = self.proj(out.last_hidden_state)  # (B, L, D)
        vecs = F.normalize(vecs, dim=-1)
        return vecs

    def score(self, q_vecs, d_vecs):
        """q_vecs: (L_q, D), d_vecs: (L_d, D)"""
        sim = q_vecs @ d_vecs.T          # (L_q, L_d)
        max_sim = sim.max(dim=1).values  # (L_q,) 每 query token 取 doc 中最相似
        return max_sim.sum()             # 标量
```

**推理流程**：

```python
# 离线：编码所有文档的 per-token 向量
doc_token_vecs = [model.encode(d) for d in all_docs]
# 存索引（每文档 L_d 个 D 维向量）

# 在线：编码 query
q_vecs = model.encode(query)  # (L_q, D)

# ANN 检索：用 query 向量找候选
# ColBERT v2 用 PLAID 优化，先用 centroid 找候选文档
candidates = retrieve_candidates(q_vecs, top_k=1000)

# 精排：对候选算 max-sim
scores = [model.score(q_vecs, d_vecs[d_id]) for d_id in candidates]
top_10 = sorted(zip(candidates, scores), key=lambda x: -x[1])[:10]
```

**代表模型**：

- **ColBERT (v1)**：Khattab & Zaharia 2020，原始版本
- **ColBERTv2**：Khattab et al. 2021，残余压缩 + 量化，索引减小 10 倍
- **PLAID ColBERT**：Santhanam et al. 2022，工程优化，延迟降到毫秒级
- **Jina-ColBERT**：多语言版本
- **Lightweight ColBERT**：减少参数，部署友好

---

## L4 · 原理深挖

### 4.1 为什么 late interaction 精度高

双塔把整句压成单一向量，丢失 token 级信息。"iPhone 15 Pro" 和 "iPhone 14 Pro" 在双塔向量空间可能很近，但用户其实想要 15。

ColBERT 保留 token 级向量，query 的 "15" 向量会去 doc 中找最像的 token 向量：

- doc "iPhone 15 Pro" 的 "15" 向量与 query "15" 向量高相似
- doc "iPhone 14 Pro" 的 "14" 向量与 query "15" 向量低相似

max-sim 捕获这种细粒度差异，精度提升。

**与 Cross-encoder 的差异**：Cross-encoder 是 query 的每个 token 与 doc 的每个 token 全 attention 交互（双向），ColBERT 是 query 的每个 token 单向找 doc 中最相似的（单向 max）。损失了一些交互深度，但大幅降低计算量。

### 4.2 max-sim 的几何意义

max-sim $\sum_i \max_j \vec{Q}_i \cdot \vec{D}_j$ 可以理解为：

- 对 query 每个 token，在 doc 中找到"最匹配"的 token
- 累加所有 query token 的最高匹配分

这是一种**软性对齐**（soft alignment），类似动态时间规整（DTW）但更简单。它假设 query 每个 token 都能在 doc 找到对应，doc 中多余的 token 不影响。

**优点**：

- doc 长度变化不影响（max 取最相似的）
- query 中重要 token 主导分数（普通 token 的 max-sim 低）

**缺点**：

- 不考虑词序（max-sim 是无序的）
- 对 query 中重复 token 敏感（每个重复 token 都贡献分数）

### 4.3 索引大小与优化

ColBERT 索引比双塔大 N 倍：

- 双塔：每文档 1 个 $D$ 维向量，$N \cdot D$ 存储
- ColBERT：每文档 $L_d$ 个 $D$ 维向量，$N \cdot L_d \cdot D$ 存储

$L_d = 200, D = 128$ 时每文档 100KB，百万文档 100GB--巨大。

**ColBERTv2 的优化**：

1. **残余压缩**：把 per-token 向量量化 + 聚类编码，压缩 6~10 倍
2. **二级索引**：文档向量化后聚成 cluster，查询时只搜相关 cluster
3. **缓存**：query token 向量与 cluster centroid 的相似度缓存

**PLAID ColBERT**：

1. **centroid 预筛**：用 cluster centroid 找候选文档，跳过明显不相关
2. **逐步精化**：先粗排（centroid 距离），再精排（per-token max-sim）
3. **量化**：FP16 / INT8 量化，内存减半

经过优化，ColBERT v2 + PLAID 在 MS MARCO 上达到接近 Cross-encoder 的精度，延迟在毫秒级（top-1000 候选精排 < 100ms）。

### 4.4 维度选择：为什么 128 维

ColBERT v1 用 128 维（不是 BERT 默认 768 维）。原因：

1. **存储**：128 维比 768 维小 6 倍
2. **检索速度**：max-sim 是 $O(L_q \cdot L_d \cdot D)$，$D$ 小则快
3. **精度损失小**：实验证明 128 维 per-token 向量在检索任务上足够

ColBERT v2 进一步优化：用 2-bit 量化把 128 维 float32 压成 128 维 2-bit，每向量 32 字节，索引大小与双塔 FP32 接近。

### 4.5 ColBERT vs 双塔 vs Cross-encoder

| 维度 | 双塔 | ColBERT | Cross-encoder |
|------|------|---------|----------------|
| 查询文档交互 | 单向量点积 | token 级 max-sim | 全 attention |
| 文档侧预计算 | 单向量 | per-token 向量 | 不行 |
| 召回阶段适用 | 行 | 行 | 不行 |
| 精排阶段适用 | 不行 | 行 | 行 |
| 索引大小 | 小 | 大（per-token） | 无 |
| 精度 | 中 | 高 | 高 |
| 典型延迟 | < 10ms（召回） | 10~50ms（top-1000） | 50~200ms（top-100） |
| 工程复杂度 | 低 | 中 | 中 |

**关键观察**：ColBERT 把"可预计算"和"细粒度交互"结合，是双塔和 Cross-encoder 的折中。

### 4.6 ColBERT 的训练

训练目标类似双塔，用对比学习：

$$
\mathcal{L} = -\log \frac{\exp(\text{score}(q, d^+) / \tau)}{\exp(\text{score}(q, d^+) / \tau) + \sum_{d^-} \exp(\text{score}(q, d^-) / \tau)}
$$

但 $\text{score}$ 是 max-sim 而非点积。训练时 query 和 doc per-token 向量都通过 BERT，反向传播更新参数。

**实践**：

- ColBERT v1 在 MS MARCO 上训练
- 难负样本对性能影响大
- 训练 batch 大（256+）效果更好

### 4.7 ColBERT 的局限

**局限 1：索引大**。即便优化后仍比双塔大数倍。大规模部署需更多存储和内存。

**局限 2：长文档处理**。doc 截断到 BERT 限制（512 tokens），长文档需切片处理。每切片独立索引，跨切片 max-sim 难统一。

**局限 3：训练复杂**。per-token 向量的训练比单一向量难收敛，需仔细调超参。

**局限 4：部署门槛高**。比双塔和 Cross-encoder 都复杂，需要专门索引（PLAID 等）。生态不如 sentence-transformers 成熟。

**局限 5：长 query 不友好**。query 太长时 max-sim 累加项多，分数被低质量 token 稀释。实践中 query 通常短，不是大问题。

### 4.8 PLAID ColBERT 的工程突破

PLAID (Politix-friendly Lightweight Algorithm for Indexing and search in Dense spaces) 是 ColBERT 的工程优化版，让 ColBERT 真正可用于生产：

**关键优化**：

1. **Centroid 预筛**：文档 token 向量聚成 K 个 cluster，查询时先找相关 cluster 再精排
2. **量化**：per-token 向量量化到 2-bit，存储大幅压缩
3. **向量化 max-sim**：用 SIMD 指令并行计算，速度提升数倍
4. **缓存策略**：query token 向量与 centroid 的相似度缓存，避免重复计算

**效果**：MS MARCO 上 top-5 召回率接近 Cross-encoder，延迟 < 100ms，索引大小可接受。是 ColBERT 生产部署的事实标准。

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **2020**：Khattab & Zaharia 提出 ColBERT（SIGIR），late interaction 概念
- **2021**：ColBERTv2（NAACL），残余压缩 + 量化，索引减小 10 倍
- **2022**：PLAID ColBERT（Santhanam et al.），工程优化让 ColBERT 生产可用
- **2022**：Jina-ColBERT，多语言版本
- **2023**：ColBERT 在 RAG 系统中作为"高精度召回+精排一体化"方案被采用
- **2024**：ColBERT 仍是 late interaction 代表，新工作如 ColBERT-X、UV-EMBEDDING 探索多模态扩展

### 5.2 使用常见坑

**坑 1：索引过大没优化**。

直接用 ColBERT v1，per-token 向量存储巨大。务必用 ColBERTv2 或 PLAID 的压缩和量化。

**坑 2：用 ColBERT 做大规模召回**。

ColBERT 索引虽可预计算，但比双塔大数倍。大规模（> 千万）召回时存储和内存压力大。建议双塔召回 + ColBERT 精排。

**坑 3：长文档不切片**。

doc 截断到 512 tokens 丢失尾部信息。索引设计时切 chunk 到合适长度，每 chunk 独立 ColBERT 索引，查询时跨 chunk 聚合（max 聚合常用）。

**坑 4：query 过长**。

query 太长时 max-sim 累加项多，低质量 token 稀释分数。query 预处理时截断或抽取关键词。

**坑 5：训练 batch 太小**。

per-token 向量训练需大 batch 才稳定。batch=256+ 推荐，gradient cache 等技巧可用。

**坑 6：评估指标错位**。

ColBERT 既可召回又可精排，评估要分清。召回指标（recall@1000）和精排指标（recall@10, MRR@10）都要看。

**坑 7：部署生态不熟**。

ColBERT 部署需用 PLAID/rml检索引擎，不像 sentence-transformers 那么"开箱即用"。团队要熟悉 PLAID 工具链。

**坑 8：与 Cross-encoder 精度差异高估**。

ColBERT 精度虽高，但 top-1 仍略低于 Cross-encoder。极致精度场景仍需 Cross-encoder 精排，ColBERT 作为精排前的中间阶段。

**坑 9：忘了归一化 per-token 向量**。

ColBERT 要求 per-token 向量归一化（max-sim 是点积，不归一化数值不稳）。训练和推理都要归一化。

**坑 10：直接复用 BERT 输出维度**。

原始 BERT 768 维做 per-token 向量太大。ColBERT 默认 128 维降维，存储和速度都更友好。

### 5.3 ColBERT 的适用场景

**适合用**：

- 中等规模（< 千万）文档
- 精度要求高（如客服问答、企业搜索）
- 接受较大索引存储
- 团队能驾驭 PLAID 工具链

**不适合**：

- 极大规模（> 亿级）召回（用双塔）
- 极致精度 top-1（用 Cross-encoder）
- 索引存储受限（用双塔 + 量化）
- 简单部署（用 sentence-transformers）

**典型架构**：

- 小规模高精度：直接 ColBERT 端到端
- 中规模：双塔召回 + ColBERT 精排（替代 Cross-encoder，更快）
- 大规模：双塔召回 + ColBERT 粗排 + Cross-encoder 精排

### 5.4 ColBERT 的延伸与变体

**ColBERTv2**：原版的工程优化版，压缩 + 量化，事实标准。

**PLAID ColBERT**：PLAID 引擎 + ColBERTv2，生产部署首选。

**Jina-ColBERT**：多语言，支持中英文等 50+ 语言。

**ColBERT-X**：跨语言版本，多语言训练。

**UV-EMBEDDING**：多模态扩展，图像 + 文本的 late interaction。

**Lightweight ColBERT**：减少 BERT 层数和参数，部署友好。

### 5.5 与混合检索的关系

ColBERT 与混合检索（BM25 + Dense）方向不同但互补：

- **混合检索**：多路召回（稀疏 + 稠密）融合，提升召回率
- **ColBERT**：单路 late interaction，提升精度

两者可叠加：BM25 + Dense + ColBERT 三路召回融合，再用 Cross-encoder 精排。但工程复杂度上升，多数场景双塔 + ColBERT 已够。

---

## 速记卡

| 维度 | ColBERT |
|------|---------|
| 架构 | per-token BERT + late interaction (max-sim) |
| 训练 | 对比学习，max-sim 替代点积 |
| 优势 | 精度高于双塔，速度远快于 Cross-encoder |
| 局限 | 索引大、部署复杂 |
| 维度 | 128 维（非 BERT 默认 768） |
| 优化 | ColBERTv2 + PLAID（压缩 + 量化 + 缓存） |
| 适用 | 中规模高精度召回 + 精排 |

**核心公式**：

$$
\text{score}(q, d) = \sum_{i=1}^{L_q} \max_{j=1}^{L_d} \vec{Q}_i \cdot \vec{D}_j
$$

**三模型对比**：

| 模型 | 交互 | 预计算 | 精度 | 速度 | 索引 |
|------|------|--------|------|------|------|
| 双塔 | 单向量点积 | 行 | 中 | 极快 | 小 |
| ColBERT | token max-sim | 行 | 高 | 中 | 大 |
| Cross-encoder | 全 attention | 不行 | 极高 | 慢 | 无 |

**一句话记忆**：ColBERT = 保留 per-token 向量 + late interaction (max-sim)。文档侧可预计算，查询侧 token 级交互，精度高于双塔、速度远快于 Cross-encoder。索引大但 PLAID 优化后生产可用。

---

> *上一篇：[交叉编码器 Cross-encoder](./cross-encoder) -- 精排主力。*
> *下一篇：[召回与精排](./recall-rerank) -- 两阶段检索的工程架构。*
