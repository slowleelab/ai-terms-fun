---
title: 双塔模型 Two-Tower
slug: two-tower
category: 检索与召回
tags: [神经检索, 双塔模型, 稠密检索, 对比学习, embedding, 召回]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 双塔模型 Two-Tower

> 五层读懂一个词。这次拆的是：**双塔模型**--稠密检索（Dense Retrieval）的主力架构，现代语义召回的根基。

---

## L1 · 一句话点破

**查询和文档分别用一个 encoder 编码成向量（两座"塔"），向量相似度即相关性。** 文档向量可预计算离线建索引，查询向量在线编码，二者点积即得分。这是稠密检索的核心范式。

---

## L2 · 通俗类比

旧式关键词检索（BM25）像"对暗号"：查询和文档要有完全相同的字才得分。"我要买手机"和"想购入智能电话"对不上暗号，0 分。

双塔模型像"画像匹配"：

- 给查询画一张"语义画像"（向量）
- 给每篇文档也画一张"语义画像"
- 两张画像越像，分数越高
- "我要买手机"和"想购入智能电话"画像是同一种意思，高分

为什么叫"双塔"？因为有两个独立的编码器（塔），分别处理查询和文档。两塔不相交，只在最后算向量点积。这个设计让文档塔可以离线把所有文档编码好存索引，查询时只需算查询向量再点积--百万级文档毫秒级检索。

但双塔也有硬伤：**查询和文档编码时彼此看不到对方**。查询里的"苹果"是水果还是公司？文档塔不知道。这种"互盲"让双塔精排时不够准，需要交叉编码器（Cross-encoder）补救。

---

## L3 · 正经定义

**双塔模型 (Two-Tower / Bi-Encoder)**：查询 $q$ 和文档 $d$ 分别通过编码器 $E_q$ 和 $E_d$（可共享参数）映射到向量空间：

$$
\vec{q} = E_q(q), \quad \vec{d} = E_d(d)
$$

相关性分数为向量内积（或余弦）：

$$
\text{score}(q, d) = \vec{q} \cdot \vec{d} = E_q(q)^\top E_d(d)
$$

**训练目标（对比学习）**：让正样本对 $(q, d^+)$ 向量靠近，负样本对 $(q, d^-)$ 远离。常用 InfoNCE loss：

$$
\mathcal{L} = -\log \frac{\exp(\vec{q} \cdot \vec{d^+} / \tau)}{\exp(\vec{q} \cdot \vec{d^+} / \tau) + \sum_{d^-} \exp(\vec{q} \cdot \vec{d^-} / \tau)}
$$

$\tau$ 是温度系数，控制分布尖锐度。

**关键性质**：

- **可分离**：$\vec{q}$ 和 $\vec{d}$ 独立计算，无交叉
- **可预计算**：文档向量 $\vec{d}$ 离线编码存索引
- **可扩展**：百万级文档用 ANN 索引毫秒级检索

**伪代码**：

```python
import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, bert, dim=768):
        super().__init__()
        self.query_encoder = bert          # 共享 BERT
        self.doc_encoder = bert            # 同一 BERT
        self.proj = nn.Linear(768, dim)

    def encode(self, text_ids, text_mask):
        out = self.query_encoder(text_ids, text_mask)
        cls = out.last_hidden_state[:, 0]   # [CLS] 向量
        return nn.functional.normalize(self.proj(cls), dim=-1)

    def forward(self, query, pos_doc, neg_docs):
        q = self.encode(query['ids'], query['mask'])
        d_pos = self.encode(pos_doc['ids'], pos_doc['mask'])
        d_neg = self.encode(neg_docs['ids'], neg_docs['mask'])  # (B, K, D)
        # InfoNCE
        sim_pos = (q * d_pos).sum(-1, keepdim=True)             # (B, 1)
        sim_neg = torch.einsum('bd,bkd->bk', q, d_neg)          # (B, K)
        logits = torch.cat([sim_pos, sim_neg], dim=1) / 0.05    # (B, 1+K)
        labels = torch.zeros(q.size(0), dtype=torch.long)       # 正样本在 0
        return nn.functional.cross_entropy(logits, labels)
```

**推理**：

```python
# 离线：编码所有文档
doc_vecs = [model.encode(d) for d in all_docs]
faiss_index.add(np.stack(doc_vecs))

# 在线：编码查询，ANN 检索
q_vec = model.encode(query)
scores, doc_ids = faiss_index.search(q_vec, top_k=100)
```

**代表模型**：

- **DPR (Dense Passage Retrieval, Karpukhin et al. 2020)**：两个独立 BERT，对比学习，Facebook 出品
- **ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation, Xiong et al. 2021)**：异步刷新 ANN 索引做 hard negative
- **Sentence-BERT / Sentence-Transformers**：通用双塔，支持多任务
- **BGE / E5 / GTE / Jina**：现代中文/多语言 embedding 模型
- **CLIP**：图文双塔，详见 CLIP 词条
- **YouTube 推荐**：早期双塔用于候选召回（Covington et al. 2016）

---

## L4 · 原理深挖

### 4.1 为什么是双塔而不是单塔？

**单塔（Cross-encoder）**：查询和文档拼接 $[\text{CLS}] q [\text{SEP}] d [\text{SEP}]$，过一遍 BERT，输出相关性分数。

**双塔（Bi-encoder）**：查询和文档分别过 BERT，输出向量，点积得分。

| 维度 | 单塔（Cross-encoder） | 双塔（Bi-encoder） |
|------|----------------------|---------------------|
| 查询文档交互 | 全交叉（attention） | 仅向量点积 |
| 文档向量预计算 | 不行 | 行 |
| 检索效率 | 慢（每对都要前向） | 快（ANN） |
| 精度 | 高 | 中 |
| 适用 | 精排（rerank top-100） | 召回（从百万级到 top-100） |

工程结论：**召回用双塔，精排用交叉编码器**。这是现代两阶段检索的标准架构。

### 4.2 双塔的训练：对比学习

双塔的核心是**对比学习**：拉近正样本对、推远负样本对。

**正样本来源**：

- 搜索日志：点击数据（query-document 对）
- 问答数据：问题-答案对
- 标注数据：人工标注相关文档

**负样本来源**：

- **随机负样本**（random negative）：从语料库随机抽，简单但信号弱（多数负样本太容易区分）
- **BM25 难负样本**（hard negative）：BM25 高分但实际不相关的文档，模型最难学
- **in-batch 负样本**：同一 batch 内其他样本的正文档作为负样本（高效，batch=64 时有 63 个免费负样本）
- **ANN 难负样本**：当前模型 ANN 检索 top-k 但实际不相关的，需异步刷新索引（ANCE 做法）

**实践**：正负样本比例 1:7 ~ 1:30 常用。难负样本权重高，随机负样本保证覆盖度。

### 4.3 InfoNCE 与温度系数

InfoNCE loss 的温度 $\tau$ 控制分布尖锐度：

- $\tau$ 小：分布尖锐，模型更努力区分正负样本，但梯度小
- $\tau$ 大：分布平滑，梯度大，但区分能力弱
- 经验：$\tau = 0.01 \sim 0.1$，CLIP 用 0.07，DPR 用 0.05

**InfoNCE 的理论**：从互信息角度，最大化正样本对的互信息下界。$\tau$ 与噪声对比估计的噪声分布相关。

### 4.4 共享参数 vs 独立参数

**共享**：query 和 doc 用同一个 encoder。

- 优点：参数少、训练数据共享
- 缺点：query 和 doc 长度/风格差异大时不够灵活

**独立**：query 和 doc 各一个 encoder。

- 优点：可针对性优化（query 短、doc 长）
- 缺点：参数翻倍，可能过拟合

**实践**：

- DPR：独立 encoder
- Sentence-BERT：共享 encoder
- CLIP：独立（图像和文本本质不同）
- BGE / E5：共享

### 4.5 pooling 策略

编码器输出序列向量后，怎么得到文档的单一向量？

- **[CLS] pooling**：取 [CLS] 位置的向量，BERT 默认
- **mean pooling**：所有 token 向量取平均，sentence-BERT 推荐
- **max pooling**：每维取最大，对关键词敏感

**实践**：mean pooling 在 sentence-BERT 系列表现最稳；CLS 在 BERT 预训练任务对齐时好用；后期 BGE / E5 用 last hidden state 的 mean pooling（不被预训练 MLM 任务污染）。

### 4.6 归一化与距离

embedding 输出后是否归一化？

- **归一化**：$\vec{v} \to \vec{v} / \|\vec{v}\|$，IP 等价于余弦
- **不归一化**：向量模长含信息（如文档质量分），但训练时需特殊处理

**主流做法**：训练时归一化，让 IP 等价于余弦相似度。FAISS / Milvus 等检索库默认 IP（内积），归一化后用 IP 等价于余弦，省一次开方。

### 4.7 双塔的局限

**局限 1：查询文档互盲**。查询编码时看不到文档，"苹果"的语义无法根据候选文档消歧。文档塔同理。这是双塔最大的硬伤。

**局限 2：词级精确匹配弱**。双塔把整句压成一个向量，丢失词级信息。"iPhone 15 Pro"和"iPhone 14 Pro"在向量空间可能很近，但用户其实想要 15。BM25 在精确匹配上更强。

**局限 3：长尾词训练不足**。embedding 模型对训练时没见过的词（如新产品名、新概念）编码能力差。BM25 不受此限。

**局限 4：单一向量压缩信息**。长文档压成一个 768 维向量，信息损失大。ColBERT 等方法用多向量（per-token 向量）缓解。

工程应对：

- **混合检索**：BM25 + Dense 融合，互补精确与语义
- **ColBERT**：late interaction，保留 token 级向量
- **Cross-encoder 精排**：双塔召回后用 cross-encoder 重排

### 4.8 训练数据的质量决定上限

双塔模型性能高度依赖训练数据。DPR 之后一系列工作证明：

- **难负样本**对性能影响最大（ANCE、RocketQA）
- **数据规模**：千万级训练对效果显著好于百万级
- **数据多样性**：单一领域训练，跨领域泛化差
- **预训练 + 微调**：先在通用语料预训练（如 SimCSE、E5），再在领域数据微调

BGE、E5、GTE 等现代中文 embedding 在大规模预训练 + 多任务微调下，MTEB 中文榜单表现接近 GPT-4 级别。

### 4.9 双塔在推荐系统的应用

双塔不只用于检索，还是推荐系统的主力召回架构：

**YouTube 推荐**（Covington et al. 2016）：

- 候选生成塔：用户向量
- 视频塔：视频向量
- 内积得分 top-k 候选

**特点**：

- 用户塔特征：观看历史、搜索历史、人口属性
- 视频塔特征：视频 ID、类别、标签
- 训练：用户看了视频为正样本，未看为负样本
- 在线：用户向量在线计算，视频向量离线建索引

推荐的双塔与检索的双塔本质相同，区别在输入特征（推荐用 user/item 特征，检索用 query/doc 文本）。

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **2015**：Huang et al. 提出 DSSM（Deep Structured Semantic Model），双塔雏形，搜索场景
- **2016**：YouTube 双塔推荐（Covington et al.）
- **2018**：Sentence-BERT（Reimers & Gurevych），让 BERT 可用于 sentence embedding
- **2019**：多语言Sentence-BERT、SimCSE 等
- **2020**：DPR（Karpukhin et al.），稠密检索超越 BM25，里程碑
- **2021**：ANCE、RocketQA，难负样本训练提升性能
- **2021**：CLIP（图文双塔）大规模对比学习
- **2022**：E5、GTE、BGE 系列，中文 embedding 突破
- **2023**：MTEB 榜单竞争白热化；BGE-M3 支持稀疏+稠密+多向量
- **2024**：LLM-based embedding 兴起（如 LLM2Vec），性能继续提升

### 5.2 训练与部署常见坑

**坑 1：忘了归一化向量**。

训练时归一化，部署时不归一化，IP 距离不等价于余弦，结果偏向长向量。统一归一化是铁律。

**坑 2：负样本太弱**。

只用随机负样本训练，模型学不到区分细微差异。要加难负样本（BM25 top-k 中非相关的、当前模型 top-k 中非相关的）。

**坑 3：in-batch 负样本 batch 太小**。

batch=16 时只有 15 个免费负样本，信号弱。用 batch=256+ 或 big-batch 训练（gradient cache 等技巧）。

**坑 4：温度系数没调**。

$\tau = 1$ 时 loss 平坦，梯度小，学不动。$\tau = 0.01 \sim 0.1$ 是经验区间。CLIP 用 0.07。

**坑 5：训练数据和部署场景不匹配**。

用通用语料训练的 embedding，部署到垂直领域（法律、医疗）效果可能差。要么领域微调，要么用 BGE 等支持领域适配的模型。

**坑 6：embedding 模型升级没做版本管理**。

换模型后向量维度/语义变了，旧索引不能用。新旧索引共存过渡，灰度切换。

**坑 7：长文档压成单一向量**。

长文档（> 512 tokens）截断后信息丢失。要么用 sliding window 切片各自编码后聚合（max / mean pooling），要么用 ColBERT 等多向量方法。

**坑 8：查询文档长度差异大用共享 encoder**。

查询通常短（< 20 tokens），文档长（200~500 tokens）。共享 encoder 训练时输入分布不一致，可能性能不佳。可考虑独立 encoder 或位置编码调整。

**坑 9：忘了难负样本挖掘**。

只用 in-batch 负样本训练，模型见过的负样本有限。定期用当前模型检索 top-k，标注后作为下一轮难负样本（ANCE 异步刷新策略）。

**坑 10：评估只用检索指标**。

双塔评估除了 recall@k、MRR，还要看向量分布质量（向量均匀性、各向同性）。分布退化（向量集中在窄锥）会让 ANN 索引失效。

### 5.3 双塔 vs Cross-encoder vs ColBERT

| 维度 | 双塔 (Bi-encoder) | 交叉编码器 (Cross-encoder) | ColBERT (Late Interaction) |
|------|--------------------|-----------------------------|----------------------------|
| 查询文档交互 | 仅向量点积 | 全交叉 attention | token 级 max-sim |
| 文档向量预计算 | 行 | 不行 | 行（per-token 向量） |
| 检索效率 | 极快（ANN） | 慢（每对前向） | 中（ANN + 精排） |
| 精度 | 中 | 高 | 高 |
| 索引大小 | 小（每文档一向量） | 无 | 大（每文档 N 向量） |
| 适用 | 召回（百万级） | 精排（top-100） | 召回 + 精排 |

工程实践：

- **小规模 + 高精度**：直接 ColBERT 端到端
- **大规模 + 通用**：双塔召回 + Cross-encoder 精排
- **极致精度**：双塔 → ColBERT → Cross-encoder 三段式

### 5.4 何时仍该用 BM25 + 双塔混合

双塔虽强，这些场景仍需 BM25 兜底：

1. **精确匹配**：专有名词、代码、ID、产品型号
2. **长尾词**：embedding 训练时没见过的词
3. **可解释性**：分数可追溯到具体词命中
4. **极低延迟**：CPU 毫秒级，双塔需 GPU 推理
5. **零样本冷启动**：无训练数据时 BM25 直接可用

现代 RAG 系统标配：**BM25 + Dense 双路召回 + RRF 融合 + Cross-encoder 精排**。

---

## 速记卡

| 维度 | 双塔模型 |
|------|----------|
| 架构 | query encoder + doc encoder，点积得分 |
| 训练 | 对比学习（InfoNCE），难负样本关键 |
| 关键参数 | 温度 $\tau$、batch size、pooling 策略 |
| 优势 | 文档向量预计算，ANN 毫秒级检索 |
| 局限 | 查询文档互盲、词级匹配弱、长尾词差 |
| 接班 | ColBERT（late interaction）、Cross-encoder（精排） |
| 仍用于 | 大规模稠密召回、推荐系统召回 |

**核心公式**：

- 训练：$\mathcal{L} = -\log \frac{\exp(\vec{q} \cdot \vec{d^+} / \tau)}{\exp(\vec{q} \cdot \vec{d^+} / \tau) + \sum_{d^-} \exp(\vec{q} \cdot \vec{d^-} / \tau)}$
- 推理：$\text{score}(q, d) = E_q(q) \cdot E_d(d)$

**一句话记忆**：双塔 = 查询塔 + 文档塔 + 向量点积。文档向量离线预计算，查询向量在线编码，ANN 毫秒级检索。召回主力，搭配 Cross-encoder 精排和 BM25 兜底。

---

> *上一篇：[向量数据库](./vector-database) -- 向量检索的工程基础设施。*
> *下一篇：[交叉编码器 Cross-encoder](./cross-encoder) -- 精排主力，弥补双塔的互盲缺陷。*
