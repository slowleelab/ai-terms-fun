---
title: 交叉编码器 Cross-encoder
slug: cross-encoder
category: 检索与召回
tags: [神经检索, 交叉编码器, 精排, rerank, BERT, 召回]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 交叉编码器 Cross-encoder

> 五层读懂一个词。这次拆的是：**Cross-encoder**--检索精排的主力，弥补双塔互盲缺陷。

---

## L1 · 一句话点破

**Cross-encoder 把查询和文档拼成一句话过同一个 transformer，让每个 token 都能 attend 到对方，精度碾压双塔，但文档向量无法预计算，只能用于精排 top-100**。

---

## L2 · 通俗类比

双塔模型像"相亲双方各自填表"：query 填一张"语义画像"表，doc 填一张，两表对照算相似度。简单快，但双方填表时彼此看不到对方--"苹果"是水果还是公司，得自己猜，猜错了就错配。

Cross-encoder 像"面对面相亲"：query 和 doc 同坐一桌，逐字逐句相互交流，每个词都能感知对方的反应。聊完才能打分。精度高得多，但慢得多--每对都要聊一次，百万级文档得聊百万次。

工程取舍：

- 召回阶段：双塔快速从百万级筛到 top-100（毫秒级）
- 精排阶段：Cross-encoder 把 top-100 重排到 top-10（百毫秒级）

两阶段配合：双塔保证召回率和速度，Cross-encoder 保证精度。

---

## L3 · 正经定义

**Cross-encoder (Cross-encoder Reranker)**：查询 $q$ 和文档 $d$ 拼接为 $[\text{CLS}] q [\text{SEP}] d [\text{SEP}]$，过一遍 transformer encoder，取 [CLS] 输出过线性层得到相关性分数。

$$
\text{score}(q, d) = W \cdot \text{Transformer}([\text{CLS}] q [\text{SEP}] d [\text{SEP}])_{[\text{CLS}]} + b
$$

**与双塔的区别**：

- 双塔：$E_q(q) \cdot E_d(d)$，两塔独立
- Cross-encoder：拼接后单塔前向，全交叉 attention

**训练**：监督学习，正负样本对，binary cross-entropy 或 contrastive loss。

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class CrossEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids, attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS]
        logit = self.classifier(cls).squeeze(-1)
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logit, labels.float())
            return loss, logit
        return logit

# 训练
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
queries = ["什么是 transformer", "RLHF 是什么"]
docs = ["Transformer 是一种基于 attention 的架构", "RLHF 是基于人类反馈的强化学习"]
labels = [1.0, 1.0]  # 1=相关，0=不相关

batch = tokenizer(queries, docs, padding=True, truncation=True, return_tensors='pt')
loss, logits = model(batch['input_ids'], batch['attention_mask'], torch.tensor(labels))
loss.backward()
```

**推理**：

```python
# 召回阶段：双塔 + ANN 得到 top-100
candidates = bi_encoder_retrieve(query, top_k=100)

# 精排阶段：Cross-encoder 重排
pairs = [(query, c.text) for c in candidates]
batch = tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
                  padding=True, truncation=True, return_tensors='pt')
scores = model(batch['input_ids'], batch['attention_mask'])
reranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:10]
```

**代表模型**：

- **BERT-based Cross-encoder**：原始 BERT 微调（MS MARCO 排序）
- **ms-marco-MiniLM-L-12-v2**：sentence-transformers 提供的预训练 reranker
- **MonoBERT / MonoT5**（Nogueira et al.）：BERT / T5 用于排序
- **ColBERT v2 reranker**：ColBERT 系列的精排变体
- **Cohere Rerank**：商业 API
- **bge-reranker-large**：BGE 系列的 reranker

---

## L4 · 原理深挖

### 4.1 为什么 Cross-encoder 精度高

双塔的根本局限是**查询文档互盲**：query encoder 编码时不知道 doc 是什么，"苹果"的语义只能猜。

Cross-encoder 把 query 和 doc 拼一起过 transformer，self-attention 让 query 的每个 token 都能 attend 到 doc 的每个 token（反过来亦然）。效果：

1. **词级消歧**：query "苹果"看到 doc "Steve Jobs, iPhone, 库克"，立刻消歧为公司
2. **细粒度匹配**：query "iPhone 15 Pro" 和 doc "iPhone 15 Pro Max" 能精确对比
3. **语义组合**：query "猫吃鱼" 和 doc "鱼被猫吃"，词序不同但语义关系通过 attention 捕获
4. **否定与修饰**：query "不喜欢苹果手机" 和 doc "苹果手机很好"，Cross-encoder 能识别否定

这些能力是双塔丢失的。Cross-encoder 用全交叉 attention 换回精度，代价是计算量。

### 4.2 计算复杂度对比

设 query 长度 $L_q$，doc 长度 $L_d$，transformer 层数 $N$，hidden size $D$。

**双塔**：

- 训练时每对：$2 \cdot N \cdot L^2 \cdot D$（两个独立 forward，$L = \max(L_q, L_d)$）
- 推理时单 query + 单 doc：$N \cdot L_q^2 \cdot D + N \cdot L_d^2 \cdot D + L_q \cdot D$（前两个是 encoder，最后是点积）
- 推理时百万 doc：只需 $N \cdot L_q^2 \cdot D$（doc 向量预计算），ANN 毫秒级

**Cross-encoder**：

- 训练和推理每对：$N \cdot (L_q + L_d)^2 \cdot D$（拼接后单次 forward）
- 推理时百万 doc：百万次 forward，不可行
- 推理时 top-100：100 次 forward，百毫秒级

**关键差异**：

- 双塔：doc 向量**预计算一次**，检索时只算 query
- Cross-encoder：每对都要前向，无法预计算

这就是为什么 Cross-encoder 只能精排 top-100，不能召回百万级。

### 4.3 Cross-encoder 的训练

**数据**：query-doc 对 + 相关性标签（0/1 或分级）。

**Loss**：

- **Binary Cross-Entropy**：每对独立 0/1 分类
  $$\mathcal{L} = -[y \log \sigma(s) + (1-y) \log(1 - \sigma(s))]$$
- **Contrastive / Multiple Negatives**：1 正 + K 负，InfoNCE
  $$\mathcal{L} = -\log \frac{\exp(s^+ / \tau)}{\exp(s^+ / \tau) + \sum_i \exp(s^-_i / \tau)}$$
- **Pairwise / Hinge**：正样本对分数比负样本对高 margin
  $$\mathcal{L} = \max(0, \text{margin} - (s^+ - s^-))$$
- **Listwise / RankNet / LambdaRank**：对一组候选排序优化

**实践**：监督数据多用 binary CE 或 InfoNCE；少量标注用 pairwise。

### 4.4 截断长度选择

Cross-encoder 输入是 query + doc 拼接，doc 可能很长。如何处理？

- **截断到 512 tokens**：BERT 默认，doc 尾部丢失
- **截断到 256 tokens**：节省算力，doc 信息丢失更多
- **Longformer / BigBird**：支持 4096+ tokens，但模型大、慢
- **滑窗 + 聚合**：doc 切片各自打分，max / mean 聚合

**实践**：

- 短 doc（标题、段落）：512 够用
- 长 doc（论文、网页全文）：滑窗或 Longformer
- 索引设计阶段就切 chunk 到合适长度（如 256/512 tokens），避免精排时再处理

### 4.5 Cross-encoder 的效率优化

Cross-encoder 慢，但有优化空间：

**优化 1：知识蒸馏到双塔**。训练一个高精度 Cross-encoder 作为 teacher，蒸馏到双塔 student。Student 性能提升但保持检索速度。这是 DistilBERT、TinyBERT 在检索场景的用法。

**优化 2：缓存**。对同一 query，候选 doc 的 Cross-encoder 分数可缓存（如 LRU cache 1000 query × 100 doc）。命中率高时大幅减少推理。

**优化 3：量化**。Cross-encoder 用 FP16 或 INT8 量化，速度快 2~4 倍。精度损失通常 < 1%。

**优化 4：批量化**。top-100 候选打包成 batch 一次前向，比逐个前向快 10 倍+。

**优化 5：early exit**。transformer 层层有 logits，前几层就高置信的样本提前退出。DeeBERT 等工作证明可行。

**优化 6：ONNX / TensorRT**。导出 ONNX 后用 TensorRT 优化，速度再提升 2~5 倍。

### 4.6 精排不是终点：多阶段排序

工业级检索系统常是多阶段：

```
1. 召回（双塔 + ANN）：百万级 -> top-1000
2. 粗排（轻量 Cross-encoder 或 LGBM）：top-1000 -> top-100
3. 精排（重 Cross-encoder）：top-100 -> top-10
4. 重排（多样性 / 业务规则）：top-10 -> 最终展示
```

每阶段过滤比例约 10:1。前阶段保 recall，后阶段求 precision。

**典型组合**：

- 召回：双塔（DPR）+ BM25 混合
- 粗排：轻量 cross-encoder（MiniLM）或 GBDT
- 精排：重 cross-encoder（large BERT）
- 重排：MMR 多样性、业务加权

### 4.7 Cross-encoder vs 双塔 vs ColBERT

| 维度 | 双塔 (Bi-encoder) | Cross-encoder | ColBERT (Late Interaction) |
|------|--------------------|----------------|----------------------------|
| 查询文档交互 | 仅向量点积 | 全交叉 attention | token 级 max-sim |
| 文档向量预计算 | 行 | 不行 | 行（per-token 向量） |
| 召回阶段适用 | 行 | 不行（太慢） | 行 |
| 精排阶段适用 | 不行（精度差） | 行 | 行 |
| 计算复杂度（单对） | $O(L^2)$ | $O((L_q + L_d)^2)$ | $O(L_q \cdot L_d)$ |
| 索引大小 | 小（每文档一向量） | 无 | 大（每文档 N 向量） |
| 精度 | 中 | 高 | 高 |
| 典型延迟 | < 10ms（召回） | 50~200ms（top-100） | 10~50ms（top-1000） |

**ColBERT 的位置**：介于双塔和 Cross-encoder 之间，文档 per-token 向量可预计算，查询时 token 级 max-sim，比双塔精但比 Cross-encoder 快。详见 ColBERT 词条。

### 4.8 Cross-encoder 的局限

**局限 1：无法召回**。每对都要前向，百万级不可行。必须配合双塔召回。

**局限 2：长文档处理差**。512 tokens 截断丢信息，长文档需滑窗或 Longformer。

**局限 3：训练数据依赖**。需要大量 query-doc 标注对，与双塔一样依赖数据质量。

**局限 4：推理成本高**。top-100 精排需百毫秒级，对延迟敏感场景需优化。

**局限 5：泛化能力**。在训练领域外可能性能下降，需持续 fine-tune。

**局限 6：解释性差**。黑盒打分，不如 BM25 能追溯到词项。

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **2015**：早期 BERT 之前，用 LSTM 做查询文档匹配（如 MatchPyramid）
- **2019**：BERT-based reranker 兴起（MonoBERT, Nogueira et al.），TREC 精度大幅提升
- **2020**：DPR 论文同时验证 Cross-encoder 精排提升 recall@5 约 5~10 个百分点
- **2020**：sentence-transformers 推出预训练 cross-encoder 系列
- **2021**：MonoT5（用 T5 生成相关 / 不相关 token 表示相关性）
- **2022**：bge-reranker、Cohere Rerank 等商业方案出现
- **2023**：LLM-based reranker（用 LLM 直接打分或生成排序）兴起
- **2024**：多阶段排序成为标配，Cross-encoder 仍是精排主力

### 5.2 使用常见坑

**坑 1：用 Cross-encoder 做召回**。

Cross-encoder 每对都要前向，百万级不可行。硬要用只能装小规模数据（< 10^4），失去扩展性。

**坑 2：top_k 太小，召回不足**。

精排前 top_k=50，若双塔召回的真正相关文档在第 51~100 位，精排再准也救不回来。建议 top_k=100~200 给精排足够候选。

**坑 3：长文档截断**。

doc 截断到 512 tokens 后尾部信息丢失。索引设计时就切 chunk 到合适长度，或用滑窗。

**坑 4：忘了批量化**。

逐个对 query-doc 前向，速度慢 10 倍+。一定要 batch 化，top-100 一次前向。

**坑 5：训练数据标签噪声**。

人工标注 0/1 标签可能有噪声。用 multi-annotator 投票或 soft label（连续值）缓解。

**坑 6：Cross-encoder 与双塔训练数据不匹配**。

双塔训练时见过的负样本分布，与 Cross-encoder 精排时见的 top-100 候选分布不同。Cross-encoder 训练时要用"双塔召回 + 难负样本"的分布，否则部署时泛化差。

**坑 7：推理未量化**。

FP32 推理慢，FP16 或 INT8 量化后速度快 2~4 倍，精度损失 < 1%。生产环境必量化。

**坑 8：缓存策略不对**。

同一 query 的候选 doc 分数可缓存，但若 doc 库频繁更新，缓存失效快。要根据数据更新频率调缓存策略。

**坑 9：评估指标错位**。

Cross-encoder 评估要用精排指标（recall@10, MRR@10 在 top-100 候选上算），不是召回指标（recall@100 在全库上算）。

**坑 10：直接用通用 Cross-encoder**。

通用预训练 reranker 在垂直领域可能差。法律、医疗、代码等需领域微调。

### 5.3 Cross-encoder 的现代变体

**MonoT5**（Nogueira et al. 2020）：把 reranker 改成生成任务，让 T5 输出 "true" 或 "false" 表示相关。性能接近 BERT-reranker，但用 T5 的生成能力可扩展。

**LLM-based reranker**：用 GPT-4 / Claude 直接对候选打分或排序。精度高但成本高、延迟大，适合离线分析或高价值场景。

**Pairwise reranker**：每次比较两个候选哪个更好，类似 LLM 的成对比较。对 top-10 重排有用，但 O(N^2) 复杂度限制规模。

**Listwise reranker**：一次性看所有候选，输出排序。RankNet、LambdaRank 等经典方法。

**Cross-encoder + 业务规则混合**：精排分数 × 业务权重（如新鲜度、点击率、个性化）。生产系统几乎都做这个。

### 5.4 何时该用 Cross-encoder

**应该用**：

- 精度敏感场景（如企业搜索、客服问答）
- top_k 候选规模可控（< 200）
- 延迟容忍百毫秒级
- 有训练数据可微调

**不该用**：

- 极致低延迟（< 50ms）
- 候选规模 > 1000（先用粗排）
- 无训练数据，零样本场景
- 资源受限（移动端、边缘设备）

**替代方案**：

- 精度要求中：双塔 + 难负样本训练
- 精度要求高 + 召回规模大：双塔 + ColBERT + Cross-encoder 三段式
- 极致精度：LLM-based reranker（成本高）

---

## 速记卡

| 维度 | Cross-encoder |
|------|---------------|
| 架构 | query + doc 拼接过 transformer |
| 训练 | binary CE / InfoNCE / pairwise |
| 优势 | 全交叉 attention，精度高 |
| 局限 | 无法预计算，只能精排 top-100 |
| 典型延迟 | 50~200ms（top-100 重排） |
| 优化 | batch / 量化 / 缓存 / ONNX |
| 适用 | 精排，与双塔召回配合 |

**核心公式**：

- 推理：$\text{score}(q, d) = W \cdot \text{Transformer}([\text{CLS}] q [\text{SEP}] d [\text{SEP}])_{[\text{CLS}]} + b$
- 复杂度：$O((L_q + L_d)^2)$ 每对

**一句话记忆**：Cross-encoder = 查询文档拼接过 transformer，全交叉 attention 高精度，但每对都要前向，只能精排 top-100。配合双塔召回的两阶段架构：双塔保召回率，Cross-encoder 保精度。

---

> *上一篇：[双塔模型 Two-Tower](./two-tower) -- 稠密召回的主力架构。*
> *下一篇：[ColBERT](./colbert) -- late interaction，介于双塔与 Cross-encoder 之间的折中方案。*
