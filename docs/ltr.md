---
title: 学习排序 LTR
slug: ltr
category: 检索与召回
tags: [学习排序, LTR, LambdaMART, RankNet, 排序模型, 检索]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# 学习排序 LTR

> 五层读懂一个词。这次拆的是：**LTR (Learning to Rank)**--用机器学习排序，融合多路特征，是检索系统精排的终极形态。

---

## L1 · 一句话点破

**LTR 用监督学习训练排序模型，输入查询和文档的多路特征（BM25 分数、Dense 相似度、点击率、文档质量等），输出相关性分数**。比 RRF/加权融合更精细，但需训练数据和工程投入，是工业级精排的天花板。

---

## L2 · 通俗类比

RRF / 加权融合像"按固定规则评分"：

- RRF：只看排名，固定公式
- 加权融合：归一化 + 固定权重

LTR 像"训练一个评委"：

- 给评委看大量"查询-文档-真实相关性"样本
- 评委学习：什么特征组合意味着高相关
- 训练好后，评委能给新查询-文档对打分

评委可以考虑更多特征：

- 文本匹配分数（BM25、Dense、Cross-encoder）
- 文档质量（长度、权威度、新鲜度）
- 用户行为（点击率、停留时间）
- 个性化（用户偏好、历史）

固定规则没法综合考虑这些，LTR 可以。

代价：

- 需要大量训练数据（标注或点击日志）
- 工程投入大（特征工程、模型训练、部署）
- 训练周期长，迭代慢

工程取舍：**小系统用 RRF/加权融合，大系统用 LTR**。Google、百度、Bing 的精排都是 LTR。

---

## L3 · 正经定义

**LTR (Learning to Rank)**：用监督学习训练排序模型 $f(q, d, \text{features}) \to \text{score}$，按 score 排序。

**输入特征**（典型）：

| 类别 | 特征 |
|------|------|
| 文本匹配 | BM25 分数、TF-IDF、查询词命中数 |
| 语义匹配 | Dense 相似度、Cross-encoder 分数 |
| 文档质量 | PageRank、文档长度、发布时间 |
| 用户行为 | 历史点击率、停留时间、跳出率 |
| 查询特征 | 查询长度、查询意图分类 |
| 个性化 | 用户画像、历史查询 |

**三种损失函数**：

**1. Pointwise**：每个文档独立打分，回归或分类

$$
\mathcal{L} = \sum_i (f(q_i, d_i) - y_i)^2
$$

$y_i$ 是相关性标签（0/1 或分级）。简单但忽略文档间相对顺序。

**2. Pairwise**：文档对比较，正确顺序的分数差应大

$$
\mathcal{L} = \sum_{i, j : y_i > y_j} \max(0, \text{margin} - (f(q, d_i) - f(q, d_j)))
$$

考虑文档对相对顺序，比 pointwise 更符合排序目标。代表：RankNet。

**3. Listwise**：直接优化排序列表指标（NDCG 等）

$$
\mathcal{L} = -\text{NDCG}(\text{ranked list by } f)
$$

直接对齐排序指标，效果最好但复杂。代表：LambdaMART、ListNet。

**代表模型**：

**经典 LTR**：

- **RankNet**（Burges et al. 2005）：pairwise，神经网络
- **LambdaRank**（Burges et al. 2006）：改进 RankNet，梯度加权 NDCG
- **LambdaMART**（Burges et al. 2010）：LambdaRank + GBDT，事实标准
- **MART / GBDT**：树模型，特征工程强

**深度 LTR**：

- **DNN LTR**：深度神经网络，可融合文本和特征
- **BERT-based LTR**：BERT 输出 + 特征，端到端
- **Wide & Deep**：Google 推荐，wide（记忆）+ deep（泛化）

**伪代码**（LambdaMART）：

```python
import lightgbm as lgb
import numpy as np

# 训练数据：每行是 (query_id, features, relevance)
# features: [bm25_score, dense_sim, cross_enc_score, doc_len, ctr, ...]
train_data = [
    # query 1: doc A (rel=3), doc B (rel=1), doc C (rel=0)
    (1, [12.5, 0.85, 0.92, 500, 0.15], 3),  # doc A
    (1, [8.0, 0.78, 0.85, 1200, 0.08], 1),  # doc B
    (1, [15.0, 0.65, 0.70, 300, 0.02], 0),  # doc C
    # query 2: ...
]

# 构造 LightGBM 数据集
X = np.array([row[1] for row in train_data])
y = np.array([row[2] for row in train_data])
group = [3, ...]  # 每 query 的文档数

train_set = lgb.Dataset(X, label=y, group=group)

# 训练 LambdaMART
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5, 10],
    'learning_rate': 0.1,
    'num_leaves': 31,
}
model = lgb.train(params, train_set, num_boost_round=100)

# 推理
def ltr_rerank(query, candidates, features_extractor):
    """用 LTR 重排候选"""
    features = [features_extractor(query, c) for c in candidates]
    scores = model.predict(features)
    reranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return reranked
```

---

## L4 · 原理深挖

### 4.1 三种损失函数的对比

**Pointwise**：

- 优点：简单，可用任意回归/分类模型
- 缺点：忽略文档间相对顺序，与排序目标不完全对齐
- 适用：相关性标签连续（如评分）

**Pairwise**：

- 优点：考虑文档对相对顺序，更符合排序
- 缺点：文档对数量 $O(N^2)$，训练慢；忽略列表整体
- 适用：相关性标签二元（相关/不相关）

**Listwise**：

- 优点：直接优化排序指标（NDCG），效果最好
- 缺点：复杂，梯度计算难
- 适用：追求极致精度

**实测**：Listwise (LambdaMART) > Pairwise (RankNet) > Pointwise。

### 4.2 LambdaMART：事实标准

**LambdaMART** 是工业 LTR 的事实标准，原理：

1. **GBDT 基模型**：用梯度提升决策树，特征工程强、可解释
2. **Lambda 梯度**：不是直接用 loss 梯度，而是用 $\lambda$（与 NDCG 相关的梯度）

**$\lambda$ 的定义**：对文档对 $(i, j)$（$i$ 比 $j$ 更相关但排序反了）：

$$
\lambda_{ij} = \frac{-1}{1 + \exp(f(q, d_i) - f(q, d_j))} \cdot |\Delta \text{NDCG}_{ij}|
$$

$|\Delta \text{NDCG}_{ij}|$ 是交换 $i, j$ 后 NDCG 变化量。NDCG 影响大的对梯度大，模型更努力修正。

**优势**：

- 直接优化 NDCG，对齐排序目标
- GBDT 处理特征非线性、缺失值
- 工程成熟（LightGBM、XGBoost 都支持）

### 4.3 LTR 的特征工程

LTR 性能高度依赖特征质量。典型特征：

**检索特征**：

- BM25 分数（多字段：title、body、anchor）
- TF-IDF 分数
- 查询词命中位置（title 比 body 权重高）
- Dense 相似度
- Cross-encoder 分数
- ColBERT max-sim

**文档特征**：

- 文档长度、词数
- PageRank、权威度
- 发布时间、新鲜度
- 文档质量分（如内容完整度）

**查询特征**：

- 查询长度、词数
- 查询意图分类（导航/信息/事务）
- 查询流行度
- 查询地理/语言

**用户行为特征**：

- 历史点击率（query-doc 对）
- 平均停留时间
- 跳出率
- 用户偏好

**交叉特征**：

- 查询-文档主题匹配度
- 用户-文档历史交互
- 个性化偏好

**特征工程经验**：

- 检索特征是基础（BM25 + Dense + Cross-encoder）
- 文档特征提升质量分排序
- 用户行为特征是金矿（点击日志）
- 交叉特征提升个性化

### 4.4 LTR 的训练数据

**数据来源**：

1. **人工标注**：高质量但贵，通常 10K~100K 样本
2. **点击日志**：大规模但噪声大（点击 ≠ 相关），需去偏
3. **搜索日志 + 人工标注混合**：常用做法
4. **合成数据**：用 LLM 生成训练对

**点击日志的去偏**：

- **位置偏置**：top 位置点击率高，不是因为更相关
- **信任偏置**：用户信任搜索引擎，top 都点
- **去偏方法**：Propensity scoring、反事实评估、click model

**典型规模**：

- 小规模：10K query × 100 doc/query = 1M 样本
- 中规模：100K query × 50 doc/query = 5M 样本
- 大规模：1M query × 20 doc/query = 20M 样本

### 4.5 LTR 在多阶段排序中的位置

LTR 通常用在精排或粗排阶段：

```
1. 召回（双塔 + ANN + BM25）：百万 -> 千
2. 粗排（轻量 LTR 或 GBDT）：千 -> 百
3. 精排（重 LTR 或 Cross-encoder）：百 -> 十
4. 重排（多样性 / 业务规则）：十 -> 展示
```

**粗排 LTR**：

- 轻量特征（无 Cross-encoder）
- GBDT 或小 DNN
- 目标：保留 top-100 中 90%+ 真正相关

**精排 LTR**：

- 全特征（含 Cross-encoder 分数）
- LambdaMART 或 DNN
- 目标：top-10 中 8+ 相关

### 4.6 LTR vs Cross-encoder

| 维度 | LTR (LambdaMART) | Cross-encoder |
|------|------------------|----------------|
| 输入 | 多路特征 | 文本对 |
| 文本理解 | 间接（依赖特征） | 直接（attention） |
| 特征工程 | 强 | 弱 |
| 可解释 | 强 | 弱 |
| 训练数据 | 点击日志 + 标注 | 标注对 |
| 部署 | 快（GBDT） | 慢（transformer） |
| 适用 | 精排（多特征） | 精排（纯文本） |

**实践**：

- 文本为主：Cross-encoder
- 多特征融合：LTR
- 极致精度：LTR + Cross-encoder 分数作为特征

### 4.7 LTR 的局限

**局限 1：训练数据依赖**。

需大量标注或点击日志，小公司难获取。点击日志有偏（位置偏置等）。

**局限 2：特征工程繁琐**。

设计、维护特征管道复杂。新特征上线需 A/B 测试验证。

**局限 3：迭代慢**。

LTR 训练周期长（小时级），新数据上线慢。不如 Cross-encoder 微调灵活。

**局限 4：泛化能力**。

训练领域外性能下降，需持续 fine-tune。

**局限 5：冷启动差**。

新文档、新查询无历史特征，LTR 表现差。

**局限 6：部署复杂**。

GBDT 相对简单，DNN LTR 需 GPU。特征管道维护成本高。

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **2005**：RankNet（Burges et al.），pairwise 神经网络
- **2006**：LambdaRank，改进梯度加权 NDCG
- **2010**：LambdaMART（Burges et al.），LambdaRank + GBDT，工业标准
- **2010s**：Yahoo Learning to Rank Challenge、Yandex 等推动 LTR 普及
- **2013**：GBDT 工具成熟（XGBoost、LightGBM）
- **2016**：Wide & Deep（Google 推荐），深度 LTR 兴起
- **2019+**：BERT-based LTR 出现，但 GBDT 仍是主力（特征工程强）
- **2023+**：LLM-based reranker 兴起，但 LTR 在多特征场景仍不可替代

### 5.2 使用常见坑

**坑 1：训练数据有偏**。

点击日志有位置偏置，直接训练 LTR 会学偏。要用 click model 或 propensity scoring 去偏。

**坑 2：特征泄漏**。

训练时用了推理时不可得的特征（如未来点击）。要严格按时间划分训练/验证集。

**坑 3：特征过多**。

几百特征训练慢且易过拟合。要做特征选择（重要性排序、相关性分析）。

**坑 4：忘了特征工程**。

直接用原始分数，不做交叉特征、时序特征。LTR 性能依赖特征质量。

**坑 5：评估只看离线指标**。

NDCG@10 提升不等于在线指标提升。要做 A/B 测试验证。

**坑 6：模型迭代慢**。

LTR 训练周期长，新数据上线慢。要建立快速迭代流水线。

**坑 7：冷启动差**。

新文档无历史特征，LTR 排名低。要用内容特征兜底，或 explore-exploit 策略。

**坑 8：特征管道不同步**。

训练时特征 A，推理时特征 A 计算方式变了。要严格版本管理。

**坑 9：Listwise 损失计算慢**。

LambdaMART 的 $\lambda$ 梯度计算 $O(N^2)$ 每 query。要限制每 query 文档数（如 50）。

**坑 10：LTR 替代 Cross-encoder**。

LTR 多特征强，但纯文本理解不如 Cross-encoder。两者应配合：Cross-encoder 分数作为 LTR 特征。

### 5.3 LTR 的现代演进

**Neural LTR**：

- 用 DNN 替代 GBDT，可融合文本 embedding
- BERT-based LTR：文本端到端 + 特征
- 性能略优于 GBDT，但工程复杂

**Multi-objective LTR**：

- 同时优化相关性、点击率、收入
- 多任务学习

**Contextual LTR**：

- 考虑用户上下文（会话、位置、设备）
- 个性化排序

**LLM-based Reranker**：

- 用 LLM 直接打分或生成排序
- 精度高但成本高
- 适合高价值场景或离线分析

**Online LTR**：

- 在线学习，实时更新
- 适应分布漂移

### 5.4 何时用 LTR

**适合用**：

- 大规模搜索系统（百万级 + 千万级用户）
- 有大量点击日志或标注数据
- 需要多特征融合（文本 + 行为 + 个性化）
- 追求极致精度
- 团队有 ML 工程能力

**不适合**：

- 小规模系统（< 10K 文档）
- 无训练数据
- 快速原型
- 单路检索（用 Cross-encoder 即可）
- 资源受限

**典型架构**：

- 小系统：BM25 + Dense 召回 + Cross-encoder 精排
- 中系统：多路召回 + GBDT LTR 精排
- 大系统：多路召回 + 多阶段 LTR（粗排 GBDT + 精排 DNN）

### 5.5 LTR 在 RAG 中的角色

RAG 系统通常简化版：

- 召回：BM25 + Dense 混合
- 精排：Cross-encoder 或 LLM-reranker
- 不用 LTR（数据量小、特征少）

但企业级 RAG（如客服系统）可能用 LTR：

- 大量用户查询日志
- 文档质量分、用户满意度
- 多特征融合提升精度

---

## 速记卡

| 维度 | LTR |
|------|-----|
| 输入 | 多路特征（检索 + 文档 + 行为） |
| 损失 | Pointwise / Pairwise / Listwise |
| 事实标准 | LambdaMART (GBDT + Listwise) |
| 优势 | 多特征融合、可解释、精度高 |
| 劣势 | 数据依赖、特征工程繁琐、迭代慢 |
| 适用 | 大规模搜索系统精排 |
| 替代 | Cross-encoder（纯文本）、LLM-reranker |

**三种损失函数**：

| 类型 | 公式 | 优势 | 代表 |
|------|------|------|------|
| Pointwise | $\sum (f - y)^2$ | 简单 | 回归 |
| Pairwise | $\max(0, \text{margin} - (f_i - f_j))$ | 考虑对序 | RankNet |
| Listwise | $-\text{NDCG}$ | 对齐指标 | LambdaMART |

**典型特征**：

- 检索特征：BM25、Dense、Cross-encoder 分数
- 文档特征：长度、权威度、新鲜度
- 行为特征：点击率、停留时间
- 交叉特征：查询-文档主题匹配

**一句话记忆**：LTR = 监督学习训练排序模型，输入多路特征输出相关性分数。LambdaMART (GBDT + Listwise) 是事实标准。比 RRF/加权融合精细，但需训练数据和特征工程。大规模搜索系统精排的终极形态。

---

> *上一篇：[加权重排 Weighted Fusion](./weighted-fusion) -- RRF 的替代方案。*
> *下一篇：[Hit Rate](./hit-rate) -- 检索评估的基础指标，检索与召回类完成。*
