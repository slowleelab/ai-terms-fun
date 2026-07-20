---
title: TF-IDF
slug: tf-idf
category: 检索与索引
tags: [检索, 关键词权重, 经典算法, 统计, 信息检索]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# TF-IDF

> 五层读懂一个词。这次拆的是：**TF-IDF**——检索系统里最经典的词项权重方案。

---

## L1 · 一句话点破

**一个词对一篇文档有多重要，由它在本文档里多频繁（TF）、在全局里多普遍（IDF）共同决定。**

频繁出现 → 重要；但到处都出现 → 没区分度。两个信号相乘，得到一个能挑出"这篇文档的代表性词"的权重。

---

## L2 · 通俗类比

想象你在给一篇论文打标签。

- "的、是、和"——论文里出现几百次，但每篇论文都有，没区分度，不算标签。
- "transformer"——出现 20 次，且只在这少数几篇里出现，是强标签。
- "model"——出现 50 次，但每篇 AI 论文都有，弱标签。

TF-IDF 就是把这个直觉量化：**本文档高频 × 全局稀有 = 该词对本文档的权重**。

再换一个比喻：你在饭局上判断一个人是不是这个圈子的核心。

- 他在这场饭局说话多（TF 高）
- 但他很少出现在别的饭局（IDF 高）

→ 他是这场饭局的代表人物。如果他每场饭局都来，那就算他说很多话，也不能代表"这场"——因为到处都是他。

---

## L3 · 正经定义

**TF-IDF (Term Frequency–Inverse Document Frequency)**：对词项 $t$ 在文档 $d$ 中的权重定义为

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \text{idf}(t)
$$

其中：

- $\text{tf}(t, d)$：词项 $t$ 在文档 $d$ 中的词频。常见取法：
  - 原始计数：$\text{tf}(t, d) = f(t, d)$
  - 归一化：$\text{tf}(t, d) = f(t, d) / |d|$（$|d|$ 为文档长度）
  - 对数缩放：$\text{tf}(t, d) = 1 + \log f(t, d)$（$f=0$ 时取 0）
  - 增强型：$0.5 + 0.5 \cdot \frac{f(t, d)}{\max_{t'} f(t', d)}$（augmented TF，防止单词主导）

- $\text{idf}(t)$：逆文档频率，衡量词项的稀有度。经典定义：

$$
\text{idf}(t) = \log \frac{N}{\text{df}(t)}
$$

  其中 $N$ 是语料库文档总数，$\text{df}(t)$ 是包含 $t$ 的文档数。$\text{df}(t) = 0$ 时需平滑处理（如 $\log \frac{N+1}{\text{df}(t)+1}$ 加 1 防零除）。

**直观含义**：

- $\text{df}(t) = N$（每个文档都有 $t$）→ $\text{idf}(t) = \log 1 = 0$ → 权重归零。
- $\text{df}(t) = 1$（只有一个文档有 $t$）→ $\text{idf}(t) = \log N$ → 高权重。

**查询打分**：对查询 $q$ 和文档 $d$ 的相关性分数为

$$
\text{score}(q, d) = \sum_{t \in q} \text{tfidf}(t, d)
$$

只累加查询中出现的词项。这是向量空间检索的基础：把文档表示成 TF-IDF 向量，查询也向量，余弦相似度即得分。

**伪代码**：

```python
import math
from collections import Counter

def tfidf_score(query, doc, df_map, N):
    """query, doc: 词列表；df_map: 词 -> 包含该词的文档数；N: 总文档数"""
    tf = Counter(doc)
    doc_len = len(doc)
    score = 0.0
    for t in query:
        if t not in tf:
            continue
        tf_t = tf[t] / doc_len                      # 归一化 TF
        idf_t = math.log((N + 1) / (df_map.get(t, 0) + 1)) + 1  # 平滑 IDF
        score += tf_t * idf_t
    return score
```

**历史地位**：Karen Spärck Jones 1972 年提出 IDF 思想，与词频思想结合成 TF-IDF，统治信息检索近三十年。Lucene 的 `ClassicSimilarity` 默认就是 TF-IDF 变体（直到后来被 BM25 取代）。

---

## L4 · 原理深挖

### 4.1 信息论视角：IDF 是"信息量"

$\text{idf}(t) = \log \frac{N}{\text{df}(t)}$ 几乎就是信息论中**自信息** $-\log p(t)$ 的形式。

把 $p(t) = \text{df}(t) / N$ 视作"随机抽一个文档，命中包含 $t$ 的概率"，则：

$$
-\log p(t) = -\log \frac{\text{df}(t)}{N} = \log \frac{N}{\text{df}(t)} = \text{idf}(t)
$$

**IDF 本质是词项的"信息熵贡献"**：稀有词携带更多信息，常见词信息量低。这不是巧合——Spärck Jones 的灵感正来自信息论。

### 4.2 TF 的饱和与对数缩放

为什么 TF 用 $1 + \log f$ 而不是原始 $f$？

因为词频与相关性的关系不是线性的，而是**次线性（sublinear）**的：一个词出现 100 次和 10 次的相关性差距，远小于 10 倍。直觉：第一次出现已经是强信号，后续重复的边际信息递减。

对数缩放把线性增长压成对数增长：

| 词频 $f$ | 原始 $f$ | $1 + \log f$ |
|----------|----------|--------------|
| 1        | 1        | 1.00         |
| 10       | 10       | 2.30         |
| 100      | 100      | 3.60         |
| 1000     | 1000     | 4.90         |

100 次和 1000 次的差距被压到 1.3 倍——这才是符合人感觉的"边际递减"。BM25 把这个思想推到更精细的 $k_1$ 参数。

### 4.3 IDF 的变体与平滑

经典 IDF 在某些场景下有问题：

**问题 1：df(t) = 0**（查询词不在语料库）→ $\log(N/0)$ 未定义。平滑方案：

$$
\text{idf}(t) = \log \frac{N + 1}{\text{df}(t) + 1} + 1
$$

+1 防零除，外加 +1 让非零项最小值仍为正。Lucene 用的是这个变体。

**问题 2：负面 IDF**。当 $\text{df}(t) > N/2$（超过一半文档包含 $t$），$\text{idf}(t) < \log 2 \approx 0.69$；极端情况 $\text{df}(t) \to N$ 时 $\text{idf}(t) \to 0$。

某些变体（如 [Robertson 2004](https://doi.org/10.1561/1500000019)）允许 IDF 为负以惩罚过于普遍的词，但工程上多数实现把它截断到 0 或加平滑。

**问题 3：概率 IDF**。BM25 用的 IDF 形式来自 BM 家族（Robertson-Spärck Jones）：

$$
\text{idf}_{\text{BM}}(t) = \log \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}
$$

这个形式从概率检索模型推出（下个词条 BM25 会详谈），可以出现负值——所以 Lucene 在实现时改写为 $\log_2(1 + \frac{N - \text{df} + 0.5}{\text{df} + 0.5})$ 避免负值。

### 4.4 向量空间模型（VSM）

TF-IDF 不只是打分函数，它定义了一种**文档表示**：

- 把文档 $d$ 表示成一个 $|V|$ 维向量（$V$ 为词表），每一维是 $\text{tfidf}(t, d)$。
- 稀疏：大多数维度为 0（该文档不含的词）。
- 查询 $q$ 也表示成同样维度的向量。
- 相关性 = 两个向量的余弦相似度：

$$
\cos(q, d) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \cdot \|\vec{d}\|} = \frac{\sum_t q_t \cdot d_t}{\sqrt{\sum_t q_t^2} \sqrt{\sum_t d_t^2}}
$$

这就是 [Salton 1975](https://doi.org/10.1145/361219.361220) 的向量空间模型。TF-IDF 是它的权重方案。这套表示后来演变成 Word2Vec、BERT embedding 的"前身"——从稀疏字面量到稠密语义向量。

### 4.5 TF-IDF 的局限

**局限 1：词独立假设（bag of words）**。TF-IDF 假设词之间独立，"猫吃鱼"和"鱼吃猫"权重完全相同。无法建模词序、句法、语义。

**局限 2：无语义相似**。"汽车"和"轿车"在 TF-IDF 里是完全不同的两个维度，没有相似度。语义检索（dense retrieval）解决这个。

**局限 3：词频的文档长度偏置**。长文档天然词频高，单纯用 $f(t,d)$ 会让长文档系统性地高分。归一化（除以 $|d|$）部分缓解但不够——BM25 用更精细的长度归一化参数 $b$。

**局限 4：查询词不在文档 = 0 分**。TF-IDF 是字面匹配，文档不含查询词就是 0 分，哪怕语义相关。这是稀疏检索的根本局限，dense retrieval 就是为了破这个。

**局限 5：不区分位置**。标题中出现的词和正文中出现的词在 TF-IDF 里权重相同。实际检索中标题更关键——Lucene 用字段加权（field boost）来补这个。

### 4.6 Lucene 的 TF-IDF 实现

Lucene 早期默认相似度 `ClassicSimilarity`（TF-IDF 变体）：

```text
score(t, d) = idf(t)^2 * tf(t,d) * norm(t,d) * queryBoost
```

其中：

- $\text{tf}(t, d) = \sqrt{f(t, d)}$（平方根，进一步压平）
- $\text{idf}(t) = 1 + \log \frac{N}{\text{df}(t) + 1}$
- $\text{norm}(t, d) = 1 / \sqrt{|d|}$（长度归一化，编码到一字节索引中）
- $\text{queryBoost}$：查询时人工加权

注意 Lucene 的 TF-IDF 把 IDF 平方了，把 TF 开方了——这些是工程调参，不是教科书公式。从 Lucene 6 起，默认改为 BM25（`BM25Similarity`），TF-IDF 退居二线，只用在特殊场景（如低延迟精确匹配）。

### 4.7 TF-IDF 的现代遗产

虽然打分被 BM25 全面超越，TF-IDF 的思想仍在多个地方活着：

1. **关键词提取**：用 TF-IDF 排序文档内所有词，取 top-k 作为关键词。简单但有效，至今仍是基线。
2. **文本分类的特征**：传统 SVM / 朴素贝叶斯分类器常用 TF-IDF 加权的词袋向量作为输入。
3. **聚类与主题模型**：LDA 等主题模型预处理时常降权高频词，思想源自 IDF。
4. **预训练数据过滤**：大模型预训练语料清洗时，常用 IDF 类指标过滤低信息量片段。
5. **嵌入方法的思想先驱**：从字面词袋 → TF-IDF → LSA（TF-IDF + SVD）→ Word2Vec → BERT，TF-IDF 是稀疏字面表示的巅峰，也是被稠密语义表示取代的转折点。

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **1957**：Hans Peter Luhn 提出"词在文档中的频率反映其重要性"——TF 思想诞生。
- **1972**：Karen Spärck Jones 提出 IDF 概念，论文 "A Statistical Interpretation of Term Specificity and Its Application in Retrieval"。这是 IDF 的起源。
- **1975**：Salton 提出向量空间模型（VSM），TF-IDF 成为标准权重方案。
- **1976–1994**：TF-IDF 主宰 IR 教科书与系统（SMART 系统、早期 Lucene）。
- **1994**：Robertson & Walker 提出 BM25，成为下一代默认。
- **2000s**：Lucene 默认 TF-IDF 变体直到 v6（2016）切到 BM25。
- **2015+**：稠密检索（DPR、ColBERT 等）兴起，TF-IDF 在主流语义检索中退居二线，但仍是关键词检索基线。

Karen Spärck Jones 因 IDF 工作 1988 年获 ACM Gerard Salton Award，2004 年获 ASIS&T Award of Merit。她是 IR 领域奠基人之一。

### 5.2 常见坑

**坑 1：忘了 IDF 平滑，df=0 时崩溃**。

教科书公式 $\log \frac{N}{\text{df}}$ 在 $\text{df}=0$（查询词语料库中从未出现）时未定义。生产实现必须平滑，比如 $\log \frac{N+1}{\text{df}+1} + 1$。Lucene / Elasticsearch 都用变体平滑。

**坑 2：直接用原始 TF，长文档系统偏高**。

文档 A 长度 10000，词 "AI" 出现 50 次；文档 B 长度 100，"AI" 出现 5 次。原始 TF A=50, B=5，但相对密度 B 更高。必须归一化——除以文档长度只是基础方案，BM25 的 $b$ 参数更精细。

**坑 3：把 IDF 当相关性，忽略 TF 方向**。

IDF 是词的属性，不是"词-文档对"的属性。"AI" 这个词 IDF 高，不代表它对每篇文档都重要——还得看该文档里 TF。新人常只算 IDF 排关键词，结果排出来全是稀有专有名词，与文档主题无关。

**坑 4：处理查询词的 TF**。

查询 "AI AI AI" 重复三次，该不该把 TF 算 3？多数实现把查询当作集合，只看是否出现，不算查询侧 TF——但有些实现会算。这是工程细节，要确认。

**坑 5：对短文本（标题、推文）TF-IDF 失效**。

短文本 TF 几乎都是 1，区分度低；IDF 又依赖全局语料统计，若语料小则 df 估计不可靠。短文本场景建议用 BM25（参数调小 $b$）或稠密检索。

**坑 6：分词不同，结果天差地别**。

中英文混合语料，"机器学习"作为一个词 vs 拆成"机器/学习"，TF-IDF 结果完全不同。分词是 TF-IDF 系统的"前置黑箱"，调分词器比调 TF-IDF 参数影响大得多。

**坑 7：用 TF-IDF 算语义相似**。

TF-IDF 是字面匹配。"我要买手机"和"想购入智能电话"在 TF-IDF 下几乎 0 相关（无共同词）。语义检索必须用稠密向量（如 sentence-BERT、BGE）。TF-IDF + LSA 可以部分缓解（用 SVD 降维后向量带些语义），但远不如现代方法。

### 5.3 TF-IDF vs BM25 vs Dense Retrieval

| 维度 | TF-IDF | BM25 | Dense Retrieval |
|------|--------|------|-----------------|
| TF 饱和 | 对数或开方 | $k_1$ 参数，更精细 | 不显式 |
| 长度归一化 | 简单除以长度 | $b$ 参数，可调 | 训练时隐式 |
| IDF | 经典 $\log(N/\text{df})$ | 概率变体 $\log\frac{N-\text{df}+0.5}{\text{df}+0.5}$ | 无 |
| 语义 | 无 | 无 | 有 |
| 词序 | 无 | 无 | 部分有（Transformer） |
| 解释性 | 强 | 强 | 弱 |
| 资源占用 | 低 | 低 | 高（需 GPU 推理） |
| 召回率 | 中 | 中 | 高（语义匹配） |
| 精确匹配 | 强 | 强 | 中（可能漏精确词） |

工程实践：**混合检索**（TF-IDF/BM25 + Dense）几乎是现代 RAG 系统的标配，互补精确与语义。

### 5.4 何时还该用 TF-IDF

虽然 BM25 是更好的"打分函数"，TF-IDF 仍在这些场景有一席之地：

1. **关键词提取**：文档内词按 TF-IDF 排序，简单、解释性高。
2. **教学/原型**：理解 IR 基础概念的最小例子。
3. **极低资源场景**：嵌入式设备、无 BM25 实现时，TF-IDF 几十行就能跑。
4. **特征工程**：传统 ML 模型的文本特征（与 Word2Vec / TF-IDF 拼接）。
5. **快速过滤**：用 IDF 高的词作候选集预过滤，再上 BM25 或稠密检索精排。

---

## 速记卡

| 维度 | TF-IDF |
|------|--------|
| 公式 | $\text{tfidf}(t,d) = \text{tf}(t,d) \cdot \text{idf}(t)$ |
| TF 缩放 | 对数 $1 + \log f$ 或开方 $\sqrt{f}$ |
| IDF 平滑 | $\log \frac{N+1}{\text{df}+1} + 1$ |
| 信息论根 | IDF = 词项自信息 $-\log p(t)$ |
| 局限 | 无语义、无词序、长文档偏置、短文本失效 |
| 接班者 | BM25（更好的打分）+ Dense Retrieval（语义） |
| 仍用于 | 关键词提取、教学、特征工程、混合检索的稀疏分支 |

**一句话记忆**：TF-IDF = 本文档高频 × 全局稀有。前者看信号强度，后者看信号纯度，相乘得到该词对该文档的"代表性权重"。

---

> *上一篇：[倒排索引](./inverted-index) -- 检索系统的底层骨架。*
> *下一篇：[BM25](./bm25) -- TF-IDF 的概率论升级版，至今仍是稀疏检索事实标准。*
