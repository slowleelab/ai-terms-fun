---
title: 倒排索引（Inverted Index）
slug: inverted-index
category: 检索与索引
tags: [倒排索引, Inverted Index, 关键词检索, Lucene, Elasticsearch]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 倒排索引（Inverted Index）

> **一句话 TL;DR**：倒排索引是"词 -> 包含该词的文档列表"的映射，是关键词检索（[BM25](./bm25)/TF-IDF）和搜索引擎（Lucene/Elasticsearch）的核心数据结构。它让"找含某词的文档"从 $O(N)$ 降到 $O(\log N)$ 甚至 $O(1)$。在向量检索流行的今天，倒排索引仍是混合检索不可或缺的一半。

---

## L1 · 一句话点破

倒排索引：**从"词"出发，记录每个词出现在哪些文档中。**

```
正排索引（文档 -> 词）:
  文档1: [猫, 喜欢, 吃, 鱼]
  文档2: [狗, 喜欢, 吃, 骨头]
  文档3: [猫, 不, 喜欢, 狗]

倒排索引（词 -> 文档）:
  猫 -> [文档1, 文档3]
  喜欢 -> [文档1, 文档2, 文档3]
  吃 -> [文档1, 文档2]
  鱼 -> [文档1]
  狗 -> [文档2, 文档3]
  ...
```

查询"猫"：直接查倒排索引，得 [文档1, 文档3]，无需扫描所有文档。

为什么叫"倒排"？因为传统索引是"文档 -> 词"（正排），倒排索引把它"倒过来"成"词 -> 文档"。这是信息检索百年老树的核心技巧。

## L2 · 通俗类比

图书馆找书：

- **正排（无索引）**：每本书翻一遍，看是否含"猫"字。慢，O(N) 扫描所有书。
- **倒排索引**：建一个目录"猫 -> 第 1、3、7 本书"，查"猫"直接看书单。快，O(1) 查询。

具体例子：

```
查询 "猫 喜欢"
1. 查 "猫" -> [文档1, 文档3]
2. 查 "喜欢" -> [文档1, 文档2, 文档3]
3. 取交集 -> [文档1, 文档3]  # 同时含两词
4. 按 [BM25](./bm25) 分数排序 -> [文档1, 文档3]
```

倒排索引让"找含某些词的文档"非常快，是关键词检索的基础。

但倒排索引只能做"精确词匹配"：

- 查"猫"找不到"小猫"（除非扩展）
- 查"cat"找不到"猫"（除非跨语言）
- 查"机器学习"找不到"ML"（除非同义词扩展）

这是 [向量检索](./vector-database)（语义检索）的补充。

## L3 · 正经定义

**倒排索引（Inverted Index）**：从词项（term）到文档列表的映射，是关键词检索的核心数据结构。

**基本结构**：

```
Dictionary (词表):
  term -> [doc_freq, posting_list_pointer]

Postings (倒排记录):
  posting_list:
    doc_id -> [term_freq, positions, ...]
```

每个 posting 包含：

- doc_id：文档 ID
- term_freq：词在文档中出现次数
- positions：词在文档中的位置（用于短语查询、邻近查询）
- 其他元数据（offset、payload 等）

**查询流程**：

1. 查询分词 -> 词项 $[t_1, t_2, ...]$
2. 对每个 $t_i$，查倒排索引得 posting list
3. 合并 posting list（AND/OR/短语）
4. 用 [BM25](./bm25) / TF-IDF 打分
5. 返回 top-k 文档

**主流实现**：

| 系统 | 特点 |
|------|------|
| **Lucene** | Apache 开源库，Java，倒排索引工业级实现 |
| **Elasticsearch** | 基于 Lucene，分布式搜索 |
| **OpenSearch** | ES 的开源分支 |
| **Solr** | 基于 Lucene，企业搜索 |
| **Tantivy** | Rust 实现，性能优秀 |

**参考资料**：
- [Manning et al. - Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) - IR 圣经
- [Lucene Documentation](https://lucene.apache.org/)
- [Elasticsearch - The Definitive Guide](https://www.elastic.co/guide/en/elasticsearch/guide/current/index.html)

## L4 · 原理深挖

### 4.1 倒排索引 vs 正排索引

**正排索引**：文档 -> 词

```
文档1 -> [猫, 喜欢, 吃, 鱼]
```

查询"含猫的文档"：扫描所有文档，看是否含"猫"。$O(N)$。

**倒排索引**：词 -> 文档

```
猫 -> [文档1, 文档3]
```

查询"含猫的文档"：直接查"猫"的 posting list。$O(\text{doc\_freq})$，通常远小于 $N$。

倒排索引是搜索引擎的基础，因为查询是"词找文档"而非"文档找词"。

### 4.2 倒排索引的构建

构建流程：

```
1. 文档收集
   [文档1, 文档2, ..., 文档N]

2. 分词
   文档1: "猫喜欢鱼" -> [猫, 喜欢, 鱼]

3. 词项归一化
   - 小写化（Cat -> cat）
   - 词干化（running -> run，英文）
   - 同义词扩展（可选）

4. 构建 posting list
   猫 -> [文档1]
   喜欢 -> [文档1]
   鱼 -> [文档1]
   ...（增量构建）

5. 排序 posting list（按 doc_id）
   便于合并操作

6. 压缩
   - doc_id 差值编码
   - 词频变长编码
```

Lucene 等工业实现有大量优化：FST（Finite State Transducer）存词表、跳表加速 posting list 合并、列式存储等。

### 4.3 查询类型

倒排索引支持多种查询：

**① 词项查询（Term Query）**

查询单个词："猫"。直接查倒排索引。

**② 布尔查询（Boolean Query）**

```
"猫" AND "鱼"  -> 两个 posting list 求交
"猫" OR "狗"   -> 两个 posting list 求并
"猫" NOT "狗"  -> 求差
```

**③ 短语查询（Phrase Query）**

"猫 喜欢" 连续出现。需要 positions 信息：

```
"猫" -> [(文档1, 位置1), (文档3, 位置1)]
"喜欢" -> [(文档1, 位置2), (文档2, 位置2), (文档3, 位置2)]
检查: "猫" 在位置 i，"喜欢" 在位置 i+1 -> 短语匹配
```

**④ 前缀/通配符查询**

"猫*" -> 所有以"猫"开头的词。需要词表的 FST 支持。

**⑤ 模糊查询（Fuzzy Query）**

"猫" 容错 1 个编辑距离，找到"猫"、"毛"等。用于拼写纠错。

### 4.4 倒排索引的打分：BM25 / TF-IDF

倒排索引返回"包含查询词的文档"，但需要打分排序。主流打分函数：

**① TF-IDF**

```
score = tf * idf
tf = 词在文档中频率
idf = log(N / df)  # N 总文档数, df 含该词的文档数
```

罕见词权重高（idf 大），高频词权重低。

**② [BM25](./bm25)**

TF-IDF 的改进，加文档长度归一：

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\text{tf}(t, d) \cdot (k_1+1)}{\text{tf}(t, d) + k_1 \cdot (1 - b + b \cdot |d|/\text{avgdl})}
$$

BM25 是 Lucene/Elasticsearch 的默认打分，见 [BM25](./bm25) 词条详述。

### 4.5 倒排索引的工程优化

**① 词表压缩：FST**

Lucene 用 FST（Finite State Transducer）存词表：

- 共享前缀/后缀，压缩比高
- 支持前缀查询、自动补全
- 内存占用小

**② Posting list 压缩**

- doc_id 差值编码（相邻 doc_id 差值小）
- 变长编码（PForDelta、Roaring Bitmap 等）
- 跳表加速合并

**③ 列式存储**

文档字段（title、body、tags 等）分开存储，按需加载。

**④ 分片与分布式**

Elasticsearch 把索引分片到多台机器，并行查询。

这些优化让 Lucene 能处理亿级文档、毫秒级查询。

### 4.6 倒排索引 vs 向量索引

| 维度 | 倒排索引 | 向量索引 (HNSW/IVF) |
|------|---------|---------------------|
| 数据类型 | 稀疏向量（词项） | 稠密向量 |
| 查询类型 | 精确词匹配 | 语义相似 |
| 数据结构 | 词表 + posting list | 图/倒排量化 |
| 代表系统 | Lucene/ES | Faiss/Milvus |
| 适合 | 关键词查询 | 语义查询 |

现代搜索引擎常同时维护两套索引，做 [混合检索](./dense-sparse-vector)：

- 倒排索引：精确匹配
- 向量索引：语义匹配
- 融合结果

Elasticsearch 8+、OpenSearch、Vespa、Weaviate 等都支持混合检索。

### 4.7 倒排索引的局限

**① 语义弱**

只能匹配字面词，不理解语义。"猫" 找不到 "小猫"（除非扩展）。

**② 依赖分词**

中文需先分词，分词错误影响检索。

**③ 词表大**

维度等于词表大小（数万到数百万），但只存非零项。

**④ 不擅长长查询**

查询词越多，posting list 合并越复杂，且每个词的 idf 衰减。

这些局限由向量检索补足，混合检索是趋势。

## L5 · 沿革与坑

### 沿革

- **1940s-1950s**：倒排索引概念在信息检索早期出现。
- **1960s-1980s**：[Manning IR 书](https://nlp.stanford.edu/IR-book/) 等系统化倒排索引理论。
- **1999**：Doug Cutting 开发 Lucene，倒排索引工业级实现。
- **2010**：Elasticsearch 基于 Lucene，倒排索引成为主流搜索后端。
- **2020-2023**：向量检索兴起，但倒排索引仍是关键词检索基础。
- **2023-2025**：混合检索流行，Elasticsearch 8+、OpenSearch 等集成向量检索，倒排 + 向量共存。

### 常见误解

- ❌ **误解**：倒排索引过时了，向量检索取代它。
  ✅ **真相**：倒排索引在关键词检索上仍不可替代。混合检索让两者共存，而非取代（4.6）。

- ❌ **误解**：倒排索引只能做精确匹配。
  ✅ **真相**：通过同义词扩展、词干化、模糊查询等，倒排索引也能做一定"近似匹配"。但语义匹配仍需向量（4.3）。

- ❌ **误解**：Elasticsearch 是数据库。
  ✅ **真相**：ES 主要是搜索引擎（基于倒排索引），不是通用数据库。虽然有存储能力，但设计目标是搜索而非事务。

- ❌ **误解**：倒排索引查询慢。
  ✅ **真相**：倒排索引查询极快（毫秒级），比暴力扫描快几个数量级。瓶颈在合并多个 posting list 和打分（4.5）。

- ❌ **误解**：倒排索引和向量索引可以互换。
  ✅ **真相**：数据结构、查询类型、适用场景都不同。互换会丢失各自优势。混合检索是正解（4.6）。

### 面试怎么考

1. **"什么是倒排索引？"** --词项到文档列表的映射。让"找含某词的文档"从 O(N) 降到 O(1)-O(log N)（L1、L3）。
2. **"倒排索引和正排索引的区别？"** --正排是文档->词，倒排是词->文档。查询场景是"词找文档"，所以用倒排（4.1）。
3. **"倒排索引怎么打分？"** --用 [BM25](./bm25)/TF-IDF。考虑词频、文档频率、文档长度（4.4）。
4. **"倒排索引支持哪些查询？"** --词项、布尔、短语、前缀、模糊查询（4.3）。
5. **"倒排索引和向量索引怎么选？"** --关键词查询用倒排，语义查询用向量。混合检索是趋势（4.6）。

## 延伸阅读

- 📚 [Manning et al. - Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
- 📝 [Lucene Documentation](https://lucene.apache.org/)
- 📝 [Elasticsearch - The Definitive Guide](https://www.elastic.co/guide/en/elasticsearch/guide/current/index.html)
- 📝 [Tantivy (Rust)](https://github.com/quickwit-oss/tantivy)

---

> *上一篇：[上下文窗口](./context-window) -- 模型的"短期记忆"边界。*
> *下一篇：[TF-IDF](./tf-idf) -- 关键词权重的经典算法。*
