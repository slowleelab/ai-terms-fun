---
title: 索引算法 HNSW / IVF / PQ / LSH
slug: ann-algorithms
category: 检索与索引
tags: [向量检索, ANN, HNSW, IVF, PQ, LSH, 索引算法]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 索引算法 HNSW / IVF / PQ / LSH

> 五层读懂一个词。这次拆的是：**ANN 索引算法四大家族**--HNSW、IVF、PQ、LSH 的工程细节与选择决策。

---

## L1 · 一句话点破

**HNSW 用图、IVF 用分区、PQ 用压缩、LSH 用哈希**--四种思路各异，从不同角度逼近同一个目标：在高维向量空间里用远小于 $O(N)$ 的代价找到 top-k 近邻。

---

## L2 · 通俗类比

继续用图书馆找相似书的比喻：

- **HNSW（图法）**：建立"相似书联络网"，每本书连到几本相似的书。从入口书出发，沿着相似度高的方向跳，几跳就逼近 top-k。像社交网络中通过朋友找朋友。
- **IVF（分区法）**：把书按主题分到 $K$ 个书架，查询时只翻最相关的几个书架。像图书馆按主题分区的物理布局。
- **PQ（压缩法）**：每本书用 8 字节"指纹"代替完整特征，先用指纹粗筛大量书，再精确比较少量候选。像先看封面再翻目录。
- **LSH（哈希法）**：把每本书通过哈希函数映射到桶里，相似书大概率落同桶。查询时只扫同桶的书。像把书按"色系"分柜，红色书找红色柜。

四种思路各有适用场景，工程上常组合使用（IVF-PQ、HNSW-PQ 等）。

---

## L3 · 正经定义

四种 ANN 算法的核心数据结构：

### HNSW (Hierarchical Navigable Small World)

**Malkov & Yashunin 2016**。多层近邻图，顶层稀疏（远距离连接、少节点），底层稠密（近距离连接、全节点）。查询时从顶层入口粗定位，逐层下沉到底层精确搜索。

**关键参数**：

- $M$：每节点邻居数（底层 $2M$）。$M=16$ 通用，$M=32$ 高召回。
- $\text{efConstruction}$：建索引探索因子。$200 \sim 500$。
- $\text{efSearch}$：查询探索因子。$50 \sim 200$。

**复杂度**：

- 构建：$O(N \cdot \text{efConstruction} \cdot \log N)$
- 查询：$O(\text{efSearch} \cdot \log N)$
- 内存：$O(N \cdot M)$（图结构）+ $O(N \cdot D)$（原始向量）

### IVF (Inverted File)

**基于 k-means 聚类的分区**。把 $N$ 个向量聚成 $K$ 个簇，每个簇维护一个倒排链（含该簇所有向量）。查询时找最近的 $n_{\text{probe}}$ 个簇心，只在这些簇的倒排链中搜索。

**关键参数**：

- $K$：簇心数。$K = \sqrt{N}$ 经验值。
- $n_{\text{probe}}$：查询时搜索的簇数。$8 \sim 64$ 常用。

**复杂度**：

- 构建：$O(N \cdot K)$（k-means 迭代）
- 查询：$O(K + n_{\text{probe}} \cdot N/K)$（找簇心 + 簇内扫描）
- 内存：$O(K \cdot D)$（簇心）+ $O(N \cdot D)$（向量）

### PQ (Product Quantization)

**Jégou et al. 2011**。把 $D$ 维向量切分成 $m$ 段，每段独立 k-means 聚成 256 个码字（1 字节）。原向量压缩成 $m$ 字节的码。

**关键参数**：

- $m$：分段数。$m \in [8, 64]$，越大精度越高、压缩比越低。
- $K_s = 256$：每段码字数（1 字节）。也可用 $K_s = 65536$（2 字节，但内存翻倍）。

**复杂度**：

- 编码：$O(D)$（查 $m$ 次码字表）
- 距离查表：$O(m)$（预算 $m \times 256$ 的距离表）
- 内存：$O(N \cdot m)$（每向量 $m$ 字节）+ $O(m \cdot K_s \cdot D/m) = O(K_s \cdot D)$（码本）

### LSH (Locality Sensitive Hashing)

**Indyk & Motwani 1998**。设计一族哈希函数 $h$，使相似向量哈希到同桶的概率高，不相似向量哈希到同桶的概率低：

$$
P[h(u) = h(v)] = \text{sim}(u, v)^\rho
$$

其中 $\rho$ 是 LSH 家族的参数（越小越好）。常用 LSH 族：

- **随机超平面 LSH**（余弦相似度）：$h(v) = \text{sign}(w \cdot v)$，$w$ 是随机超平面
- **p-stable LSH**（欧氏距离）：$h(v) = \lfloor \frac{w \cdot v + b}{r} \rfloor$，$w$ 服从 p-stable 分布
- **MinHash**（Jaccard 相似度）：$h(v) = \min_{x \in v} \pi(x)$，$\pi$ 是随机排列

**关键参数**：

- $L$：哈希表数。$L$ 越大召回越高、查询越慢。
- $k$：每表的哈希函数串联数。$k$ 越大冲突率越低、召回越低。

**复杂度**：

- 查询：$O(L)$（每表一次哈希 + 桶扫描）
- 内存：$O(N \cdot L)$（多张哈希表）

---

## L4 · 原理深挖

### 4.1 HNSW 为什么这么快：小世界图理论

HNSW 的前身是 NSW (Navigable Small World)。小世界图的核心性质：

- **短路径**：任意两节点间路径长度 $\sim \log N$
- **可导航**：贪心搜索（每步选最近邻）能找到短路径

HNSW 加了**层次结构**：把图分成多层，顶层只保留少数"长程连接"节点，底层包含所有节点。查询时：

1. 顶层入口：贪心走最近邻
2. 找到本层局部最优后，下沉到下一层
3. 重复，直到最底层

效果：每层减少搜索范围，整体复杂度从 NSW 的 $O(N^\rho)$ 降到 $O(\log N)$。

**关键设计：邻居选择策略**。HNSW 不只是简单连最近邻，而是用启发式选邻居：在候选集中选能"扩展视野"的邻居（距离远但与已选邻居不重复）。这让图既有局部稠密又有长程连接，是 HNSW 性能的关键。

### 4.2 IVF 的概率分析

IVF 的召回率取决于"真实 top-k 是否都在查询的 $n_{\text{probe}}$ 个最近簇内"。

假设向量分布近似均匀，簇心覆盖半径 $R$，查询到最近簇心距离 $r_q$。若真实近邻在距 $q$ 距离 $r$ 内，要求该近邻所在簇心与查询簇心距离 $\le R + r$。

**经验值**：

- $K = \sqrt{N}, n_{\text{probe}} = \sqrt{K}$ 时，recall@10 通常 0.85~0.95
- $n_{\text{probe}} = K$（全扫）时退化为暴力

**IVF 的优化变体**：

- **IVF-Flat**：簇内精确搜索，速度与簇大小成正比
- **IVF-PQ**：簇内用 PQ 压缩向量，速度极大提升
- **IVF-HNSW**：用 HNSW 找最近簇心（$K$ 很大时需要），FAISS 集成
- **IVF-ADC**：IVF + PQ 异步距离计算（Asymmetric Distance Computation）

### 4.3 PQ 的距离计算：ADC vs SDC

PQ 压缩后距离计算有两种方式：

**SDC (Symmetric Distance Computation)**：查询和文档都压缩，距离用码字-码字距离表（$256 \times 256 \times m$）查表。

$$
d_{\text{SDC}}(\hat{q}, \hat{v}) = \sum_{i=1}^{m} \text{table}_i[\hat{q}_i, \hat{v}_i]
$$

缺点：查询也压缩，引入额外误差。

**ADC (Asymmetric Distance Computation)**：查询不压缩，只压缩文档。预算查询每段到 256 个码字的距离（$m \times 256$ 表），查表累加。

$$
d_{\text{ADC}}(q, \hat{v}) = \sum_{i=1}^{m} \text{table}_i[\hat{v}_i]
$$

ADC 精度更高（查询未压缩），是 FAISS 默认实现。

**OPQ (Optimized PQ)**：PQ 前做旋转 $R$，让各段方差均衡，减少量化误差。$v \to R v$ 再 PQ。OPQ 通常比 PQ 减少 20~40% 误差。

### 4.4 LSH 的概率保证

LSH 有理论保证：对相似度 $\text{sim}$ 的 LSH 族，串联 $k$ 个哈希函数、并行 $L$ 张表，对真实近邻 $v^*$ 与查询 $q$：

$$
P[v^* \text{ 与 } q \text{ 在某表同桶}] = 1 - (1 - \text{sim}(q, v^*)^{k \rho})^L
$$

调 $k, L$ 可控制召回。但 LSH 的实战问题：

1. **维度灾难下性能急剧下降**：高维下相似与不相似的区分度低，需要 $L$ 指数级增长才能保持召回。
2. **内存占用大**：多张哈希表，每张存全量向量 ID。
3. **实践被 HNSW 全面碾压**：ann-benchmarks 上 LSH 几乎被淘汰。

LSH 现在更多作为**理论参考**和**特殊场景**（如 Jaccard 相似度的 MinHash 仍主流用于集合相似）。

### 4.5 算法对比矩阵

| 算法 | 构建复杂度 | 查询复杂度 | 内存 | 召回 | 增删 | 适用规模 |
|------|------------|------------|------|------|------|----------|
| HNSW | $O(N \log N)$ | $O(\log N)$ | 高 | 高 | 中 | $< 10^8$ |
| IVF | $O(NK)$ | $O(\sqrt{N})$ | 中 | 中 | 中 | $< 10^9$ |
| IVF-PQ | $O(NK)$ | $O(\sqrt{N}/m)$ | 低 | 中低 | 中 | $< 10^{10}$ |
| PQ (单层) | $O(N)$ | $O(N/m)$ | 极低 | 低 | 易 | $< 10^7$ |
| LSH | $O(NL)$ | $O(L)$ | 高 | 低 | 易 | $< 10^6$ |
| DiskANN | $O(N \log N)$ | $O(\log N)$ | 低（磁盘） | 高 | 难 | $< 10^{10}$ |
| 暴力 KNN | $O(1)$ | $O(N)$ | 中 | 100% | 易 | $< 10^4$ |

### 4.6 组合索引：现代向量库的标准配置

实际生产中，单算法很少用，常组合：

**FAISS `IVF1024,PQ32`**：

- 1024 个 IVF 簇
- 每簇内向量用 PQ 压缩成 32 字节
- 查询时找 $n_{\text{probe}}=16$ 个簇，簇内用 PQ 距离查表
- 内存：$N \times 32$ 字节 + $1024 \times D$ 字节

**FAISS `HNSW32,IVF1024,PQ32`**：

- 用 HNSW 找 1024 个簇心中最近的 $n_{\text{probe}}$ 个
- 再用 IVF-PQ 在这些簇内搜索
- 适合 $K$ 很大时（$K = 65536$ 等）

**Milvus / Qdrant 默认 HNSW**：

- 单机或小集群直接用 HNSW
- 内存够用就 HNSW，最简单
- 需要磁盘存储时切 DiskANN

### 4.7 算法选择的工程经验

**默认 HNSW**：$N < 10^8$ 且内存够用时，HNSW 几乎总是最优。简单、召回高、速度好。

**IVF-PQ 当内存紧**：$N > 10^7$ 且单机内存装不下原始向量时，IVF-PQ 压缩后内存可控。

**DiskANN 当超大规模**：$N > 10^8$ 且不想分布式时，DiskANN 单机 + SSD 是优选。

**GPU 加速当吞吐密集**：FAISS-GPU、cuVS 用 GPU 并行 IVF-PQ 暴力检索，单 GPU 千万级向量毫秒级。

**别用 LSH**：除非是 Jaccard 相似度（MinHash）或教学场景，LSH 在高维向量检索已被淘汰。

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **1998**：Indyk & Motwani 提出 LSH，第一篇 ANN 理论论文
- **2010**：SIFT1M 数据集发布，ANN 评测有标准数据
- **2011**：Jégou et al. 提出 PQ，"Product Quantization for Nearest Neighbor Search"
- **2011**：NSW (Navigable Small World) 提出，图法 ANN 起点
- **2013**：FAISS 前身的 IVFADC 等方案成熟
- **2014**：NSG (Navigating Spreading-out Graph) 提出
- **2016**：Malkov & Yashunin 提出 HNSW，"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"，性能碾压同期所有方法
- **2017**：FAISS 开源（Facebook AI Research），IVF-PQ 工业化
- **2018**：ann-benchmarks 上线，HNSW 霸榜
- **2019**：DiskANN 提出（微软），支持 10 亿级向量磁盘 ANN
- **2020**：ScaNN（Google）提出各向异性量化，部分场景超过 HNSW
- **2021+**：Milvus 2.0、Qdrant、Weaviate 等向量数据库成熟

### 5.2 常见坑

**坑 1：HNSW 拿来装海量数据**。

$N = 10^9$ 时 HNSW 内存几百 GB，单机装不下。这时该用 DiskANN 或分布式向量库，不是硬上 HNSW。

**坑 2：IVF 的 $K$ 设过大或过小**。

$K = N$（每向量一簇）退化为暴力；$K = 1$ 退化为全扫。经验 $K \in [\sqrt{N}, 4\sqrt{N}]$。

**坑 3：PQ 的 $m$ 设过小**。

$m = 8$ 把 768 维压到 8 字节，每段 96 维聚成 256 类，码字过粗，距离误差大。768 维建议 $m \ge 24$。误差大会让 recall 显著下降。

**坑 4：HNSW 的 $\text{efConstruction}$ 设太小**。

设 $50$ 看似建索引快，但图结构差，查询时即使 $\text{efSearch}$ 调大也召回不足。建议 $200 \sim 500$，建索引慢但图质量好。

**坑 5：忘了归一化向量**。

很多库的"COSINE"实际是 IP + 自动归一化。若你自己实现且没归一化，IP 不等价于余弦，结果偏向长向量。embedding 一律先 L2 normalize。

**坑 6：跨参数比较 recall 不公平**。

调参时常比 recall@10 vs QPS 曲线。但 recall 必须在同一数据集、同一 ground truth 上算。不同实现算 recall 的方式不同（如是否包含训练集），要统一口径。

**坑 7：用 benchmark 数据调参，部署后失效**。

ann-benchmarks 用 SIFT、GloVe 等公开数据集，你的生产 embedding 分布可能不同。务必用自己的评估集调参。

**坑 8：增量更新后图退化**。

HNSW 删节点后图结构会留"死链接"（边指向已删节点），新插入节点的邻居选择基于现有图，可能不够优。生产中定期重建索引。

**坑 9：LSH 还在用**。

除非 Jaccard 相似度场景，高维向量检索别用 LSH。HNSW 全面碾压，调参还更简单。

**坑 10：相信论文报告的 QPS**。

论文常在优化过的数据集、单线程、特定硬件上测 QPS，生产环境（多线程、并发 IO、混合负载）QPS 通常打个 3~5 折。务必自己测。

### 5.3 算法选择决策树

```
N < 10^4?
  └─ 暴力 KNN（FAISS IndexFlat）

N < 10^8 且内存够?
  └─ HNSW（M=16, efSearch=100）
      召回不足?
        └─ 增大 M 或 efSearch

N < 10^8 但内存紧?
  └─ IVF-PQ（K=√N, m=D/32, n_probe=16）

N > 10^8?
  └─ DiskANN（单机 + SSD）
      或
  └─ 分布式向量库（Milvus / Qdrant 集群）
```

### 5.4 新兴方向

**ScaNN (Google)**：各向异性量化，对方向敏感的量化误差加权，在部分数据集上超过 HNSW。Google 内部生产用，开源版本 (2020) 但生态不如 FAISS。

**DiskANN**：磁盘 ANN，10 亿级向量单机服务。微软 2019 提出，2023 进入 FAISS。

**Vamana**：DiskANN 用的图算法，比 HNSW 稀疏，配合 SSD 随机读优化。

**学习索引（learned index）**：用神经网络学习索引结构。理论有潜力，工程尚未击败 HNSW。

**GPU ANN**：FAISS-GPU、cuVS 用 GPU 并行暴力或 IVF-PQ。GPU 适合吞吐密集型，CPU 适合延迟敏感型。

---

## 速记卡

| 算法 | 核心思想 | 关键参数 | 优势 | 劣势 |
|------|----------|----------|------|------|
| HNSW | 多层近邻图 | $M, \text{efSearch}$ | 召回速度双优 | 内存大 |
| IVF | k-means 分区 | $K, n_{\text{probe}}$ | 内存友好 | 召回略低 |
| PQ | 向量分段量化 | $m$ | 极致压缩 | 距离失真 |
| LSH | 局部敏感哈希 | $L, k$ | 理论保证 | 实战已淘汰 |

**组合索引**：

- **IVF-PQ**：分区 + 压缩，大规模低内存首选
- **HNSW-IVF**：HNSW 找簇心，IVF 簇内搜，大 $K$ 场景
- **HNSW-PQ**：图结构 + 向量压缩，内存极致优化

**一句话记忆**：HNSW 用图、IVF 用分区、PQ 用压缩、LSH 用哈希。$N < 10^8$ 默认 HNSW，内存紧用 IVF-PQ，超大规模用 DiskANN，LSH 别用。

---

> *上一篇：[KNN / ANN](./knn-ann) -- 精确最近邻到近似最近邻的演化。*
> *下一篇：[ANN 库：FAISS / Milvus / Qdrant](./ann-libraries) -- 主流 ANN 库的工程实现对比。*
