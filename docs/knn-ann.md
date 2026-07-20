---
title: KNN / ANN
slug: knn-ann
category: 检索与索引
tags: [向量检索, 最近邻, 近似算法, 高维向量, HNSW, IVF, LSH]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# KNN / ANN

> 五层读懂一个词。这次拆的是：**KNN / ANN**--向量检索从精确到近似的工程演化。

---

## L1 · 一句话点破

**KNN 是精确最近邻（找真正的 top-k 最近），ANN 是近似最近邻（用索引结构以极小精度损失换巨大速度提升）。** 高维向量场景下，KNN 不可扩展，ANN 是工业唯一选择。

---

## L2 · 通俗类比

你在图书馆找"和你这本最像的 10 本书"。

**KNN 做法（暴力法 / brute force）**：把图书馆每本书都翻一遍，两两比较，挑出最像的 10 本。准确，但图书馆有 1000 万本书时根本不可行。

**ANN 做法（带索引）**：

- **分区法（IVF）**：先把书按主题分到 1000 个区，找你这本书所在区及周边几个区的书。绝大多数最像的书都在附近分区，远区的基本不会是 top-10。
- **图法（HNSW）**：建立"相似图书网络"，每本连到几本相似的书。从任意起点出发，沿着相似度高的方向跳，几跳就逼近 top-10。
- **压缩法（PQ）**：每本书用 8 字节"指纹"代替完整特征，先按指纹粗筛，再精确比较少量候选。

代价：可能漏掉 1~2 本真正很相似但被分区/图跳过的书（recall < 100%）。

工程权衡：**用 95% 的召回率换 1000 倍的速度**。生产中 ANN 通常能到 recall@10 = 0.95+，延迟 < 10ms，KNN 则要秒级甚至更慢。

---

## L3 · 正经定义

**KNN (k-Nearest Neighbors)**：给定查询向量 $q$、向量集合 $V$、距离函数 $d(\cdot, \cdot)$，找出 $V$ 中距离 $q$ 最近的 $k$ 个向量。形式化：

$$
\text{KNN}_k(q, V) = \arg\min_{S \subseteq V, |S|=k} \sum_{v \in S} d(q, v)
$$

朴素实现：计算 $q$ 到 $V$ 中每个向量的距离，排序取前 $k$。复杂度 $O(N \cdot D)$（$N$ = 向量数，$D$ = 维度）。$N = 10^7, D = 768$ 时单次查询 ~30 GFLOPs，CPU 毫秒级几乎不可能。

**ANN (Approximate Nearest Neighbor)**：在可接受的精度损失下（如 recall@10 ≥ 0.95），用索引结构加速近邻搜索。核心思路三种：

1. **分区（partition-based）**：把向量空间划分为多个区域，查询时只搜可能含近邻的区域。代表：IVF。
2. **图（graph-based）**：构建近邻图，查询时图上贪心游走逼近近邻。代表：HNSW、NSG、Vamana。
3. **压缩（compression-based）**：把向量压缩成短码，用短码快速估算距离筛候选。代表：PQ、SQ、OPQ。

**ANN 的评价指标**：

- **Recall@k**：返回的 top-k 中真正属于真实 top-k 的比例。
- **QPS (Queries Per Second)**：每秒查询数。
- **延迟分布**：p50 / p99 延迟。
- **索引大小**：内存/磁盘占用。
- **构建时间**：建索引耗时。

经典 trade-off：**Recall 越高，QPS 越低**。通过调索引参数在曲线上选点。

**伪代码**（HNSW 查询骨架）：

```python
def hnsw_search(q, graph, entry_point, ef, top_k):
    """从 entry_point 出发，维护大小 ef 的候选集，返回 top_k"""
    visited = {entry_point}
    candidates = [(dist(q, entry_point), entry_point)]  # min-heap
    results = [(-dist(q, entry_point), entry_point)]    # max-heap
    while candidates:
        d_c, c = heappop(candidates)
        d_w, w = results[0]  # 当前最差结果
        if d_c > -d_w:        # 候选已比最差结果还远，停止
            break
        for neighbor in graph[c]:
            if neighbor not in visited:
                visited.add(neighbor)
                d_n = dist(q, neighbor)
                if len(results) < ef or d_n < -results[0][0]:
                    heappush(candidates, (d_n, neighbor))
                    heappush(results, (-d_n, neighbor))
                    if len(results) > ef:
                        heappop(results)
    return sorted([(-d, n) for d, n in results])[:top_k]
```

参数 `ef`（exploration factor）越大，召回越高但越慢。

---

## L4 · 原理深挖

### 4.1 为什么 KNN 在高维不可扩展

**计算复杂度**：$O(N \cdot D)$ 每次查询。$N = 10^7, D = 768$ 时一次查询需 $7.68 \times 10^9$ 次浮点运算，CPU 上至少几百毫秒。

**维度灾难**：高维空间中，所有点之间的距离趋于相同，"最近"的区分度变低。具体地，对随机点：

$$
\lim_{D \to \infty} \frac{\max_v d(q, v) - \min_v d(q, v)}{\min_v d(q, v)} \to 0
$$

意思是高维下"最近"和"最远"的相对差距趋于 0，距离函数失去区分力。这是 KD-Tree 等空间划分树在高维失效的根本原因--在低维（< 20 维）有效，高维退化成线性扫描。

**存储瓶颈**：$N = 10^7, D = 768$，float32，向量化存储 $28.6$ GB。暴力检索要把这些数据全部加载并扫描，IO 和算力都吃不消。

工程结论：**$D \ge 100$ 且 $N \ge 10^5$ 时，必须用 ANN**。

### 4.2 三大 ANN 流派

#### 4.2.1 分区法：IVF (Inverted File)

**思路**：用 k-means 把 $N$ 个向量聚成 $K$ 个簇（$K \approx \sqrt{N}$），查询时只搜最近 $n_{\text{probe}}$ 个簇。

```python
def ivf_search(q, centroids, inverted_lists, n_probe, top_k):
    # 1. 找最近的 n_probe 个簇心
    dists = [(dist(q, c), i) for i, c in enumerate(centroids)]
    nearest_clusters = sorted(dists)[:n_probe]
    # 2. 在这些簇的倒排链中找 top_k
    candidates = []
    for _, c_id in nearest_clusters:
        candidates.extend(inverted_lists[c_id])
    return sorted([(dist(q, v), v) for v in candidates])[:top_k]
```

**参数权衡**：

- $K$ 大：每个簇小，查询快但 recall 低（簇心密度高，可能漏召回）
- $K$ 小：每个簇大，查询慢但 recall 高
- $n_{\text{probe}}$ 大：recall 高，QPS 低
- 经验：$K = \sqrt{N}$，$n_{\text{probe}} = 8 \sim 64$

**变体**：

- **IVF-Flat**：簇内精确搜索
- **IVF-PQ**：簇内用 PQ 压缩向量，进一步加速
- **IVF-HNSW**：用 HNSW 找簇心（K-means 簇心很多时，找最近簇心本身也需要 ANN）

#### 4.2.2 图法：HNSW (Hierarchical Navigable Small World)

**思路**：构建多层近邻图。顶层稀疏（少边、远距离连接），底层稠密（多边、近距离连接）。查询时从顶层入口粗定位，逐层下沉到底层精确搜索。

**为什么 HNSW 现在是默认选择**：

- 召回率高（recall@10 = 0.95+ 在多数数据集上轻松达到）
- 查询快（QPS 数千到数万）
- 增删容易（不像 IVF 要重训聚类）
- 支持过滤查询

**关键参数**：

- $M$：每节点邻居数。$M = 16$ 通用，$M = 32$ 高召回。决定内存和图密度。
- $\text{efConstruction}$：建索引时探索因子。$200 \sim 500$ 常用，大则建索引慢但图质量好。
- $\text{efSearch}$：查询时探索因子。$50 \sim 200$ 常用，大则慢但准。

**HNSW 缺点**：

- 内存占用大（图结构 + 原始向量）
- 索引难以序列化（图结构跨版本兼容性差）
- 不支持高效范围搜索（要 top-k，不要半径内全部）

#### 4.2.3 压缩法：PQ (Product Quantization)

**思路**：把 $D$ 维向量切分成 $m$ 段，每段独立用 k-means 聚成 256 个码字（1 字节）。原向量压缩成 $m$ 字节的码。

```python
def pq_encode(vec, codebooks):
    """vec: D 维；codebooks: m 个段，每段 256 个 D/m 维码字"""
    codes = []
    for i, cb in enumerate(codebooks):
        segment = vec[i * D//m : (i+1) * D//m]
        c = argmin_l2(segment, cb)  # 最近码字索引
        codes.append(c)
    return codes  # 长度 m 的字节数组

def pq_distance(q, codes, codebooks):
    """预计算 q 到所有码字的距离表，O(m) 查表"""
    dist = 0
    for i, c in enumerate(codes):
        dist += table[i][c]  # table 预算好
    return dist
```

**PQ 关键性质**：

- 压缩比：$D \times 4$ 字节 $\to m$ 字节，通常 32~64 倍压缩
- 距离计算：$O(m)$ 查表，比 $O(D)$ 浮点运算快 50 倍+
- 误差：有损，距离估计有偏差，但 ANN 容忍这个偏差

**PQ 通常和 IVF 合用**：IVF-PQ = 分区 + 压缩。先用 IVF 找候选簇，再用 PQ 在簇内快速估算距离。FAISS 的 `IVFx,PQy` 索引就是这个组合。

### 4.3 距离函数的选择

向量检索常用三种距离：

| 距离 | 公式 | 适用 |
|------|------|------|
| 欧氏距离 (L2) | $\sqrt{\sum_i (q_i - v_i)^2}$ | 通用 |
| 内积 (IP) | $\sum_i q_i \cdot v_i$ | 已归一化时等价于余弦 |
| 余弦相似度 | $\frac{q \cdot v}{\|q\| \|v\|}$ | 文本语义 |

**关键**：**embedding 向量归一化后，三种距离等价排序**。生产实践：embedding 出来直接 L2 normalize，用 IP（内积）检索，省一次开方和除法。

**FAISS / Milvus / Qdrant 支持的三种 metric**：`L2` / `IP` / `COSINE`。COSINE 实现通常是先归一化再用 IP。

### 4.4 ANN 的过滤查询（Filtered ANN）

实际场景常需"在满足 metadata 过滤的子集中找 top-k"。如"在我订阅的作者中找最相似的文章"。

**朴素做法**：先 ANN 找大候选集，再过滤。问题：候选集大部分被过滤掉，top-k 实际数量不足。

**改进做法**：

- **Pre-filter**：先过滤出子集，再在子集上 ANN。问题：子集太小或分布偏斜时 ANN 索引失效。
- **In-filter**：ANN 检索时动态检查过滤条件，丢弃不满足的候选。Qdrant、Milvus 用这个。
- **Hybrid**：根据过滤比例自动选择 pre 或 in。Weaviate、Pinecone 用这个。

工程结论：**过滤查询是 ANN 工程化的硬骨头**，选向量数据库时要看它的过滤查询实现。

### 4.5 分布式 ANN

单机内存装不下时需分片。

**分片策略**：

- **随机分片**：向量均匀分配到 shard。查询时所有 shard 并行，merge top-k。简单但每个 shard 都要查。
- **聚类分片**：按聚类中心分片，查询只查相关 shard。省查询但维护成本高。

**常见架构**：

- **Milvus**：分布式集群，存储计算分离，支持十亿级向量
- **Qdrant**：单机性能强，分布式通过 raft 同步
- **Pinecone**：托管服务，自动分片和复制
- **Weaviate**：图数据库基因，混合检索友好

### 4.6 ANN 基准评测：ann-benchmarks

[ann-benchmarks.com](http://ann-benchmarks.com/) 是事实标准 ANN 评测平台，在多个公开数据集（SIFT、GIST、MNIST、Fashion-MNIST、NYTimes、Glove）上比较各算法的 recall-QPS 曲线。

**主要结论**（截至 2024）：

1. **HNSW 在中低规模（N < 10^7）几乎全胜**，recall-QPS 曲线最靠右上
2. **IVF-PQ 在大规模 + 内存受限场景占优**，PQ 压缩让 10 亿向量可装单机
3. **图算法家族（HNSW、NSG、Vamana）总体优于分区算法**，但内存占用大
4. **新算法如 DiskANN、SPANN 在超大规模 + 磁盘场景占优**

---

## L5 · 沿革与坑

### 5.1 历史脉络

- **1970s**：KNN 在统计模式识别中成熟，KD-Tree (Bentley 1975) 是早期加速结构
- **1990s**：LSH (Locality Sensitive Hashing, Indyk & Motwani 1998) 提出，理论保证概率召回
- **2010**：SIFT1M 数据集发布，催生 ANN 工业评测
- **2011**：PQ (Jégou et al. "Product Quantization for Nearest Neighbor Search") 提出，压缩 ANN 里程碑
- **2013**：FAISS 前身 IVFADC 等方案在学术界成熟
- **2014**：NSG 等图方法陆续提出
- **2016**：Malkov & Yashunin 提出 HNSW，性能碾压同期所有方法
- **2017**：FAISS 开源，IVF-PQ 工业化
- **2018**：ann-benchmarks 上线，HNSW 霸榜至今
- **2019**：DiskANN (Subramanya et al.) 提出磁盘 ANN，支持 10 亿级向量
- **2021**：Milvus 2.0、Qdrant、Weaviate 等向量数据库工业级落地
- **2023+**：向量数据库爆发，成为 RAG / LLM 应用基础设施

### 5.2 常见坑

**坑 1：忘了归一化就用余弦**。

很多库的"COSINE"实际是 IP + 自动归一化，但若你自己实现且没归一化，IP 不等价于余弦，结果会偏向长向量。**embedding 一律先 L2 normalize** 是铁律。

**坑 2：HNSW 参数默认不调，召回不够怪算法**。

HNSW 默认 $M=16, \text{efConstruction}=200, \text{efSearch}=50$。召回不够时优先调 $\text{efSearch}$ 到 100~200，QPS 下降但仍远高于暴力。建索引时 $\text{efConstruction}$ 调到 400+ 让图更稠密。

**坑 3：IVF 不调 $n_{\text{probe}}$**。

$n_{\text{probe}}=1$ 时只查一个簇，召回很低。要根据 recall 需求调到 8~64。生产中常写脚本在评估集上扫参数找最佳点。

**坑 4：PQ 压缩太狠，距离估计失真**。

$m = 8$ 把 768 维压到 8 字节，每段 96 维聚成 256 类，码字过粗，距离误差大。经验：每段不超过 32 维，$m$ 至少 $D/32$。768 维建议 $m \ge 24$。

**坑 5：增量更新导致索引退化**。

HNSW 增删节点后图结构会退化（边指向已删除节点、新节点邻居不够好）。生产中定期重建索引，或用支持高效增删的实现（如 Qdrant）。

**坑 6：过滤查询召回偏低**。

预过滤会让 ANN 在小子集上失效；后过滤会让 top-k 不足。要根据过滤比例选策略：高过滤比例用 pre-filter，低过滤比例用 in-filter。

**坑 7：距离函数不一致**。

训练 embedding 时用余弦相似度，检索时用 L2 距离。即使归一化后排序等价，但分数值不同，下游处理（如阈值过滤、加权融合）会失真。统一使用同一种度量。

**坑 8：拿 recall@10 = 0.95 当 100%**。

ANN 的 5% 漏召回集中在"边缘相关"的文档上。对长尾查询影响小，但对头部高频查询可能持续漏特定结果。关键场景建议结合 BM25 兜底。

**坑 9：benchmark 数据不代表生产数据**。

ann-benchmarks 用 SIFT、GloVe 等公开数据集，你的生产 embedding 分布可能完全不同（更稠密、更稀疏、cluster 结构不同）。务必在自己的评估集上调参。

**坑 10：用 HNSW 装超大语料**。

$N = 10^9$ 时 HNSW 内存动辄几百 GB，单机装不下。这时应选 DiskANN（磁盘）或分布式向量数据库，不要硬上 HNSW。

### 5.3 KNN vs ANN 选择决策

| 场景 | 选择 | 原因 |
|------|------|------|
| $N < 10^4$ | KNN | 朴素暴力即可，毫秒级，无索引维护成本 |
| $N \in [10^4, 10^6]$ | HNSW | 单机内存装得下，HNSW 召回速度都最佳 |
| $N \in [10^6, 10^8]$ | HNSW 或 IVF-PQ | 内存够用 HNSW，内存紧用 IVF-PQ |
| $N \in [10^8, 10^{10}]$ | DiskANN 或分布式 | 单机内存装不下，需磁盘或分片 |
| 极低延迟（< 1ms） | IVF-PQ + 量化 | 压缩后向量小，cache 友好 |
| 极高召回（> 0.99） | HNSW 大 $\text{efSearch}$ 或重排 | 图算法高 recall 容易达到 |
| 增删频繁 | HNSW | 图结构增删比 IVF 重新聚类容易 |

### 5.4 ANN 的下一代

**DiskANN / Vamana**：微软研究院 2019 提出，图算法 + 磁盘存储，支持 10 亿级向量单机服务。Vamana 图比 HNSW 稀疏，配合 SSD 随机读优化。

**SPANN**：微软 2021 提出的分布式 ANN，用聚类分片 + 倒排，每分片独立 DiskANN，支持 10 亿+ 向量近实时检索。

**学习索引**：用神经网络学习索引结构（如 RMI、learned KD-tree），理论上有潜力但工程上尚未击败 HNSW。

**GPU ANN**：FAISS-GPU、cuVS（RAPIDS）等用 GPU 并行暴力或 IVF-PQ，单 GPU 千万级向量毫秒级。GPU 适合吞吐密集型，CPU 适合延迟敏感型。

**量子 ANN**：理论探讨阶段，工程无关。

---

## 速记卡

| 维度 | KNN | ANN |
|------|-----|-----|
| 准确性 | 100% recall | 90%~99% recall |
| 复杂度 | $O(ND)$ | $O(\log N)$ 到 $O(\sqrt{N})$ |
| 扩展性 | 不可扩展 | 10 亿级可扩展 |
| 工程实现 | 朴素 | IVF / HNSW / PQ / DiskANN |
| 默认选择 | $N < 10^4$ | $N \ge 10^4$ |
| 关键指标 | — | recall@k, QPS, 索引大小 |

**三大 ANN 流派**：

| 流派 | 代表算法 | 优势 | 劣势 |
|------|----------|------|------|
| 分区 | IVF, IVF-PQ | 内存友好、压缩比高 | 召回略低、聚类维护 |
| 图 | HNSW, NSG, Vamana | 召回速度双优 | 内存大、增删退化 |
| 压缩 | PQ, SQ, OPQ | 极致压缩 | 距离失真 |

**一句话记忆**：KNN 是精确但慢，ANN 是用索引结构以极小精度损失换巨大速度提升。HNSW 是当前 $N < 10^7$ 的默认选择，IVF-PQ 是大规模 + 内存紧的选择，DiskANN 是超大规模 + 磁盘的选择。

---

> *上一篇：[BM25](./bm25) -- 稀疏检索的事实标准。*
> *下一篇：[索引算法 HNSW / IVF / PQ / LSH](./ann-algorithms) -- ANN 算法的工程细节对比。*
