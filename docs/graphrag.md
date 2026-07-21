---
title: GraphRAG 图谱增强检索
slug: graphrag
category: 进阶专题
tags: [GraphRAG, 知识图谱, RAG, 社区聚类, 多跳推理, Microsoft]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# GraphRAG 图谱增强检索

> 五层读懂一个词。这次拆的是：**GraphRAG**--把知识图谱塞进 RAG，用社区聚类和实体关系解决向量检索答不了的"全局性问题"。

---

## L1 · 一句话点破

**GraphRAG = 知识图谱构建 + 社区聚类 + 层级摘要 + 图谱检索**。Naive RAG 检索 chunk，GraphRAG 检索"实体关系网络"，专治全局性、多跳、聚合类问题。

---

## L2 · 通俗类比

Naive RAG 像在图书馆里按关键词找几张最相关的卡片，适合"RLHF 是什么"这种点状问题。但遇到"这个领域的主要流派有哪些"、"全书的核心论点是什么"这种需要跨多张卡片综合的问题，它就拉胯--因为向量检索只会返回局部相似的片段，不会主动去聚合。

GraphRAG 的做法是先把整本书拆解成一张**关系网**：

- 把每个句子里的**实体**（人、机构、概念）抠出来当节点
- 把实体间的关系当**边**（"A 发明了 B"、"C 隶属于 D"）
- 用社区发现算法把关系网**聚成几个圈**（社区），每个圈是一个主题簇
- 给每个社区、每个层级生成**摘要**
- 查询时按问题类型选择：局部问题走实体检索，全局问题走社区摘要

**关键区别**：

| 问题类型 | Naive RAG | GraphRAG |
|----------|-----------|----------|
| "RLHF 的损失函数是什么" | ✅ 直接命中 chunk | 不如 Naive 快 |
| "强化学习对齐有哪些主流路线" | ❌ 召回零散片段 | ✅ 社区摘要直接给答案 |
| "A 和 B 谁的影响更大" | ❌ 需要多跳推理 | ✅ 图谱多跳路径 |

**代价**：构建知识图谱本身很贵（LLM 调用爆炸），索引时间从分钟级涨到小时级，存储从 chunk 扩到 chunk + 图 + 摘要。所以 GraphRAG 不是 Naive RAG 的替代品，而是**全局性问题的补丁**。

---

## L3 · 正经定义

**GraphRAG**：由 Microsoft Research 于 2024 年 4 月提出的 RAG 增强方案（Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"），核心是用知识图谱 + 社区层级摘要扩展 RAG 处理全局性问题的能力。

**处理流水线**：

1. **文本切分**：源文档切 chunks
2. **实体与关系抽取**：LLM 从每个 chunk 抽 `(实体, 实体, 关系描述, 时间戳)` 四元组
3. **图谱构建**：实体为节点，关系为边，同义实体合并
4. **社区发现**：用 Leiden 算法对图分层聚类，得到多层社区层级
5. **社区摘要**：LLM 为每个社区生成报告式摘要（叶子层 -> 中间层 -> 根层）
6. **检索与生成**：
   - **Global Search**：用社区摘要做 map-reduce 式汇总，回答全局问题
   - **Local Search**：从问题实体出发，检索关联的 chunk + 实体 + 关系，回答局部问题

**与 Naive RAG 的本质区别**：Naive RAG 检索单元是 chunk（文本片段），GraphRAG 检索单元是"实体 + 关系 + 社区摘要"的混合结构，且引入了**显式的全局聚合机制**（社区摘要）。

**参考资料**：

- 📄 Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, arXiv:2404.16130, 2024
- 🔧 Microsoft GraphRAG 官方实现：https://github.com/microsoft/graphrag
- 📄 Traag et al., *From Louvain to Leiden: guaranteeing well-connected communities*, Scientific Reports, 2019
- 📄 Microsoft, *GraphRAG: Unlocking LLM discovery across narrative private datasets*, 2024 博客

---

## L4 · 原理深挖

### 4.1 为什么 Naive RAG 答不了全局问题

Naive RAG 的检索是**局部相似度匹配**：query 和 chunk 算余弦，取 top-k。这种范式有两个结构性缺陷：

**缺陷 1：召回片段化**。全局问题（"全书的主题有哪些"）的答案分散在几百个 chunk 里，top-k 只能召回最相似的几个，丢掉大量相关片段。

**缺陷 2：无聚合能力**。即使召回了所有相关 chunk，LLM 也要在 context 里自己综合，但 context 长度有限，且 LLM 在长 context 下有 "lost in the middle" 问题，聚合质量打折。

GraphRAG 的解法是**离线预聚合**：把分散的信息提前组织成社区摘要，查询时直接取摘要，而不是临时从 chunk 聚合。

### 4.2 实体与关系抽取

GraphRAG 用 LLM 做 zero-shot 抽取，prompt 大致是：

```
给定以下文本，识别其中的所有实体（人/组织/地点/概念），
以及它们之间的关系。输出 JSON：
{
  "entities": [{"name": "...", "type": "...", "description": "..."}],
  "relationships": [{"source": "...", "target": "...", "description": "..."}]
}
```

**关键工程点**：

- **迭代抽取**：同一 chunk 多次抽取取并集，降低漏抽
- **实体归一**："OpenAI"、"OpenAI Inc"、"openai" 合并为同一节点（用名称相似度 + LLM 判断）
- **描述聚合**：同一实体在多个 chunk 出现，描述合并成一段
- **抽取 prompt 调参**：`max_gleanings`（每 chunk 迭代次数）、实体类型限定（限定为"组织/人/地点"可减少噪声）

**代价**：这是 GraphRAG 最贵的步骤。一篇 100k token 的文档，抽取阶段可能要 10M+ LLM token。

### 4.3 图谱构建与社区发现

抽取完得到一个**异构图** $G = (V, E)$：

- $V$：实体节点，带 `name / type / description` 属性
- $E$：关系边，带 `description / weight`（weight = 关系共现次数）

**Leiden 算法**做社区发现：

- 目标：最大化模块度 $Q = \frac{1}{2m} \sum_{ij} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$
- $A_{ij}$：邻接矩阵，$k_i$：节点 $i$ 的度，$m$：总边数，$\delta$：同社区指示函数
- Leiden 比 Louvain 多一步"细化"，保证社区内部连通性

**层级聚类**：Leiden 递归分裂，得到多层社区。例如：

```
Level 0: 1 个社区（全图）
Level 1: 8 个社区
Level 2: 35 个社区
Level 3: 120 个叶子社区
```

每层社区数指数级增长，叶子层粒度细，根层粒度粗。

### 4.4 社区摘要生成

**自底向上**为每个社区生成摘要：

- **叶子社区**：把社区内所有实体描述 + 关系描述拼起来，LLM 生成报告式摘要（含关键实体、主题、关系）
- **中间层社区**：把子社区摘要 + 跨子社区的关系拼起来，LLM 生成上层摘要
- **根社区**：全图总览

**摘要的内容结构**（Microsoft GraphRAG 默认模板）：

```
## 主题概述
...
## 关键实体
- entity1: ...
- entity2: ...
## 关键关系
- A -> B: ...
## 时间线 / 演化
...
```

**摘要长度**：叶子层 ~1500 token，上层递增。这是预聚合的核心产出。

### 4.5 检索策略：Global vs Local

GraphRAG 区分两类查询：

**Global Search**（全局问题，如"全书主要主题有哪些"）：

```python
def global_search(query, community_reports, llm, top_k_communities=10):
    # 1. Map: 每个社区摘要独立答一遍
    partial_answers = []
    for report in community_reports[:top_k_communities]:
        prompt = f"基于以下社区报告，回答问题：{query}\n\n报告：{report}"
        partial = llm.generate(prompt)
        partial_answers.append(partial)
    # 2. Reduce: 汇总所有部分答案
    final_prompt = f"综合以下部分答案，给出最终答案：\n" + "\n".join(partial_answers)
    return llm.generate(final_prompt)
```

这是 **map-reduce** 范式：map 阶段每个社区并行答，reduce 阶段汇总。社区数多时 map 阶段 LLM 调用爆炸，可抽样 top-k 社区。

**Local Search**（局部问题，如"X 机构的 RLHF 工作是什么"）：

1. 识别 query 中的实体
2. 从图里找这些实体的 1-hop 邻居（关联实体 + 关系）
3. 检索这些实体所在的 chunk
4. 拼接 chunk + 实体描述 + 关系描述喂 LLM

Local Search 本质是**带图谱上下文的 Naive RAG**，图谱提供实体关系的结构化背景。

### 4.6 成本与收益的权衡

GraphRAG 的索引成本是 Naive RAG 的 **10-100 倍**：

| 维度 | Naive RAG | GraphRAG |
|------|-----------|----------|
| 索引阶段 LLM 调用 | 仅 embedding | 抽取 + 摘要（10-100x） |
| 索引时间 | 分钟 | 小时 |
| 存储 | chunk + 向量索引 | + 图 + 社区摘要 |
| 查询成本 | 1 次 LLM | Global: N+1 次（N=社区数）|
| 全局问题质量 | 差 | 好 |
| 局部问题质量 | 好 | 相当或略好 |

**实践建议**：

- 数据量小（<100 chunks）：Naive RAG 够用
- 数据量大 + 全局性问题多：GraphRAG
- 混合场景：两套并存，路由器分流

### 4.7 GraphRAG 的变体与演进

- **HippoRAG**（NeurIPS 2024）：用个性化 PageRank 模拟海马体记忆机制，比 GraphRAG 轻
- **LightRAG**（2024）：简化版 GraphRAG，去掉社区聚类，双层检索（实体 + 关系）
- **nano-graphrag**：千行级极简实现，适合学习
- **GraphRAG + 向量**：Microsoft 官方也在融合向量检索，做 hybrid

### 4.8 何时用 GraphRAG

**适合**：

- 全书/全库级综述问题
- 多跳推理（"A 的导师的导师是谁"）
- 主题演化分析（"X 技术从 2020 到 2024 怎么演化的"）
- 私有语料探索（财报、专利、案件卷宗）

**不适合**：

- 点状事实查询（"RLHF 损失函数"）--Naive RAG 更快更准
- 数据量小
- 实时性要求高（图谱构建慢）
- 数据频繁更新（重建图成本高）

---

## L5 · 沿革与坑

### 5.1 沿革

- **2023**：RAG 概念普及，但全局性问题短板暴露
- **2024-04**：Microsoft 发表 GraphRAG 论文，提出社区聚类 + 层级摘要范式
- **2024-05**：Microsoft 开源 graphrag 官方实现
- **2024 下半年**：HippoRAG / LightRAG / nano-graphrag 等变体涌现
- **2025**：GraphRAG 与向量检索融合（hybrid）、与 Agentic RAG 结合成为趋势

### 5.2 常见坑

**坑 1：抽取 prompt 没调好，图全是噪声**。实体类型不限会导致"日期"、"数字"都被抽成实体。限定 `entity_types = ["organization", "person", "location", "event"]` 可大幅降噪。

**坑 2：忘了实体归一**。同义词实体不合并，图碎成孤岛。要做 name normalization + LLM 实体合并。

**坑 3：社区层级选错**。叶子社区太细答不了全局问题，根社区太粗答不了局部问题。实践：Global Search 用中层社区，Local Search 用叶子社区。

**坑 4：索引成本爆炸**。100k token 文档抽取阶段可能烧 10M LLM token。要先小样本估算成本再全量跑。

**坑 5：Global Search map 阶段 LLM 调用爆炸**。社区数 100+ 时 map 阶段 100+ 次调用。用 top-k 社区抽样或异步并发。

**坑 6：数据更新痛苦**。文档改一段，相关实体/关系/社区摘要都得重建。增量更新是开放问题，目前多数实现是全量重建。

**坑 7：拿 GraphRAG 当 Naive RAG 替代品**。点状问题用 GraphRAG 反而更慢更贵。两类查询分流才是正解。

**坑 8：评估只看全局问题**。GraphRAG 在局部问题上不一定优于 Naive RAG，要分场景评估。

**坑 9：社区摘要质量不监控**。LLM 生成的摘要可能漏关键信息或编造。要做摘要质量抽检。

**坑 10：图太稀疏**。文档本身实体关系少时，Leiden 聚不出有意义的社区。要先看图密度。

### 5.3 面试怎么考

1. **GraphRAG 解决了 Naive RAG 的什么问题？** 答：全局性/聚合类问题，通过社区摘要预聚合。
2. **GraphRAG 的索引流水线？** 答：切分 -> LLM 抽实体关系 -> 建图 -> Leiden 社区发现 -> 层级摘要。
3. **Global Search 和 Local Search 的区别？** 答：Global 用社区摘要做 map-reduce，Local 从实体出发做带图上下文的检索。
4. **GraphRAG 什么时候不适用？** 答：点状查询、小数据、高实时性、频繁更新。
5. **Leiden 算法在 GraphRAG 里的作用？** 答：把实体关系图聚成层级社区，是预聚合的结构基础。

---

## 速记卡

| 阶段 | 产出 | 成本 |
|------|------|------|
| 抽取 | 实体 + 关系四元组 | LLM 调用最贵 |
| 建图 | 异构图 $G=(V,E)$ | 中 |
| 社区发现 | 多层社区（Leiden） | 低 |
| 社区摘要 | 层级报告 | LLM 调用次贵 |
| Global Search | map-reduce 答全局 | 查询时 N+1 次 LLM |
| Local Search | 实体邻居 + chunk | 查询时 1 次 LLM |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| `chunk_size` | 1200-3000 token | 抽取粒度 |
| `max_gleanings` | 1-3 | 抽取召回率 vs 成本 |
| `entity_types` | 4-8 类 | 图噪声 |
| `community_level` | 2-3 层 | 摘要粒度 |
| `top_k_communities` | 10-50 | Global Search 质量 vs 成本 |

**一句话记忆**：GraphRAG = LLM 抽实体关系建图 + Leiden 社区聚类 + 层级摘要 + Global/Local 双路检索。专治 Naive RAG 答不了的全局性/多跳问题，代价是索引成本 10-100x。点状查询还是 Naive RAG 快。

---

> *上一篇：[知识库 Knowledge Base](./knowledge-base) -- RAG 的知识存储与管理，进阶专题的起点。*
> *下一篇：[Self-RAG 自反思检索](./self-rag) -- 让 LLM 自己判断要不要检索、检索结果好不好。*
