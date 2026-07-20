---
title: 分块 Chunking
slug: chunking
category: 数据表示与编码
tags: [Chunking, 分块, RAG, 滑动窗口, 语义分块]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 分块 Chunking

> **一句话 TL;DR**：分块是把长文档切成较小片段（chunk）的过程，主要服务于 [RAG](./rag) 和长文档处理。切多大切在哪直接影响检索质量和生成效果。主流方法从最朴素的固定长度切分，到滑动窗口、语义分块、层级分块等。Chunking 是 RAG 工程里被低估却极其关键的一环。

---

## L1 · 一句话点破

Chunking：**把长文本切成较小、相对独立的片段（chunk），每个 chunk 单独 [embedding](./embedding) 和检索。**

为什么需要？因为：

- 文档可能很长（几万字甚至更多），但 [embedding](./embedding) 模型有长度限制（通常 512 token）
- 检索时希望返回"精准片段"而非整篇文档
- 生成时希望 LLM 上下文聚焦，不被无关内容干扰

```
长文档 -> 切成 chunks -> 每个 chunk 单独 embedding -> 入向量库
查询 -> 查向量库 -> 返回 top-k chunks -> 拼到 LLM prompt -> 生成答案
```

chunk 多大、怎么切，直接决定 RAG 能不能找到"对的那一段"。

## L2 · 通俗类比

图书馆找资料：

- **不分块**：把整本书塞进索引，查询时返回整本书。问题：书太大，LLM 上下文装不下，且大部分内容无关
- **按章分块**：每章一个 chunk。问题：章太大，相关性稀释；章太小可能跨章信息丢失
- **按段分块**：每段一个 chunk。问题：段间关联丢失（如"上一段提到的方案"）
- **滑动窗口分块**：固定长度切，相邻 chunk 重叠一部分。问题：可能从句子中间切断
- **语义分块**：按内容主题切换处分块。最贴近人类阅读，但实现复杂

每种切法权衡：

- **chunk 大**：上下文全，但相关性稀释、embedding 不准、LLM 上下文压力大
- **chunk 小**：相关性高、检索准，但可能丢失上下文、chunk 数量多成本高

没有"最优"chunk 大小，需根据数据、任务、模型实验。

## L3 · 正经定义

**Chunking**：把长文本 $D$ 切分成片段集合 $\{c_1, c_2, ..., c_n\}$ 的过程，每个 $c_i$ 是相对独立的语义单元。

**主流方法**：

| 方法 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **固定长度** | 每 N 字符一个 chunk | 简单 | 可能切断句子 |
| **固定长度+重叠** | 每 N 字符，相邻 chunk 重叠 M | 保留边界上下文 | 冗余存储 |
| **按句/段** | 自然语言单位切 | 不破坏句子 | 长度不均 |
| **滑动窗口** | 步长 S，窗口 W | 平衡长度和连续性 | 仍可能语义断裂 |
| **语义分块** | 按主题/语义跳变切 | 最贴近语义 | 实现复杂、慢 |
| **层级分块** | 多粒度（句-段-章） | 适应不同查询 | 工程复杂 |
| **结构化分块** | 按 Markdown 标题/HTML 标签 | 保留文档结构 | 依赖文档格式 |

**典型参数**：

- chunk 大小：200-1000 token（典型 500）
- 重叠：50-200 token（约 chunk 大小的 10-20%）

**参考资料**：
- [Lewis et al., 2020 - DPR (Dense Passage Retrieval)](https://arxiv.org/abs/2004.04906)
- [Karpukhin et al., 2020 - DPR](https://arxiv.org/abs/2004.04906)
- [Gao et al., 2023 - RAG Survey](https://arxiv.org/abs/2312.10997)
- [Greg Kamradt - Chunking Strategies](https://github.com/FullStackRetrieval-com/RetrievalTutorials)

## L4 · 原理深挖

### 4.1 为什么 chunk 大小很关键

chunk 大小直接影响 RAG 的两个核心指标：

**① 检索精度（找到对的内容）**

- chunk 太大：单个 chunk 包含多个主题，embedding 被"稀释"，相似度不准
- chunk 太小：单个 chunk 信息不足，可能错过查询所需上下文

**② 生成质量（LLM 拿到对的上下文）**

- chunk 太大：LLM 上下文装不下太多 chunk，且无关信息干扰
- chunk 太小：LLM 缺少上下文，可能误读片段

实证：典型场景 chunk 500-1000 token 较优。但具体需实验。

### 4.2 固定长度 + 重叠：最朴素的方案

最简单的 chunking：

```python
def fixed_size_chunk(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap  # 步长 = size - overlap
    return chunks
```

特点：

- 实现简单
- 长度均匀（便于批处理）
- 重叠避免边界信息丢失

问题：

- 可能在句子中间切断（"我喜欢机器学习。它" | "很有趣"）
- 不考虑语义

实务：作为基线，多数 RAG 教程的默认方案。

### 4.3 按句/段切分：保留自然边界

改进：在句号、换行等自然边界切分：

```python
def sentence_chunk(text, max_size=500):
    sentences = split_sentences(text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_size:
            current += sent
        else:
            chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks
```

特点：

- 不破坏句子完整性
- chunk 长度不均（但都在 max_size 内）

进阶：递归切分（LangChain 的 RecursiveCharacterTextSplitter）。优先按段落、其次按句子、最后按字符切，尽量保留自然边界。

### 4.4 语义分块：按主题切换

[Semantic Chunking (Greg Kamradt)](https://github.com/FullStackRetrieval-com/RetrievalTutorials) 的思路：在"语义跳变点"切分。

```
1. 把文本按句切分
2. 每句计算 embedding
3. 计算相邻句的 embedding 相似度
4. 相似度低于阈值处 = 语义跳变点 = chunk 边界
```

效果：chunk 内语义连贯，chunk 间主题切换。最贴近人类阅读理解。

代价：

- 需要 embedding 每句话，成本高
- 阈值需调
- 慢

适合高质量 RAG 场景（如研究、法律）。

### 4.5 层级分块：多粒度检索

单一 chunk 大小难以适应所有查询：

- "公司财报净利润" -> 想要小 chunk（精准数字）
- "公司战略方向" -> 想要大 chunk（上下文全）

**层级分块**：同时存多个粒度（句、段、章），检索时按查询选择合适粒度。

```
查询 "净利润" -> 检索句级 chunk
查询 "战略" -> 检索章级 chunk
```

代表：RAPTOR（recursive summarization）、MultiVector Retriever 等。

代价：存储成本高（多份）、工程复杂。

### 4.6 结构化分块：利用文档结构

很多文档有天然结构（Markdown 标题、HTML 标签、PDF 段落）。按结构切：

```
# 第一章
## 1.1 节
内容...
## 1.2 节
内容...
```

按标题层级切，每个 chunk 带上"它在文档中的位置"（如"第一章 > 1.1 节"）。

优势：

- 保留文档逻辑结构
- chunk 自带上下文（标题、层级）
- 符合人类阅读习惯

代表：Unstructured、LlamaIndex 的 MarkdownNodeParser 等。

### 4.7 Chunking 的工程权衡

实际选择 chunking 策略的考量：

| 维度 | 考虑 |
|------|------|
| 文档类型 | 文章 vs 代码 vs 法律文书 vs 对话 |
| 查询类型 | 事实查找 vs 主题理解 vs 关系推理 |
| 模型限制 | embedding 模型最大长度、LLM 上下文窗口 |
| 成本 | chunk 数量影响 embedding 和存储成本 |
| 质量 | 越精细的 chunking 越准但越慢越贵 |

经验法则：

- **简单场景**（FAQ、知识库）：固定 500 token + 50 重叠
- **结构化文档**（Markdown、法律）：按结构切
- **高质量场景**（研究、医疗）：语义分块或层级分块
- **多查询类型**：层级分块

### 4.8 Chunking 与 RAG 的整体关系

Chunking 是 RAG 流程的第一步，影响后续所有环节：

```
文档 -> Chunking -> Embedding -> 入向量库 -> 检索 -> Rerank -> LLM 生成
```

- chunk 差 -> embedding 差 -> 检索差 -> 生成差
- chunk 好 -> 整个 RAG 链路受益

实务中 chunking 的优化 ROI 极高：简单调整 chunk 大小或方法，可能比换 embedding 模型效果更显著。但常被忽视，大家关注 embedding 模型而忽略 chunking。

## L5 · 沿革与坑

### 沿革

- **2018-2020**：早期 RAG（[DPR](https://arxiv.org/abs/2004.04906)）用 100-token chunk。
- **2020-2022**：LangChain、LlamaIndex 等框架提供多种 chunking 工具，固定长度 + 重叠成为默认。
- **2023**：[RAG Survey (Gao et al.)](https://arxiv.org/abs/2312.10997) 系统化 RAG 流程，chunking 作为关键环节。
- **2023-2024**：语义分块、层级分块、结构化分块流行，chunking 质量成为 RAG 优化焦点。
- **2024-2025**：长上下文 LLM（128K、1M）出现，"长上下文 vs 精准 chunking"的权衡讨论。但实践证明精准 chunking 仍优于"全塞进去"。

### 常见误解

- ❌ **误解**：chunk 越大上下文越全，效果越好。
  ✅ **真相**：chunk 太大稀释 embedding 相关性，且 LLM 上下文压力大。最优 chunk 大小需实验，典型 500-1000 token（4.1）。

- ❌ **误解**：固定长度切分就够了，复杂方法没必要。
  ✅ **真相**：固定长度在句子中间切断会损失语义。简单场景够用，高质量场景需要语义/结构化分块（4.3-4.6）。

- ❌ **误解**：长上下文 LLM 出现，chunking 没用了。
  ✅ **真相**：长上下文 LLM 能装更多，但"全塞进去"有 lost-in-the-middle 问题、成本高、注意力分散。精准 chunking 仍优于全塞（5）。

- ❌ **误解**：chunking 只是预处理，影响小。
  ✅ **真相**：chunking 是 RAG 的基础，影响 embedding、检索、生成全链路。优化 chunking 的 ROI 极高（4.8）。

- ❌ **误解**：chunk 数越多检索越准。
  ✅ **真相**：chunk 多 = 存储和检索成本高，且小 chunk 信息不足。需平衡（4.1）。

- ❌ **误解**：所有文档用同一种 chunking 策略。
  ✅ **真相**：不同文档类型（文章、代码、法律、对话）适合不同策略。需按数据特点选（4.7）。

### 面试怎么考

1. **"什么是 chunking？为什么需要？"** --把长文本切成片段，每个单独 embedding/检索。因为 embedding 模型有长度限制、检索要精准片段、LLM 上下文聚焦（L1）。
2. **"chunk 大小怎么选？"** --权衡检索精度和上下文完整。太大稀释相关性、太小丢上下文。典型 500-1000 token，需实验（4.1）。
3. **"固定长度和语义分块的区别？"** --固定长度简单但可能切断句子；语义分块按主题跳变切，最贴近语义但慢（4.2、4.4）。
4. **"为什么 chunk 间要重叠？"** --避免边界信息丢失。相邻 chunk 重叠 10-20%（4.2）。
5. **"长上下文 LLM 还需要 chunking 吗？"** --需要。长上下文有 lost-in-the-middle、成本高、注意力分散问题。精准 chunking 仍优于全塞（5）。

## 延伸阅读

- 📄 [Lewis et al., 2020 - DPR](https://arxiv.org/abs/2004.04906)
- 📄 [Gao et al., 2023 - RAG Survey](https://arxiv.org/abs/2312.10997)
- 📝 [Greg Kamradt - Chunking Strategies](https://github.com/FullStackRetrieval-com/RetrievalTutorials)
- 📝 [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

---

> *上一篇：[Token 词元](./token) -- 模型的最小处理单位。*
> *下一篇：[高维向量](./high-dim-vector) -- 为什么 embedding 是几百上千维。*
