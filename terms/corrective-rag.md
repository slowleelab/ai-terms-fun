---
title: Corrective RAG 检索纠错
slug: corrective-rag
category: 进阶专题
tags: [CRAG, Corrective RAG, 检索纠错, web 回退, RAG, 评估器]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# Corrective RAG 检索纠错

> 五层读懂一个词。这次拆的是：**Corrective RAG (CRAG)**--检索回来先别急着用，让评估器打分，垃圾 chunk 触发 web 检索兜底，模糊的 chunk 做知识精炼，干净的 chunk 直接用。

---

## L1 · 一句话点破

**CRAG = 检索质量评估器 + 三档策略（正确/模糊/错误）+ 知识精炼 + web 回退**。在 Naive RAG 检索和生成之间插一个"质检员"，过滤垃圾检索、补救失败检索，即插即用不需要重训 LLM。

---

## L2 · 通俗类比

Naive RAG 是个"拿到资料就开答"的实习生：检索回什么就用什么，哪怕资料是垃圾也硬着头皮答，结果经常一本正经地胡说。

CRAG 给这个实习生配了一个**质检流程**：

1. 检索回来的 chunk，先丢给一个**评估器**（小模型）打分
2. 评估器给出三档判断：
   - **正确（Correct）**：chunk 跟问题高度相关，直接用
   - **错误（Incorrect）**：chunk 跟问题不相关，丢掉，转去**web 搜索**兜底
   - **模糊（Ambiguous）**：chunk 有点相关但不全，做**知识精炼**（strip 噪声，留精华）
3. 三档分别走不同后处理，再把精炼后的内容喂 LLM 生成

**三档策略直观对比**：

| 评估结果 | chunk 质量 | 处理 | 后果 |
|---------|-----------|------|------|
| Correct | 高 | 直接用 | 高质量答案 |
| Incorrect | 低（垃圾） | 丢掉，web 搜 | 避免幻觉 |
| Ambiguous | 中 | 知识精炼 | 降噪提纯 |

**和 Self-RAG 的区别**：Self-RAG 把质检能力"烧进" LLM（用反思 token），要重训。CRAG 用**外部评估器**，不动 LLM，即插即用，更像一个独立的"检索后过滤"模块。

**代价**：多一个评估器调用 + 可能多一次 web 检索，延迟增加。但比起重训 LLM，工程成本低得多。

---

## L3 · 正经定义

**Corrective RAG (CRAG)**：Yan et al. (ICLR 2024) 提出，在 Naive RAG 的检索与生成之间插入一个**轻量检索评估器（retrieval evaluator）**，将检索质量分三档，分别触发"直接使用 / 知识精炼 / web 检索兜底"三种策略，从而提升 RAG 鲁棒性。

**核心组件**：

1. **检索评估器（T5-Large 微调）**：输入 `(query, chunk)`，输出相关性分数 $\in [0, 1]$
2. **阈值分档**：
   - $s \geq \theta_{high}$：Correct，直接用
   - $s \leq \theta_{low}$：Incorrect，触发 web 搜索
   - $\theta_{low} < s < \theta_{high}$：Ambiguous，做知识精炼
3. **知识精炼（Knowledge Refinement）**：用 LLM 对 chunk 做"去噪 + 抽要点"，输出精炼版
4. **web 检索兜底**：检索失败时调用 web 搜索（如 Google / Bing API），取回的网页再走一次评估

**检索-评估-纠错流水线**：

```
Query
  ↓ 向量检索
top-k chunks
  ↓ 评估器打分
分档（Correct / Ambiguous / Incorrect）
  ├─ Correct: 直接用
  ├─ Ambiguous: 知识精炼
  └─ Incorrect: web 搜索 + 再评估
  ↓ 合并精炼后的上下文
LLM 生成答案
```

**参考资料**：

- 📄 Yan et al., *Corrective Retrieval Augmented Generation*, ICLR 2024, arXiv:2401.15884
- 🔧 CRAG 官方实现：https://github.com/HuskyInSalt/CRAG
- 📄 Asai et al., *Self-RAG*, ICLR 2024（对比工作）

---

## L4 · 原理深挖

### 4.1 为什么需要检索纠错

Naive RAG 的检索不可能 100% 准：

- query 表述模糊，检索回垃圾
- 知识库覆盖不全，相关问题无答案
- embedding 模型对某些 query 失效
- chunk 切分不当，召回的 chunk 语义残缺

**问题**：Naive RAG 拿到垃圾 chunk 照样硬答，产生幻觉。这种"沉默失败"是 RAG 落地最大的痛点之一。

**CRAG 的思路**：与其指望检索永远准，不如承认检索会错，加一个**质检 + 补救**环节。失败时主动 fallback 到 web 搜索，比硬答幻觉强。

### 4.2 检索评估器的训练

CRAG 的评估器是一个**轻量分类器**，判断 `(query, chunk)` 相关性。

**模型选择**：T5-Large（770M），比 LLM 小得多，推理快。

**训练数据构造**：

- 正例：MS MARCO / Natural Questions 等数据集里"query 真实相关 chunk"
- 负例：
  - 困难负例：同主题但不相关的 chunk（BM25 召回的 top-k 但人工标注无关）
  - 简单负例：随机 chunk
- 训练目标：二分类（相关 / 不相关），或回归到 $[0, 1]$ 分数

**评分输出**：

```python
def retrieval_evaluator(query, chunk, model):
    input_text = f"Query: {query} Document: {chunk} Relevant:"
    score = model.predict(input_text)  # sigmoid 输出 [0, 1]
    return score
```

**阈值分档**（论文默认）：

- $\theta_{high} = 0.79$：Correct
- $\theta_{low} = 0.50$：Incorrect
- 中间区间：Ambiguous

阈值可根据数据集调整，高风险场景（医疗）调严，宽松场景（闲聊）调松。

### 4.3 知识精炼（Knowledge Refining）

对 Ambiguous 档的 chunk，CRAG 不直接用也不丢弃，而是做**精炼**：

```python
def knowledge_refinement(chunk, llm):
    prompt = f"""以下文档可能含噪声，请提取与问题相关的核心信息，
    去除无关内容，输出精炼版：

    原文档：{chunk}
    """
    refined = llm.generate(prompt)
    return refined
```

**精炼操作**：

- 去除无关段落
- 提取关键句
- 修正格式错误
- 保留事实陈述，删除主观评价

**效果**：Ambiguous chunk 从"半垃圾"变"精华"，提升生成质量。

### 4.4 web 检索兜底

对 Incorrect 档，CRAG 触发 **web 检索**：

```python
def web_search_fallback(query, web_retriever):
    # 1. web 搜索
    web_results = web_retriever.search(query, top_k=3)
    # 2. 对 web 结果再走评估器
    refined_results = []
    for result in web_results:
        score = retrieval_evaluator(query, result)
        if score >= theta_low:
            refined = knowledge_refinement(result) if score < theta_high else result
            refined_results.append(refined)
    return refined_results
```

**关键设计**：

- web 结果**再走一次评估器**，避免引入新噪声
- web + 本地结果合并，权重可调
- web 不可用（隐私 / 内网）时，Incorrect 档直接拒答或回退到"我不知道"

**隐私场景的妥协**：医疗、金融等隐私场景不能 web 搜索，CRAG 退化为"过滤 + 精炼"，没有兜底。这种情况 Incorrect 档只能拒答。

### 4.5 CRAG 的完整算法

```python
def crag_generate(query, local_retriever, web_retriever, evaluator, llm,
                  theta_high=0.79, theta_low=0.50):
    # 1. 本地检索
    chunks = local_retriever.search(query, top_k=5)
    
    # 2. 评估 + 分档处理
    refined_contexts = []
    for chunk in chunks:
        score = evaluator(query, chunk)
        if score >= theta_high:
            # Correct: 直接用
            refined_contexts.append(chunk)
        elif score <= theta_low:
            # Incorrect: web 兜底
            web_chunks = web_search_fallback(query, web_retriever, evaluator)
            refined_contexts.extend(web_chunks)
        else:
            # Ambiguous: 知识精炼
            refined = knowledge_refinement(chunk, llm)
            refined_contexts.append(refined)
    
    # 3. 合并上下文，生成
    context = "\n".join(refined_contexts)
    response = llm.generate(f"Context: {context}\nQuery: {query}\nAnswer:")
    return response
```

### 4.6 CRAG vs Self-RAG vs Naive RAG

| 维度 | Naive RAG | Self-RAG | CRAG |
|------|-----------|----------|------|
| 检索质检 | 无 | LLM 反思 token | 外部评估器 |
| 失败补救 | 无 | 重检索 / 重写 | web 搜索兜底 |
| 训练成本 | 0 | SFT + DPO 重训 LLM | 仅训评估器 |
| 即插即用 | - | 否（要重训 LLM） | 是 |
| 推理开销 | 低 | 高（反思 token） | 中（评估器 + 可能的 web 检索） |
| web 兜底 | 无 | 无 | 有 |
| 隐私友好 | 是 | 是 | 部分场景需禁用 web |

**核心差异**：

- Self-RAG 把"质检"能力烧进 LLM，能力强但训练贵
- CRAG 用外部评估器，能力稍弱但即插即用
- CRAG 独创"web 兜底"，Self-RAG 没有

### 4.7 评估器的局限

**局限 1：评估器本身会错**。T5-Large 评估器准确率 ~85%，仍有 15% 误判。误判会触发错误策略（垃圾判成 Correct，好 chunk 判成 Incorrect）。

**局限 2：阈值难调**。$\theta_{high}$ / $\theta_{low}$ 需要在验证集上 grid search，不同数据集最优阈值不同。

**局限 3：跨域失效**。评估器在 MS MARCO 训练，迁移到医疗 / 法律等垂直域可能失效，需重训。

**局限 4：web 检索引入噪声**。web 结果质量参差，即使再评估也可能引入错误信息。高风险场景慎用 web 兜底。

**局限 5：延迟增加**。评估器 + 可能的 web 检索 + 知识精炼，端到端延迟比 Naive RAG 高 2-3 倍。

### 4.8 CRAG 的工程化建议

**1. 评估器选型**：

- 轻量：T5-Large / DeBERTa-v3-large
- 重量：Cross-encoder（如 BGE-reranker），精度更高但慢
- 极简：用 LLM 自己当评估器（prompt："这个 chunk 跟 query 相关吗"），省一个模型但贵

**2. 阈值调优**：

- 在验证集上 grid search $\theta_{high} \in [0.6, 0.9]$，$\theta_{low} \in [0.3, 0.6]$
- 高风险场景调严（少 Correct 多 Ambiguous）
- 宽松场景调松（多 Correct 少 Ambiguous）

**3. web 检索的妥协**：

- 隐私场景：禁用 web 兜底，Incorrect 档拒答
- 半隐私：用企业内网搜索替代 web
- 开放场景：直接用 Google / Bing API

**4. 评估器冷启动**：

- 没有标注数据时，用 LLM 自动标注（GPT-4 当评估器生成训练数据）
- 部署后用用户反馈（点赞 / 踩）做在线学习

**5. 多档 vs 二档**：

- 三档（Correct / Ambiguous / Incorrect）是论文默认
- 简化版可二档（Correct / Incorrect），Ambiguous 归入 Correct，减少精炼开销

### 4.9 CRAG 的演进

- **CRAG** (2024-01)：原始论文，三档 + web 兜底
- **CRAG + Self-RAG 融合**：用 Self-RAG 的反思 token 做评估，CRAG 的 web 兜底做补救
- **CRAG + GraphRAG**：图谱检索失败时 web 兜底
- **Agentic CRAG**：把 CRAG 包装成 Agent 工具，由 LLM Agent 决定何时触发纠错

---

## L5 · 沿革与坑

### 5.1 沿革

- **2024-01**：Yan et al. 发 CRAG 论文，提出"检索评估 + 三档纠错 + web 兜底"
- **2024-02**：与 Self-RAG / Adaptive RAG 并列成为 RAG 鲁棒性代表工作
- **2024 下半年**：CRAG 思想被 LangChain / LlamaIndex 等框架吸收，成为标准 RAG 组件
- **2025**：CRAG 与 Agent 框架结合，演化为 Agentic RAG 的纠错能力

### 5.2 常见坑

**坑 1：评估器跨域直接用**。MS MARCO 训的评估器在医疗 / 法律域可能失效，要做域内微调或重训。

**坑 2：阈值没调**。直接用论文默认 $\theta_{high}=0.79$ / $\theta_{low}=0.50$，在自己数据集上分档错乱。要在验证集上 grid search。

**坑 3：web 兜底引入新幻觉**。web 结果质量参差，没二次评估就引入 context，LLM 被 web 噪声误导。web 结果必须再走评估器。

**坑 4：隐私场景忘禁 web**。医疗 / 金融场景误用 web 兜底，泄露隐私。要按场景配置 web 开关。

**坑 5：Ambiguous 档精炼成本高**。每个 Ambiguous chunk 都调 LLM 精炼，token 开销大。可简化为二档（去掉 Ambiguous，归入 Correct）。

**坑 6：评估器延迟没估**。T5-Large 评估器每次检索多一次前向，top-5 chunk 多 5 次评估，延迟显著增加。

**坑 7： Incorrect 档全靠 web**。web 搜索失败或返回噪声时，没有第二层兜底。要设计"web 也失败 -> 拒答"的策略。

**坑 8：忘了评估器自身评估**。评估器准确率不监控，悄悄退化没人知道。要定期用标注数据测评估器准确率。

**坑 9：CRAG 当万能解**。CRAG 提升鲁棒性但不提升上限。知识库本身差，CRAG 也救不回来，只能多兜底几次 web。

**坑 10：评估器和 reranker 混淆**。CRAG 的评估器是"二分类 / 打分"，reranker 是"排序"。两者可结合但作用不同：reranker 先排序取 top-k，CRAG 评估器再分档。

### 5.3 面试怎么考

1. **CRAG 的三档是什么？** 答：Correct（直接用）/ Ambiguous（知识精炼）/ Incorrect（web 兜底）。
2. **CRAG 和 Self-RAG 的区别？** 答：CRAG 用外部评估器即插即用，Self-RAG 把反思能力烧进 LLM 要重训；CRAG 独创 web 兜底。
3. **CRAG 的检索评估器怎么训？** 答：T5-Large，用 MS MARCO 等数据集的相关 / 不相关 chunk 二分类训练。
4. **CRAG 在隐私场景怎么用？** 答：禁用 web 兜底，Incorrect 档直接拒答，退化为"过滤 + 精炼"。
5. **CRAG 的局限？** 答：评估器会错、阈值难调、跨域失效、web 引入噪声、延迟增加 2-3 倍。

---

## 速记卡

| 档位 | 评估分数 | 处理策略 | 输出 |
|------|---------|---------|------|
| Correct | $s \geq 0.79$ | 直接用 | 原 chunk |
| Ambiguous | $0.50 < s < 0.79$ | 知识精炼 | 精炼版 chunk |
| Incorrect | $s \leq 0.50$ | web 兜底 + 再评估 | web 精炼 chunk |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| $\theta_{high}$ | 0.79 | Correct 阈值 |
| $\theta_{low}$ | 0.50 | Incorrect 阈值 |
| 评估器模型 | T5-Large | 准确率 vs 延迟 |
| web top_k | 3 | 兜底召回 |
| 本地 top_k | 5 | 初始检索 |

**一句话记忆**：CRAG = 检索后插一个评估器，三档分流（直接用 / 精炼 / web 兜底），即插即用不重训 LLM。代价是多一次评估器调用 + 可能的 web 检索，延迟翻倍。专治 Naive RAG 沉默失败，隐私场景禁 web 退化为过滤模式。

---

> *上一篇：[Self-RAG 自反思检索](./self-rag) -- 把质检能力烧进 LLM，反思 token 控制检索。*
> *下一篇：[Agentic RAG 智能体检索](./agentic-rag) -- 把 Self-RAG / CRAG 的纠错思想扩展到 Agent 框架，多轮自主检索。*
