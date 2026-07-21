---
title: Multi-modal RAG 多模态检索
slug: multimodal-rag
category: 进阶专题
tags: [Multi-modal RAG, 多模态, CLIP, 跨模态检索, 表格, 图像]
author: ai-terms-fun
created: 2026-07-21
updated: 2026-07-21
---

# Multi-modal RAG 多模态检索

> 五层读懂一个词。这次拆的是：**Multi-modal RAG**--把 RAG 从纯文本扩展到文本 + 图像 + 表格 + 音视频，用跨模态 embedding 让"用文字搜图"、"用图搜文"成为可能。

---

## L1 · 一句话点破

**Multi-modal RAG = 多模态 embedding + 跨模态检索 + 多模态生成**。把图像 / 表格 / 音频和文本对齐到同一向量空间，检索时跨模态匹配，生成时多模态输入，专治"图文混排文档"和"跨模态查询"。

---

## L2 · 通俗类比

Naive RAG 只会处理文字：文档里的图片、表格、公式被解析成文字（或直接丢弃），用户也只能用文字提问。这在 PDF / PPT / 网页这类**图文混排**的真实文档前拉胯：

- 财报 PDF 里的柱状图，OCR 出来是一堆数字，丢失"趋势"语义
- 论文里的架构图，文字描述远不如图像本身
- 用户想"找那张展示 Transformer 架构的图"，纯文本检索无能为力

Multi-modal RAG 把检索的"语言"从单一文字扩展到**多模态**：

- **图像**：用 CLIP / SigLIP 把图像 embed 到向量空间
- **表格**：结构化编码或转 markdown
- **文本**：传统 text embedding
- **音频**：用 Whisper 转文字或 CLAP 直接 embed
- **视频**：抽关键帧 + 音频

**核心魔法：跨模态对齐**。CLIP 训练后，"一只猫"的文字 embedding 和猫图片的 embedding 在向量空间里很近。所以：

- 用户输入文字"Transformer 架构图"，能直接检索到那张图
- 用户上传一张架构图，能找到描述它的文字段落
- 用户问"这张表里哪行数据异常"，能检索到表格 + 相关文字说明

**和 Naive RAG 的区别**：

| 维度 | Naive RAG | Multi-modal RAG |
|------|-----------|-----------------|
| 文档类型 | 纯文本 | 文本 + 图 + 表 + 音视频 |
| embedding | text-only | 跨模态对齐 |
| 查询 | 文字 | 文字 / 图 / 表 / 语音 |
| 检索 | 文字相似度 | 跨模态相似度 |
| 生成 | 文字 | 文字 + 多模态输出 |

**代价**：多模态 embedding 模型大、推理慢；图像 / 表格的切分和索引复杂；多模态 LLM（GPT-4V / Gemini）调用贵。所以 Multi-modal RAG 适合确实有跨模态需求的场景，纯文本文档用 Naive RAG 更划算。

---

## L3 · 正经定义

**Multi-modal RAG**：在 RAG 框架中引入多模态数据（图像 / 表格 / 音频 / 视频），通过跨模态 embedding 模型（CLIP / SigLIP / CLAP 等）将不同模态对齐到统一向量空间，实现跨模态检索，并通过多模态 LLM（GPT-4V / Gemini / Claude 3.5 Sonnet）处理多模态上下文生成答案。

**核心组件**：

1. **多模态文档解析**：
   - PDF：PyMuPDF / Unstructured 提取文本 + 图像 + 表格
   - PPT：python-pptx 提取每页图 + 文
   - 网页：BeautifulSoup 提取 img / table / text
2. **多模态切分**：
   - 文本：按段落 / 句子
   - 图像：整图或区域切片
   - 表格：整表或行级
3. **多模态 embedding**：
   - 图像 + 文本：CLIP / SigLIP / BGE-VL
   - 音频：CLAP / Whisper
   - 视频：关键帧 + 音频
4. **多模态索引**：向量库（Milvus / Qdrant）支持多模态向量
5. **跨模态检索**：query 和文档可不同模态
6. **多模态 LLM 生成**：GPT-4V / Gemini / Claude 接收图 + 文 context

**三种主流架构**：

- **架构 1：统一 embedding**（CLIP-based）：所有模态 embed 到同一空间，统一检索
- **架构 2：模态分离 + 路由**：每个模态独立 embedding + 独立索引，路由器选择
- **架构 3：转文本 + 文本 RAG**：图像 OCR / 表格转 markdown / 音频转文字，全转文本后走 Naive RAG

**参考资料**：

- 📄 Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021
- 📄 Chang et al., *WebGLM: Towards An Efficient Web-Enhanced Question Answering System*, 2023（多模态 RAG 早期工作）
- 📄 Zhao et al., *Bubbling: Debiasing retrieval-augmented generation via adaptive planning*, 2024
- 🔧 LlamaIndex Multi-modal RAG：https://docs.llamaindex.ai/en/stable/use_cases/multimodal/
- 🔧 LangChain Multi-modal RAG：https://python.langchain.com/docs/use_cases/multimodal

---

## L4 · 原理深挖

### 4.1 跨模态对齐：CLIP 的魔法

Multi-modal RAG 的基础是**跨模态 embedding**，CLIP 是开山之作。

**CLIP 训练**：

- 数据：4 亿 (图像, 文字描述) 对
- 目标：对比学习，拉近匹配的 (图, 文)，推远不匹配的

$$
\mathcal{L}_{CLIP} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)} \right]
$$

其中 $I_i$ 是图像 embedding，$T_i$ 是文本 embedding，$\text{sim}$ 是余弦相似度，$\tau$ 是温度。

**对齐效果**：训练后，"一只猫坐在沙发上"的文字 embedding 和对应猫图片的 embedding 在向量空间里很近。这就让**跨模态检索**成为可能：

```python
def cross_modal_search(query_text, image_index, clip_model):
    # 文字 query -> embedding
    query_vec = clip_model.encode_text(query_text)
    # 在图像索引里检索
    results = image_index.search(query_vec, top_k=5)
    return results  # 返回最相关的图片
```

**CLIP 的局限**：

- 训练数据偏网络图片，对专业领域（医疗影像 / 工业图纸）效果差
- 文字描述长度有限（~77 token），长文档对齐弱
- 细粒度区分弱（"哈士奇" vs "阿拉斯加"容易混）

**后续改进**：

- **SigLIP**（Google）：sigmoid 替代 softmax，batch size 不受限，效果更好
- **BGE-VL**（智源）：中文多模态强
- **Jina-CLIP**：长文本支持好

### 4.2 多模态文档解析与切分

真实文档（PDF / PPT）的解析是 Multi-modal RAG 最大的工程难点。

**PDF 解析**：

```python
def parse_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # PyMuPDF
    chunks = []
    for page in doc:
        # 文本
        text = page.get_text()
        chunks.extend(text_chunk(text, chunk_size=500))
        # 图像
        for img in page.get_images():
            image = extract_image(img)
            chunks.append(MultimodalChunk(type="image", content=image))
        # 表格
        tables = page.find_tables()
        for table in tables:
            chunks.append(MultimodalChunk(type="table", content=table.to_markdown()))
    return chunks
```

**关键挑战**：

- **图文混排**：图和文在版面上相邻，语义关联强，但切分后分离。解决：保留图文位置关系，做"图文对"chunk
- **表格识别**：PDF 表格被解析成乱码。解决：用 Camelot / Table Transformer 专用工具
- **公式图像**：公式被当图像。解决：用 MathPix / Nougat 把公式转 LaTeX
- **扫描件**：纯图像 PDF，要 OCR。解决：Tesseract / PaddleOCR

**切分策略**：

| 模态 | 切分单元 | 理由 |
|------|---------|------|
| 文本 | 段落 / 句子 | 语义完整 |
| 图像 | 整图 / 区域 | 整图保语义，区域提精度 |
| 表格 | 整表 / 行 | 整表保结构，行级提粒度 |
| 视频 | 关键帧 | 时间采样 |

### 4.3 多模态索引

Multi-modal RAG 需要存多种类型的向量：

```python
class MultimodalIndex:
    def __init__(self):
        self.text_index = VectorIndex(dim=768)    # BGE text
        self.image_index = VectorIndex(dim=512)   # CLIP image
        self.table_index = VectorIndex(dim=768)   # BGE text (表格转 markdown)
    
    def add(self, chunks):
        for chunk in chunks:
            if chunk.type == "text":
                vec = text_embedder.encode(chunk.content)
                self.text_index.add(vec, metadata=chunk.metadata)
            elif chunk.type == "image":
                vec = clip_model.encode_image(chunk.content)
                self.image_index.add(vec, metadata=chunk.metadata)
            elif chunk.type == "table":
                vec = text_embedder.encode(chunk.content)
                self.table_index.add(vec, metadata=chunk.metadata)
    
    def search(self, query, top_k=5):
        # 文字 query 同时检三个索引
        text_vec = text_embedder.encode(query)
        image_vec = clip_model.encode_text(query)  # CLIP 文字编码器
        return {
            "text": self.text_index.search(text_vec, top_k),
            "image": self.image_index.search(image_vec, top_k),
            "table": self.table_index.search(text_vec, top_k)
        }
```

**两种架构对比**：

| 架构 | 优势 | 劣势 |
|------|------|------|
| 统一 embedding（CLIP） | 跨模态检索自然 | 文本精度不如专用 text embedder |
| 模态分离 | 每个模态用最优 embedder | 跨模态检索难（要融合） |
| 转文本 + 文本 RAG | 工程简单 | 丢失图像 / 表格语义 |

**实践**：模态分离 + 融合是主流。文字走 BGE，图像走 CLIP / SigLIP，表格转 markdown 走 BGE。检索后融合（RRF / 加权）。

### 4.4 跨模态检索的典型场景

**场景 1：文字搜图**。用户问"找 Transformer 架构图"，文字 query 经 CLIP text encoder，在图像索引里检索。

**场景 2：图搜文**。用户上传一张架构图，CLIP image encoder 编码，在文本索引里检索描述该图的段落。

**场景 3：图文混合查询**。用户问"这张表的数据和哪段文字描述对应"，表格 + 文字双路检索。

**场景 4：视频检索**。用户问"找讲解 RLHF 的视频片段"，query 编码后在视频关键帧 + 音频索引里检索。

### 4.5 多模态 LLM 生成

检索回的多模态 chunk 喂给多模态 LLM：

```python
def multimodal_rag_generate(query, retrieved_chunks, vlm):
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": f"问题：{query}"})
    messages[0]["content"].append({"type": "text", "text": "参考资料："})
    for chunk in retrieved_chunks:
        if chunk.type == "text":
            messages[0]["content"].append({"type": "text", "text": chunk.content})
        elif chunk.type == "image":
            messages[0]["content"].append({"type": "image_url", "image_url": chunk.content})
        elif chunk.type == "table":
            messages[0]["content"].append({"type": "text", "text": f"表格：\n{chunk.content}"})
    response = vlm.chat(messages)
    return response
```

**主流多模态 LLM**：

- **GPT-4V / GPT-4o**（OpenAI）：图像理解强
- **Gemini 1.5 Pro**（Google）：原生多模态，支持视频
- **Claude 3.5 Sonnet**（Anthropic）：图表理解强
- **Qwen-VL**（阿里）：中文多模态强
- **LLaVA**：开源多模态

**关键能力差异**：

| 能力 | GPT-4o | Gemini 1.5 | Claude 3.5 |
|------|--------|------------|------------|
| 图像理解 | 强 | 强 | 强 |
| 表格理解 | 中 | 强 | 强 |
| 视频理解 | 弱 | 原生支持 | 弱 |
| 长文档 | 中 | 原生 2M context | 200k |
| 中文 | 中 | 中 | 中 |

### 4.6 表格的特殊处理

表格是 Multi-modal RAG 的难点。三种处理方式：

**方式 1：转 markdown**。简单，但复杂表格（合并单元格）失真。

**方式 2：结构化编码**。把表格转成 `[row, col, value]` 三元组，保留结构。

**方式 3：表格 embedding**。用专用表格 embedding 模型（如 TAPAS）。

**实践**：转 markdown + 文本 embedding 是工程主流，简单够用。复杂表格用结构化编码 + SQL 查询。

### 4.7 三种架构的工程权衡

**架构 1：统一 CLIP embedding**

```
文本 -> CLIP text encoder -> 统一空间
图像 -> CLIP image encoder -> 统一空间
检索：统一向量检索
```

- 优势：跨模态检索自然，一个索引搞定
- 劣势：文本检索精度不如专用 text embedder（CLIP 文本能力弱）

**架构 2：模态分离 + 融合**

```
文本 -> BGE -> 文本索引
图像 -> CLIP -> 图像索引
表格 -> BGE (markdown) -> 表格索引
检索：多路 + RRF 融合
```

- 优势：每个模态用最优 embedder，精度高
- 劣势：工程复杂，跨模态融合难调

**架构 3：转文本 + Naive RAG**

```
图像 -> OCR / VLM 描述 -> 文本
表格 -> markdown -> 文本
音频 -> Whisper -> 文本
统一走 Naive RAG
```

- 优势：工程最简单，复用 Naive RAG 全套
- 劣势：丢失图像 / 表格的视觉语义，细粒度查询差

**选型建议**：

- 跨模态查询需求强（图搜文 / 文搜图）：架构 2
- 主要文字 + 少量图：架构 3（转文本 + Naive RAG）
- 极简快速原型：架构 1（统一 CLIP）

### 4.8 Multi-modal RAG 的评估

**检索指标**：

- 跨模态 Recall@K：文字 query 召回正确图像的比例
- 跨模态 Hit Rate@K：图搜文命中相关文的概率

**生成指标**：

- 答案准确率：基于多模态 context 答对比例
- 视觉理解准确率：LLM 是否正确理解图像内容
- 表格推理准确率：LLM 是否正确读表

**评估集**：

- MM-RAG benchmark：多模态 RAG 评估集
- 自建：图文混排文档 + 跨模态 query + 金标准答案

### 4.9 成本与延迟

Multi-modal RAG 比 Naive RAG 贵 **5-20 倍**：

| 维度 | Naive RAG | Multi-modal RAG |
|------|-----------|-----------------|
| embedding | text-only | text + image + table |
| 检索 | 1 次 | 多路（3-5 次） |
| LLM | 文本 LLM | 多模态 LLM（贵 3-5x） |
| 索引大小 | 文本向量 | + 图像向量（大） |
| 解析 | 简单 | 复杂（PDF / 表格 / OCR） |

**降本策略**：

- 图像用更小的 embedding 模型（CLIP ViT-B 替代 ViT-L）
- 表格转文本走 Naive RAG，只有跨模态查询才上多模态
- 多模态 LLM 只在最终生成调用，检索用轻量 embedder

---

## L5 · 沿革与坑

### 5.1 沿革

- **2021**：CLIP 发布，跨模态对齐基础奠定
- **2023**：GPT-4V / Gemini 发布，多模态 LLM 成熟
- **2023 下半年**：LlamaIndex / LangChain 推出 Multi-modal RAG 官方支持
- **2024**：SigLIP / BGE-VL 等更强多模态 embedder 涌现；表格 / 视频检索能力增强
- **2025**：Multi-modal RAG 成为企业级 RAG 标配，与 Agentic RAG 结合处理复杂多模态场景

### 5.2 常见坑

**坑 1：PDF 解析丢图丢表**。用简单 `pdfplumber` 只提文本，图表全丢。要用 PyMuPDF / Unstructured 提多模态。

**坑 2：CLIP 文本能力弱**。直接用 CLIP text encoder 做文本检索，精度不如 BGE。文字检索要用专用 text embedder。

**坑 3：图像切分破坏语义**。把一张架构图切成几块分别 embed，整体语义丢失。整图 embed 优先，区域切片慎用。

**坑 4：表格 OCR 失真**。复杂表格（合并单元格）OCR 成乱码。要用 Camelot / Table Transformer 专用工具。

**坑 5：跨模态融合难调**。多路检索结果用 RRF 融合，但不同模态分数分布不同，融合权重难调。要做归一化。

**坑 6：多模态 LLM 贵**。GPT-4V / Gemini 调用比文本 LLM 贵 3-5x，且图像 token 消耗大。要控制图像数量 + 分辨率。

**坑 7：图像 embedding 维度不一致**。CLIP 512 维，BGE 768 维，不能直接放一个索引。要分库或投影到同维。

**坑 8：忘了图文位置关联**。图文 chunk 独立存储，丢失版面相邻关系。要保留位置元数据，检索时关联。

**坑 9：扫描件 OCR 错误传播**。OCR 错字被 embed 进向量，检索错。要做 OCR 后处理 + 人工抽检。

**坑 10：评估只看文字答案**。多模态 RAG 的图像理解 / 表格推理能力没单独评估，问题被掩盖。要分模态评估。

**坑 11：视频检索只抽关键帧**。关键帧之间动作丢失，时序问题答不了。要结合音频 + 关键帧 + 时间戳。

**坑 12：跨模态 query 路由错**。用户上传图问问题，系统还走文字检索。要识别 query 模态并路由。

### 5.3 面试怎么考

1. **Multi-modal RAG 和 Naive RAG 的区别？** 答：多模态 embedding + 跨模态检索 + 多模态 LLM，支持图文混排文档和跨模态查询。
2. **CLIP 怎么实现跨模态对齐？** 答：对比学习，4 亿 (图, 文) 对训练，拉近匹配对、推远不匹配对，最终图文 embedding 在同一空间。
3. **Multi-modal RAG 的三种架构？** 答：统一 CLIP embedding / 模态分离 + 融合 / 转文本 + Naive RAG。各自权衡精度 vs 工程复杂度。
4. **表格在 Multi-modal RAG 里怎么处理？** 答：转 markdown（简单）/ 结构化编码（保结构）/ 专用表格 embedder（TAPAS）。
5. **Multi-modal RAG 的成本为什么高？** 答：多模态 embedding 慢、多路检索、多模态 LLM 贵 3-5x、解析复杂。

---

## 速记卡

| 模态 | embedder | 切分单元 |
|------|----------|---------|
| 文本 | BGE / E5 | 段落 / 句子 |
| 图像 | CLIP / SigLIP | 整图 / 区域 |
| 表格 | BGE (markdown) | 整表 / 行 |
| 音频 | CLAP / Whisper | 片段 |
| 视频 | 关键帧 + 音频 | 时间采样 |

**三种架构**：

| 架构 | 优势 | 劣势 |
|------|------|------|
| 统一 CLIP | 跨模态自然 | 文本精度弱 |
| 模态分离 + 融合 | 精度最高 | 工程复杂 |
| 转文本 + Naive RAG | 最简单 | 丢视觉语义 |

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| CLIP 维度 | 512 / 768 | 精度 vs 存储 |
| 图像分辨率 | 224 / 336 | 精度 vs 速度 |
| 多路 top_k | 3-5 per modality | 召回 vs 融合难度 |
| 图像 token | 512-2048 | VLM 成本 |

**主流多模态 LLM**：GPT-4o / Gemini 1.5 Pro / Claude 3.5 Sonnet / Qwen-VL / LLaVA

**一句话记忆**：Multi-modal RAG = 跨模态 embedding（CLIP 对齐图文）+ 多模态索引 + 多模态 LLM 生成。三种架构（统一 CLIP / 模态分离 / 转文本），模态分离 + 融合是主流。专治图文混排文档和跨模态查询，代价是 embedding 慢、多路检索、VLM 贵 3-5x、PDF 解析复杂。

---

> *上一篇：[Agentic RAG 智能体检索](./agentic-rag) -- 多轮自主推理循环，专治复杂问题。*
> *下一篇预告：Long-context RAG / Modular RAG -- 长上下文 RAG 与模块化架构，进阶专题的后续扩展方向。*
