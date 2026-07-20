---
title: CLIP（对比语言-图像预训练）
slug: clip
category: 数据表示与编码
tags: [CLIP, 多模态, 对比学习, 图文对齐, 零样本分类]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# CLIP（对比语言-图像预训练）

> **一句话 TL;DR**：CLIP 是 OpenAI 2021 年提出的多模态模型，用对比学习把图像和文本对齐到同一个 [embedding](./embedding) 空间。它让"找一张猫的图"和"找'猫'这个词"用同一个向量表示，实现零样本图像分类、图文检索、多模态 RAG。CLIP 是多模态 AI 的奠基之作，影响了后续 BLIP、LLaVA、GPT-4V 等。

---

## L1 · 一句话点破

CLIP（Contrastive Language-Image Pre-training）：**训练两个编码器（图像编码器 + 文本编码器），让"匹配的图文对"在向量空间接近，"不匹配的图文对"远离。**

```
图像 "猫的照片" -> 图像编码器 -> 图像向量 [0.3, -0.2, ...]
文本 "a photo of a cat" -> 文本编码器 -> 文本向量 [0.28, -0.18, ...]
   -> 两者余弦相似度高（匹配）

图像 "猫的照片" + 文本 "a photo of a dog" -> 余弦相似度低（不匹配）
```

训练后，图像和文本在同一向量空间，可以用文本查图、用图查文本、做零样本分类。

## L2 · 通俗类比

教一个孩子认动物，但不用标注"这是猫"：

- **传统监督学习**：拿一万张猫的图，告诉孩子"这是猫"。学到的只是"猫"这个标签。
- **CLIP 的方式**：拿网上自然出现的图文对（图 + 配文"a cute cat"），让孩子理解"图里这只动物"和"配文里的 cat 词"对应。学到的不仅是"猫"，还有"猫"和语言中所有相关概念的关系。

CLIP 学到的是**图文对齐**：图像内容和语言描述之间的对应关系。这让 CLIP 能做"零样本"任务：

- **零样本分类**：给一张新图，问"这是猫、狗还是鸟？"，CLIP 把图和"a photo of a cat/dog/bird"分别算相似度，选最高的。
- **图文检索**：给一段文字，找最匹配的图；给一张图，找最匹配的文字描述。
- **多模态 RAG**：用图作为知识源，文字查询检索相关图。

为什么 CLIP 重要？因为它**不依赖标注数据**（用网上自然图文对），且**泛化到没见过的类别**（零样本）。这打破了传统视觉模型"每个类别都要标注"的限制。

## L3 · 正经定义

**CLIP (Radford et al., 2021)**：双编码器架构，用对比学习对齐图像和文本。

**架构**：

- **图像编码器**：ViT（Vision Transformer）或 ResNet，把图像编码为向量 $\mathbf{I} \in \mathbb{R}^d$
- **文本编码器**：Transformer，把文本编码为向量 $\mathbf{T} \in \mathbb{R}^d$
- 两个编码器输出投影到同一 $d$ 维空间（典型 $d=512$ 或 $768$）

**训练目标（InfoNCE 对比损失，见 [损失函数](./loss-function)）**：

给定 batch 内 $N$ 个图文对 $(\mathbf{I}_i, \mathbf{T}_i)$：

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[\log \frac{\exp(\mathbf{I}_i \cdot \mathbf{T}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{I}_i \cdot \mathbf{T}_j / \tau)} + \log \frac{\exp(\mathbf{I}_i \cdot \mathbf{T}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{I}_j \cdot \mathbf{T}_i / \tau)}\right]
$$

- 分子：匹配对的相似度
- 分母：所有可能配对的相似度和
- 让匹配对相似度高，不匹配对相似度低

**训练数据**：4 亿图文对（WIT，Web Image Text），从互联网爬取。

**能力**：

- **零样本分类**：用文本 prompt（"a photo of a {class}"）做分类
- **图文检索**：文搜图、图搜文
- **多模态 embedding**：图和文在同一空间，可混合检索
- **作为视觉骨干**：为 BLIP、LLaVA、GPT-4V 等提供视觉编码

**参考资料**：
- [Radford et al., 2021 - CLIP](https://arxiv.org/abs/2103.00020) - 必读
- [Jia et al., 2021 - ALIGN](https://arxiv.org/abs/2102.05918) - 类似工作，Google
- [Li et al., 2022 - BLIP](https://arxiv.org/abs/2201.12086)
- [Liu et al., 2023 - LLaVA](https://arxiv.org/abs/2304.08485)

## L4 · 原理深挖

### 4.1 对比学习：CLIP 的核心

CLIP 的核心是**对比学习（contrastive learning）**：

- 正样本：匹配的图文对（图 + 真实描述）
- 负样本：不匹配的图文对（图 + 其他图的描述）
- 目标：拉近正样本，推远负样本

batch 内 $N$ 个图文对，每个图有 1 个正样本（自己的文本）和 $N-1$ 个负样本（其他文本）。对称地，每个文本有 1 个正样本和 $N-1$ 个负样本。

InfoNCE 损失（见 [损失函数](./loss-function)）形式上像分类：把"哪个文本匹配这个图"当作 $N$ 类分类问题。

对比学习的关键：

- **大批次**：负样本越多越好，CLIP 用 batch=32K
- **数据多样性**：图文对应覆盖广泛概念
- **对称设计**：图查文 + 文查图双向

### 4.2 零样本分类：CLIP 的标志性能力

传统图像分类：每个类别需标注数据训练。

CLIP 零样本分类：

```
输入: 一张图
候选类别: ["cat", "dog", "bird", "car"]
prompt 模板: "a photo of a {class}"
计算: 图向量 vs "a photo of a cat" 向量
      图向量 vs "a photo of a dog" 向量
      ...
选: 相似度最高的类
```

效果：CLIP 在 ImageNet 零样本分类 76.2%，接近有监督 ResNet（76.5%）。但 CLIP 没用 ImageNet 训练数据，完全零样本。

零样本分类的关键：

- **prompt 工程**："a photo of a {class}" 比 "{class}" 效果好
- **类名描述**：详细描述（"a small domesticated carnivorous mammal"）比单字（"cat"）好
- **类别空间**：CLIP 见过的概念才能分类

### 4.3 CLIP 的训练数据：WIT

CLIP 用 4 亿图文对训练，来自互联网爬取的 WIT（Web Image Text）。

为什么需要这么多数据？

- 图文对噪声大（配文不一定准确描述图）
- 概念覆盖要广
- 对比学习需要大量负样本

WIT 不公开，但 [LAION-5B](https://arxiv.org/abs/2210.08314) 等开源数据集让社区能复现 CLIP 类模型（如 [OpenCLIP](https://github.com/mlfoundations/open_clip)）。

### 4.4 CLIP 的视觉编码器：ViT vs ResNet

CLIP 论文尝试两种图像编码器：

- **ResNet**：CNN 架构，经典视觉模型
- **ViT (Vision Transformer)**：把图切成 patch，用 Transformer 处理

实验：ViT 版本通常更好，特别是大规模。CLIP-ViT 成为多模态模型的视觉骨干。

后续模型：

- **BLIP/BLIP-2**：用 CLIP 视觉编码器 + Q-bridge 连接 LLM
- **LLaVA**：CLIP-ViT + LLaMA，多模态对话
- **GPT-4V**：闭源，但思路类似（视觉编码器 + LLM）

CLIP 的视觉编码器成为多模态 LLM 的事实标准组件之一。

### 4.5 CLIP 的局限

**① 计数能力弱**

CLIP 难以精确计数（"3 个苹果" vs "4 个苹果" 常分错）。

**② 空间关系弱**

"猫在桌子上" vs "桌子在猫上" 难以区分。

**③ 细粒度分类弱**

对训练时没见过的细分类别（如不同鸟种）效果差。

**④ 偏见**

训练数据来自互联网，包含社会偏见（如性别、种族刻板印象）。CLIP 输出可能放大这些偏见。

**⑤ 不擅长生成**

CLIP 是判别模型（对齐），不是生成模型。要生成图像需配合 DALL-E、Stable Diffusion 等（Stable Diffusion 用 CLIP 的文本编码器）。

### 4.6 CLIP 的衍生与后续

**CLIP 衍生模型**：

- **OpenCLIP**：开源复现，多种规模
- **Chinese-CLIP**：中文优化
- **EVA-CLIP**：更强性能
- **SigLIP**：用 sigmoid 损失替代 softmax，效果更好

**CLIP 思想的扩展**：

- **视频 CLIP**：VideoCLIP 等，对齐视频和文本
- **音频 CLIP**：CLAP 等，对齐音频和文本
- **多模态 LLM**：BLIP、LLaVA、GPT-4V 等用 CLIP 视觉编码器

CLIP 的对比学习思想已扩展到几乎所有模态对齐任务。

### 4.7 CLIP 在 RAG 中的应用

CLIP 让多模态 RAG 成为可能：

```
1. 索引阶段:
   文档中的图 -> CLIP 图像编码器 -> 图向量 -> 入向量库
   文档中的文本 -> CLIP 文本编码器 -> 文本向量 -> 入向量库

2. 查询阶段:
   文字查询 -> CLIP 文本编码器 -> 查询向量
   检索向量库 -> 返回相关文本和图

3. 生成阶段:
   检索结果（文本+图）拼到多模态 LLM -> 生成答案
```

应用场景：

- **图文混合知识库**：产品手册（图+文）、医学影像+报告
- **视觉问答**：基于图像库回答问题
- **内容审核**：检索可疑图文

CLIP 是多模态 RAG 的基础组件。

## L5 · 沿革与坑

### 沿革

- **2021 年 1 月**：[Radford et al. - CLIP](https://arxiv.org/abs/2103.00020) 发表，零样本分类震撼社区。
- **2021 年 5 月**：[ALIGN (Google)](https://arxiv.org/abs/2102.05918) 类似工作，用 10 亿图文对。
- **2022 年**：[BLIP/BLIP-2](https://arxiv.org/abs/2201.12086) 用 CLIP 视觉编码器做多模态理解。
- **2022-2023**：Stable Diffusion 等用 CLIP 文本编码器做文本到图像生成。
- **2023 年**：[LLaVA](https://arxiv.org/abs/2304.08485) 等 多模态 LLM 用 CLIP-ViT 视觉编码器。
- **2023 年**：[SigLIP](https://arxiv.org/abs/2303.15343) 用 sigmoid 损失改进 CLIP。
- **2024-2025**：CLIP 成为多模态 AI 标配组件。多模态 RAG、视觉搜索等产品依赖 CLIP 类模型。

### 常见误解

- ❌ **误解**：CLIP 是图像分类模型。
  ✅ **真相**：CLIP 是图文对齐模型，零样本分类只是它的一个应用。它还能做图文检索、多模态 embedding、视觉骨干等（L1、L3）。

- ❌ **误解**：CLIP 用了标注数据。
  ✅ **真相**：CLIP 用网上爬取的自然图文对，无需人工标注。这是它零样本能力的关键（4.3）。

- ❌ **误解**：CLIP 能生成图像。
  ✅ **真相**：CLIP 是判别/对齐模型，不生成图像。生成需 DALL-E、Stable Diffusion 等（但 Stable Diffusion 用 CLIP 的文本编码器）（4.5）。

- ❌ **误解**：CLIP 在所有视觉任务上都强。
  ✅ **真相**：CLIP 在零样本和泛化上强，但计数、空间关系、细粒度分类等任务上弱。专有监督模型在特定任务上常优于 CLIP（4.5）。

- ❌ **误解**：CLIP 已经过时。
  ✅ **真相**：CLIP 思想仍是多模态 AI 基础。SigLIP、EVA-CLIP 等改进延续 CLIP 路线。多模态 LLM 仍用 CLIP 视觉编码器（4.6）。

- ❌ **误解**：CLIP 的训练数据是公开的。
  ✅ **真相**：CLIP 的 WIT 数据集不公开。但 LAION 等开源数据集让社区能复现（OpenCLIP）（4.3）。

### 面试怎么考

1. **"什么是 CLIP？核心思想？"** --双编码器对比学习，对齐图像和文本到同一向量空间。匹配对相似度高，不匹配低（L1、L3）。
2. **"CLIP 怎么做零样本分类？"** --把图和"a photo of a {class}"算相似度，选最高。无需训练数据（4.2）。
3. **"CLIP 的训练损失是什么？"** --InfoNCE 对比损失。batch 内每个图 1 正样本 N-1 负样本（4.1）。
4. **"CLIP 训练数据为什么大？"** --对比学习需大量负样本；图文对噪声大需数据量补偿；概念覆盖要广（4.3）。
5. **"CLIP 在多模态 LLM 里起什么作用？"** --提供视觉编码器。BLIP、LLaVA、GPT-4V 等用 CLIP-ViT 编码图像（4.4）。
6. **"CLIP 有什么局限？"** --计数、空间关系、细粒度分类弱；有偏见；不擅长生成（4.5）。

## 延伸阅读

- 📄 [Radford et al., 2021 - CLIP](https://arxiv.org/abs/2103.00020) - 必读
- 📄 [Jia et al., 2021 - ALIGN](https://arxiv.org/abs/2102.05918)
- 📄 [Li et al., 2022 - BLIP](https://arxiv.org/abs/2201.12086)
- 📄 [Liu et al., 2023 - LLaVA](https://arxiv.org/abs/2304.08485)
- 📝 [OpenCLIP](https://github.com/mlfoundations/open_clip)

---

> *上一篇：[稠密 vs 稀疏向量](./dense-sparse-vector) -- 两种向量表示的取舍。*
> *下一篇：[上下文窗口](./context-window) -- 模型一次能看多长。*
