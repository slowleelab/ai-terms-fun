---
title: 自回归生成（Autoregressive Generation）
slug: autoregressive
category: 推理与生成
tags: [自回归, 自回归生成, KV Cache, GPT, 因果注意力]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 自回归生成（Autoregressive Generation）

> **一句话 TL;DR**：自回归生成是 GPT 类模型的核心机制--每次基于已生成的内容预测下一个 token，再把新 token 拼回去继续预测，一个字一个字往外吐。它是 GPT 能"无限续写"的本质，也解释了为什么 GPT 回答慢（每生成一个 token 都要跑一次前向）、为什么有 [上下文窗口](./context-window) 限制、为什么 [幻觉](./hallucination) 难以根治。

---

## L1 · 一句话点破

自回归（Autoregressive，AR）：**用过去预测未来。** 模型把已生成的序列 $x_{<t}$ 作为输入，输出下一个 token $x_t$ 的概率分布，采样一个 $x_t$，把它拼到序列末尾，再预测 $x_{t+1}$，循环直到结束。

形式化：

$$
P(x_{1:T}) = \prod_{t=1}^T P(x_t | x_{<t})
$$

GPT/LLaMA/Qwen 等都是自回归模型。它们的"生成"本质是**不断重复"预测下一个 token"这一件事**。看懂这一句，就懂了 GPT 的工作原理。

## L2 · 通俗类比

打字时的"输入法联想"--你打"今天天气"，输入法弹出"很好/不错/真热"候选词；你选了"很好"，输入法继续基于"今天天气很好"预测下一个词。

自回归生成就这样：

```
输入: "今天天气"
模型预测下一个: "很" (P=0.6)
拼上: "今天天气很"
模型预测下一个: "好" (P=0.7)
拼上: "今天天气很好"
模型预测下一个: "，" (P=0.5)
...
```

每一步都把"到目前为止的所有内容"塞回模型，让它预测下一个。生成 1000 字，就是把这个过程重复 1000 次。

这带来几个直接后果：

1. **慢**：每个 token 都要跑一次完整前向，1000 字 = 1000 次前向
2. **上下文有限**：每次都要塞"所有历史"，但模型有 [上下文窗口](./context-window) 限制（如 8K、32K、128K）
3. **错误累积**：早期生成错一个字，后续都基于这个错字继续，[幻觉](./hallucination) 越滚越大
4. **不能回头**：生成完一个 token 就不能改，自回归是单向的

对比一下：BERT 是"双向"的，能同时看到前后文，所以做理解任务强；GPT 是"单向"的，只能看前文，但能生成。这就是 [编码器-解码器](./encoder-decoder) 词条里讲的"理解 vs 生成"分工。

## L3 · 正经定义

**自回归生成**：模型按时间步依次生成 token，每步条件于已生成的全部 token。对 decoder-only 模型（GPT/LLaMA）：

$$
x_t \sim P_\theta(\cdot | x_{<t})
$$

完整生成流程：

```
1. 输入 prompt x_{1:k}
2. for t = k+1, k+2, ...:
     logits = model(x_{1:t-1})
     P_t = softmax(logits[-1])
     x_t = sample(P_t)   # 或 argmax（贪心解码）
     if x_t == EOS: break
3. 返回 x_{k+1:T}
```

**因果注意力（Causal Attention）**：decoder-only 模型的核心。每个位置只能看到自己及之前的位置，不能看未来。实现上是对 [自注意力](./self-attention) 的注意力矩阵加一个上三角 mask：

$$
\text{Attn}_{ij} = 0 \quad \text{if } j > i
$$

这保证训练时能并行（一次前向算所有位置的 loss），推理时严格按因果顺序生成。

**KV Cache**：自回归推理的关键优化。每步计算 $x_t$ 时，前面 $x_{<t}$ 的 K、V 向量不变，缓存起来复用，避免重复计算。这让推理从 $O(T^2)$ 降到 $O(T)$（每步只算新 token 的 K、V）。

**参考资料**：
- [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 与因果 mask
- [Radford et al., 2018/2019 - GPT/GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - decoder-only 自回归
- [Kwon et al., 2023 - vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) - KV Cache 管理优化
- [Pope et al., 2022 - Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)

## L4 · 原理深挖

### 4.1 因果 mask：训练并行、推理串行的根源

decoder-only Transformer 训练时，整条序列 $x_{1:T}$ 一次前向算完所有位置的预测：

$$
\text{loss} = \sum_t \text{CE}(P_\theta(x_t | x_{<t}), x_t)
$$

每个位置的预测只依赖 $x_{<t}$（通过因果 mask 保证）。训练效率高，因为所有 token 一次性处理。

但推理时无法并行：要生成 $x_t$ 必须先有 $x_{<t}$，因为 $x_t$ 的预测依赖 $x_{t-1}$ 的实际值（采样结果）。这是自回归生成的根本瓶颈。

**为什么训练能并行？** 训练时所有 $x_t$ 都是已知的（来自训练数据），不需要"生成"，只需要"预测每个位置的条件概率"。预测之间没有依赖，所以并行。

**为什么推理不能并行？** 推理时 $x_t$ 未知，要采样。$x_t$ 的采样结果决定 $x_{t+1}$ 的输入，串行依赖。除非用 speculative decoding 等技巧（猜多个 token 再验证）。

### 4.2 KV Cache：把推理复杂度从 O(T²) 降到 O(T)

朴素自回归推理：生成第 $t$ 个 token 时，把 $x_{1:t-1}$ 全部塞进模型，重新算所有位置的 K、V。复杂度 $O(T^2)$（每步 $O(t)$，总共 $O(T^2)$）。

观察：每步只新增一个 token $x_{t-1}$，前面 $x_{1:t-2}$ 的 K、V 不变。把它们缓存：

```
cache = {}  # {layer: [K, V]}
for t = k+1, k+2, ...:
    # 只算新 token x_{t-1} 的 K、V
    new_K, new_V = compute_kv(x_{t-1})
    # 拼接到缓存
    cache[layer] = [cat(cache_K, new_K), cat(cache_V, new_V)]
    # 注意力只用新 token 的 Q 查询所有缓存的 K
    logits = attention(Q=new_Q, K=cache_K, V=cache_V)
```

每步复杂度 $O(1)$（只算一个 token 的 K、V），总复杂度 $O(T)$。

**KV Cache 是现代 LLM 推理的标配**，但带来新挑战：

- **显存占用大**：KV Cache 随序列长度线性增长。LLaMA-70B、上下文 32K、batch=1 时，KV Cache 可达数 GB
- **batch 大小受限**：多个请求的 KV Cache 共占显存，限制吞吐
- **长上下文成本**：上下文越长，KV Cache 越大，推理越贵

[vLLM (PagedAttention)](https://arxiv.org/abs/2309.06180) 等推理引擎用类似操作系统的分页机制管理 KV Cache，大幅提升显存利用率。这是 2023 年推理优化的核心突破。

### 4.3 采样策略：从概率分布到具体 token

模型每步输出下一个 token 的概率分布 $P_t$。怎么从分布里选 token 决定生成的"风格"：

- **贪心解码**：选概率最高的 token。确定性强，但容易重复、机械
- **Top-k 采样**：只在前 k 个高概率 token 里采。增加多样性
- **Top-p 采样（nucleus）**：在累计概率前 p 的 token 里采。自适应多样性
- **[温度](./temperature)**：调节 softmax 的"陡峭度"。高温随机、低温确定

这些策略见 [贪心解码](./greedy-decoding)、[Top-k](./top-k-sampling)、[Top-p](./top-p-sampling)、[温度](./temperature) 各词条。

采样的本质权衡：**多样性 vs 一致性**。完全贪心 = 枯燥重复；完全随机 = 乱讲。好的采样在两者间平衡。

### 4.4 自回归的局限

**① 错误累积**

每步采样都可能选错。错字进入历史，后续预测基于错字，错上加错。这是 [幻觉](./hallucination) 的机制根源之一。

缓解：beam search 保留多条候选、约束解码（grammar constraints）、self-consistency（多次采样取多数）。

**② 不能回头**

生成 $x_t$ 后不能改。但人写作时会回头改。自回归模型没有"编辑"能力，只能"往前续"。

缓解：迭代式生成（generate-then-edit）、填充式生成（如 GLM 的 prefix-LM）、tree of thoughts 等搜索式生成。

**③ 上下文窗口限制**

KV Cache 随长度线性增长，注意力计算随长度平方增长（虽有 FlashAttention 等优化）。模型有最大上下文限制（如 8K、32K、128K、1M）。

长上下文难题见 [上下文窗口](./context-window) 词条。

**④ 推理慢**

每 token 一次前向，长生成耗时。speculative decoding（小模型先猜、大模型验证）、continuous batching 等优化旨在加速。

### 4.5 非自回归生成：另一种思路

自回归不是唯一生成方式。**非自回归（Non-Autoregressive, NAR）** 生成一次性输出所有 token：

$$
x_{1:T} = f_\theta(\text{prompt}) \quad \text{(一次前向)}
$$

优势：快（一次前向 vs T 次前向）。
劣势：质量差，因为 token 之间没有依赖建模。

NAR 在机器翻译（[Gu et al., 2017](https://arxiv.org/abs/1711.02281)）等任务有研究，但质量远不及 AR。目前大模型生成仍以 AR 为主，NAR 只在特定场景（如语音合成、部分翻译）使用。

**半自回归（Semi-Autoregressive）** 是折中：每次生成一小段（如 4 个 token），再迭代。代表如 [Mask-Predict](https://arxiv.org/abs/1904.09324)。

### 4.6 自回归 vs 双向：GPT 和 BERT 的根本区别

回顾 [编码器-解码器](./encoder-decoder) 词条讲的架构分工：

| 维度 | BERT（双向） | GPT（自回归） |
|------|-------------|--------------|
| 注意力 | 全可见（每位置看全部） | 因果 mask（只看前文） |
| 训练目标 | MLM（完形填空） | Next-token CE |
| 优势 | 理解任务强 | 生成能力强 |
| 劣势 | 不能直接生成 | 理解任务略弱（但大模型后差距缩小） |

因果 mask 让 GPT 训练时也能并行，但限制了它"看后文"的能力。这是 GPT 在纯理解任务（如分类、NER）早期略弱于 BERT 的原因。

大模型时代，GPT 类模型通过规模 + [指令微调](./instruction-tuning) 在理解任务上追平甚至超越 BERT，自回归的"通用性"胜出。BERT 类模型主要保留在专用理解任务（如 [Embedding](./embedding) 编码）。

## L5 · 沿革与坑

### 沿革

- **2017**：[Transformer](https://arxiv.org/abs/1706.03762) 提出因果 mask，原论文用于 decoder 部分。
- **2018**：[GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 用纯 decoder（自回归）做预训练，证明 AR 路线可行。
- **2019**：[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 扩大规模，展示 zero-shot 生成能力。
- **2020**：[GPT-3](https://arxiv.org/abs/2005.14165) 进一步扩大，自回归 + in-context learning 引爆生成式 AI。
- **2022-2023**：[FlashAttention](https://arxiv.org/abs/2205.14135)、[vLLM/PagedAttention](https://arxiv.org/abs/2309.06180) 等推理优化爆发，KV Cache 管理成为核心工程问题。
- **2023**：Speculative decoding（[Leviathan et al.](https://arxiv.org/abs/2211.17192)）流行，加速自回归推理。
- **2024-2025**：长上下文（128K、1M）+ 推理模型（o1、DeepSeek-R1）让自回归生成在"推理任务"上展现新能力。但 AR 的根本瓶颈（慢、错误累积）仍是研究焦点。

### 常见误解

- ❌ **误解**：自回归模型训练时也是逐 token 生成的。
  ✅ **真相**：训练时所有 token 已知，可以一次前向并行计算所有位置的 loss（通过因果 mask 保证依赖）。只有推理（生成）才是逐 token 串行（4.1）。

- ❌ **误解**：自回归 = GPT，BERT 不是自回归。
  ✅ **真相**：BERT 是双向的，不是自回归生成。但 BERT 的 MLM 也可以看作"给定上下文预测被 mask 的 token"，是某种"非顺序的自回归"。严格说自回归指"按时间顺序、条件于过去"的生成，GPT 是典型代表。

- ❌ **误解**：KV Cache 是可选优化。
  ✅ **真相**：现代 LLM 推理没有 KV Cache 基本不可用（慢 100 倍以上）。它是标配，不是可选。所有主流推理框架（vLLM、TensorRT-LLM、SGLang）都内置。

- ❌ **误解**：自回归模型能"理解"它生成的内容。
  ✅ **真相**：模型只是基于统计规律预测下一个 token，没有真正的"理解"。生成流畅 ≠ 理解。这是 [幻觉](./hallucination) 的根源--模型生成看似合理但实际错误的内容。

- ❌ **误解**：上下文越长模型越聪明。
  ✅ **真相**：上下文长只意味着能处理更长的输入，不代表"更聪明"。模型在长上下文上还有"lost in the middle"现象（中间位置信息容易被忽略）。上下文长度是工具，不是能力。

- ❌ **误解**：自回归生成的每个 token 都是模型"想说的"。
  ✅ **真相**：模型输出的是概率分布，采样的随机性让结果有不确定性。同一 prompt 多次生成可能差异很大（高温下尤其明显）。"想说的"是分布，具体 token 是采样结果。

### 面试怎么考

1. **"什么是自回归生成？为什么 GPT 是自回归的？"** --每次基于已生成内容预测下一个 token，循环生成。GPT 用因果 mask 保证训练时能并行，推理时按顺序生成（L1、L3）。
2. **"训练时能并行，为什么推理时不能？"** --训练时所有 token 已知，预测之间无依赖；推理时 $x_t$ 未知需采样，$x_{t+1}$ 依赖 $x_t$，串行（4.1）。
3. **"什么是 KV Cache？为什么需要？"** --每步只新增一个 token，前面 token 的 K、V 不变，缓存复用。把推理从 $O(T^2)$ 降到 $O(T)$（4.2）。
4. **"自回归生成有什么局限？"** --慢、错误累积、不能回头、上下文有限（4.4）。
5. **"GPT 和 BERT 的注意力有什么区别？"** --GPT 用因果 mask 只看前文，BERT 全可见。前者能生成，后者理解强（4.6）。
6. **"为什么大模型推理慢？"** --每 token 一次前向，长生成需多次前向；KV Cache 显存大；注意力计算随长度增长。

## 延伸阅读

- 📄 [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 📄 [Radford et al., 2018 - GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- 📄 [Kwon et al., 2023 - vLLM](https://arxiv.org/abs/2309.06180)
- 📄 [Dao et al., 2022 - FlashAttention](https://arxiv.org/abs/2205.14135)
- 📄 [Leviathan et al., 2022 - Speculative Decoding](https://arxiv.org/abs/2211.17192)

---

> *上一篇：[过拟合 & 正则化](./overfitting) -- Dropout / 权重衰减为什么有效。*
> *下一篇：[贪心解码](./greedy-decoding) -- 自回归生成最简单的选词策略。*
