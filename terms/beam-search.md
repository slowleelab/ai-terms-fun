---
title: 束搜索 Beam Search
slug: beam-search
category: 推理与生成
tags: [Beam Search, 束搜索, 解码策略, 序列概率, 多样性束搜索]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 束搜索 Beam Search

> **一句话 TL;DR**：beam search 是 [贪心解码](./greedy-decoding) 的改进版--每步不只保留 1 条最优路径，而是保留 B 条（beam width B）候选，扩展探索后选累计概率最高的 B 条继续。它比贪心更接近全局最优，但仍可能陷入重复退化。在翻译、摘要等"有标准答案"的任务上效果好，在开放式对话上反而常不如采样。

---

## L1 · 一句话点破

Beam search：**每步保留 B 条候选序列，对每条候选扩展所有可能的下一个 token，从 B×|V| 条新候选里选累计概率最高的 B 条，循环直到结束。**

形式化：

- 维护 B 条候选序列 $\{y^{(1)}, ..., y^{(B)}\}$ 及其累计对数概率
- 每步对每条候选扩展所有 token $v \in V$，得到 $B \times |V|$ 条新候选
- 选累计对数概率最高的 B 条，丢弃其他
- 直到所有候选生成 EOS 或达最大长度

B=1 时退化为 [贪心解码](./greedy-decoding)。B 越大越接近全局最优（穷举搜索的近似）。

## L2 · 通俗类比

下棋时有两种思路：

- **贪心**：每步只看眼前最优，选完就走，不回头。简单但容易陷入死路。
- **Beam search**：每步保留 B 个"看起来不错的局面"，对每个局面推演所有可能的下一步，从所有结果里选 B 个最优继续。像"宽限 B 条路同时走，看哪条最终最优"。

具体例子（B=2，词表只 3 个词 "好/坏/中"）：

```text
Step 1: 从空扩展
  - "好" P=0.5
  - "坏" P=0.3
  - "中" P=0.2
  保留 top-2: ["好", "坏"]

Step 2: 对每条候选扩展所有 token
  - "好好"   P = 0.5 * 0.4 = 0.20
  - "好坏"   P = 0.5 * 0.3 = 0.15
  - "好中"   P = 0.5 * 0.3 = 0.15
  - "坏好"   P = 0.3 * 0.6 = 0.18
  - "坏坏"   P = 0.3 * 0.2 = 0.06
  - "坏中"   P = 0.3 * 0.2 = 0.06
  保留 top-2: ["好好"(0.20), "坏好"(0.18)]

Step 3: 继续扩展...
```

对比贪心（B=1）：第一步选"好"（0.5 最高），第二步基于"好"选最高，可能错过"坏好"这条路（第一步次高但第二步更高）。

Beam search 的代价：每步计算量是贪心的 B 倍。B=5、B=10 是常见取值。

## L3 · 正经定义

**Beam Search**：宽度为 B 的束搜索，维护 B 条候选序列，每步扩展后保留 top-B。

```python
def beam_search(model, prompt, B, max_len):
    beams = [(prompt, 0.0)]  # (seq, log_prob)
    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == EOS:
                new_beams.append((seq, score))
                continue
            logits = model(seq)[-1]
            log_probs = log_softmax(logits)
            for v in vocab:
                new_beams.append((seq + [v], score + log_probs[v]))
        beams = sorted(new_beams, key=lambda x: -x[1])[:B]
        if all(b[0][-1] == EOS for b in beams): break
    return max(beams, key=lambda x: x[1])[0]
```

**关键变体**：

| 变体 | 改进 | 用途 |
|------|------|------|
| Length-Normalized Beam Search | 用序列长度归一化分数，避免偏好短序列 | 通用 |
| Diverse Beam Search | 在 beam 间加多样性惩罚 | 开放式生成 |
| Beam Search with Length Penalty | 显式长度惩罚/奖励 | 翻译、摘要 |
| Constrained Beam Search | 强制包含/排除某些 token | 关键词生成 |

**分数计算**：

朴素：$\text{score}(y) = \sum_t \log P(y_t | y_{<t})$。问题：长序列累积负值大，beam search 偏好短序列。

长度归一化：$\text{score}(y) = \frac{1}{|y|^\alpha} \sum_t \log P(y_t | y_{<t})$，$\alpha \in [0.6, 0.7]$ 常用。

**参考资料**：
- [Jurafsky & Martin - SLP3 第 13 章](https://web.stanford.edu/~jurafsky/slp3/)
- [Vijayakumar et al., 2016 - Diverse Beam Search](https://arxiv.org/abs/1610.02424)
- [Wu et al., 2016 - Google's NMT System](https://arxiv.org/abs/1609.08144) - length normalization 实务
- [Holtzman et al., 2019 - Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

## L4 · 原理深挖

### 4.1 序列概率与对数空间

beam search 的分数用**对数概率**而非原始概率。原因：

原始概率乘积 $P = \prod P_t$ 在长序列下会下溢（多个小数相乘趋近 0）。取对数变加法：

$$
\log P = \sum_t \log P_t
$$

数值稳定。比较 $\log P$ 大小等价比较 $P$（对数单调）。

但 $\log P$ 始终为负（$P < 1$），长序列累积绝对值大，beam search 倾向短序列。这就是为什么需要长度归一化（4.4）。

### 4.2 为什么 beam search 比贪心更优

[贪心解码](./greedy-decoding) 每步只看当前最优，可能错过"第一步次高但后续更高"的序列。

反例：

```text
路径 A: 0.5 * 0.1 * 0.1 = 0.005  (贪心选，第一步 0.5 最高)
路径 B: 0.4 * 0.9 * 0.9 = 0.324  (整体最优，贪心不会选)
```

贪心选了第一步最高的 0.5，但后续 0.1、0.1 极低，整体 0.005。beam search（B>=2）会同时保留"0.5 路径"和"0.4 路径"，在第二步看到"0.4*0.9=0.36"远高于"0.5*0.1=0.05"，淘汰贪心的选择，最终选 B 路径。

**beam search 找的是近似全局最优**。B 越大越接近穷举搜索（exhaustive search，复杂度 $O(|V|^T)$，不可行）。B=5-10 是工程上质量与成本的平衡。

但 beam search 仍不是真正的全局最优：

- 仍是贪心的"扩展-剪枝"启发式
- B 有限，可能剪掉最终最优的路径
- 不回溯，剪掉的路径不能恢复

### 4.3 Beam search 的退化问题

beam search 虽然比贪心更优，但仍有重复退化（[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751)）：

**① 重复更严重**

beam search 偏好"高概率"序列，而高概率序列常是重复的（模型对已见 pattern 给高概率）。B 越大，越容易找到"高概率的退化序列"。

**② 缺乏多样性**

B 条候选常高度相似，前缀相同只末尾几个 token 不同。在需要多样选择的场景（如对话、写作）不合适。

**③ 不匹配人类语言分布**

[Holtzman et al., 2019](https://arxiv.org/abs/1904.09751) 的关键发现：人类语言的下一 token 概率分布是**双峰**的--少数高概率 token + 长尾低概率 token。beam search 永远选高概率区，生成的文本"太可预测"，不符合人类语言的"合理但适度意外"特性。

这是 ChatGPT 等对话模型默认用 [Top-p 采样](./top-p-sampling) 而非 beam search 的根本原因。

### 4.4 长度归一化：避免偏好短序列

朴素 beam search 偏好短序列（对数概率累加，越长越负）。length normalization 修正：

$$
\text{score}(y) = \frac{1}{|y|^\alpha} \sum_t \log P(y_t | y_{<t})
$$

$\alpha$ 通常 0.6-0.7。$\alpha = 1$ 完全平均（每 token 平均对数概率），$\alpha = 0$ 退化为朴素。

[Wu et al., 2016 - Google NMT](https://arxiv.org/abs/1609.08144) 提出这个技巧，是翻译质量提升的关键。后续几乎所有 beam search 实现都默认带 length normalization。

### 4.5 Beam search 的适用与不适用场景

**适用**：

- **机器翻译**：有标准答案，目标是最接近参考译文，beam search 普遍提升 BLEU
- **摘要生成**：目标是准确概括原文，beam search 提升准确度
- **结构化输出**：如表头生成、SQL 生成，需要确定性高质量输出

**不适用**：

- **开放式对话**：需要多样性、创意，beam search 反而枯燥退化
- **创意写作**：同理
- **长篇生成**：beam search 在长序列上计算成本高且退化严重

经验法则：**有"标准答案"或"参考输出"用 beam search；开放式生成用采样**。

### 4.6 Diverse Beam Search 与其他改进

针对 beam search 缺乏多样性的问题：

**Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))**

把 B 条 beam 分成 G 组，组间加多样性惩罚：

$$
\text{score}_g = \text{score} - \lambda \sum_{g' \in \{1, \dots, g-1\}} \text{sim}(y^{(g)}, y^{(g')})
$$

强迫不同组生成不同内容。提升多样性，但实现复杂。

**Top-k / Top-p 采样**

更彻底的解决：放弃搜索，引入随机性。见 [Top-k 采样](./top-k-sampling)、[Top-p 采样](./top-p-sampling) 词条。

**Speculative Beam Search**

结合 speculative decoding 加速 beam search，2024 年研究热点。

### 4.7 大模型时代的 beam search 地位

GPT-3 之前：beam search 是 NLP 生成任务的标配，几乎所有翻译、摘要系统都用。

GPT-3 之后：

- **对话场景**：几乎清一色采样（Top-p + 温度），beam search 罕见
- **结构化任务**（代码、SQL）：beam search 仍用，但贪心（temp=0）更常见
- **翻译/摘要**：专用模型仍用 beam search，但大模型 zero-shot 在这些任务上常直接采样

beam search 在大模型时代地位下降，但在专用、有标准答案的任务上仍是默认。

## L5 · 沿革与坑

### 沿革

- **1970s**：beam search 在语音识别、早期机器翻译中是标准解码算法。
- **2014-2017**：seq2seq + attention 时代，beam search（B=5）是翻译、摘要标配，BLEU/ROUGE 提升的关键。
- **2016**：[Diverse Beam Search](https://arxiv.org/abs/1610.02424) 提出多样性变体。
- **2016**：[Google NMT](https://arxiv.org/abs/1609.08144) 系统化 length normalization 实务。
- **2019**：[Holtzman et al.](https://arxiv.org/abs/1904.09751) 揭示 beam search 退化问题，nucleus sampling 兴起。
- **2020-2022**：GPT-3、ChatGPT 等大模型默认采样，beam search 在对话场景式微。
- **2023-2024**：专用任务（翻译、代码、SQL）仍用 beam search；研究焦点转向"speculative decoding 加速"和"约束解码"。
- **2025**：推理模型（o1、DeepSeek-R1）用 best-of-n 采样（多次采样取最优）替代 beam search，beam search 进一步边缘化。

### 常见误解

- ❌ **误解**：beam search 总比贪心好。
  ✅ **真相**：在翻译、摘要等"有标准答案"任务上通常更好；在开放式对话上反而更易退化。任务性质决定（4.5）。

- ❌ **误解**：B 越大效果越好。
  ✅ **真相**：B 增大带来边际收益递减，且更易找到"高概率退化序列"。B=5-10 是多数任务最佳，再大性价比低。

- ❌ **误解**：beam search 找到的是全局最优序列。
  ✅ **真相**：仍是启发式近似。B 有限时会剪掉最终最优路径，不回溯。真正的全局最优需要 $O(|V|^T)$ 穷举，不可行。

- ❌ **误解**：大模型时代 beam search 没用了。
  ✅ **真相**：专用任务（翻译、摘要、SQL）仍用。大模型开放式对话用采样，不代表 beam search 淘汰（4.7）。

- ❌ **误解**：beam search 能解决重复退化。
  ✅ **真相**：beam search 反而比贪心更易退化，因为它偏好高概率序列，而高概率序列常是重复的。需要 repetition penalty 等额外机制（4.3）。

### 面试怎么考

1. **"什么是 beam search？和贪心的区别？"** --保留 B 条候选，每步扩展后选 top-B。B=1 退化为贪心。比贪心更接近全局最优（L1、L3）。
2. **"beam search 为什么比贪心好？举例。"** --序列概率是乘积，贪心每步最大不等价整体最大。反例：0.5*0.1*0.1 < 0.4*0.9*0.9（4.2）。
3. **"什么是 length normalization？为什么需要？"** --用 $|y|^\alpha$ 归一化对数概率，避免偏好短序列（4.4）。
4. **"beam search 有什么问题？"** --重复退化、缺乏多样性、不匹配人类语言分布（4.3）。
5. **"什么场景用 beam search，什么场景用采样？"** --有标准答案的任务用 beam search；开放式生成用采样（4.5）。

## 延伸阅读

- 📄 [Vijayakumar et al., 2016 - Diverse Beam Search](https://arxiv.org/abs/1610.02424)
- 📄 [Wu et al., 2016 - Google NMT](https://arxiv.org/abs/1609.08144)
- 📄 [Holtzman et al., 2019 - Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- 📝 [Jurafsky & Martin - SLP3 第 13 章](https://web.stanford.edu/~jurafsky/slp3/)

---

> *上一篇：[贪心解码](./greedy-decoding) -- 自回归生成最简单的选词策略。*
> *下一篇：[Top-k 采样](./top-k-sampling) -- 只在高概率候选里随机。*
