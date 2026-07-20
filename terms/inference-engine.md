---
title: 推理引擎（vLLM / TensorRT-LLM）
slug: inference-engine
category: 模型压缩与加速
tags: [推理引擎, vLLM, TensorRT-LLM, KV Cache, PagedAttention, Continuous Batching]
author: ai-terms-fun
created: 2026-07-20
updated: 2026-07-20
---

# 推理引擎（vLLM / TensorRT-LLM）

> **一句话 TL;DR**：推理引擎是大模型生产部署的"加速器"--把训练好的模型高效跑起来。核心技术是 PagedAttention（管理 [KV Cache](./autoregressive)）、Continuous Batching（动态批处理）、Speculative Decoding（投机解码）。代表项目 vLLM（开源主流）、TensorRT-LLM（NVIDIA 官方）、SGLang。它们让 LLM 推理吞吐提升 5-20 倍，是 ChatGPT 等服务能扛高并发的工程基础。

---

## L1 · 一句话点破

推理引擎：**优化大模型推理过程的系统软件，解决"如何让 LLM 在生产环境扛高并发、低延迟、低成本"的工程问题。**

训练框架（PyTorch）关注"算得对"，推理引擎关注"算得快、算得省"。它们处理：

- **KV Cache 管理**（见 [自回归生成](./autoregressive)）：显存大头，需高效管理
- **批处理**：多个请求合并，提升吞吐
- **量化/稀疏加速**：配合 [量化](./quantization)、[剪枝](./pruning)
- **解码加速**：speculative decoding 等

vLLM 是开源主流，TensorRT-LLM 是 NVIDIA 官方方案，SGLang 是新兴竞争者。

## L2 · 通俗类比

餐厅运营：

- **训练框架（PyTorch）**：菜谱研发。厨师慢慢试菜，关注菜的质量，不计较速度。
- **朴素推理（HuggingFace transformers）**：单点厨房。一个顾客点单，厨师做一道，做完再做下一个。简单但慢，高峰期排队。
- **推理引擎（vLLM）**：高效连锁厨房。
  - **批处理**：多个订单合并处理，一锅多菜
  - **PagedAttention**：食材（KV Cache）分页存放，按需取用，不浪费冰箱空间
  - **Continuous Batching**：随到随做，不等整批到齐
  - **Speculative Decoding**：副厨先猜主厨会做什么，主厨只验证不重做

效果：同样厨房（GPU），能服务的顾客数（吞吐）提升 5-20 倍。

为什么需要推理引擎？LLM 推理有几个朴素实现的瓶颈：

1. **KV Cache 显存浪费**：朴素实现预分配最大长度，多数请求用不满，浪费严重
2. **批处理效率低**：传统 batching 等所有请求结束才下一批，长尾请求拖慢整体
3. **解码慢**：每 token 一次前向（见 [自回归生成](./autoregressive)），长生成慢
4. **量化支持差**：朴素框架不优化 INT4/INT8 计算

推理引擎针对这些瓶颈逐一优化。

## L3 · 正经定义

**推理引擎（Inference Engine）**：专门优化 LLM 推理的系统软件，提供高吞吐、低延迟的推理服务。

**主流推理引擎对比**：

| 引擎 | 维护方 | 特点 | 适用 |
|------|--------|------|------|
| **vLLM** | UC Berkeley | 开源主流，PagedAttention 鼻祖 | 通用，开源首选 |
| **TensorRT-LLM** | NVIDIA | 硬件深度优化 | NVIDIA GPU 极致性能 |
| **SGLang** | UC Berkeley | 结构化生成、RadixAttention | 复杂 prompt 程序 |
| **llama.cpp** | ggerganov | CPU/边缘部署 | 消费端、Mac |
| **TGI** | HuggingFace | 易用，HF 生态 | HF 用户 |
| **DeepSpeed-FastGen** | Microsoft | DeepSpeed 生态 | 微软系 |

**核心技术**：

| 技术 | 解决问题 | 出处 |
|------|---------|------|
| **PagedAttention** | KV Cache 显存浪费 | [vLLM, 2023](https://arxiv.org/abs/2309.06180) |
| **Continuous Batching** | 批处理长尾 | Orca, 2022 |
| **Speculative Decoding** | 自回归解码慢 | [Leviathan et al., 2022](https://arxiv.org/abs/2211.17192) |
| **Prefix Caching** | 共享前缀重复计算 | SGLang |
| **Chunked Prefill** | 长 prompt 阻塞 | vLLM/SGLang |

**参考资料**：
- [Kwon et al., 2023 - vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) - 必读
- [Yu et al., 2022 - Orca / Continuous Batching](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Leviathan et al., 2022 - Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [NVIDIA - TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Zheng et al., 2023 - SGLang](https://arxiv.org/abs/2312.07104)

## L4 · 原理深挖

### 4.1 PagedAttention：KV Cache 的分页管理

朴素 LLM 推理的显存浪费：

```
请求 A（最大长度 2048）：实际生成 100 token，预分配 2048 槽位 -> 浪费 1948
请求 B（最大长度 2048）：实际生成 2000 token，预分配 2048 槽位 -> 浪费 48
请求 C（最大长度 2048）：实际生成 50 token，预分配 2048 槽位 -> 浪费 1998
```

平均浪费 50%+ 显存。朴素实现预分配最大长度，多数请求用不满。

[PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) 借鉴操作系统虚拟内存的分页机制：

- KV Cache 分成固定大小的"块"（block，如 16 个 token 的 K、V）
- 每个请求按需申请块，用多少申请多少
- 块可以不连续（逻辑块映射到物理块）
- 共享前缀（如 system prompt）可共享块

效果：显存利用率从 ~50% 提升到 ~95%，batch 大小翻倍，吞吐提升 2-4 倍。

PagedAttention 是 vLLM 的核心创新，也是开源 LLM 推理的标志性技术。

### 4.2 Continuous Batching：动态批处理

传统 batching（static batching）：

```
batch = [请求 A, 请求 B, 请求 C]  # 等齐
开始处理
请求 A 生成完（100 token），但 B、C 还在生成
A 完成后空等，直到 B、C 都完成
batch 结束，下一批
```

问题：长尾请求（C 生成 2000 token）拖慢整个 batch，已完成的请求空等。

[Continuous Batching (Orca, 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) 改为动态：

```
batch = [A, B, C]
开始处理
A 完成 -> 立即移出，新请求 D 加入 batch
B 完成 -> 立即移出，新请求 E 加入 batch
...
batch 永远满，无空等
```

效果：吞吐提升 5-10 倍（实测 vLLM 比 HF transformers 快 10-20 倍）。

Continuous Batching + PagedAttention 是 vLLM 等推理引擎的基础组合。

### 4.3 Speculative Decoding：投机解码

[自回归生成](./autoregressive) 的瓶颈：每 token 一次前向。1000 token = 1000 次前向。

[Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192) 的洞察：**用小模型（draft model）先猜多个 token，大模型（target model）一次验证多个**。

```
1. 小模型快速生成 k 个 token: x_1, x_2, ..., x_k
2. 大模型一次前向，算出每个位置的真实分布
3. 对比：
   - 小模型猜对的 token：接受（大模型也认可）
   - 小模型猜错的 token：拒绝，用大模型分布重采样
4. 接受的前缀 + 重采样的 token = 输出
```

关键：大模型一次前向就能验证 k 个 token（因为 [自回归](./autoregressive) 训练时本就并行），所以验证 k 个 token 的成本约等于生成 1 个。

效果：如果小模型猜得准（如 70%），吞吐提升 2-3 倍，且输出和纯大模型完全一致（无损）。

Speculative Decoding 的变种：

- **Medusa**：训练多个"头"同时预测多个未来 token，不需小模型
- **EAGLE**：用更准的 draft 模型，提升接受率
- **Lookahead Decoding**：并行解码，无需 draft 模型

这是 2023-2024 年推理加速的热点。

### 4.4 Prefix Caching：共享前缀复用

很多请求有共享前缀（如 system prompt、few-shot 示例）。朴素实现每个请求都重新计算前缀的 KV Cache。

[Prefix Caching (SGLang)](https://arxiv.org/abs/2312.07104) 缓存共享前缀的 KV Cache，多个请求复用：

```
请求 A: [system prompt] + [用户问题 1]
请求 B: [system prompt] + [用户问题 2]
请求 C: [system prompt] + [用户问题 3]

朴素: 每个请求都计算 system prompt 的 KV（3 次）
Prefix Caching: 计算 1 次，缓存复用 3 次
```

效果：对长 system prompt + 多请求场景，首 token 延迟（TTFT）降低 5-10 倍。

SGLang 的 RadixAttention 进一步用基数树管理前缀，支持任意前缀共享。

### 4.5 Chunked Prefill：长 prompt 不阻塞

长 prompt（如 32K 输入）的 prefill 阶段计算量大，会阻塞其他请求的 decode。

[Chunked Prefill](https://arxiv.org/abs/2401.01125) 把 prefill 分块，与 decode 交错执行：

```
朴素: [长 prefill 阻塞 5s] -> [decode] -> [长 prefill 阻塞 5s] -> [decode]
Chunked: [prefill chunk 1] [decode] [prefill chunk 2] [decode] ...
```

让长 prompt 不阻塞已生成的请求，整体延迟更稳定。vLLM、SGLang 都支持。

### 4.6 推理引擎的选型

实际选型的考量：

| 维度 | vLLM | TensorRT-LLM | SGLang | llama.cpp |
|------|------|--------------|--------|-----------|
| 硬件 | 通用 GPU | NVIDIA 优化 | 通用 GPU | CPU/边缘 |
| 性能 | 强 | 极强（NVIDIA） | 强（结构化） | 中 |
| 易用 | 易 | 复杂 | 中 | 易 |
| 量化 | GPTQ/AWQ | 深度优化 | GPTQ/AWQ | GGUF |
| 生态 | 开源主流 | NVIDIA 系 | 新兴 | 消费端主流 |

经验法则：

- **服务端 NVIDIA GPU**：vLLM（易用）或 TensorRT-LLM（极致）
- **结构化输出/复杂 prompt 程序**：SGLang
- **消费端/Mac/CPU**：llama.cpp
- **HF 生态深度用户**：TGI

### 4.7 推理引擎的未来

2024-2025 年的趋势：

**① 推理模型（o1、DeepSeek-R1）的引擎支持**

推理模型生成超长思维链，对推理引擎的 KV Cache 管理、长序列优化提出新要求。vLLM 等在跟进。

**② 多模态推理**

图文模型（如 GPT-4V 开源版）的推理引擎，需处理图像编码 + 文本生成。

**③ 端侧推理**

手机、PC 上的本地推理引擎（如 Apple MLX、Intel OpenVINO），让 LLM 真正离线可用。

**④ 推理-训练融合**

 disaggregated prefill/decode、test-time 训练等新范式，模糊训练和推理边界。

推理引擎是大模型从"研究"到"生产"的关键一环，工程价值不亚于模型本身。

## L5 · 沿革与坑

### 沿革

- **2022**：[Orca - Continuous Batching](https://www.usenix.org/conference/osdi22/presentation/yu) 提出，奠定动态批处理基础。
- **2023 年 6 月**：[vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) 发布，开源 LLM 推理革命，吞吐提升 10-20 倍。
- **2023 年 8 月**：NVIDIA 发布 TensorRT-LLM，硬件深度优化。
- **2023 年 11 月**：[Speculative Decoding](https://arxiv.org/abs/2211.17192) 论文 + 开源实现流行。
- **2023 年 12 月**：[SGLang](https://arxiv.org/abs/2312.07104) 发布，RadixAttention 和结构化生成。
- **2024 年**：vLLM、SGLang 快速迭代，成为开源推理主流。Medusa、EAGLE 等 speculative decoding 变体涌现。
- **2024-2025 年**：推理模型（o1、DeepSeek-R1）推动长序列推理优化。端侧推理引擎（Apple MLX 等）兴起。

### 常见误解

- ❌ **误解**：HuggingFace transformers 直接部署就够了。
  ✅ **真相**：HF transformers 是研究框架，生产部署吞吐低 5-20 倍。生产应用推理引擎（vLLM 等）是标配。

- ❌ **误解**：vLLM 一定比 TensorRT-LLM 慢。
  ✅ **真相**：在 NVIDIA 最新硬件上 TensorRT-LLM 通常更快（深度优化），但 vLLM 在通用 GPU 上性能也很好，且易用、生态好。按场景选（4.6）。

- ❌ **误解**：PagedAttention 是 vLLM 的"魔法"。
  ✅ **真相**：PagedAttention 是借鉴 OS 虚拟内存的分页机制。本质是工程创新，不是理论突破。但工程实现到位让它产生巨大价值（4.1）。

- ❌ **误解**：Speculative Decoding 会损失质量。
  ✅ **真相**：经典 speculative decoding 是无损的--大模型验证每个 token，输出和纯大模型完全一致。只是用小模型"猜"加速（4.3）。

- ❌ **误解**：推理引擎只服务端用。
  ✅ **真相**：llama.cpp 等是消费端推理引擎，让 LLM 跑在 Mac、手机、CPU 上。端侧推理是重要方向（4.7）。

- ❌ **误解**：KV Cache 管理是小优化。
  ✅ **真相**：KV Cache 是 LLM 推理显存大头（70B 模型 32K 上下文可达数十 GB）。PagedAttention 让显存利用率从 50% 到 95%，batch 翻倍，是吞吐提升的核心（4.1）。

### 面试怎么考

1. **"vLLM 为什么比 HF transformers 快？"** --PagedAttention（显存利用率翻倍）+ Continuous Batching（无空等）+ 量化/稀疏优化（4.1、4.2）。
2. **"什么是 PagedAttention？"** --借鉴 OS 分页机制管理 KV Cache，按需分配块，显存利用率从 50% 到 95%（4.1）。
3. **"Continuous Batching 和 Static Batching 的区别？"** --Static 等整批完成；Continuous 随到随走，无空等，吞吐提升 5-10 倍（4.2）。
4. **"Speculative Decoding 怎么加速？"** --小模型猜多个 token，大模型一次验证多个。猜对的接受，猜错的重采样。无损，2-3x 加速（4.3）。
5. **"Prefix Caching 解决什么？"** --共享前缀（system prompt 等）的 KV Cache 复用，避免重复计算。首 token 延迟降低 5-10 倍（4.4）。
6. **"vLLM 和 TensorRT-LLM 怎么选？"** --vLLM 易用、开源主流；TensorRT-LLM 在 NVIDIA 硬件极致优化但复杂。按场景选（4.6）。

## 延伸阅读

- 📄 [Kwon et al., 2023 - vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) - 必读
- 📄 [Yu et al., 2022 - Orca / Continuous Batching](https://www.usenix.org/conference/osdi22/presentation/yu)
- 📄 [Leviathan et al., 2022 - Speculative Decoding](https://arxiv.org/abs/2211.17192)
- 📄 [Zheng et al., 2023 - SGLang](https://arxiv.org/abs/2312.07104)
- 📝 [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- 📝 [vLLM 文档](https://docs.vllm.ai/)

---

> *上一篇：[剪枝](./pruning) -- 删掉没用的连接。*
> *下一篇：[Tokenizer](./tokenizer) -- 文本怎么变成模型能吃的数字。*
