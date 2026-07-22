---
title: Continuous Batching 连续批处理
slug: continuous-batching
category: 进阶专题
tags: [Continuous Batching, Dynamic Batching, 吞吐量, vLLM, 推理服务]
author: ai-terms-fun
created: 2026-07-22
updated: 2026-07-22
---

# Continuous Batching 连续批处理

> 五层读懂一个词。这次拆的是：**Continuous Batching**--LLM 推理服务的吞吐量秘籍。不等整个 batch 都完成才接新请求，而是谁完成谁走、新请求随时插队，GPU 利用率从 ~30% 飙到 ~90%。

---

## L1 · 一句话点破

**Continuous Batching = 动态拼 batch + 请求级迭代调度**。每个 iteration 级别动态组装 batch：完成的请求立即返回、新请求随时加入，不等整个 batch 同步。配合 PagedAttention，把 GPU 利用率从 ~30% 提升到 ~90%。

---

## L2 · 通俗类比

传统 batching 像一辆**定点班车**：

- 凑满一车乘客（batch）才发车
- 每个乘客目的地不同（序列长度不同）
- 要等**最远的乘客**到站，整班车才能返程接下一批
- 短途乘客到了也得等长途乘客，GPU 闲置

**问题**：LLM 生成是自回归，不同请求生成长度差异大。一个请求生成 10 token，另一个生成 1000 token，整 batch 要等 1000 token 的那个完成，GPU 大量时间在等最长请求。

**Continuous Batching 像一辆公交车**：

- 不定点发车，持续运行
- 每站（iteration）有人下车（完成的请求）就下
- 同时有人上车（新请求）就上
- 车一直满载运行，不空等

**具体机制**：

- 每个 iteration（生成一个 token）级别调度
- 某请求生成完（遇到 EOS），立即从 batch 移除，返回结果
- 新请求随时加入 batch，填补空位
- batch 大小动态变化，GPU 始终满载

**对比**：

| 维度 | Static Batching | Continuous Batching |
|------|----------------|---------------------|
| 同步点 | 整 batch 完成 | 每 iteration |
| 短请求等待长请求 | 是 | 否 |
| 新请求等待 | 等当前 batch 完成 | 随时加入 |
| GPU 利用率 | ~30% | ~90% |
| 吞吐量 | 基线 | 2-10x |

**配合 PagedAttention**：

- PagedAttention 让 KV-Cache 按需分配，请求可以随时加入/移除
- Continuous Batching 动态调度，配合 PagedAttention 的灵活内存
- 两者是 vLLM 高吞吐的两大支柱

**代价**：调度开销；实现复杂；单请求延迟可能略增（batch 内竞争）。

---

## L3 · 正经定义

**Continuous Batching**（又称 Dynamic Batching / Iteration-Level Batching）：LLM 推理服务技术，在 iteration（token）级别动态组装 batch。每个 iteration 检查哪些请求完成（生成 EOS）、哪些新请求等待加入，动态调整 batch 组成，不等整个 batch 同步完成。

**与传统 Static Batching 的对比**：

**Static Batching**：

```
t=0:    batch = [req1(10 tok), req2(1000 tok), req3(50 tok)]
t=10:   req1 完成，但 req2/req3 还在，batch 不变
t=50:   req3 完成，req2 还在
t=1000: req2 完成，整 batch 结束，才能接新请求
```

**Continuous Batching**：

```
t=0:    batch = [req1(10 tok), req2(1000 tok), req3(50 tok)]
t=10:   req1 完成，移除，加入 req4
        batch = [req2, req3, req4]
t=50:   req3 完成，移除，加入 req5
        batch = [req2, req4, req5]
t=1000: req2 完成，移除，加入 req6
        batch = [req4, req5, req6]
        ... 持续运行
```

**关键调度决策**（每个 iteration）：

1. 哪些请求完成？移除
2. 哪些新请求等待？按优先级选加入
3. batch 大小是否超限？超限则不加新请求
4. 显存是否够？不够则不加新请求

**参考资料**：

- 📄 Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023（vLLM，Continuous Batching 实现）
- 📄 Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022（Continuous Batching 最早提出）
- 🔧 vLLM 文档：https://docs.vllm.ai/
- 🔧 TGI（Text Generation Inference）：https://github.com/huggingface/text-generation-inference

---

## L4 · 原理深挖

### 4.1 Static Batching 的低效

**问题 1：木桶效应**。batch 内最长请求决定整 batch 时间，短请求完成后 GPU 闲置但仍占用资源。

**问题 2：无法动态加入**。batch 一旦开始，新请求必须等整 batch 完成才能加入。

**问题 3：padding 浪费**。不同请求序列长度不同，要 padding 到最长，计算浪费。

**实测**（Llama-7B，batch=8，请求长度 10-1000 token 均匀分布）：

- Static Batching：GPU 利用率 ~30%，吞吐量 100 token/s
- Continuous Batching：GPU 利用率 ~90%，吞吐量 300+ token/s

### 4.2 Continuous Batching 的调度

```python
class ContinuousBatchingScheduler:
    def __init__(self, max_batch_size, max_num_tokens):
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens  # 总 token 数限制
        self.running_batch = []  # 正在运行的请求
        self.waiting_queue = []  # 等待的请求
    
    def schedule(self):
        # 1. 移除完成的请求
        self.running_batch = [r for r in self.running_batch if not r.finished]
        
        # 2. 尝试从等待队列加入新请求
        while (len(self.running_batch) < self.max_batch_size and 
               self.waiting_queue):
            # 检查显存是否够
            if self.check_memory(self.waiting_queue[0]):
                req = self.waiting_queue.pop(0)
                self.running_batch.append(req)
            else:
                break  # 显存不够，停止加入
        
        # 3. 当前 iteration 处理 running_batch
        return self.running_batch
    
    def check_memory(self, new_request):
        # 估算新请求的 KV-Cache 显存需求
        estimated_kv_cache = self.estimate_kv_cache(new_request)
        return self.current_memory + estimated_kv_cache < self.memory_limit
```

**关键调度策略**：

- **FCFS（先来先服务）**：简单，默认
- **最短任务优先**：短请求优先，减少等待
- **优先级调度**：VIP 请求优先
- **公平调度**：保证各类请求都能被处理

### 4.3 Prefill 和 Decode 的分离调度

**Prefill 阶段**（处理 prompt）和 **Decode 阶段**（生成 token）特性不同：

| 阶段 | 计算量 | 显存 | 瓶颈 |
|------|--------|------|------|
| Prefill | 大（一次算 prompt 全部 token） | 增长 | 计算 |
| Decode | 小（每次算 1 token） | 稳定 | 内存 |

**混合 batch 的问题**：

- Prefill 请求计算密集，decode 请求内存密集
- 同 batch 混合时，prefill 拖慢 decode，decode 占内存拖累 prefill

**分离调度**（vLLM 的 chunked prefill）：

- Prefill 和 decode 分开调度
- 同一 iteration 可以有 prefill batch 和 decode batch
- 或将 prefill 切片，和 decode 混合

**效果**：吞吐量再提升 20-50%。

### 4.4 Continuous Batching 的内存管理

Continuous Batching 要配合 PagedAttention 才能发挥最大效果：

**为什么需要 PagedAttention**：

- Continuous Batching 动态加入/移除请求，KV-Cache 动态变化
- 传统连续分配无法应对动态变化（碎片化）
- PagedAttention 的按需分页完美匹配 Continuous Batching 的动态特性

**配合机制**：

```
新请求加入：
    1. Scheduler 决定加入
    2. Block Manager 分配 prefill 的 KV-Cache blocks
    3. Prefill 计算，填充 blocks
    4. 加入 running batch，开始 decode

请求完成：
    1. 生成 EOS，标记完成
    2. 从 running batch 移除
    3. Block Manager 释放 KV-Cache blocks
    4. 返回结果给用户
```

### 4.5 Continuous Batching 的吞吐量分析

**吞吐量公式**：

$$
\text{Throughput} = \frac{\text{batch\_size} \times \text{tokens\_per\_iteration}}{\text{iteration\_time}}
$$

**Static Batching 的限制**：

- batch_size 被最长请求限制
- 长 request 拖慢整 batch
- 大量时间在 padding 和等待

**Continuous Batching 的提升**：

- batch_size 动态满载
- 短请求完成立即移除，不拖累长请求
- 新请求立即填补，GPU 不空转

**实测对比**（vLLM 论文，Llama-7B，A10G）：

| 并发请求数 | Static | Continuous | 提升 |
|-----------|--------|-----------|------|
| 1 | 30 tok/s | 30 tok/s | 1x |
| 10 | 80 tok/s | 250 tok/s | 3x |
| 50 | 60 tok/s | 400 tok/s | 7x |
| 100 | 40 tok/s | 450 tok/s | 11x |

**并发越高，Continuous Batching 优势越大**。

### 4.6 Continuous Batching 的延迟权衡

**吞吐量 vs 延迟的 trade-off**：

- Continuous Batching 提升吞吐量，但单请求延迟可能增加
- 原因：batch 内请求竞争计算资源
- 大 batch 时，单请求 decode 慢

**延迟优化**：

- 限制 max_batch_size，控制单请求延迟
- 优先级调度，VIP 请求小 batch
- 延迟敏感场景用小 batch，吞吐敏感场景用大 batch

**实践**：

- 对话场景：延迟敏感，max_batch_size = 8-16
- 批处理场景：吞吐敏感，max_batch_size = 64-256
- 混合场景：动态调整

### 4.7 Continuous Batching 的实现

**主流推理引擎**：

| 引擎 | Continuous Batching | PagedAttention |
|------|--------------------|----|
| vLLM | ✅ | ✅ |
| TGI | ✅ | ✅ |
| TensorRT-LLM | ✅ | ✅ |
| SGLang | ✅ | ✅ |
| HF Transformers | ❌（static） | ❌ |

**vLLM 的实现**：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7B")

# 批量请求，vLLM 自动用 continuous batching
prompts = ["prompt1", "prompt2", "prompt3", ...]
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

# vLLM 内部自动调度，用户无需关心 batch
outputs = llm.generate(prompts, sampling_params)
```

**服务模式**：

```bash
# 启动 OpenAI 兼容 API 服务
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7B

# 用户请求自动进入 continuous batching 队列
curl http://localhost:8000/v1/completions -d '{"prompt": "...", ...}'
```

### 4.8 Continuous Batching 的局限

**局限 1：调度开销**。每 iteration 调度决策有开销，小模型上比例较高。

**局限 2：单请求延迟增加**。大 batch 时单请求 decode 变慢，延迟敏感场景受影响。

**局限 3：实现复杂**。要处理并发、内存、调度、错误恢复，工程量大。

**局限 4：请求间干扰**。大 batch 内请求可能互相影响（如某个请求生成长，拖慢其他）。

**局限 5：冷启动延迟**。新请求加入要 prefill，prefill 计算密集，可能拖慢同 batch 的 decode。

---

## L5 · 沿革与坑

### 5.1 沿革

- **2022-07**：Orca 论文（OSDI）首次提出 iteration-level scheduling
- **2023-06**：vLLM 把 Continuous Batching + PagedAttention 结合，引爆社区
- **2023 下半年**：TGI、TensorRT-LLM 等纷纷跟进
- **2024**：chunked prefill、分离调度等优化出现
- **2024-2025**：Continuous Batching 成为 LLM 推理服务标配，所有主流引擎支持

### 5.2 常见坑

**坑 1：小并发用 Continuous Batching 期望大提升**。并发 < 4 时优势不明显，它是为高并发设计的。

**坑 2：max_batch_size 设错**。太大单请求延迟高，太小吞吐量低。要按场景调。

**坑 3：没配合 PagedAttention**。Continuous Batching 不配 PagedAttention，KV-Cache 碎片化严重，效果打折。

**坑 4：prefill 和 decode 没分离调度**。混合 batch 时 prefill 拖慢 decode。要用 chunked prefill。

**坑 5：延迟敏感场景盲目追求吞吐**。大 batch 高吞吐但单请求延迟高，对话场景用户感受差。要限 batch size。

**坑 6：调度策略选错**。FCFS 在长请求多时饿死短请求。要按场景选调度策略。

**坑 7：显存预算没考虑动态变化**。Continuous Batching 下请求数动态变化，显存需求波动。要预留 buffer。

**坑 8：请求优先级没处理**。VIP 请求和普通请求同队，VIP 延迟高。要用优先级队列。

**坑 9：错误恢复机制缺失**。某请求出错影响整 batch。要隔离错误，不影响其他请求。

**坑 10：只看吞吐不看 P99 延迟**。平均延迟好但 P99 高（某些请求等很久）。要监控 P99。

**坑 11：并发控制不当**。无限制接请求导致 OOM。要有显存预算 + 并发上限。

**坑 12：流式输出和 batching 冲突**。流式输出每个 token 要返回，batching 希望攒一批。要设计合理的流式 batching 机制。

### 5.3 面试怎么考

1. **Continuous Batching 解决什么问题？** 答：Static Batching 的木桶效应（短请求等长请求）和无法动态加入新请求，用 iteration 级调度，GPU 利用率从 30% 提到 90%。
2. **Continuous Batching 和 PagedAttention 的关系？** 答：两者配合--PagedAttention 提供灵活的 KV-Cache 内存管理，Continuous Batching 动态调度请求，共同实现高吞吐。
3. **Continuous Batching 的调度决策？** 答：每 iteration 检查完成请求（移除）、等待请求（按优先级加入）、batch 上限和显存限制。
4. **Continuous Batching 的 trade-off？** 答：吞吐量 vs 单请求延迟。大 batch 高吞吐但单请求延迟增加，要按场景调 max_batch_size。
5. **Prefill 和 Decode 为什么分离调度？** 答：Prefill 计算密集、Decode 内存密集，混合 batch 互相拖累。分离调度或 chunked prefill 提升效率。

---

## 速记卡

| 维度 | Static Batching | Continuous Batching |
|------|----------------|---------------------|
| 同步点 | 整 batch | 每 iteration |
| 新请求 | 等 batch 完成 | 随时加入 |
| 短请求 | 等长请求 | 完成立走 |
| GPU 利用率 | ~30% | ~90% |
| 吞吐量 | 基线 | 2-10x |

**调度决策**（每 iteration）：

```
1. 移除完成请求
2. 检查显存 / batch 上限
3. 从等待队列加入新请求
4. 执行 batch 生成 1 token
```

**关键参数**：

| 参数 | 典型值 | 影响 |
|------|--------|------|
| max_batch_size | 8-256 | 吞吐 vs 延迟 |
| max_num_tokens | 8192-32768 | 总 token 限制 |
| 调度策略 | FCFS / 优先级 | 公平性 vs 延迟 |
| chunked prefill | 开启 | prefill/decode 分离 |

**主流引擎**：

| 引擎 | Continuous Batching | PagedAttention |
|------|--------------------|----|
| vLLM | ✅ | ✅ |
| TGI | ✅ | ✅ |
| TensorRT-LLM | ✅ | ✅ |
| HF Transformers | ❌ | ❌ |

**一句话记忆**：Continuous Batching = iteration 级动态拼 batch，谁完成谁走、新请求随时插队，GPU 利用率从 30% 飙到 90%，吞吐量 2-10 倍。配合 PagedAttention 的灵活 KV-Cache 管理效果最佳。高并发场景利器，但要权衡吞吐量 vs 单请求延迟，对话场景限 batch size，批处理场景放开 batch。vLLM/TGI/TensorRT-LLM 标配。

---

> *上一篇：[PagedAttention 分页注意力](./paged-attention) -- PagedAttention 提供灵活内存，Continuous Batching 动态调度，两者配合。*
> *下一篇：[Speculative Decoding 推测解码](./speculative-decoding) -- 用小模型猜 token，大模型验证，降低单请求延迟。*
