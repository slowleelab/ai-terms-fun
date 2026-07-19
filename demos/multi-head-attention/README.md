# 多头注意力 最小可运行 Demo

> 30 行 PyTorch 实现多头注意力，并对比单头，直观看到"多个头学到了不同的注意力模式"。

## 它演示了什么

- 多头注意力的高效实现（一次大投影 + reshape，避免循环 $h$ 次）
- 多个头的注意力矩阵可视化：每个头关注不同的关系
- 与单头注意力的对比

## 安装

```bash
pip install torch matplotlib
```

## 运行

```bash
python multi_head_attention_demo.py
```

运行后会打印每个头的注意力矩阵形状，并保存一张图 `multi_head_weights.png`，展示 $h$ 个头各自的注意力模式。

## 代码

见 [`multi_head_attention_demo.py`](./multi_head_attention_demo.py)。

## 自己动手改造

1. **改头数**：把 `h=8` 改成 `h=1`（退化为单头）或 `h=16`，观察注意力模式的多样性变化。
2. **实现 GQA**：让多个头共享 K/V 投影，看 KV cache 大小变化（提示：K/V 的投影矩阵从 $h$ 组减到 $g$ 组）。
3. **剪枝实验**：训练一个简单任务后，去掉一半头，看效果掉多少。
