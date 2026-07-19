# 位置编码 最小可运行 Demo

> 实现正弦位置编码和 RoPE，可视化不同位置的编码向量，直观感受"相邻位置相似、相远位置差异大"，以及 RoPE 旋转的几何意义。

## 它演示了什么

- 正弦/余弦位置编码的生成与可视化
- 可学习位置编码的概念
- RoPE 旋转矩阵的实现
- 位置编码相似度矩阵：相邻位置的编码确实更相似

## 安装

```bash
pip install torch matplotlib numpy
```

## 运行

```bash
python positional_encoding_demo.py
```

会生成两张图：
- `sinusoidal_pe.png`：正弦位置编码的热力图（位置 × 维度）
- `pe_similarity.png`：不同位置编码之间的余弦相似度矩阵，可见对角线附近最亮（相邻位置相似）

## 代码

见 [`positional_encoding_demo.py`](./positional_encoding_demo.py)。

## 自己动手改造

1. **超过训练长度**：把 `max_len` 从 128 调到 1024，观察正弦编码的相似度模式是否保持--验证"理论上可外推"。
2. **实现 RoPE**：本 demo 已包含 RoPE 的旋转实现，试着对一个 query/key 对应用 RoPE，验证点积只依赖相对位置。
3. **NTK-aware scaling**：调整 RoPE 的 base（从 10000 改成更大的值），观察外推时相似度模式的变化。
