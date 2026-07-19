# 自注意力 最小可运行 Demo

> 20 行 PyTorch 实现一个能跑的自注意力，并打印注意力矩阵，直观看到"代词 it 找到 animal"这件事真的发生了。

## 它演示了什么

- Q / K / V 三个投影矩阵的作用
- 注意力矩阵的计算（点积 → 缩放 → softmax）
- 可视化：哪个 token 关注了哪个 token

## 安装

```bash
pip install torch matplotlib
```

## 运行

```bash
python self_attention_demo.py
```

运行后会打印注意力权重矩阵，并保存一张热力图 `attention_weights.png`。

## 代码

见 [`self_attention_demo.py`](./self_attention_demo.py)。

## 自己动手改造

1. **不除以 sqrt(d_k)**：把 `scores / sqrt(d_k)` 改成 `scores`，看权重矩阵是不是变得极端（一行几乎全是 1，其余全是 0）--这就是 softmax 饱和。
2. **去掉 W_Q/W_K/W_V**：直接用 X 算 `softmax(X @ X.T) X`，看效果差异。
3. **换一句更长的句子**：观察注意力矩阵的模式变化。
