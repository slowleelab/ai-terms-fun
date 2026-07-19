# Embedding 最小可运行 Demo

> 30 行代码，亲眼看见"语义相近的句子向量也相近"，并戳破"直接平均词向量就行"的幻觉。

## 它演示了什么

- 用 sentence-transformers 把句子编码成向量
- 用余弦相似度衡量语义接近度
- 对比两个反直觉案例：
  - 反例 1：`not good` 与 `good` 在词平均下被算成近邻
  - 正例 2：不同句式表达同一意思，sentence embedding 仍能识别

## 安装

```bash
pip install sentence-transformers numpy
```

## 运行

```bash
python embedding_demo.py
```

## 代码

见 [`embedding_demo.py`](./embedding_demo.py)。

## 自己动手改造

1. **换模型**：把 `all-MiniLM-L6-v2` 换成 `bge-small-zh-v1.5`，看中文相似度排序有没有变化。
2. **加个反例**：自己造一组 `1+1=2` 和 `1+1=3`，看相似度是不是高得离谱--直观感受"embedding 不编码真伪"。
3. **画个热力图**：把 6 个句子的相似度矩阵画出来，比看数字直观。
