# RAG 最小可运行 Demo

> 60 行 Python，从零跑通一个能用的 RAG。不依赖任何云服务，本地 CPU 就能跑。

## 它演示了什么

- 文档切分（chunking）
- 用 sentence-transformers 做 embedding
- 用 FAISS 做向量检索
- 把检索结果拼进 prompt 交给 LLM 生成

生成模型这里用一个**可替换的桩（stub）**，方便你接 OpenAI / 本地 Ollama / 任何 LLM。重点在检索链路，那是 RAG 的灵魂。

## 安装

```bash
pip install sentence-transformers faiss-cpu numpy
```

> 首次运行会自动下载一个 ~80MB 的 embedding 模型（`all-MiniLM-L6-v2`）。

## 运行

```bash
python rag_demo.py
```

## 代码

见 [`rag_demo.py`](./rag_demo.py)。

## 自己动手改造

跑通之后，建议你按 RAG 词条 L4 里提到的三个翻车点，依次制造问题再修复：

1. **改 chunk_size**：把 `chunk_size=200` 调成 `50`，看检索质量怎么变差。
2. **换 query**：把 "退款政策" 改成 "我能退货吗"，观察陈述句文档和疑问句 query 的相似度差异。
3. **改 top_k**：从 `k=2` 调到 `k=5`，看噪声是否变多。

制造问题比解决问题更能学到东西。
