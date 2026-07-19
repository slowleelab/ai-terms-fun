"""
RAG 最小可运行 Demo
===================

从零实现一个能用的检索增强生成链路：
    文档 -> 切分 -> embedding -> FAISS 索引
    query -> embedding -> top-k 检索 -> 拼 prompt -> LLM 生成

无云依赖，CPU 可跑。生成部分留成可替换桩。

依赖：pip install sentence-transformers faiss-cpu numpy
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# ---------------------------------------------------------------------------
# 1. 知识库：假装这是公司的内部文档
# ---------------------------------------------------------------------------
CORPUS = [
    "本公司的退款政策如下：商品在签收后 7 天内可申请全额退款，需保留原包装。",
    "退款到账时间一般为 3 到 5 个工作日，原路返回至支付账户。",
    "对于定制类商品，一经生产不予退款，但可在生产前修改需求。",
    "会员等级达到金卡后，退款处理优先级提升，通常 1 个工作日内到账。",
    "如商品在运输过程中损坏，请在签收 24 小时内拍照并联系客服，可补发或退款。",
    "我们的客服工作时间为周一至周日 9:00-21:00，法定节假日除外。",
    "退货物流费用：质量问题由公司承担，非质量问题由买家承担。",
]


# ---------------------------------------------------------------------------
# 2. 简单的 chunking：按句号切，演示用。生产环境别这么粗暴。
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_len: int = 200) -> List[str]:
    """按句号切分，超长再硬切。"""
    sentences = [s.strip() for s in text.replace("；", "。").split("。") if s.strip()]
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= max_len:
            buf = f"{buf}。{s}" if buf else s
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


# ---------------------------------------------------------------------------
# 3. 索引：把每个 chunk 编码成向量，丢进 FAISS
# ---------------------------------------------------------------------------
@dataclass
class RagIndex:
    chunks: List[str]
    index: faiss.IndexFlatIP  # 内积索引（向量需先归一化 -> 等价于余弦相似度）
    embedder: SentenceTransformer


def build_index(corpus: List[str], embedder: SentenceTransformer) -> RagIndex:
    chunks: List[str] = []
    for doc in corpus:
        chunks.extend(chunk_text(doc))

    vecs = embedder.encode(chunks, normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return RagIndex(chunks=chunks, index=index, embedder=embedder)


# ---------------------------------------------------------------------------
# 4. 检索
# ---------------------------------------------------------------------------
def retrieve(rag: RagIndex, query: str, k: int = 2) -> List[str]:
    q_vec = rag.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    _, idx = rag.index.search(q_vec, k)
    return [rag.chunks[i] for i in idx[0]]


# ---------------------------------------------------------------------------
# 5. 生成：这里用桩。真实环境换成 OpenAI / Ollama / 任何 LLM 调用。
# ---------------------------------------------------------------------------
def generate(query: str, contexts: List[str]) -> str:
    context_block = "\n".join(f"- {c}" for c in contexts)
    prompt = (
        f"请根据以下资料回答问题。如果资料里没有答案，请明确说"资料不足"，不要编造。\n\n"
        f"【资料】\n{context_block}\n\n"
        f"【问题】{query}\n\n"
        f"【回答】"
    )
    # ---- 桩：打印 prompt 让你看清楚 RAG 到底喂了什么给 LLM ----
    print("===== 实际喂给 LLM 的 prompt =====")
    print(prompt)
    print("==================================")
    return "[这里替换成你的 LLM 调用。上面就是 RAG 的全部秘密：检索完，拼进 prompt。]"


# ---------------------------------------------------------------------------
# 6. 跑一遍
# ---------------------------------------------------------------------------
def main():
    print("加载 embedding 模型中（首次会下载）...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("构建索引中...")
    rag = build_index(CORPUS, embedder)
    print(f"索引完成，共 {len(rag.chunks)} 个 chunk\n")

    queries = [
        "我买的东西坏了，能退款吗？",
        "退款几天能到账？",
        "定制商品能退吗？",
    ]
    for q in queries:
        print(f"\n>>> 用户提问：{q}")
        contexts = retrieve(rag, q, k=2)
        print(f"    检索到的 top-2 chunk：")
        for i, c in enumerate(contexts, 1):
            print(f"      [{i}] {c}")
        answer = generate(q, contexts)
        print(f"    回答：{answer}")


if __name__ == "__main__":
    main()
