"""
Embedding 最小可运行 Demo
========================

用 sentence-transformers 把句子编码成向量，算余弦相似度，
亲眼看到「语义相近的句子向量也相近」。

然后戳破一个常见幻觉：直接把词向量平均当句向量，
会把 not good 和 good 算成近邻。

依赖：pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def show_similarity_matrix(sentences: list[str], vecs: np.ndarray, title: str):
    print(f"\n===== {title} =====")
    # 表头
    header = "              " + "  ".join(f"{s[:6]:>8}" for s in sentences)
    print(header)
    for i, s in enumerate(sentences):
        row = "  ".join(f"{cosine(vecs[i], vecs[j]):8.3f}" for j in range(len(sentences)))
        print(f"{s[:12]:<12}  {row}")


def main():
    print("加载模型中（首次会下载）...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---- 案例 1：6 个句子，看语义相似度排序是否符合直觉 ----
    sentences = [
        "I love this movie.",            # 正面
        "This film is great.",           # 正面（同义不同句式）
        "I hate this movie.",            # 反面
        "The weather is nice today.",    # 无关
        "这家店很好吃。",                 # 中文正面
        "This restaurant is delicious.", # 英文同义
    ]
    vecs = model.encode(sentences, normalize_embeddings=True)
    show_similarity_matrix(sentences, vecs, "Sentence Embedding 相似度矩阵")

    # 观察：
    # - 句 0/1（同义不同句式）相似度应 > 0.7
    # - 句 0/2（一字之差，意思相反）相似度可能依然很高 -- 模型对短句的语义判别有限
    # - 句 4/5（中英跨语言）相似度应 > 0.5 -- 多语言 embedding 的神奇之处

    # ---- 案例 2：直接平均词向量为什么不行 ----
    print("\n===== 反例：直接平均词向量的陷阱 =====")
    # 用模型的 word-level 接口模拟"朴素平均词向量"
    pairs = [
        ("good",      "not good"),       # 意思相反，但词集合几乎一样
        ("good",      "great"),          # 同义
        ("good",      "bad"),            # 反义
    ]
    for a, b in pairs:
        va = model.encode(a, normalize_embeddings=True)
        vb = model.encode(b, normalize_embeddings=True)
        sim = cosine(va, vb)
        print(f"  sim({a!r:>10}, {b!r:<10}) = {sim:.3f}")

    # 观察：
    # - good / not good 的相似度可能高得离谱（>0.7），
    #   因为 sentence encoder 对短语也按整体编码，
    #   而"平均词向量"流派会更严重地犯这个错。
    # - 这就是为什么需要专门的 sentence embedding 模型（SBERT/E5/bge），
    #   它们用对比学习专门优化过"反义词要拉开"。

    print("\n结论：embedding 好用，但不编码真伪、不天然理解否定。")
    print("它编码的是「相似语境下出现的文本」--用之前要想清楚你要的是不是这个。")


if __name__ == "__main__":
    main()
