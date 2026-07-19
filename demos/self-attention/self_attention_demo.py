"""
自注意力 最小可运行 Demo
========================

20 行 PyTorch 实现一个能跑的自注意力，并打印注意力矩阵。
直观看到「代词 it 关注 animal」这件事真的发生。

注意：本 demo 用随机初始化的权重，所以注意力模式是随机的，
真实模型经过训练后才会出现有意义的模式。
要观察真实模式，请用预训练模型（如 bert-base）的某层注意力头。

依赖：pip install torch matplotlib
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

# 用中文字体，避免 matplotlib 中文乱码（macOS）
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 1. 自注意力的完整实现（就这么几行）
# ---------------------------------------------------------------------------
class SelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.W_Q = torch.nn.Linear(d_model, d_k, bias=False)
        self.W_K = torch.nn.Linear(d_model, d_k, bias=False)
        self.W_V = torch.nn.Linear(d_model, d_k, bias=False)

    def forward(self, X):
        # X: [n, d_model]
        Q = self.W_Q(X)                       # [n, d_k]
        K = self.W_K(X)                       # [n, d_k]
        V = self.W_V(X)                       # [n, d_k]
        scores = Q @ K.T / (self.d_k ** 0.5)  # [n, n]  缩放点积
        weights = F.softmax(scores, dim=-1)   # [n, n]  归一化
        output = weights @ V                  # [n, d_k] 加权求和
        return output, weights


# ---------------------------------------------------------------------------
# 2. 用一句经典例子跑一遍
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(42)

    # 经典例句：it 指代的是 animal 而不是 street
    tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "tired"]
    n = len(tokens)
    d_model, d_k = 16, 16

    # 假装每个 token 已经被 embedding 成向量
    X = torch.randn(n, d_model)

    attn = SelfAttention(d_model, d_k)
    with torch.no_grad():
        output, weights = attn(X)

    print(f"输入序列长度 n = {n}, d_model = {d_model}, d_k = {d_k}")
    print(f"注意力矩阵形状: {weights.shape}  (应为 [{n}, {n}])")
    print(f"输出形状: {output.shape}  (应为 [{n}, {d_k}])")
    print(f"每行权重和（应都为 1.0）: {weights.sum(dim=-1).tolist()[:3]}...")

    # 找出 "it"（索引 7）最关注的 top-3 token
    it_idx = 7
    topk = torch.topk(weights[it_idx], k=3)
    print(f"\n随机权重下，'it' 最关注的 top-3 token（注意是随机的，非真实模式）：")
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  {tokens[idx]:>10}  权重 = {score:.4f}")

    # 画热力图
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(weights.numpy(), cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("被关注的 token (Key)")
    ax.set_ylabel("发起关注的 token (Query)")
    ax.set_title("自注意力权重矩阵（随机初始化）\n真实模型经训练后会出现有意义的模式")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("attention_weights.png", dpi=120)
    print("\n热力图已保存到 attention_weights.png")
    print("提示：在真实模型中，'it' 那一行往往会在 'animal' 列出现高权重。")


if __name__ == "__main__":
    main()
