"""
多头注意力 最小可运行 Demo
==========================

30 行 PyTorch 实现多头注意力（高效写法：一次大投影 + reshape），
并可视化多个头的注意力矩阵，直观看到「不同头关注不同的关系」。

注意：随机初始化下注意力模式是随机的，真实模型训练后才会出现
有意义的分化（如位置头、句法头等）。

依赖：pip install torch matplotlib
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, h: int):
        super().__init__()
        assert d_model % h == 0, "d_model 必须能被 h 整除"
        self.h = h
        self.d_k = d_model // h
        # 一次性大投影，比循环 h 次高效得多
        self.W_Q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_K = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_V = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_O = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        # X: [n, d_model]
        n, d = X.shape
        Q = self.W_Q(X).view(n, self.h, self.d_k).transpose(0, 1)  # [h, n, d_k]
        K = self.W_K(X).view(n, self.h, self.d_k).transpose(0, 1)  # [h, n, d_k]
        V = self.W_V(X).view(n, self.h, self.d_k).transpose(0, 1)  # [h, n, d_k]

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)  # [h, n, n]
        weights = F.softmax(scores, dim=-1)                   # [h, n, n]
        out = weights @ V                                     # [h, n, d_k]

        out = out.transpose(0, 1).reshape(n, d)               # 拼回 [n, d]
        return self.W_O(out), weights


def main():
    torch.manual_seed(42)

    tokens = ["The", "animal", "didn't", "cross", "the", "street",
              "because", "it", "was", "tired"]
    n, d_model, h = len(tokens), 32, 8
    X = torch.randn(n, d_model)

    mha = MultiHeadAttention(d_model, h)
    with torch.no_grad():
        out, weights = mha(X)

    print(f"输入: n={n}, d_model={d_model}, h={h}, d_k={d_model//h}")
    print(f"输出形状: {out.shape}  (应为 [{n}, {d_model}])")
    print(f"注意力权重形状: {weights.shape}  (应为 [{h}, {n}, {n}])")
    print(f"\n关键点：总计算量 = h × (n² · d_k) = {h} × ({n}² × {d_model//h}) = {h*n*n*d_model//h}")
    print(f"等价于单头 n² · d = {n}² × {d_model} = {n*n*d_model}")
    print(f"-> 多头和单头的 FLOPs 几乎相同，但视角多了 {h} 个")

    # 可视化前 4 个头的注意力矩阵
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for i, ax in enumerate(axes.flat):
        if i < h:
            im = ax.imshow(weights[i].numpy(), cmap="viridis", vmin=0, vmax=weights[i].max().item())
            ax.set_title(f"head {i}", fontsize=11)
            ax.set_xticks(range(n))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels(tokens, fontsize=8)
        else:
            ax.axis("off")
    fig.suptitle(f"多头注意力的 {h} 个头（随机初始化）\n真实模型训练后，不同头会分化出不同功能",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("multi_head_weights.png", dpi=120)
    print(f"\n热力图已保存到 multi_head_weights.png")
    print("提示：观察 8 个头的差异--即使是随机初始化，模式也已经不同。")


if __name__ == "__main__":
    main()
