"""
位置编码 最小可运行 Demo
========================

实现正弦/余弦位置编码和 RoPE，可视化编码向量，
直观看到「相邻位置编码相似、相远位置差异大」。

依赖：pip install torch matplotlib numpy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 1. 正弦/余弦位置编码（原论文 Vaswani et al., 2017）
# ---------------------------------------------------------------------------
def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """生成 [max_len, d_model] 的正弦位置编码。"""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
    # 频率：10000^(2i/d) 的倒数
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维用 sin
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维用 cos
    return pe


# ---------------------------------------------------------------------------
# 2. RoPE 旋转位置编码（Su et al., 2021）
# ---------------------------------------------------------------------------
def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    对 [n, d] 的向量应用 RoPE 旋转。
    把 d 维看成 d/2 个二维平面，每个平面按位置旋转。
    """
    n, d = x.shape
    assert d % 2 == 0
    half = d // 2
    # 每个平面的旋转频率
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float) / half))  # [half]
    positions = torch.arange(n, dtype=torch.float)  # [n]
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [n, half]
    cos = angles.cos()  # [n, half]
    sin = angles.sin()  # [n, half]
    # 把 x 拆成 (x_even, x_odd)，旋转
    x1 = x[:, 0::2]  # [n, half]
    x2 = x[:, 1::2]  # [n, half]
    rotated_even = x1 * cos - x2 * sin
    rotated_odd = x1 * sin + x2 * cos
    out = torch.zeros_like(x)
    out[:, 0::2] = rotated_even
    out[:, 1::2] = rotated_odd
    return out


# ---------------------------------------------------------------------------
# 3. 可视化
# ---------------------------------------------------------------------------
def cosine_sim_matrix(a: torch.Tensor) -> np.ndarray:
    a = a / (a.norm(dim=1, keepdim=True) + 1e-9)
    return (a @ a.T).numpy()


def main():
    max_len, d_model = 128, 64

    pe = sinusoidal_positional_encoding(max_len, d_model)
    print(f"正弦位置编码形状: {pe.shape}  (应为 [{max_len}, {d_model}])")
    print(f"第 0 个位置的编码前 8 维: {pe[0, :8].tolist()}")
    print(f"第 1 个位置的编码前 8 维: {pe[1, :8].tolist()}")
    print(f"-> 相邻位置的编码不同，但相似度较高")

    # 验证 RoPE 的核心性质：点积只依赖相对位置
    torch.manual_seed(0)
    q = torch.randn(1, d_model)
    k = torch.randn(1, d_model)
    # 把 q 放在位置 0，k 放在位置 5 vs q 在 10、k 在 15（相对距离都是 5）
    q_seq = q.expand(16, d_model)
    k_seq = k.expand(16, d_model)
    q_rot = apply_rope(q_seq)
    k_rot = apply_rope(k_seq)
    # 比较 (q@pos=0, k@pos=5) 和 (q@pos=10, k@pos=15) 的点积
    dot_05 = (q_rot[0] * k_rot[5]).sum().item()
    dot_10_15 = (q_rot[10] * k_rot[15]).sum().item()
    print(f"\nRoPE 验证：相对距离都是 5")
    print(f"  (q@0) · (k@5)   = {dot_05:.4f}")
    print(f"  (q@10) · (k@15) = {dot_10_15:.4f}")
    print(f"  两者应几乎相等（差异来自数值精度），证明 RoPE 点积只依赖相对位置")

    # 画图 1：正弦位置编码热力图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    im = ax.imshow(pe.numpy()[:64, :32], cmap="RdBu", aspect="auto",
                   vmin=-1, vmax=1)
    ax.set_xlabel("维度 index")
    ax.set_ylabel("位置 index")
    ax.set_title("正弦位置编码（前 64 位置 × 32 维度）")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 画图 2：位置编码间的余弦相似度
    ax = axes[1]
    sim = cosine_sim_matrix(pe[:64])
    im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xlabel("位置 j")
    ax.set_ylabel("位置 i")
    ax.set_title("位置编码间的余弦相似度\n对角线附近最亮（相邻位置相似）")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("positional_encoding.png", dpi=120)
    print("\n图已保存到 positional_encoding.png")


if __name__ == "__main__":
    main()
