"""
特征扩散模块 — Feature Diffusion Module v0.4

基于度量场定义的测地亲和度矩阵，对原子特征进行确定性平滑。
使特征平滑成为度量场学习的自然涌现。

数学原理:
    中点度量: d²_ij = (μ_i - μ_j)ᵀ g((μ_i + μ_j)/2) (μ_i - μ_j)
    亲和矩阵: A_ij = exp(-d²_ij / (2σ_iσ_j)) · sigmoid((τ_max - d_ij) / s)
    自适应 σ: σ_i = d_g(μ_i, μ_{neighbor_K}), K=5~10
    特征扩散: F^{(t+1)} = (1-α)F^{(t)} + α·S·F^{(t)},  S = D⁻¹A

限制:
    - 测地高斯核不保证 PSD（Schoenberg 定理仅适用于欧几里得嵌入）。
      行随机 S = D⁻¹A 用于马尔可夫扩散时，非 PSD 不影响收敛性
      （Perron-Frobenius 定理保证 S 的谱半径 ≤ 1）。
    - 如需谱聚类，在 eigenvalue decomposition 前对 A 进行特征值截断。
    - A 通过 A = (A + Aᵀ)/2 强制对称。
"""

import torch


def compute_geodesic_affinity(mus, metric_field, K=5, tau_max_factor=3.0, s_factor=0.1):
    """
    对称测地亲和矩阵 + sigmoid 软掩码 + 自适应 sigma。

    Args:
        mus: (N, D) 原子位置张量
        metric_field: MetricField2D 实例，可微的度量场
        K: 自适应 sigma 的近邻数，默认 5
        tau_max_factor: τ_max = tau_max_factor * mean(sigma)，默认 3.0
        s_factor: s = s_factor * τ_max，默认 0.1

    Returns:
        A: (N, N) 对称亲和矩阵
    """
    N = mus.shape[0]
    if N < 2:
        return torch.zeros(N, N, device=mus.device)

    # ── 中点度量 ──
    mids = (mus.unsqueeze(0) + mus.unsqueeze(1)) / 2  # (N, N, D)
    dx = mus.unsqueeze(0) - mus.unsqueeze(1)           # (N, N, D)

    # 批量评估度量场: (N*N, D) → (N, N, D, D)
    g_mid = metric_field(mids.reshape(-1, mus.shape[-1])).reshape(N, N, mus.shape[-1], mus.shape[-1])

    # 测地距离平方: d²_ij = dxᵀ g_mid dx
    d2 = torch.einsum('ijm,ijmn,ijn->ij', dx, g_mid, dx).clamp(min=0)  # (N, N)
    d = torch.sqrt(d2 + 1e-8)                                             # (N, N)

    # ── 自适应 sigma：每个原子取第 K 近邻的测地距离 ──
    # topk 返回 (values, indices)，第 0 近是自身 (d=0)
    d_sorted = d.topk(k=K + 1, dim=1, largest=False)[0]  # (N, K+1)
    sigma_i = d_sorted[:, K]                              # (N,) 第 K 近邻距离

    sigma_prod = sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)  # (N, N)

    # ── 指数衰减亲和度 ──
    A = torch.exp(-d2 / (2 * sigma_prod + 1e-8))

    # ── sigmoid 软掩码（替代硬截断） ──
    tau_max = tau_max_factor * sigma_i.mean()
    s = s_factor * tau_max
    soft_mask = torch.sigmoid((tau_max - d) / s)

    A = A * soft_mask

    # ── 强制对称（消除数值误差产生的不对称）──
    A = 0.5 * (A + A.t())

    # 排除自身
    A.fill_diagonal_(0.0)

    return A


def feature_diffusion(F, A, alpha=0.5, T=2):
    """
    可微特征扩散。

    F^{(t+1)} = (1-α)F^{(t)} + α·S·F^{(t)}
    其中 S = D⁻¹A 为行归一化转移矩阵。

    Args:
        F: (N, d) 原始特征
        A: (N, N) 亲和矩阵
        alpha: 扩散步长，默认 0.5
        T: 迭代次数，默认 2

    Returns:
        F_diffused: (N, d) 扩散后特征
    """
    # 行归一化: S = D⁻¹A
    D = A.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
    S = A / D                                        # (N, N)

    F_new = F
    for _ in range(T):
        F_new = (1 - alpha) * F_new + alpha * (S @ F_new)

    return F_new
