"""
Feature Diffusion Module — 特征扩散损失。

通过度量场引导的特征平滑，让空间邻近且度量场认为"属于同一物体"
的原子特征互相扩散（趋同），同时保留不同物体间的特征差异。

数学原理:
    L_diff = Σ_{i<j} M_ij * ||f_i - f_j||² / Σ M_ij

其中:
    - M_ij = sigmoid(α * (1 - d_metric(i,j)))  → 软掩码，
      度量距离小（同一物体）≈ 1，度量距离大（不同物体）≈ 0
    - d_metric(i,j) = tr(g((p_i + p_j)/2))  → 中点度量迹
    - 自适应 sigma 由 KNN 距离决定: σ_i = scale * d_k(i)
"""

import torch
import torch.nn.functional as F


def pairwise_midpoint_metric(atom_positions, metric_field):
    """
    计算每对原子中点的度量迹，作为"是否属于同一区域"的信号。

    Args:
        atom_positions: (N, 2) 原子位置张量，归一化到 [0, 1]
        metric_field: MetricField2D 实例

    Returns:
        d_metric: (N, N) 对称张量，d_metric[i,j] = tr(g(midpoint(i,j)))
    """
    N = atom_positions.shape[0]
    
    # 计算所有对的中点: (N, N, 2)
    p_i = atom_positions.unsqueeze(1)  # (N, 1, 2)
    p_j = atom_positions.unsqueeze(0)  # (1, N, 2)
    midpoints = (p_i + p_j) / 2.0       # (N, N, 2)
    
    # 展平为 (N*N, 2) 批量查询度量场
    midpoints_flat = midpoints.reshape(-1, 2)  # (N*N, 2)
    
    with torch.no_grad():
        # 度量场的迹 tr(g) 不需要梯度（只做软掩码权重，不做优化目标）
        trace_flat = metric_field.trace(midpoints_flat)  # (N*N,)
    
    d_metric = trace_flat.reshape(N, N)  # (N, N)
    
    # 保证对称性
    d_metric = (d_metric + d_metric.T) / 2.0
    
    return d_metric


def adaptive_sigma_knn(pos, K=8, scale=0.5, eps=1e-8):
    """
    基于 KNN 距离的自适应 sigma，每个原子有自己的 sigma。
    sigma 大 → 稀疏区域的扩散允许更大特征差异。
    sigma 小 → 密集区域的扩散要求更接近的特征。

    Args:
        pos: (N, 2) 位置张量
        K: 近邻数
        scale: sigma = scale * d_k (第 K 近邻距离)
        eps: 数值稳定

    Returns:
        sigma: (N,) 每个原子的自适应 sigma
    """
    N = pos.shape[0]
    if N <= K + 1:
        return torch.full((N,), scale * 0.1, device=pos.device)
    
    # 计算所有对距离
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 2)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + eps)  # (N, N)
    
    # 取第 K+1 近（排除自身）
    # kthvalue 是 1-indexed，所以我们取第 K+1 个（第 1 个最小的是自身距离 0）
    K_eff = min(K + 1, N)
    d_k, _ = torch.kthvalue(dist, K_eff, dim=1)  # (N,)
    
    sigma = scale * d_k
    
    return sigma


def feature_diffusion_loss(atoms, metric_field, H, W,
                           sigma_scale=0.5, K=8, alpha=1.0, eps=1e-8):
    """
    特征扩散损失函数。

    通过度量场引导的软掩码，鼓励同一物体内原子特征趋同，
    同时利用自适应 sigma 防止特征完全坍缩。

    Args:
        atoms: 原子列表 (每个原子有 .position 和 ._feature)
        metric_field: MetricField2D 实例
        H, W: 图像分辨率（用于将 position [0,1] 映射到度量场坐标）
        sigma_scale: sigma = sigma_scale × 第K近邻距离
        K: 自适应 sigma 的 KNN 参数
        alpha: sigmoid 陡度参数
        eps: 数值稳定

    Returns:
        loss: 标量损失
    """
    N = len(atoms)
    if N < 2:
        return torch.tensor(0.0, device=metric_field.params.device)
    
    # 获取原子位置和特征
    positions = torch.stack([a.position for a in atoms])  # (N, 2)
    features = torch.stack([a._feature for a in atoms])   # (N, D)
    
    # 1. 计算中点度量迹（软掩码的输入）
    d_metric = pairwise_midpoint_metric(positions, metric_field)  # (N, N)
    
    # 2. 计算软掩码: M_ij = sigmoid(alpha * (1 - d_metric[i,j]))
    #    d_metric 小（同一物体）→ M_ij ≈ 1
    #    d_metric 大（不同物体）→ M_ij ≈ 0
    mask = torch.sigmoid(alpha * (1.0 - d_metric))  # (N, N)
    
    # 3. 计算特征距离平方 ||f_i - f_j||²
    f_i = features.unsqueeze(1)  # (N, 1, D)
    f_j = features.unsqueeze(0)  # (1, N, D)
    feat_dist_sq = ((f_i - f_j) ** 2).sum(dim=-1)  # (N, N)
    
    # 4. 计算自适应 sigma
    sigma = adaptive_sigma_knn(positions, K=K, scale=sigma_scale, eps=eps)  # (N,)
    
    # 5. 每个 pair 的 sigma 用几何平均: sqrt(sigma_i * sigma_j)
    sigma_pair = torch.sqrt(
        sigma.unsqueeze(1) * sigma.unsqueeze(0) + eps
    )  # (N, N)
    
    # 6. 扩散损失: L = Σ M_ij * ||f_i - f_j||² / Σ M_ij
    #    用 sigmoid 软掩码做自适应的邻居选择
    weighted_diff = mask * feat_dist_sq  # (N, N)
    
    # 排除自身 (i=j)
    diag_mask = torch.eye(N, device=positions.device, dtype=torch.bool)
    weighted_diff = weighted_diff * (~diag_mask).float()
    mask_offdiag = mask * (~diag_mask).float()
    
    sum_weighted = weighted_diff.sum()
    sum_mask = mask_offdiag.sum() + eps
    
    loss = sum_weighted / sum_mask
    
    return loss


def feature_diffusion_loss_v2(atoms, metric_field, H, W,
                               sigma_scale=0.5, K=8, alpha=1.0, eps=1e-8):
    """
    特征扩散损失 v2 — 加入自适应 sigma 归一化。

    与 v1 的区别: 在分子中除以 sigma_pair²，使得
    稀疏区域（sigma 大）允许更大特征差异，不至于过度平滑。

    L = Σ [M_ij * ||f_i - f_j||² / (σ_i * σ_j)] / Σ M_ij

    Args:
        同 feature_diffusion_loss
    """
    N = len(atoms)
    if N < 2:
        return torch.tensor(0.0, device=metric_field.params.device)
    
    positions = torch.stack([a.position for a in atoms])  # (N, 2)
    features = torch.stack([a._feature for a in atoms])   # (N, D)
    
    # 中点度量迹
    d_metric = pairwise_midpoint_metric(positions, metric_field)  # (N, N)
    
    # 软掩码
    mask = torch.sigmoid(alpha * (1.0 - d_metric))  # (N, N)
    
    # 特征距离
    f_i = features.unsqueeze(1)
    f_j = features.unsqueeze(0)
    feat_dist_sq = ((f_i - f_j) ** 2).sum(dim=-1)  # (N, N)
    
    # 自适应 sigma
    sigma = adaptive_sigma_knn(positions, K=K, scale=sigma_scale, eps=eps)  # (N,)
    sigma_pair_sq = sigma.unsqueeze(1) * sigma.unsqueeze(0) + eps  # (N, N)
    
    # 归一化损失
    weighted_diff = mask * feat_dist_sq / sigma_pair_sq
    
    # 排除自身
    diag_mask = torch.eye(N, device=positions.device, dtype=torch.bool)
    weighted_diff = weighted_diff * (~diag_mask).float()
    mask_offdiag = mask * (~diag_mask).float()
    
    sum_weighted = weighted_diff.sum()
    sum_mask = mask_offdiag.sum() + eps
    
    return sum_weighted / sum_mask
