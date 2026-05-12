import torch


def coherence_loss(atoms, metric_field, feature_sigma=1.0, repulsion_weight=1.0):
    """
    凝聚损失。
    
    吸引项（C）：鼓励测地距离近且特征相似的原子对形成簇。
    排斥项（R）：最大化特征方差，防止所有原子坍缩到同一特征。
    
    Args:
        atoms: 原子列表 [Atom2D, ...]
        metric_field: MetricField2D 实例
        feature_sigma: 特征高斯核的带宽
        repulsion_weight: 排斥项权重（推荐 1.0，与吸引项尺度匹配）
    
    Returns:
        loss: 标量
    """
    N = len(atoms)
    if N < 2:
        return torch.tensor(0.0, device=atoms[0].position.device)
    
    mus = torch.stack([a.position for a in atoms])
    feats = torch.stack([a._feature for a in atoms])
    eps_vals = torch.stack([a.existence_prob for a in atoms])
    radii = torch.stack([a.radius for a in atoms])
    
    # ── 吸引项：特征亲和度 × 测地邻接 ──
    feat_diff = (feats.unsqueeze(0) - feats.unsqueeze(1)) ** 2
    feat_affinity = torch.exp(-feat_diff.sum(dim=-1) / (2 * feature_sigma ** 2))
    
    g_centers = metric_field(mus)
    dx = mus.unsqueeze(0) - mus.unsqueeze(1)
    # d2[i,j] = dx[i,j]^T @ g[i] @ dx[i,j]
    # dx: (N, N, 2), g_centers: (N, 2, 2)
    d2 = torch.einsum('ijm,imn,ijn->ij', dx, g_centers, dx)  # (N, N)
    d2 = d2.clamp(min=0.0)
    
    r_sum = radii.unsqueeze(0) + radii.unsqueeze(1)
    soft_adj = torch.sigmoid((r_sum ** 2 - d2) / (r_sum ** 2 + 1e-6))
    
    mask = 1.0 - torch.eye(N, device=mus.device)
    soft_adj = soft_adj * mask
    feat_affinity = feat_affinity * mask
    
    eps_mat = eps_vals.unsqueeze(0) * eps_vals.unsqueeze(1)
    C_raw = (eps_mat * feat_affinity * soft_adj).sum()
    C = C_raw / (N * (N - 1) / 2 + 1.0)
    
    # ── 排斥项：最大化特征方差 ──
    centered = feats - feats.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / N
    feature_variance = torch.trace(cov)
    R = -feature_variance
    
    loss = -C + repulsion_weight * R / feats.shape[1]
    
    return loss
