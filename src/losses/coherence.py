import torch
import math


def coherence_loss(atoms, metric_field, feature_sigma=1.0, repulsion_weight=0.01):
    """
    凝聚损失。
    
    鼓励测地距离近且特征相似的原子对形成簇。
    包含吸引项（鼓励凝聚）和排斥项（防止所有原子合并成一个簇）。
    
    Args:
        atoms: 原子列表 [Atom2D, ...]
        metric_field: MetricField2D 实例
        feature_sigma: 特征高斯核的带宽
        repulsion_weight: 排斥项权重
    
    Returns:
        loss: 标量
    """
    N = len(atoms)
    if N < 2:
        return torch.tensor(0.0, device=atoms[0].position.device)
    
    # 收集原子参数
    mus = torch.stack([a.position for a in atoms])  # (N, 2)
    feats = torch.stack([a._feature for a in atoms])  # (N, D)
    eps_vals = torch.stack([a.existence_prob for a in atoms])  # (N,)
    radii = torch.stack([a.radius for a in atoms])  # (N,)
    
    # 特征亲和度矩阵 (N, N)
    feat_diff = (feats.unsqueeze(0) - feats.unsqueeze(1)) ** 2  # (N, N, D)
    K = torch.exp(-feat_diff.sum(dim=-1) / (2 * feature_sigma ** 2))  # (N, N)
    
    # 测地距离：利用度量在原子中心处的值
    g_centers = metric_field(mus)  # (N, 2, 2)
    dx = mus.unsqueeze(0) - mus.unsqueeze(1)  # (N, N, 2)
    
    # 批量计算 dx^T g dx
    # g_centers: (N, 2, 2), dx: (N, N, 2)
    # 需要对每个 i,j 计算 dx[i,j]^T g[i] dx[i,j]
    # 使用广播：g_centers.unsqueeze(0): (1, N, 2, 2), dx.unsqueeze(-1): (N, N, 2, 1)
    gx = torch.matmul(g_centers.unsqueeze(0), dx.unsqueeze(-1)).squeeze(-1)  # (N, N, 2)
    d2 = (dx * gx).sum(dim=-1)  # (N, N)
    d2 = d2.clamp(min=0.0)
    
    # 邻接矩阵：测地距离小于截断半径之和（使用 soft 近似保持可微性）
    r_sum = radii.unsqueeze(0) + radii.unsqueeze(1)  # (N, N)
    # 使用 sigmoid 软化：当 d2 < r_sum^2 时 sigmoid > 0.5
    soft_adj = torch.sigmoid((r_sum ** 2 - d2) / (r_sum ** 2 + 1e-6))  # (N, N)
    
    # 排除自环
    mask = 1.0 - torch.eye(N, device=mus.device)
    soft_adj = soft_adj * mask
    K = K * mask
    
    # 凝聚密度：C = sum(eps_i * eps_j * K_ij * adj_ij)
    eps_mat = eps_vals.unsqueeze(0) * eps_vals.unsqueeze(1)  # (N, N)
    C = (eps_mat * K * soft_adj).sum()
    
    # 损失 = -吸引 + 排斥
    # 吸引项：鼓励高凝聚
    attraction = -C
    
    # 排斥项：惩罚所有原子合并为一个簇
    # 使用 C^2 / N 作为简单的体积惩罚
    repulsion = repulsion_weight * C ** 2 / N
    
    loss = attraction + repulsion
    
    return loss
