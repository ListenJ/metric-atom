import torch


def contrastive_coherence_loss(atoms, metric_field, tau=0.5, pos_thresh=0.3, neg_thresh=2.0,
                                var_weight=0.1):
    """
    有界对比凝聚损失 (InfoNCE 风格)。

    对每个锚点原子 i:
      - 正样本: 归一化测地距离 < pos_thresh 的原子 j
      - 负样本: 归一化测地距离 > neg_thresh 的原子 k
    损失: -log( exp(sim_ij/tau) / (exp(sim_ij/tau) + sum_neg exp(sim_ik/tau)) )

    同时保留弱全局方差正则化防止特征完全坍缩。

    Args:
        atoms: 原子列表
        metric_field: MetricField2D 实例
        tau: InfoNCE 温度系数
        pos_thresh: 正样本归一化距离阈值
        neg_thresh: 负样本归一化距离阈值
        var_weight: 方差正则化权重

    Returns:
        loss: 标量
    """
    N = len(atoms)
    if N < 4:
        return torch.tensor(0.0, device=atoms[0].position.device)

    mus = torch.stack([a.position for a in atoms])       # (N, 2)
    feats = torch.stack([a._feature for a in atoms])     # (N, D)
    radii = torch.stack([a.radius for a in atoms])       # (N,)

    # ── 归一化特征（余弦相似度用） ──
    feats_norm = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)

    # ── 成对归一化测地距离 ──
    g_centers = metric_field(mus)                       # (N, 2, 2)
    dx = mus.unsqueeze(0) - mus.unsqueeze(1)            # (N, N, 2)
    d2 = torch.einsum('ijm,imn,ijn->ij', dx, g_centers, dx)
    d2 = d2.clamp(min=0.0)                              # (N, N)

    # 归一化距离（除以半径和）
    r_sum = radii.unsqueeze(0) + radii.unsqueeze(1)     # (N, N)
    norm_d = torch.sqrt(d2) / (r_sum + 1e-8)            # (N, N)

    mask = 1.0 - torch.eye(N, device=mus.device)        # 去掉自身

    # ── 正负样本掩码 ──
    pos_mask = (norm_d < pos_thresh).float() * mask
    neg_mask = (norm_d > neg_thresh).float() * mask

    # 确保每个锚点至少有一个正样本和一个负样本
    has_pos = pos_mask.sum(dim=-1) > 0
    has_neg = neg_mask.sum(dim=-1) > 0
    valid = has_pos & has_neg
    if valid.sum() == 0:
        # 保底：若无有效锚点，使用最接近/最远的作为正负样本
        valid = torch.ones(N, dtype=torch.bool, device=mus.device)
        for i in range(N):
            sorted_idx = torch.argsort(norm_d[i])
            pos_mask[i, sorted_idx[1:max(3, N//10)]] = 1.0  # 最近几个为正
            neg_mask[i, sorted_idx[-max(3, N//4):]] = 1.0    # 最远几个为负

    # ── 余弦相似度 ──
    sim = feats_norm @ feats_norm.T                        # (N, N), [-1, 1]

    # ── InfoNCE 损失 ──
    loss_total = 0.0
    count = 0
    for i in range(N):
        if not valid[i]:
            continue
        pos_idx = torch.where(pos_mask[i] > 0)[0]
        neg_idx = torch.where(neg_mask[i] > 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        pos_sim = sim[i, pos_idx]                          # (P,)
        neg_sim = sim[i, neg_idx]                          # (Q,)

        # 对每个正样本计算 InfoNCE
        for ps in pos_sim:
            numerator = torch.exp(ps / tau)
            denominator = numerator + torch.exp(neg_sim / tau).sum()
            loss_total += -torch.log(numerator / (denominator + 1e-8))
            count += 1

    loss_nce = loss_total / max(count, 1)

    # ── 弱方差正则化（防止完全坍缩） ──
    centered = feats - feats.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / N
    feature_variance = torch.trace(cov)
    # 鼓励方差不低于 0.05
    var_loss = torch.relu(0.05 - feature_variance / feats.shape[1])

    loss = loss_nce + var_weight * var_loss

    return loss
