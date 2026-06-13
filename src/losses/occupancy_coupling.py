import torch


def occupancy_coupling_loss(g_field, occupancy_mask, g_occ_target=1.0, g_bg_target=10.0):
    """
    占位耦合损失。
    
    鼓励物体内部（占位区）的度量值接近小值（高密度），
    背景区域（空区域）的度量值接近大值（低密度）。
    
    使用度量矩阵的迹作为简单指标。
    
    Args:
        g_field: MetricField2D 实例
        occupancy_mask: (H, W) 占位掩码，1表示物体内部，0表示背景
        g_occ_target: 物体内部的目标迹值（小）
        g_bg_target: 背景区域的目标迹值（大）
    
    Returns:
        loss: 标量
    """
    # 计算每个像素的度量迹
    trace = g_field.trace()  # (H, W)
    
    # 占位区损失
    occ_loss = ((trace - g_occ_target) ** 2 * occupancy_mask).mean()
    
    # 背景区损失
    bg_loss = ((trace - g_bg_target) ** 2 * (1.0 - occupancy_mask)).mean()
    
    return occ_loss + bg_loss


def trace_contrast_loss(g_field, occupancy_mask, target_in=1.0, target_bg=5.0):
    """
    显式推拉度量场迹的对比损失 (hinge-based)。

    与 occupancy_coupling_loss 的区别:
      - occupancy_coupling_loss 用 MSE 推向绝对目标 (1.0 vs 10.0),
        当 trace 距目标远时梯度大, 但接近目标时梯度消失。
      - trace_contrast_loss 用 hinge 确保最小分离度,
        物体内 trace 必须 < target_in + margin,
        背景 trace 必须 > target_bg - margin。
        即使绝对值不理想, 只要满足 margin 就不产生梯度,
        避免与平滑正则的过度对抗。

    Args:
        g_field: MetricField2D 实例
        occupancy_mask: (H, W) 占位掩码
        target_in: 物体内 trace 目标上界
        target_bg: 背景 trace 目标下界

    Returns:
        loss: 标量
    """
    trace = g_field.trace()  # (H, W)
    occ = occupancy_mask
    bg = 1.0 - occupancy_mask

    n_obj = occ.sum().clamp(min=1)
    n_bg = bg.sum().clamp(min=1)

    # 物体内: 鼓励 trace < target_in (惩罚超出的部分)
    loss_in = ((trace - target_in).clamp(min=0) ** 2 * occ).sum() / n_obj

    # 背景: 鼓励 trace > target_bg (惩罚不足的部分)
    loss_bg = ((target_bg - trace).clamp(min=0) ** 2 * bg).sum() / n_bg

    return loss_in + loss_bg
