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
