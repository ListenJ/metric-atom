import torch


def occupancy_coupling_loss(g_field, occupancy_mask, g_occ_target=1.0, g_bg_target=10.0):
    """
    占位耦合损失（网格版本）。
    
    鼓励物体内部（占位区）的度量值接近小值（高密度），
    背景区域（空区域）的度量值接近大值（低密度）。
    
    Args:
        g_field: MetricField2D 实例
        occupancy_mask: (H, W) 占位掩码，1表示物体内部，0表示背景
        g_occ_target: 物体内部的目标迹值（小）
        g_bg_target: 背景区域的目标迹值（大）
    
    Returns:
        loss: 标量
    """
    trace = g_field.trace()
    
    occ_loss = ((trace - g_occ_target) ** 2 * occupancy_mask).mean()
    bg_loss = ((trace - g_bg_target) ** 2 * (1.0 - occupancy_mask)).mean()
    
    return occ_loss + bg_loss


def occupancy_coupling_loss_mlp(g_field, occupancy_mask, H, W,
                                  g_occ_target=1.0, g_bg_target=10.0,
                                  num_samples=500):
    """
    占位耦合损失（MLP版本）— 基于采样点计算。
    
    Args:
        g_field: MLPMetricField2D 实例
        occupancy_mask: (H, W) 占位掩码
        H, W: 图像尺寸
        g_occ_target: 物体内部目标迹值
        g_bg_target: 背景区域目标迹值
        num_samples: 每个区域的采样点数
    
    Returns:
        loss: 标量
    """
    device = occupancy_mask.device
    
    occ_coords = torch.nonzero(occupancy_mask > 0.5).float()
    bg_coords = torch.nonzero(occupancy_mask <= 0.5).float()
    
    if len(occ_coords) == 0 or len(bg_coords) == 0:
        return torch.tensor(0.0, device=device, requires_grad=False)
    
    occ_idx = torch.randperm(len(occ_coords), device=device)[:min(num_samples, len(occ_coords))]
    bg_idx = torch.randperm(len(bg_coords), device=device)[:min(num_samples, len(bg_coords))]
    
    occ_pixels = occ_coords[occ_idx]
    bg_pixels = bg_coords[bg_idx]
    
    occ_coords_norm = occ_pixels.flip(-1) / torch.tensor([W, H], device=device, dtype=torch.float32)
    bg_coords_norm = bg_pixels.flip(-1) / torch.tensor([W, H], device=device, dtype=torch.float32)
    
    trace_occ = g_field.trace(occ_coords_norm)
    trace_bg = g_field.trace(bg_coords_norm)
    
    occ_loss = ((trace_occ - g_occ_target) ** 2).mean()
    bg_loss = ((trace_bg - g_bg_target) ** 2).mean()
    
    return occ_loss + bg_loss
