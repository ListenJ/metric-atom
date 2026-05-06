import torch


def metric_smoothness_loss(g_field):
    """
    度量平滑损失。
    
    鼓励度量场空间连续，惩罚相邻像素间的剧烈变化。
    使用有限差分计算梯度。
    
    Args:
        g_field: MetricField2D 实例
    
    Returns:
        loss: 标量
    """
    # 获取参数 (H, W)
    l11, l21, l22 = g_field.get_params_at_pixels()
    
    # 计算每个参数的空间梯度
    def spatial_gradient(param):
        """计算参数在 x 和 y 方向的梯度"""
        # x 方向梯度 (水平)
        gx = param[:, 1:] - param[:, :-1]  # (H, W-1)
        # y 方向梯度 (垂直)
        gy = param[1:, :] - param[:-1, :]  # (H-1, W)
        return gx, gy
    
    loss = 0.0
    for param in [l11, l21, l22]:
        gx, gy = spatial_gradient(param)
        loss += (gx ** 2).mean()
        loss += (gy ** 2).mean()
    
    return loss
