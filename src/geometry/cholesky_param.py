import torch


def cholesky_to_metric(l11, l21, l22, eps=1e-4):
    """
    从Cholesky参数构建2x2正定度量矩阵的元素。
    
    g = L L^T + eps * I
    L = [[l11, 0], [l21, l22]]
    
    Args:
        l11, l21, l22: 标量或张量
        eps: 极小偏置保证严格正定
    
    Returns:
        g11, g12, g22: 度量矩阵的独立元素
    """
    g11 = l11**2 + eps
    g12 = l11 * l21
    g22 = l21**2 + l22**2 + eps
    return g11, g12, g22


def cholesky_to_metric_3d(l11, l21, l22, l31, l32, l33, eps=1e-4):
    """
    从Cholesky参数构建3x3正定度量矩阵的元素。
    
    g = L L^T + eps * I
    L = [[l11,  0,   0 ],
         [l21, l22,  0 ],
         [l31, l32, l33]]
    
    Args:
        l11, l21, l22, l31, l32, l33: 标量或张量
        eps: 极小偏置保证严格正定
    
    Returns:
        g11, g12, g13, g22, g23, g33: 度量矩阵的独立元素（上三角）
    """
    g11 = l11**2 + eps
    g12 = l11 * l21
    g13 = l11 * l31
    g22 = l21**2 + l22**2 + eps
    g23 = l21 * l31 + l22 * l32
    g33 = l31**2 + l32**2 + l33**2 + eps
    return g11, g12, g13, g22, g23, g33
