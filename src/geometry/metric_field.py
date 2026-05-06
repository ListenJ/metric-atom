import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry.cholesky_param import cholesky_to_metric


class MetricField2D(nn.Module):
    """
    2D 黎曼度量场 g(x)。
    
    参数化方式：Cholesky 分解 L(x)，保证正定性。
    每个像素存储下三角矩阵的三个元素 (l11, l21, l22)，
    通过 g = L L^T + eps*I 构建正定度量张量。
    
    支持双线性插值采样，输入坐标归一化到 [0, 1]。
    """
    
    def __init__(self, H, W, init_scale=1.0, eps=1e-4):
        super().__init__()
        self.H = H
        self.W = W
        self.eps = eps
        
        # 将三个参数存储为 (1, 3, H, W) 的张量，便于使用 grid_sample
        # 通道0: l11, 通道1: l21, 通道2: l22
        self.params = nn.Parameter(torch.zeros(1, 3, H, W))
        
        # 初始化：对角元素为 init_scale，非对角为 0
        with torch.no_grad():
            self.params[0, 0].fill_(init_scale)  # l11
            self.params[0, 2].fill_(init_scale)  # l22
            self.params[0, 1].zero_()             # l21
    
    def forward(self, coords):
        """
        对给定坐标采样度量张量。
        
        Args:
            coords: (N, 2) 张量，坐标归一化到 [0, 1]
        
        Returns:
            g: (N, 2, 2) 正定度量张量
        """
        if coords.dim() != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be (N, 2), got {coords.shape}")
        
        if coords.min() < 0.0 or coords.max() > 1.0:
            raise ValueError("coords must be normalized to [0, 1]")
        
        N = coords.shape[0]
        
        # grid_sample 需要 (N, H_out, W_out, 2) 格式的 grid
        # coords 是 [0,1]，需要映射到 [-1,1]
        grid = coords.unsqueeze(1).unsqueeze(1) * 2.0 - 1.0  # (N, 1, 1, 2)
        
        # 使用双线性插值采样参数
        # 需要 batch size 匹配，将 params 扩展为 (N, 3, H, W)
        params_expanded = self.params.expand(N, -1, -1, -1)
        # sampled: (N, 3, 1, 1)
        sampled = F.grid_sample(
            params_expanded, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # 重塑为 (N, 3)
        sampled = sampled.squeeze(-1).squeeze(-1)  # (N, 3)
        l11, l21, l22 = sampled[:, 0], sampled[:, 1], sampled[:, 2]
        
        # 构建正定度量矩阵
        g11, g12, g22 = cholesky_to_metric(l11, l21, l22, self.eps)
        
        g = torch.zeros(N, 2, 2, device=coords.device, dtype=coords.dtype)
        g[:, 0, 0] = g11
        g[:, 0, 1] = g12
        g[:, 1, 0] = g12
        g[:, 1, 1] = g22
        
        return g
    
    def get_params_at_pixels(self):
        """
        获取每个像素上的原始Cholesky参数。
        
        Returns:
            l11, l21, l22: 每个都是 (H, W) 张量
        """
        return self.params[0, 0], self.params[0, 1], self.params[0, 2]
    
    def trace(self, coords=None):
        """
        计算度量矩阵的迹。如果提供坐标则采样，否则返回所有像素的迹。
        
        Args:
            coords: 可选，(N, 2) 坐标
        
        Returns:
            trace: (N,) 或 (H, W)
        """
        if coords is not None:
            g = self.forward(coords)
            return g[:, 0, 0] + g[:, 1, 1]
        else:
            l11, l21, l22 = self.get_params_at_pixels()
            g11, _, g22 = cholesky_to_metric(l11, l21, l22, self.eps)
            return g11 + g22
