import torch
import torch.nn as nn

from src.geometry.cholesky_param import cholesky_to_metric


class MLPMetricField2D(nn.Module):
    """
    连续度量场 — 4层MLP，分辨率无关。
    
    输入：(u, v) ∈ [0, 1]
    输出：Cholesky元素 (l11, l21, l22) → 正定度量矩阵
    """
    
    def __init__(self, hidden_dim=128, num_layers=4, init_scale=1.0, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.init_scale = init_scale
        
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.mlp = nn.Sequential(*layers)
        
        with torch.no_grad():
            self.mlp[-1].weight.data *= 0.1
            self.mlp[-1].bias.data = torch.tensor([init_scale, 0.0, init_scale])
    
    def forward(self, coords):
        """
        Args:
            coords: (N, 2) ∈ [0, 1]
        Returns:
            g: (N, 2, 2) 正定度量张量
        """
        if coords.min() < 0.0 or coords.max() > 1.0:
            coords = coords.clamp(0.0, 1.0)
        
        params = self.mlp(coords)
        l11, l21, l22 = params[:, 0], params[:, 1], params[:, 2]
        
        g11, g12, g22 = cholesky_to_metric(l11, l21, l22, self.eps)
        
        g = torch.zeros(coords.shape[0], 2, 2, device=coords.device, dtype=coords.dtype)
        g[:, 0, 0] = g11
        g[:, 0, 1] = g12
        g[:, 1, 0] = g12
        g[:, 1, 1] = g22
        
        return g
    
    def trace(self, coords):
        """
        计算度量矩阵的迹。
        
        Args:
            coords: (N, 2) 坐标
        
        Returns:
            trace: (N,)
        """
        g = self.forward(coords)
        return g[:, 0, 0] + g[:, 1, 1]