import torch
import torch.nn as nn
import math

from src.atoms.base_atom import BaseAtom


class Atom2D(BaseAtom):
    """
    2D 感知原子。
    
    参数：
        mu: 中心位置 (2,)
        log_r: 对数半径（可学习）
        color: RGB颜色 (3,)
        feature: 特征向量 (feature_dim,)
        logit_eps: 存在概率的logit（可学习）
    
    空间支持场使用 smoothstep 截断，软化宽度 δ=0.2。
    距离计算基于局部度量场提供的马氏距离。
    """
    
    def __init__(self, mu, radius, color, feature_dim=16, eps=0.5):
        super().__init__()
        
        if mu.dim() != 1 or mu.shape[0] != 2:
            raise ValueError(f"mu must be (2,), got {mu.shape}")
        if color.dim() != 1 or color.shape[0] != 3:
            raise ValueError(f"color must be (3,), got {color.shape}")
        
        self._mu = nn.Parameter(mu.clone())
        self._log_r = nn.Parameter(torch.log(torch.tensor(radius, dtype=mu.dtype, device=mu.device)))
        self._color = nn.Parameter(color.clone())
        self._feature = nn.Parameter(torch.randn(feature_dim, dtype=mu.dtype, device=mu.device) * 0.1)
        self._logit_eps = nn.Parameter(torch.logit(torch.tensor(eps, dtype=mu.dtype, device=mu.device)))
    
    @property
    def position(self):
        return self._mu
    
    @property
    def radius(self):
        return torch.exp(self._log_r)
    
    @property
    def existence_prob(self):
        return torch.sigmoid(self._logit_eps)
    
    def forward(self, x, metric_fn):
        """
        Args:
            x: (N, 2) 查询点
            metric_fn: 返回 g(x) 的函数
        
        Returns:
            weight: (N,) 空间支持权重 [0, 1]
            density: (N,) 密度值
            feature_contrib: (N, feature_dim) 特征贡献
        """
        N = x.shape[0]
        
        # 在原子中心处采样度量（原子形状由其局部几何决定）
        g = metric_fn(self._mu.unsqueeze(0))  # (1, 2, 2)
        
        # 位移向量
        dx = x - self._mu.unsqueeze(0)  # (N, 2)
        
        # 马氏距离平方: dx^T g dx
        # g: (1, 2, 2), dx: (N, 2) -> gx: (N, 2)
        gx = torch.matmul(g, dx.unsqueeze(-1)).squeeze(-1)
        d2 = (dx * gx).sum(dim=-1)  # (N,)
        d2 = d2.clamp(min=0.0)  # 防止数值误差
        
        r = torch.exp(self._log_r)
        
        # smoothstep 截断函数
        t = torch.sqrt(d2) / r
        delta = 0.2
        t1 = 1.0 - delta
        
        weight = torch.where(
            t < t1,
            torch.ones_like(t),
            torch.where(
                t < 1.0,
                0.5 + 0.5 * torch.cos(math.pi * (t - t1) / delta),
                torch.zeros_like(t)
            )
        )
        
        eps_val = torch.sigmoid(self._logit_eps)
        density = eps_val * weight
        
        feature_contrib = self._feature.unsqueeze(0) * density.unsqueeze(-1)
        
        return weight, density, feature_contrib
