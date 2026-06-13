import torch
import torch.nn as nn
import math

from src.atoms.base_atom import BaseAtom


class Atom2D(BaseAtom):
    """
    2D 自组织原子。

    参数：
        mu: 中心位置 (2,)
        log_r: 对数半径（可学习）
        color: RGB颜色 (3,)
        state: 内部状态 (state_dim,) — 编码 "我是什么/属于哪个物体"
        logit_eps: 存在概率的logit（可学习）

    状态 s_i 通过两个信号更新：
        1. 预测误差信号 — 我的状态能帮助预测 masked 像素吗？
        2. 邻居传播信号 — 测地近邻原子的状态聚合

    空间支持场使用 smoothstep 截断，软化宽度 δ=0.2。
    距离计算基于局部度量场提供的马氏距离。
    """
    
    def __init__(self, mu, radius, color, state_dim=16, eps=0.5):
        super().__init__()
        
        if mu.dim() != 1 or mu.shape[0] != 2:
            raise ValueError(f"mu must be (2,), got {mu.shape}")
        if color.dim() != 1 or color.shape[0] != 3:
            raise ValueError(f"color must be (3,), got {color.shape}")
        
        self._mu = nn.Parameter(mu.clone())
        self._log_r = nn.Parameter(torch.log(torch.tensor(radius, dtype=mu.dtype, device=mu.device)))
        self._color = nn.Parameter(color.clone())
        # State encodes "what/which object" and is decoded to color for rendering.
        # state_dim=3 is kept as a special case for direct RGB encoding (no decoder).
        # state_dim>3 is used with an external state_decoder in the renderer.
        self._state = nn.Parameter(torch.rand(state_dim, dtype=mu.dtype, device=mu.device))
        self._logit_eps = nn.Parameter(torch.logit(torch.tensor(eps, dtype=mu.dtype, device=mu.device)))
        self._state_dim = state_dim
    
    @property
    def position(self):
        return self._mu
    
    @property
    def radius(self):
        return torch.exp(self._log_r)
    
    @property
    def existence_prob(self):
        return torch.sigmoid(self._logit_eps)
    
    @property
    def state(self):
        return self._state

    def forward(self, x, metric_fn):
        """
        Args:
            x: (N, 2) 查询点
            metric_fn: 返回 g(x) 的函数
        
        Returns:
            weight: (N,) 空间支持权重 [0, 1]
            density: (N,) 密度值
            state_contrib: (N, state_dim) 状态贡献
        """
        N = x.shape[0]
        
        g = metric_fn(self._mu.unsqueeze(0))
        dx = x - self._mu.unsqueeze(0)
        gx = torch.matmul(g, dx.unsqueeze(-1)).squeeze(-1)
        d2 = (dx * gx).sum(dim=-1).clamp(min=0.0)
        
        r = torch.exp(self._log_r)
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
        
        state_contrib = self._state.unsqueeze(0) * density.unsqueeze(-1)
        
        return weight, density, state_contrib

    def get_color(self, state_decoder=None):
        """
        Return RGB color for rendering.
        
        If state_dim == 3, state directly parametrizes RGB.
        Otherwise, an external state_decoder must be provided to map state -> color.
        """
        if state_decoder is not None:
            return torch.sigmoid(state_decoder(self._state))
        # Clamp to valid RGB range
        return torch.sigmoid(self._state)
