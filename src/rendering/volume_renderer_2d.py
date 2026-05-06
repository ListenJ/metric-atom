import torch
import torch.nn as nn

from src.atoms.base_atom import BaseAtom


def volume_render_2d(rays_o, rays_d, atoms, metric_field, num_samples=256, near=0.0, far=2.0):
    """
    2D 体积渲染器。
    
    对每条光线采样并积分，计算像素颜色和深度。
    
    Args:
        rays_o: (H*W, 2) 射线起点
        rays_d: (H*W, 2) 射线方向（归一化）
        atoms: 原子列表 [Atom2D, ...]
        metric_field: MetricField2D 实例
        num_samples: 每条光线采样点数
        near, far: 光线采样范围
    
    Returns:
        rendered_color: (H*W, 3) RGB
        rendered_depth: (H*W,) 深度
        rendered_alpha: (H*W,) 不透明度（用于损失计算）
    """
    H = rays_o.shape[0]
    device = rays_o.device
    
    # 在 [near, far] 范围内均匀采样
    t_vals = torch.linspace(near, far, num_samples, device=device)  # (S,)
    dt = t_vals[1] - t_vals[0]
    
    # 采样点: (H, S, 2)
    # rays_o: (H, 1, 2), rays_d: (H, 1, 2), t_vals: (1, S, 1)
    samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.view(1, -1, 1)
    samples_flat = samples.reshape(-1, 2)  # (H*S, 2)
    
    # 归一化到 [0, 1] 用于度量场查询
    # 假设场景范围是 [0, 2]
    samples_normalized = samples_flat / 2.0
    samples_normalized = samples_normalized.clamp(0.0, 1.0)
    
    # 累积密度和颜色
    sigma = torch.zeros(H, num_samples, device=device)  # (H, S)
    color = torch.zeros(H, num_samples, 3, device=device)  # (H, S, 3)
    
    # 遍历所有原子计算贡献
    for atom in atoms:
        _, density, feat = atom(samples_flat, metric_field)
        density = density.reshape(H, num_samples)  # (H, S)
        
        sigma += density
        
        # 颜色贡献 = 密度 * 原子颜色
        atom_color = atom._color.unsqueeze(0).expand(H * num_samples, -1)
        color_contrib = density.reshape(-1, 1) * atom_color  # (H*S, 3)
        color += color_contrib.reshape(H, num_samples, 3)
    
    # 体积积分（alpha 合成）
    # alpha = 1 - exp(-sigma * dt)
    alpha = 1.0 - torch.exp(-sigma * dt)  # (H, S)
    alpha = alpha.clamp(0.0, 1.0)
    
    # 透射率 T = prod(1 - alpha)
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=1)  # (H, S)
    # 将 T 向后移一位：T[0] = 1, T[1] = 1 - alpha[0], ...
    T = torch.cat([torch.ones(H, 1, device=device), T[:, :-1]], dim=1)  # (H, S)
    
    # 权重 = T * alpha
    weights = T * alpha  # (H, S)
    
    # 渲染颜色
    rendered_color = (weights.unsqueeze(-1) * color).sum(dim=1)  # (H, 3)
    
    # 渲染深度
    rendered_depth = (weights * t_vals.unsqueeze(0)).sum(dim=1)  # (H,)
    
    # 不透明度 = 1 - T_final
    rendered_alpha = 1.0 - T[:, -1]  # (H,)
    
    return rendered_color, rendered_depth, rendered_alpha
