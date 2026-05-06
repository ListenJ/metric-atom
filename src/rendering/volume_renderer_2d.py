import torch
import torch.nn as nn

from src.atoms.base_atom import BaseAtom


def volume_render_2d(rays_o, rays_d, atoms, metric_field, num_samples=256, near=0.0, far=1.0, scene_size=1.0):
    """
    2D 体积渲染器。
    
    对每条光线采样并积分，计算像素颜色和深度。
    坐标空间：所有采样点和原子坐标均在 scene_size 范围内，
    metric_field 期望 [0,1] 坐标，内部自动归一化。
    
    Args:
        rays_o: (H*W, 2) 射线起点，坐标在 [0, scene_size]
        rays_d: (H*W, 2) 射线方向（归一化）
        atoms: 原子列表 [Atom2D, ...]
        metric_field: MetricField2D 实例
        num_samples: 每条光线采样点数
        near, far: 光线采样范围
        scene_size: 场景大小（用于内部归一化）
    
    Returns:
        rendered_color: (H*W, 3) RGB
        rendered_depth: (H*W,) 深度
        rendered_alpha: (H*W,) 不透明度
    """
    N_rays = rays_o.shape[0]
    device = rays_o.device
    
    # 在 [near, far] 范围内均匀采样
    t_vals = torch.linspace(near, far, num_samples, device=device)  # (S,)
    dt = t_vals[1] - t_vals[0]
    
    # 采样点: (H, S, 2)
    samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.view(1, -1, 1)
    samples_flat = samples.reshape(-1, 2)  # (H*S, 2)
    
    # 累积密度和颜色
    sigma = torch.zeros(N_rays, num_samples, device=device)  # (H, S)
    color = torch.zeros(N_rays, num_samples, 3, device=device)  # (H, S, 3)
    
    # 遍历所有原子计算贡献
    for atom in atoms:
        _, density, feat = atom(samples_flat, metric_field)
        density = density.reshape(N_rays, num_samples)  # (H, S)
        
        sigma += density
        
        # 颜色贡献 = 密度 * 原子颜色
        atom_color = atom._color.unsqueeze(0).expand(N_rays * num_samples, -1)
        color_contrib = density.reshape(-1, 1) * atom_color  # (H*S, 3)
        color += color_contrib.reshape(N_rays, num_samples, 3)
    
    # 体积积分（alpha 合成）
    alpha = 1.0 - torch.exp(-sigma * dt)  # (H, S)
    alpha = alpha.clamp(0.0, 1.0)
    
    # 透射率
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=1)  # (H, S)
    T = torch.cat([torch.ones(N_rays, 1, device=device), T[:, :-1]], dim=1)
    
    # 权重 = T * alpha
    weights = T * alpha  # (H, S)
    
    # 渲染颜色
    rendered_color = (weights.unsqueeze(-1) * color).sum(dim=1)  # (H, 3)
    
    # 渲染深度
    rendered_depth = (weights * t_vals.unsqueeze(0)).sum(dim=1)  # (H,)
    
    # 不透明度
    rendered_alpha = 1.0 - T[:, -1]  # (H,)
    
    return rendered_color, rendered_depth, rendered_alpha
