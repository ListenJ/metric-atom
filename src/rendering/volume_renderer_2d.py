import torch
import torch.nn as nn


def volume_render_2d(rays_o, rays_d, atoms, metric_field, num_samples=256,
                     near=0.0, far=1.0, scene_size=1.0, return_per_atom=False):
    """
    2D 体积渲染器（向量化版本）。
    
    Args:
        return_per_atom: 若为 True，额外返回 (A,) 的每个原子总密度贡献
    """
    N_rays = rays_o.shape[0]
    N_atoms = len(atoms)
    device = rays_o.device
    
    empty_return = (
        torch.zeros(N_rays, 3, device=device),
        torch.zeros(N_rays, device=device),
        torch.zeros(N_rays, device=device)
    )
    if return_per_atom:
        empty_return += (torch.zeros(0, device=device),)
    
    if N_atoms == 0:
        return empty_return
    
    # 光线采样
    t_vals = torch.linspace(near, far, num_samples, device=device)
    dt = t_vals[1] - t_vals[0]
    
    samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.view(1, -1, 1)  # (H, S, 2)
    samples_flat = samples.reshape(-1, 2)  # (H*S, 2)
    N_points = samples_flat.shape[0]
    
    # 收集原子参数（向量化）
    mus = torch.stack([a.position for a in atoms])  # (A, 2)
    radii = torch.stack([a.radius for a in atoms])  # (A,)
    colors = torch.stack([a._color for a in atoms])  # (A, 3)
    eps_vals = torch.stack([a.existence_prob for a in atoms])  # (A,)
    
    # 批量计算每个原子的度量（一次调用）
    g_centers = metric_field(mus)  # (A, 2, 2)
    
    # 计算所有采样点对所有原子的马氏距离
    # dx: (A, N_points, 2) — 每个原子到每个采样点的位移
    dx = samples_flat.unsqueeze(0) - mus.unsqueeze(1)  # (A, N_points, 2)
    
    # dx^T @ g @ dx，批量计算
    # g_centers: (A, 2, 2), dx: (A, N_points, 2)
    gx = torch.matmul(g_centers.unsqueeze(1), dx.unsqueeze(-1)).squeeze(-1)  # (A, N_points, 2)  -- but wrong dims
    
    # 正确做法：使用 einsum 批量计算 dx[i,n]^T @ g[i] @ dx[i,n]
    # gx[i, n, k] = sum_j g[i, k, j] * dx[i, n, j]
    gx = torch.einsum('akj,anj->ank', g_centers, dx)  # (A, N_points, 2)
    
    d2 = (dx * gx).sum(dim=-1).clamp(min=0.0)  # (A, N_points)
    
    # smoothstep 截断（向量化）
    t = torch.sqrt(d2) / (radii.unsqueeze(-1) + 1e-8)  # (A, N_points)
    delta = 0.2
    t1 = 1.0 - delta
    
    weight = torch.where(
        t < t1,
        torch.ones_like(t),
        torch.where(
            t < 1.0,
            0.5 + 0.5 * torch.cos(3.141592653589793 * (t - t1) / delta),
            torch.zeros_like(t)
        )
    )  # (A, N_points)
    
    # 密度 = eps * weight
    density = eps_vals.unsqueeze(-1) * weight  # (A, N_points)
    
    # 总密度（所有原子累加）
    sigma = density.sum(dim=0).reshape(N_rays, num_samples)  # (H, S)
    
    # 颜色（加权混合）
    # density: (A, N_points), colors: (A, 3) → (A, N_points, 3)
    color_contrib = density.unsqueeze(-1) * colors.unsqueeze(1)  # (A, N_points, 3)
    color = color_contrib.sum(dim=0).reshape(N_rays, num_samples, 3)  # (H, S, 3)
    
    # alpha 合成
    alpha = 1.0 - torch.exp(-sigma * dt)
    alpha = alpha.clamp(0.0, 1.0)
    
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
    T = torch.cat([torch.ones(N_rays, 1, device=device), T[:, :-1]], dim=1)
    
    weights = T * alpha
    
    rendered_color = (weights.unsqueeze(-1) * color).sum(dim=1)
    rendered_depth = (weights * t_vals.unsqueeze(0)).sum(dim=1)
    rendered_alpha = 1.0 - T[:, -1]
    
    if return_per_atom:
        density_reshaped = density.reshape(N_atoms, N_rays, num_samples)
        weights_expanded = weights.unsqueeze(0).expand(N_atoms, -1, -1)
        per_atom_total = (density_reshaped * weights_expanded).sum(dim=(1, 2))
        return rendered_color, rendered_depth, rendered_alpha, per_atom_total
    
    return rendered_color, rendered_depth, rendered_alpha
