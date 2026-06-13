import torch
import torch.nn as nn


def volume_render_2d(rays_o, rays_d, atoms, metric_field, num_samples=256,
                     near=0.0, far=1.0, scene_size=1.0, return_per_atom=False,
                     atom_chunk_size=None, state_decoder=None):
    """
    2D 体积渲染器（向量化 + 原子切片版本）。

    通过 atom_chunk_size 将原子分批处理，大幅降低显存峰值。
    每批原子独立计算 density/color，然后累加到 sigma/color。

    Args:
        atom_chunk_size: 同时处理的原子数上限，None=不分批。
            对 4GB VRAM 建议 10-20，对 8GB 建议 30-50。
        return_per_atom: 若为 True，额外返回 (A,) 的每个原子总密度贡献
        state_decoder: 可选 (nn.Module) state_dim -> 3 颜色解码器。
            如果提供，使用 decoder(atom._state) 作为颜色；否则使用
            atom.get_color() 直接返回 sigmoid(_state)。
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

    # 光线采样（所有原子共享）
    t_vals = torch.linspace(near, far, num_samples, device=device)
    dt = t_vals[1] - t_vals[0]

    samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.view(1, -1, 1)  # (N_rays, S, 2)
    samples_flat = samples.reshape(-1, 2)  # (N_rays*S, 2)
    N_points = samples_flat.shape[0]

    # 收集原子参数
    mus_all = torch.stack([a.position for a in atoms])  # (A, 2)
    radii_all = torch.stack([a.radius for a in atoms])  # (A,)
    # Color source: atom.get_color() uses state directly (STATE=COLOR)
    # If an external decoder is provided, use it to map state to color.
    if state_decoder is not None:
        colors_all = state_decoder(torch.stack([a.state for a in atoms]))  # (A, 3)
    else:
        colors_all = torch.stack([a.get_color() for a in atoms])  # (A, 3)
    eps_all = torch.stack([a.existence_prob for a in atoms])  # (A,)

    # 原子切片：确定批大小
    if atom_chunk_size is None or atom_chunk_size >= N_atoms:
        atom_chunk_size = N_atoms

    # 累积器
    sigma = torch.zeros(N_rays, num_samples, device=device)
    color = torch.zeros(N_rays, num_samples, 3, device=device)
    per_atom_contribs = [] if return_per_atom else None

    for a_start in range(0, N_atoms, atom_chunk_size):
        a_end = min(a_start + atom_chunk_size, N_atoms)
        Na = a_end - a_start

        mus = mus_all[a_start:a_end]
        radii = radii_all[a_start:a_end]
        colors = colors_all[a_start:a_end]
        eps_vals = eps_all[a_start:a_end]

        # 度量场查询（仅当前批原子）
        g_centers = metric_field(mus, batch_size=min(512, Na * 4))  # (Na, 2, 2)

        # 马氏距离
        dx = samples_flat.unsqueeze(0) - mus.unsqueeze(1)  # (Na, N_points, 2)
        gx = torch.einsum('akj,anj->ank', g_centers, dx)  # (Na, N_points, 2)
        d2 = (dx * gx).sum(dim=-1).clamp(min=0.0)  # (Na, N_points)

        # smoothstep
        t = torch.sqrt(d2) / (radii.unsqueeze(-1) + 1e-8)  # (Na, N_points)
        delta_s = 0.2
        t1 = 1.0 - delta_s
        pi_val = 3.141592653589793
        weight = torch.where(
            t < t1,
            torch.ones_like(t),
            torch.where(
                t < 1.0,
                0.5 + 0.5 * torch.cos(pi_val * (t - t1) / delta_s),
                torch.zeros_like(t)
            )
        )  # (Na, N_points)

        # 密度
        density_a = eps_vals.unsqueeze(-1) * weight  # (Na, N_points)

        # 累加到总密度
        sigma += density_a.sum(dim=0).reshape(N_rays, num_samples)

        # 颜色贡献
        color += (density_a.unsqueeze(-1) * colors.unsqueeze(1)).sum(dim=0).reshape(N_rays, num_samples, 3)

        # 每原子贡献（可选）
        if return_per_atom:
            per_atom_contribs.append(density_a.reshape(Na, N_rays, num_samples))

        # 释放本批中间张量
        del dx, gx, d2, t, weight, density_a, g_centers

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
        # 合并各批的 per_atom 贡献
        density_all = torch.cat(per_atom_contribs, dim=0)  # (N_atoms, N_rays, num_samples)
        weights_expanded = weights.unsqueeze(0)  # (1, N_rays, num_samples)
        per_atom_total = (density_all * weights_expanded).sum(dim=(1, 2))  # (N_atoms,)
        del density_all, weights_expanded
        return rendered_color, rendered_depth, rendered_alpha, per_atom_total

    return rendered_color, rendered_depth, rendered_alpha


def volume_render_3d(rays_o, rays_d, atoms, metric_field, num_samples=256,
                     near=0.0, far=1.0, scene_size=1.0, return_per_atom=False,
                     state_decoder=None):
    """
    3D 体积渲染器（向量化版本）。

    Args:
        return_per_atom: 若为 True，额外返回 (A,) 的每个原子总密度贡献
        state_decoder: 可选 (nn.Module) state_dim -> 3 颜色解码器。
            如果提供，使用 decoder(atom._state) 作为颜色。
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

    samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.view(1, -1, 1)  # (N_rays, S, 3)
    samples_flat = samples.reshape(-1, 3)  # (N_rays*S, 3)
    N_points = samples_flat.shape[0]

    # 收集原子参数（向量化）
    mus = torch.stack([a.position for a in atoms])  # (A, 3)
    radii = torch.stack([a.radius for a in atoms])  # (A,)
    # Color source: atom.get_color() uses state directly (STATE=COLOR)
    if state_decoder is not None:
        colors = state_decoder(torch.stack([a.state for a in atoms]))  # (A, 3)
    else:
        colors = torch.stack([a.get_color() for a in atoms])  # (A, 3)
    eps_vals = torch.stack([a.existence_prob for a in atoms])  # (A,)

    # 批量计算每个原子的度量
    g_centers = metric_field(mus)  # (A, 3, 3)

    # 计算所有采样点对所有原子的马氏距离
    dx = samples_flat.unsqueeze(0) - mus.unsqueeze(1)  # (A, N_points, 3)

    # einsum: dx[i,n]^T @ g[i] @ dx[i,n]
    gx = torch.einsum('akj,anj->ank', g_centers, dx)  # (A, N_points, 3)
    d2 = (dx * gx).sum(dim=-1).clamp(min=0.0)  # (A, N_points)

    # smoothstep 截断
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
    sigma = density.sum(dim=0).reshape(N_rays, num_samples)  # (N_rays, S)

    # 颜色混合
    color_contrib = density.unsqueeze(-1) * colors.unsqueeze(1)  # (A, N_points, 3)
    color = color_contrib.sum(dim=0).reshape(N_rays, num_samples, 3)  # (N_rays, S, 3)

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
