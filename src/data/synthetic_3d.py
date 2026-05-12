"""
MetricAtom 3D 合成数据生成。

场景：多个彩色球体在 3D 空间中，从不同相机视角渲染为 2D 图像。
每个视角使用 pinhole 相机模型，通过光线-球体相交渲染。
"""

import numpy as np
import torch


def generate_3d_scene(H=128, W=128, num_objects=2, seed=42):
    """
    生成 3D 场景定义（球体参数）。

    Args:
        H, W: 图像分辨率（仅用于确定物体大小比例）
        num_objects: 物体数量 (2-4)
        seed: 随机种子

    Returns:
        spheres: list of dicts，每个包含 {'center': (3,), 'radius': float, 'color': (3,)}
        scene_bounds: (6,) [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    np.random.seed(seed)
    
    colors_pool = [
        [1.0, 0.2, 0.2],  # 红
        [0.2, 0.2, 1.0],  # 蓝
        [0.2, 1.0, 0.2],  # 绿
        [1.0, 0.8, 0.0],  # 黄
        [1.0, 0.0, 1.0],  # 紫
    ]
    
    scene_size = 2.0  # [-1, 1]^3 场景
    
    spheres = []
    for k in range(num_objects):
        # 随机位置在 [-0.6, 0.6]^3 内（避免太靠近边界）
        center = np.random.uniform(-0.6, 0.6, 3)
        radius = np.random.uniform(0.15, 0.35)
        color = np.array(colors_pool[k % len(colors_pool)], dtype=np.float32)
        spheres.append({'center': center, 'radius': radius, 'color': color})
    
    scene_bounds = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    
    return spheres, scene_bounds


def ray_sphere_intersect(ray_o, ray_d, sphere_center, sphere_radius):
    """
    光线-球体相交检测（向量化版本）。

    Args:
        ray_o: (N, 3) 光线起点
        ray_d: (N, 3) 光线方向（需归一化）
        sphere_center: (3,) 球心
        sphere_radius: float

    Returns:
        t: (N,) 相交距离（无相交则为 -1）
    """
    oc = ray_o - sphere_center  # (N, 3)
    a = (ray_d * ray_d).sum(dim=-1)  # (N,) — 已归一化则全为 1
    b = 2.0 * (oc * ray_d).sum(dim=-1)  # (N,)
    c = (oc * oc).sum(dim=-1) - sphere_radius ** 2  # (N,)
    disc = b ** 2 - 4 * a * c
    
    t = torch.full_like(b, -1.0)
    mask = disc >= 0
    sqrt_disc = torch.sqrt(disc.clamp(min=0))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    t_pos = torch.where(t1 > 0, t1, t2)
    t = torch.where(mask & (t_pos > 0), t_pos, t)
    return t


def render_spheres_scene(rays_o, rays_d, spheres, H, W, near=0.1, far=4.0):
    """
    渲染包含多个球体的场景（光线-球体相交 + alpha 合成）。

    Args:
        rays_o: (N_rays, 3) 光线起点
        rays_d: (N_rays, 3) 光线方向
        spheres: list of dict，每个包含 'center', 'radius', 'color'
        H, W: 图像分辨率
        near, far: 光线起止范围

    Returns:
        image: (N_rays, 3) 渲染 RGB
        depth: (N_rays,) 深度
        mask: (N_rays,) 是否击中任何物体
    """
    device = rays_o.device
    N_rays = rays_o.shape[0]
    
    bg_color = torch.tensor([0.9, 0.9, 0.9], device=device, dtype=torch.float32)
    
    # 逐物体渲染并 alpha 合成
    acc_color = torch.zeros(N_rays, 3, device=device)
    acc_alpha = torch.zeros(N_rays, 1, device=device)
    acc_depth = torch.full((N_rays,), far, device=device)
    
    for sphere in spheres:
        center = torch.tensor(sphere['center'], device=device, dtype=torch.float32)
        radius = sphere['radius']
        color = torch.tensor(sphere['color'], device=device, dtype=torch.float32)
        
        t = ray_sphere_intersect(rays_o, rays_d, center, radius)  # (N_rays,)
        
        hit = (t > 0) & (t < far)
        if not hit.any():
            continue
        
        # 球体法线方向
        hit_o = rays_o[hit]
        hit_d = rays_d[hit]
        hit_t = t[hit]
        hit_pts = hit_o + hit_d * hit_t.unsqueeze(-1)
        normals = (hit_pts - center) / radius
        
        # 简单漫反射光照（+0.5 环境光）
        light_dir = torch.tensor([0.5, 0.5, 1.0], device=device, dtype=torch.float32)
        light_dir = light_dir / light_dir.norm()
        diffuse = (normals * light_dir.unsqueeze(0)).sum(dim=-1).clamp(min=0.0) * 0.5 + 0.5
        
        sphere_color = color.unsqueeze(0) * diffuse.unsqueeze(-1)  # (hit_count, 3)
        
        # 添加到合成缓冲区
        prev_rgb = acc_color[hit].clone()
        prev_alpha = acc_alpha[hit]
        
        new_alpha = 1.0
        acc_color[hit] = prev_rgb + new_alpha * (1.0 - prev_alpha) * sphere_color
        acc_alpha[hit] = prev_alpha + new_alpha * (1.0 - prev_alpha)
        
        # 最近击中深度
        closer = (hit_t < acc_depth[hit])
        acc_depth[hit] = torch.where(closer, hit_t, acc_depth[hit])
    
    # 背景色填充
    image = acc_color + (1.0 - acc_alpha) * bg_color.unsqueeze(0)
    mask = (acc_alpha.squeeze(-1) > 0.5).float()
    
    return image, acc_depth, mask


def generate_multi_view_3d(H=128, W=128, num_objects=2, num_views=8, seed=42):
    """
    生成多视角 3D 训练数据。

    球体位于 [-0.5, 0.5]^3 内，相机均匀分布在半径为 3.0 的球壳上，
    统一看向原点。

    Args:
        H, W: 分辨率
        num_objects: 场景球体数
        num_views: 视角数
        seed: 随机种子

    Returns:
        images: (V, H, W, 3) RGB 图像 [0, 1]
        masks: (V, H, W) 占位掩码 (0/1)
        cameras: list of dicts, 每个含 pos, rot, fx, fy
        spheres: 场景球体定义
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    spheres, scene_bounds = generate_3d_scene(H, W, num_objects, seed)
    
    # 相机参数
    fx = fy = float(W)
    radius_cam = 3.0  # 相机距离场景中心距离
    
    # 在球壳上均匀采样视角
    images = []
    masks = []
    cameras = []
    
    for v in range(num_views):
        # 使用 Fibonacci 球体算法在球壳上均匀采样
        gold_ratio = (1 + np.sqrt(5)) / 2
        theta = 2 * np.pi * v / gold_ratio
        phi = np.arccos(1 - 2 * (v + 0.5) / num_views)
        
        cam_pos = np.array([
            radius_cam * np.sin(phi) * np.cos(theta),
            radius_cam * np.sin(phi) * np.sin(theta),
            radius_cam * np.cos(phi),
        ], dtype=np.float32)
        
        cam_pos_t = torch.from_numpy(cam_pos).float().to(device)
        
        # 构建 look-at 旋转矩阵
        target_t = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        up_default = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
        
        z = (cam_pos_t - target_t) / (cam_pos_t - target_t).norm()
        x = torch.cross(up_default, z)
        x = x / x.norm()
        y = torch.cross(z, x)
        rot = torch.stack([x, y, z], dim=0)  # (3, 3)
        
        # 生成光线
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        x_cam = (xs - W / 2.0) / fx
        y_cam = (ys - H / 2.0) / fy
        z_cam = torch.ones_like(x_cam)
        
        dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)
        dirs_world = (rot @ dirs_cam.unsqueeze(-1)).squeeze(-1)
        
        rays_o = cam_pos_t.unsqueeze(0).unsqueeze(0).expand(H, W, -1)
        rays_d = dirs_world
        
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        # 渲染
        image, depth, mask = render_spheres_scene(rays_o_flat, rays_d_flat, spheres, H, W)
        
        images.append(image.cpu().reshape(H, W, 3).numpy())
        masks.append(mask.cpu().reshape(H, W).numpy())
        cameras.append({
            'pos': cam_pos.copy(),
            'rot': rot.cpu().numpy().copy(),
            'fx': fx,
            'fy': fy,
        })
    
    images = np.stack(images, axis=0).astype(np.float32)
    masks = np.stack(masks, axis=0).astype(np.float32)
    
    return images, masks, cameras, spheres


def get_occupancy_3d(masks):
    """
    从多视角掩码生成 3D 占位体素。
    简化：使用第一帧的 2D 掩码作为后续训练用占位，
    实际 3D 训练中不再需要全局 3D 占位（改为通过多视图监督）。

    Returns:
        occupancy: (H, W) 第一帧的占位掩码（用于 2D 兼容）
    """
    return masks[0].copy()
