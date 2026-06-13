"""
MetricAtom 3D 训练脚本 — 多视角球体场景重建。

从 3D 球体场景的多个 2D 渲染中学习原子表示 + 3D 度量场。
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from contextlib import nullcontext
from torch.amp import GradScaler
from scipy.ndimage import distance_transform_edt

from src.geometry.metric_field import MetricField3D
from src.atoms.atom_3d import Atom3D
from src.rendering.ray_sampler import RaySampler3D
from src.rendering.volume_renderer_2d import volume_render_3d
from src.losses.reconstruction import l1_loss
from src.losses.metric_regularizer import metric_smoothness_loss_3d
from src.losses.occupancy_coupling import occupancy_coupling_loss
# [HISTORICAL] coherence_loss removed — 3D clustering is TBD (DirectClusterLoss 3D extension pending)
from src.data.synthetic_3d import generate_multi_view_3d
from src.visualization.plot_3d import (
    plot_render_comparison_3d, plot_atom_scatter_3d,
    plot_metric_slice_3d, plot_atom_position_3d,
    plot_loss_curves_3d, generate_3d_evaluation_report
)


def create_atoms_3d(num_atoms, device, seed=42, radius_min=0.15, radius_max=0.30):
    """初始化 3D 原子，在场景空间中均匀分布。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    atoms = []
    for i in range(num_atoms):
        mu = torch.tensor([
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.1, 0.9),
        ], device=device, dtype=torch.float32)
        radius = radius_min + torch.rand(1, device=device).item() * (radius_max - radius_min)
        color = torch.rand(3, device=device, dtype=torch.float32)
        atom = Atom3D(mu, radius=radius, color=color, feature_dim=16, eps=0.5)
        atom.birth_epoch = 0
        atoms.append(atom)
    print(f"  [Init] {num_atoms} 个 3D 原子初始化在场景空间")
    return atoms


def seed_atoms_smart_3d(atoms, pred_color, target_img, H, W, device,
                         metric_field, occupancy_frame, epoch,
                         num_seeds=12, radius_min=0.10, radius_max=0.20):
    """
    3D 智能播种：在 2D 渲染误差大的像素对应 3D 光线附近播种新原子。
    
    简化：在误差高的光线方向上随机深度处创建新原子。
    """
    N = len(atoms)
    error = (pred_color.detach() - target_img).abs().mean(dim=-1).reshape(H, W)
    
    # 高斯平滑误差图
    from torch.nn.functional import conv2d
    kernel_size = 19
    sigma = 3.0
    kernel_1d = torch.exp(-torch.linspace(-3, 3, kernel_size, device=device)**2 / (2 * sigma**2))
    kernel = kernel_1d.outer(kernel_1d)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    error_smooth = conv2d(error.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2).squeeze()
    
    # 选择高误差区域
    threshold = torch.quantile(error_smooth, 0.9)
    high_error = error_smooth > threshold
    if high_error.sum() < num_seeds:
        high_error = error_smooth > torch.quantile(error_smooth, 0.8)
    if high_error.sum() == 0:
        return atoms, N
    
    coords = torch.nonzero(high_error).float()  # (M, 2) = (y, x)
    idx = torch.randperm(len(coords))[:num_seeds]
    selected = coords[idx]  # (k, 2)
    
    # 在随机深度处播种新原子
    new_atoms = []
    for k in range(len(selected)):
        y, x = selected[k, 0].item(), selected[k, 1].item()
        px = int(x)
        py = int(y)
        target_rgb = target_img.detach().reshape(H, W, 3)[py, px]
        
        mu = torch.tensor([
            np.random.uniform(0.2, 0.8),
            np.random.uniform(0.2, 0.8),
            np.random.uniform(0.2, 0.8),
        ], device=device, dtype=torch.float32)
        
        atom = Atom3D(mu,
                      radius_min + torch.rand(1, device=device).item() * (radius_max - radius_min),
                      target_rgb.clone(), feature_dim=16, eps=0.5)
        atom.birth_epoch = epoch
        new_atoms.append(atom)
    
    print(f"  [Seed] +{len(new_atoms)} (error-guided, {N}→{N+len(new_atoms)})")
    return atoms + new_atoms, len(new_atoms)


def prune_atoms_contrib_3d(contrib, atoms, birth_epochs, epoch,
                            threshold=0.1, min_atoms=30, protection=300):
    """3D 原子剪枝（同 2D 版本，但保护期更长）。"""
    N = len(atoms)
    if N <= min_atoms:
        return atoms, birth_epochs
    
    protect = [(i, birth_epochs.get(id(a), 0)) for i, a in enumerate(atoms)]
    protect_mask = torch.tensor([(epoch - be >= protection) for _, be in protect],
                                device=contrib.device)
    protect_mask = protect_mask.to(contrib.dtype)
    
    if protect_mask.sum() < min_atoms // 2:
        return atoms, birth_epochs
    
    contrib_adjusted = contrib * protect_mask
    thresh = torch.quantile(contrib_adjusted[protect_mask > 0], threshold) if protect_mask.any() else 0
    keep = (contrib_adjusted > thresh) | (protect_mask <= 0)
    
    kept = [a for i, a in enumerate(atoms) if keep[i]]
    new_epochs = {id(a): birth_epochs.get(id(a), 0) for i, a in enumerate(atoms) if keep[i]}
    
    pruned = len(atoms) - len(kept)
    if pruned > 0:
        print(f"  [Prune] -{pruned} (prot={protection}, thresh={thresh:.2f}, contrib_m={contrib.mean():.2f})")
    
    return kept, new_epochs


def train_scene_3d(H=128, W=128, res_x=32, res_y=32, res_z=32,
                   num_atoms=200, num_epochs=3000, num_views=16,
                   phase2_start=1200, lr=1e-3, device='cuda',
                   output_dir='outputs/3d_128x128', bf16=False,
                   num_samples=128, seed_every=25, prune_every=None,
                   render_chunk_size=4096,
                   parametrization='cholesky'):
    """
    3D 完整训练流程。
    
    从多视角 2D 渲染中学习 3D 原子 + 3D 度量场。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if bf16 else nullcontext()
    scaler = GradScaler('cuda', enabled=bf16)
    
    # ── 生成合成 3D 数据 ──
    print(f"[1/5] 生成 3D 合成数据 ({H}x{W}, {num_views} 视角)...")
    images_np, masks_np, cameras, spheres = generate_multi_view_3d(
        H=H, W=W, num_objects=2, num_views=num_views, seed=42
    )
    images = torch.from_numpy(images_np).float().to(device)
    masks = torch.from_numpy(masks_np).float().to(device)
    
    # 3D 占位体素：在场景空间创建网格，标记每个体素是否包含球体
    print(f"[1.5/5] 预计算 3D 占位体素 ({res_x}x{res_y}x{res_z})...")
    occ_voxels = torch.zeros(res_z, res_y, res_x, device=device)
    xs = torch.linspace(0, 1, res_x, device=device)
    ys = torch.linspace(0, 1, res_y, device=device)
    zs = torch.linspace(0, 1, res_z, device=device)
    
    # 场景坐标 [0,1] 映射到世界坐标 [-1, 1]
    for sphere in spheres:
        center_world = sphere['center']  # (3,) in [-0.6, 0.6]
        # 映射到 [0, 1]
        cx = (center_world[0] + 1.0) / 2.0
        cy = (center_world[1] + 1.0) / 2.0
        cz = (center_world[2] + 1.0) / 2.0
        r = sphere['radius'] / 2.0  # 归一化半径
        
        for iz, z in enumerate(zs.cpu().numpy()):
            for iy, y in enumerate(ys.cpu().numpy()):
                for ix, x in enumerate(xs.cpu().numpy()):
                    dx, dy, dz = x - cx, y - cy, z - cz
                    if dx*dx + dy*dy + dz*dz < r*r:
                        occ_voxels[iz, iy, ix] = 1.0
    
    # ── 初始化 3D 度量场和原子 ──
    print(f"[2/5] 初始化 3D 度量场 ({res_x}x{res_y}x{res_z}) + {num_atoms} 个原子...")
    metric_field = MetricField3D(res_x, res_y, res_z, init_scale=1.0,
                                  parametrization=parametrization).to(device)
    atoms = create_atoms_3d(num_atoms, device, seed=42, radius_min=0.15, radius_max=0.30)
    
    atom_params = [p for a in atoms for p in a.parameters()]
    
    optimizer = torch.optim.Adam([
        {'params': metric_field.parameters(), 'lr': lr},
        {'params': atom_params, 'lr': lr * 3},
    ])
    
    # ── 预计算所有视角的光线 ──
    print(f"[3/5] 预计算 {num_views} 个视角的光线...")
    all_rays_o = []
    all_rays_d = []
    for v in range(num_views):
        cam = cameras[v]
        pos_t = torch.tensor(cam['pos'], device=device, dtype=torch.float32)
        rot_t = torch.tensor(cam['rot'], device=device, dtype=torch.float32)
        fx, fy = cam['fx'], cam['fy']
        rays_o, rays_d = RaySampler3D.generate_rays(
            H, W, fx, fy, pos_t, rot_t, near=0.1, far=5.0, device=device
        )
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
    
    # ── 损失权重 ──
    w_met = 0.01
    w_vol = 0.02
    w_coh = 2.0
    repulsion_weight = 5.0
    
    # 3D 距离变换用于位置正则化
    occ_np = occ_voxels.cpu().numpy()
    dist_to_obj = distance_transform_edt(1 - occ_np)  # 0 在物体内, >0 远离
    dist_max = max(res_x, res_y, res_z)
    dist_to_obj = np.clip(dist_to_obj / dist_max, 0.0, 1.0)
    dist_map = torch.from_numpy(dist_to_obj).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    
    # 预计算每帧的 2D 占位掩码
    frame_occupancy = masks.clone()  # (V, H, W)
    
    # ── 初始化跟踪变量 ──
    losses_log = []
    atom_contrib_accum = torch.zeros(len(atoms), device=device)
    atom_birth_epochs = {id(a): 0 for a in atoms}
    
    prune_interval = prune_every if prune_every else max(num_epochs // 10, 50)
    protection_epochs = 300  # 3D 更长的保护期
    seed_freq = seed_every
    
    print(f"[4/5] 开始 3D 训练 ({num_epochs} epochs, Phase 2 @ epoch {phase2_start})...")
    print(f"       Prune every {prune_interval}, Seed every {seed_freq}, Protection={protection_epochs}")
    
    for epoch in range(num_epochs):
        view_idx = epoch % num_views
        target_img = images[view_idx].reshape(-1, 3)
        
        do_prune = (epoch > 0 and epoch % prune_interval == 0)
        do_seed = (epoch > 0 and epoch % seed_freq == 0 and epoch >= 50)
        
        optimizer.zero_grad()
        
        # ── 分块渲染（同 2D） ──
        rays_o = all_rays_o[view_idx]
        rays_d = all_rays_d[view_idx]
        N_rays = rays_o.shape[0]
        cs = render_chunk_size if render_chunk_size else N_rays
        num_chunks = (N_rays + cs - 1) // cs
        loss_render_val = 0.0
        pred_color_parts = []
        
        for chunk_idx, chunk_start in enumerate(range(0, N_rays, cs)):
            chunk_end = min(chunk_start + cs, N_rays)
            n_chunk = chunk_end - chunk_start
            chunk_weight = n_chunk / N_rays
            
            with amp_ctx:
                render_result = volume_render_3d(
                    rays_o[chunk_start:chunk_end], rays_d[chunk_start:chunk_end],
                    atoms, metric_field,
                    num_samples=num_samples, near=0.1, far=5.0,
                    return_per_atom=do_prune
                )
                pred_color_c, _, _ = render_result[:3]
                pred_color_parts.append(pred_color_c.detach())
                
                loss_render_c = l1_loss(pred_color_c, target_img[chunk_start:chunk_end])
                
                if do_prune:
                    per_atom_c = render_result[3]
                    if chunk_idx == 0:
                        per_atom_frame = per_atom_c.detach()
                    else:
                        per_atom_frame += per_atom_c.detach()
            
            scaler.scale(loss_render_c * chunk_weight).backward()
            loss_render_val += loss_render_c.detach().item() * chunk_weight
        
        pred_color = torch.cat(pred_color_parts, dim=0)
        
        if do_prune:
            atom_contrib_accum += per_atom_frame
        
        # ── 正则化损失 ──
        coh_val = 0.0
        loss_pos_t = torch.tensor(0.0, device=device)
        with amp_ctx:
            loss_met = metric_smoothness_loss_3d(metric_field) * w_met
            # occupancy_coupling_loss 需要 3D 占位和 3D trace
            occ_3d = occ_voxels.unsqueeze(0)  # (1, D, H, W) for trace comparison
            trace_3d = metric_field.trace()  # (D, H, W)
            loss_vol = ((trace_3d - 1.0) ** 2 * occ_voxels).mean() + \
                       ((trace_3d - 10.0) ** 2 * (1.0 - occ_voxels)).mean()
            loss_vol = loss_vol * w_vol
            
            if len(atoms) > 0:
                atom_positions = torch.stack([a.position for a in atoms])
                grid = atom_positions.view(1, -1, 1, 1, 3) * 2 - 1
                grid = grid.permute(0, 2, 3, 1, 4)  # (1,1,1,N,3)
                pos_dist = F.grid_sample(dist_map, grid, mode='bilinear',
                                         padding_mode='border', align_corners=False)
                pos_dist = pos_dist.squeeze()
                if pos_dist.dim() == 0:
                    pos_dist = pos_dist.unsqueeze(0)
                loss_pos_t = pos_dist.mean() * 5.0  # w_pos=5.0
            
            loss_reg = loss_met + loss_vol + loss_pos_t
            
            # [TBD] 3D DirectClusterLoss — 当前禁用，coh_val 保持 0.0
        
        scaler.scale(loss_reg).backward()
        
        # ── 优化 ──
        scaler.unscale_(optimizer)
        all_params = [p for pg in optimizer.param_groups for p in pg['params']]
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # ── 剪枝 + 播种 ──
        if do_prune:
            atoms, atom_birth_epochs = prune_atoms_contrib_3d(
                atom_contrib_accum, atoms, atom_birth_epochs, epoch,
                threshold=0.1, min_atoms=40, protection=protection_epochs
            )
            atom_contrib_accum = atom_contrib_accum[:len(atoms)]
            atom_contrib_accum.zero_()
            
            if do_seed:
                seed_count = max(8, H // 8)
                atoms, added = seed_atoms_smart_3d(
                    atoms, pred_color, target_img, H, W, device,
                    metric_field, frame_occupancy[view_idx], epoch,
                    num_seeds=seed_count, radius_min=0.10, radius_max=0.20
                )
                extra = torch.zeros(added, device=device)
                atom_contrib_accum = torch.cat([atom_contrib_accum, extra])
                for a in atoms[-added:]:
                    atom_birth_epochs[id(a)] = epoch
                
                new_atom_params = []
                existing_ids = set()
                for pg in optimizer.param_groups:
                    for p in pg['params']:
                        existing_ids.add(id(p))
                for a in atoms:
                    for p in a.parameters():
                        if id(p) not in existing_ids:
                            new_atom_params.append(p)
                if new_atom_params:
                    optimizer.add_param_group({'params': new_atom_params, 'lr': lr * 3})
        
        # ── 日志 ──
        losses_log.append({
            'epoch': epoch,
            'total': loss_render_val + loss_reg.item(),
            'render': loss_render_val,
            'met': loss_met.item(),
            'vol': loss_vol.item() / w_vol if w_vol > 0 else 0,
            'coh': coh_val,
            'pos': loss_pos_t.item(),
        })
        
        if epoch % 100 == 0:
            feats = torch.stack([a._feature.detach() for a in atoms])
            feat_std = feats.std(dim=0).mean().item()
        
        if epoch % 200 == 0 or epoch == num_epochs - 1:
            log = losses_log[-1]
            phase = "2" if epoch >= phase2_start else "1"
            print(f"  [{epoch:4d}/{num_epochs}|P{phase}] "
                  f"T={log['total']:7.3f} R={log['render']:.3f} "
                  f"M={log['met']:.3f} V={log['vol']:.3f} "
                  f"C={log['coh']:.3f} P={log['pos']:.4f} "
                  f"A={len(atoms)} FS={feat_std:.4f}")
            
            # 使用 view 0 渲染可视化
            view_eval = 0
            rays_oe = all_rays_o[view_eval]
            rays_de = all_rays_d[view_eval]
            with torch.no_grad():
                with amp_ctx:
                    pred_eval, _, _ = volume_render_3d(
                        rays_oe, rays_de, atoms, metric_field,
                        num_samples=num_samples, near=0.1, far=5.0
                    )
            
            plot_render_comparison_3d(
                pred_eval, images[view_eval].reshape(-1, 3),
                H, W, epoch, output_path
            )
            plot_atom_scatter_3d(atoms, H, W, epoch, output_path)
            plot_atom_position_3d(atoms, epoch, output_path)
        
        if epoch % 500 == 0 or epoch == num_epochs - 1:
            plot_metric_slice_3d(metric_field, res_x, res_y, res_z, epoch, output_path)
    
    # ── 完成 ──
    print(f"[5/5] 训练完成。保存模型并评估...")
    torch.save({
        'metric_field': metric_field.state_dict(),
        'atoms': [atom.state_dict() for atom in atoms],
        'losses_log': losses_log,
    }, output_path / 'checkpoint.pt')
    
    plot_loss_curves_3d(losses_log, output_path, phase2_start)
    
    metrics = generate_3d_evaluation_report(
        atoms, metric_field, images_np, masks_np, losses_log,
        H, W, phase2_start, output_path / 'final'
    )
    
    return atoms, metric_field, losses_log, metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MetricAtom 3D Training')
    parser.add_argument('--resolution', type=int, default=128,
                        help='Render resolution')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Number of epochs')
    parser.add_argument('--bf16', action='store_true', default=True,
                        help='Enable BF16 mixed precision')
    parser.add_argument('--atom', type=int, default=200,
                        help='Initial atom count')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--samples', type=int, default=128,
                        help='Ray samples per ray')
    parser.add_argument('--chunk-size', type=int, default=4096,
                        help='Render chunk size for VRAM')
    parser.add_argument('--views', type=int, default=16,
                        help='Number of camera views')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--voxels', type=int, default=32,
                        help='MetricField3D voxel resolution (cubic)')
    parser.add_argument('--parametrization', type=str, default='cholesky',
                        choices=['cholesky', 'matrix_exp'],
                        help="Metric field parametrization (matrix_exp is very slow in 3D)")
    args = parser.parse_args()
    
    H = W = args.resolution
    res = args.voxels
    
    output_dir = args.output if args.output else f'outputs/3d_{H}x{W}_v{args.views}'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bf16_enabled = args.bf16 and device == 'cuda' and torch.cuda.is_bf16_supported()
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"CUDA: {torch.version.cuda}  |  BF16: {'Enabled' if bf16_enabled else 'Disabled'}")
    
    print(f"3D Training: {H}x{W}, Atoms={args.atom}, Epochs={args.epochs}")
    print(f"Views={args.views}, MetricField voxels={res}^3")
    print(f"Output: {output_dir}")
    
    atoms, field, log, metrics = train_scene_3d(
        H=H, W=W,
        res_x=res, res_y=res, res_z=res,
        num_atoms=args.atom,
        num_epochs=args.epochs,
        num_views=args.views,
        phase2_start=args.epochs * 2 // 5,
        lr=args.lr,
        device=device,
        output_dir=output_dir,
        bf16=bf16_enabled,
        num_samples=args.samples,
        render_chunk_size=args.chunk_size,
        parametrization=args.parametrization,
    )
