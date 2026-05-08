"""
MetricAtom 2D 训练脚本 — 128x128 完整验证。

目标：验证度量驱动聚类假设。
"""

import torch
import numpy as np
from pathlib import Path

from src.geometry.metric_field import MetricField2D
from src.atoms.atom_2d import Atom2D
from src.rendering.ray_sampler import RaySampler2D
from src.rendering.volume_renderer_2d import volume_render_2d
from src.losses.reconstruction import l1_loss
from src.losses.metric_regularizer import metric_smoothness_loss
from src.losses.occupancy_coupling import occupancy_coupling_loss
from src.losses.coherence import coherence_loss
from src.data.synthetic_2d import generate_multi_view, get_occupancy
from src.visualization.plot_metric import (
    plot_render_comparison, plot_atom_distribution, plot_metric_field,
    plot_feature_similarity, plot_loss_curves, generate_evaluation_report
)
from src.visualization.plot_atoms import plot_atom_scatter


def create_atoms(num_atoms, device, seed=42, radius_min=0.12, radius_max=0.18):
    torch.manual_seed(seed)
    atoms = []
    grid_size = int(np.ceil(np.sqrt(num_atoms)))
    for i in range(grid_size):
        for j in range(grid_size):
            if len(atoms) >= num_atoms:
                break
            u = (i + 0.5) / grid_size + torch.randn(1).item() * 0.03
            v = (j + 0.5) / grid_size + torch.randn(1).item() * 0.03
            u = np.clip(u, 0.1, 0.9)
            v = np.clip(v, 0.1, 0.9)
            mu = torch.tensor([u, v], device=device, dtype=torch.float32)
            radius = radius_min + torch.rand(1, device=device, dtype=torch.float32).item() * (radius_max - radius_min)
            color = torch.rand(3, device=device, dtype=torch.float32)
            atom = Atom2D(mu, radius=radius, color=color, feature_dim=16, eps=0.5)
            atom.birth_epoch = 0
            atoms.append(atom)
        if len(atoms) >= num_atoms:
            break
    return atoms


def seed_atoms_smart(atoms, pred_color, target_img, H, W, device,
                     metric_field, occupancy, epoch,
                     num_seeds=12, radius_min=0.06, radius_max=0.12,
                     blur_sigma=3.0):
    """
    智能播种：优先覆盖高渲染误差 + 低原子密度 + 物体内部区域。

    结合三张热力图：
    1) 渲染误差（L1）
    2) 原子密度空间分布（高斯核平滑）
    3) 占位掩码（鼓励在物体上播种）
    """
    N = len(atoms)
    error = (pred_color.detach() - target_img).abs().mean(dim=-1).reshape(H, W)
    
    from torch.nn.functional import conv2d
    kernel_size = int(blur_sigma * 6 + 1) | 1
    kernel = torch.exp(-torch.linspace(-3, 3, kernel_size, device=device)**2 / (2 * blur_sigma**2))
    kernel = kernel.outer(kernel)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    error_smooth = conv2d(error.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2).squeeze()
    
    density_map = torch.zeros(H, W, device=device)
    if N > 0:
        mus = torch.stack([a.position for a in atoms])
        radii = torch.stack([a.radius for a in atoms])
        px = (mus[:, 0] * W).clamp(0, W-1).long()
        py = (mus[:, 1] * H).clamp(0, H-1).long()
        radius_px = (radii * W).clamp(min=1)
        for i in range(N):
            r = int(radius_px[i].item())
            y_min = max(0, py[i] - r)
            y_max = min(H, py[i] + r + 1)
            x_min = max(0, px[i] - r)
            x_max = min(W, px[i] + r + 1)
            density_map[y_min:y_max, x_min:x_max] += 1
    
    density_smooth = conv2d(density_map.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2).squeeze()
    
    # 组合得分：高误差 × (1/原子密度) × 占位权重
    score = error_smooth / (density_smooth + 1.0) * (occupancy + 0.3)
    
    high_score = score > torch.quantile(score, 0.9)
    if high_score.sum() < num_seeds:
        high_score = score > torch.quantile(score, max(0.95 - N * 0.001, 0.7))
    if high_score.sum() == 0:
        return atoms, N
    
    coords = torch.nonzero(high_score).float()
    idx = torch.randperm(len(coords))[:num_seeds]
    new_mus = coords[idx].flip(-1)
    new_mus[:, 0] = new_mus[:, 0] / W
    new_mus[:, 1] = new_mus[:, 1] / H
    
    target_rgb = target_img.detach().reshape(H, W, 3)
    new_colors = torch.stack([target_rgb[int(y), int(x)] for y, x in coords[idx].flip(-1)]).to(device)
    
    new_atoms = []
    for k in range(len(new_mus)):
        atom = Atom2D(new_mus[k], radius_min + torch.rand(1, device=device).item() * (radius_max - radius_min),
                       new_colors[k], feature_dim=16, eps=0.5)
        atom.birth_epoch = epoch
        new_atoms.append(atom)
    
    print(f"  [Seed] +{len(new_atoms)} (score-based, μ_score={score.max():.3f}, dens={density_smooth.mean():.1f})")
    return atoms + new_atoms, len(new_atoms)


def prune_atoms_contrib(contrib, atoms, birth_epochs, epoch, 
                          threshold=0.1, min_atoms=30, protection=200):
    """
    渲染贡献剪枝 + 保护期。
    只删 birth_epoch + protection < epoch 的原子。
    """
    N = len(atoms)
    if N <= min_atoms:
        return atoms, birth_epochs
    
    protect = [(i, birth_epochs.get(id(a), 0)) for i, a in enumerate(atoms)]
    protect_mask = torch.tensor([(epoch - be >= protection) for _, be in protect],
                                device=contrib.device)
    
    if protect_mask.sum() < min_atoms // 2:
        return atoms, birth_epochs
    
    contrib_adjusted = contrib * protect_mask.float()
    thresh = torch.quantile(contrib_adjusted[protect_mask], threshold) if protect_mask.any() else 0
    keep = (contrib_adjusted > thresh) | ~protect_mask
    
    kept = [a for i, a in enumerate(atoms) if keep[i]]
    new_epochs = {id(a): birth_epochs.get(id(a), 0) for i, a in enumerate(atoms) if keep[i]}
    
    pruned = len(atoms) - len(kept)
    if pruned > 0:
        print(f"  [Prune] -{pruned} (prot={protection}, thresh={thresh:.2f}, contrib_m={contrib.mean():.2f})")
    
    return kept, new_epochs


def train_scene(H=128, W=128, num_atoms=200, num_epochs=2000, num_views=8,
                phase2_start=800, lr=5e-4, device='cpu', output_dir='outputs/2d_full'):
    """完整训练流程"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scene_size = 1.0
    
    print(f"[1/5] 生成合成数据 ({H}x{W}, {num_views} 视角)...")
    images_np, masks_np, transforms = generate_multi_view(
        H=H, W=W, num_objects=2, num_views=num_views, seed=42
    )
    images = torch.from_numpy(images_np).float().to(device)
    masks = torch.from_numpy(masks_np).float().to(device)
    occupancy = torch.from_numpy(get_occupancy(masks_np)).float().to(device)
    
    print(f"[2/5] 初始化度量场 ({H}x{W}) + {num_atoms} 个原子...")
    metric_field = MetricField2D(H, W, init_scale=1.0).to(device)
    atoms = create_atoms(num_atoms, device, seed=42)
    
    all_params = list(metric_field.parameters())
    for atom in atoms:
        all_params.extend(atom.parameters())
    
    optimizer = torch.optim.Adam(all_params, lr=lr)
    
    print(f"[3/5] 预计算光线...")
    rays_o, rays_d = RaySampler2D.generate_rays_orthographic(
        H, W, scene_size=scene_size, device=device
    )
    
    w_met = 0.01
    w_vol = 0.1
    w_coh = 1.0
    repulsion_weight = 0.3
    
    losses_log = []
    atom_contrib_accum = torch.zeros(len(atoms), device=device)
    atom_birth_epochs = {id(a): 0 for a in atoms}
    
    print(f"[4/5] 开始训练 ({num_epochs} epochs, Phase 2 @ epoch {phase2_start})...")
    
    prune_every = max(num_epochs // 10, 50)
    
    for epoch in range(num_epochs):
        frame_idx = epoch % num_views
        target_img = images[frame_idx].reshape(-1, 3)
        
        do_prune = (epoch > 0 and epoch % prune_every == 0)
        
        render_result = volume_render_2d(
            rays_o, rays_d, atoms, metric_field,
            num_samples=24, near=0.0, far=scene_size, scene_size=scene_size,
            return_per_atom=do_prune
        )
        pred_color, pred_depth, pred_alpha = render_result[:3]
        if do_prune:
            per_atom = render_result[3]
            atom_contrib_accum += per_atom.detach()
        
        loss_render = l1_loss(pred_color, target_img)
        loss_met = metric_smoothness_loss(metric_field) * w_met
        loss_vol = occupancy_coupling_loss(metric_field, occupancy,
                                           g_occ_target=1.0, g_bg_target=10.0) * w_vol
        loss = loss_render + loss_met + loss_vol
        
        coh_val = 0.0
        if epoch >= phase2_start:
            loss_coh = coherence_loss(atoms, metric_field, repulsion_weight=repulsion_weight) * w_coh
            coh_val = loss_coh.item()
            loss += loss_coh
            
            if epoch == phase2_start:
                with torch.no_grad():
                    feats = torch.stack([a._feature for a in atoms])
                    noise = torch.randn_like(feats) * 0.01
                    for i, a in enumerate(atoms):
                        a._feature.add_(noise[i])
                print(f"  [Inject] Initial feature noise at Phase 2 start")
            
            if epoch > phase2_start and epoch % 100 == 0:
                with torch.no_grad():
                    feats = torch.stack([a._feature for a in atoms])
                    noise = torch.randn_like(feats) * 0.02
                    for i, a in enumerate(atoms):
                        a._feature.add_(noise[i])
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        
        if do_prune:
            atoms, atom_birth_epochs = prune_atoms_contrib(
                atom_contrib_accum, atoms, atom_birth_epochs, epoch,
                threshold=0.1, min_atoms=30, protection=200
            )
            atom_contrib_accum = atom_contrib_accum[:len(atoms)]
            atom_contrib_accum.zero_()
            
            if epoch >= 50 and epoch <= 1800:
                atoms, added = seed_atoms_smart(
                    atoms, pred_color, target_img, H, W, device,
                    metric_field, occupancy, epoch,
                    num_seeds=12, radius_min=0.04, radius_max=0.10
                )
                extra = torch.zeros(added, device=device)
                atom_contrib_accum = torch.cat([atom_contrib_accum, extra])
                for a in atoms[-added:]:
                    atom_birth_epochs[id(a)] = epoch
            
            new_params = []
            existing_ids = {id(p) for p in all_params}
            for atom in atoms:
                for p in atom.parameters():
                    if id(p) not in existing_ids:
                        new_params.append(p)
                        all_params.append(p)
                        existing_ids.add(id(p))
            if new_params:
                optimizer.add_param_group({'params': new_params})
        
        losses_log.append({
            'epoch': epoch,
            'total': loss.item(),
            'render': loss_render.item(),
            'met': loss_met.item(),
            'vol': loss_vol.item(),
            'coh': coh_val,
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
                  f"C={log['coh']:.3f} A={len(atoms)} FS={feat_std:.4f}")
            
            plot_render_comparison(pred_color, target_img, H, W, epoch, output_path)
            plot_metric_field(metric_field, H, W, epoch, output_path)
            plot_atom_scatter(atoms, H, W, epoch, output_path)
        
        if epoch >= phase2_start and epoch == num_epochs - 1:
            plot_atom_distribution(atoms, H, W, epoch, output_path)
            plot_feature_similarity(atoms, epoch, output_path)
    
    print(f"[5/5] 训练完成。保存模型并评估...")
    
    # 保存模型状态
    torch.save({
        'metric_field': metric_field.state_dict(),
        'atoms': [atom.state_dict() for atom in atoms],
        'losses_log': losses_log,
    }, output_path / 'checkpoint.pt')
    
    # 保存训练曲线
    plot_loss_curves(losses_log, output_path, phase2_start)
    
    # 生成完整评估报告
    metrics = generate_evaluation_report(
        atoms, metric_field, images_np, masks_np, losses_log,
        H, W, phase2_start, output_path / 'final'
    )
    
    return atoms, metric_field, losses_log, metrics


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    atoms, field, log, metrics = train_scene(
        H=48, W=48,
        num_atoms=100,
        num_epochs=300,
        num_views=6,
        phase2_start=120,
        lr=5e-3,
        device=device,
        output_dir='outputs/2d_coverage_boost'
    )
