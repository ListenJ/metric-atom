"""
MetricAtom 2D 训练脚本 — 128x128 完整验证。

目标：验证度量驱动聚类假设。
"""

import torch
import numpy as np
import os
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


def create_atoms(num_atoms, device, seed=42):
    """网格初始化原子，确保覆盖整个场景"""
    torch.manual_seed(seed)
    atoms = []
    
    # 使用粗略的网格布局 + 小随机扰动
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
            radius = 0.08 + torch.rand(1, device=device, dtype=torch.float32).item() * 0.07
            color = torch.rand(3, device=device, dtype=torch.float32)
            atom = Atom2D(mu, radius=radius, color=color, feature_dim=16, eps=0.5)
            atoms.append(atom)
        if len(atoms) >= num_atoms:
            break
    
    return atoms


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
    
    losses_log = []
    
    print(f"[4/5] 开始训练 ({num_epochs} epochs, Phase 2 @ epoch {phase2_start})...")
    
    for epoch in range(num_epochs):
        frame_idx = epoch % num_views
        target_img = images[frame_idx].reshape(-1, 3)
        
        pred_color, pred_depth, pred_alpha = volume_render_2d(
            rays_o, rays_d, atoms, metric_field,
            num_samples=48, near=0.0, far=scene_size, scene_size=scene_size
        )
        
        loss_render = l1_loss(pred_color, target_img)
        loss_met = metric_smoothness_loss(metric_field) * w_met
        loss_vol = occupancy_coupling_loss(metric_field, occupancy,
                                           g_occ_target=1.0, g_bg_target=10.0) * w_vol
        loss = loss_render + loss_met + loss_vol
        
        coh_val = 0.0
        if epoch >= phase2_start:
            loss_coh = coherence_loss(atoms, metric_field) * w_coh
            coh_val = loss_coh.item()
            loss += loss_coh
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)  # 梯度裁剪
        optimizer.step()
        
        losses_log.append({
            'epoch': epoch,
            'total': loss.item(),
            'render': loss_render.item(),
            'met': loss_met.item(),
            'vol': loss_vol.item(),
            'coh': coh_val,
        })
        
        # 日志和可视化
        if epoch % 200 == 0 or epoch == num_epochs - 1:
            log = losses_log[-1]
            phase = "2" if epoch >= phase2_start else "1"
            print(f"  [{epoch:4d}/{num_epochs}|P{phase}] "
                  f"T={log['total']:7.3f} R={log['render']:.3f} "
                  f"M={log['met']:.3f} V={log['vol']:.3f} C={log['coh']:.3f}")
            
            if epoch <= 600 or epoch % 200 == 0:  # 减频可视化
                plot_render_comparison(pred_color, target_img, H, W, epoch, output_path)
                plot_metric_field(metric_field, H, W, epoch, output_path)
                plot_atom_scatter(atoms, H, W, epoch, output_path)
        
        if epoch % 400 == 0 and epoch > 0:
            plot_atom_distribution(atoms, H, W, epoch, output_path)
            if epoch >= phase2_start:
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
    
    n_threads = min(os.cpu_count() or 6, 8)
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)
    os.environ.setdefault('MKL_NUM_THREADS', str(n_threads))
    os.environ.setdefault('OMP_NUM_THREADS', str(n_threads))
    os.environ.setdefault('KMP_BLOCKTIME', '0')
    os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  |  Threads: {n_threads}  |  MKL: {torch.backends.mkl.is_available()}")
    
    atoms, field, log, metrics = train_scene(
        H=64, W=64,
        num_atoms=80,
        num_epochs=300,
        num_views=6,
        phase2_start=150,
        lr=5e-3,
        device=device,
        output_dir='outputs/2d_final'
    )
