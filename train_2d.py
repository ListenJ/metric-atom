"""
MetricAtom 2D 训练脚本 — 最小可行验证。

目标：在合成 2D 几何形状上验证度量驱动聚类假设。
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def create_atoms(num_atoms, scene_size, device, seed=42):
    """
    在场景中随机初始化原子。
    原子位置偏向物体可能出现的位置：中心区域 (0.25, 0.25) 到 (0.75, 0.75)。
    """
    torch.manual_seed(seed)
    atoms = []
    for _ in range(num_atoms):
        mu = torch.rand(2, device=device) * 0.5 + 0.25  # [0.25, 0.75]
        radius = 0.05 + torch.rand(1, device=device).item() * 0.1  # [0.05, 0.15]
        color = torch.rand(3, device=device)  # [0, 1]
        atom = Atom2D(mu, radius=radius, color=color, feature_dim=16, eps=0.5)
        atoms.append(atom)
    return atoms


def train_scene(H=128, W=128, num_atoms=100, num_epochs=300, num_views=4,
                lr=1e-3, device='cpu', output_dir='outputs'):
    """
    在单个合成场景上训练。
    
    Args:
        H, W: 图像分辨率
        num_atoms: 原子数量
        num_epochs: 训练步数
        num_views: 多视角数量
        lr: 学习率
        device: 计算设备
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/5] 生成合成数据...")
    images_np, masks_np, transforms = generate_multi_view(
        H=H, W=W, num_objects=2, num_views=num_views, seed=42
    )
    images = torch.from_numpy(images_np).float().to(device)  # (V, H, W, 3)
    masks = torch.from_numpy(masks_np).float().to(device)    # (V, H, W, K)
    occupancy = torch.from_numpy(get_occupancy(masks_np)).float().to(device)  # (H, W)
    
    scene_size = 1.0
    
    print(f"[2/5] 初始化度量场和原子...")
    metric_field = MetricField2D(H, W, init_scale=1.0).to(device)
    atoms = create_atoms(num_atoms, scene_size, device, seed=42)
    
    # 收集所有可优化参数
    all_params = list(metric_field.parameters())
    for atom in atoms:
        all_params.extend(atom.parameters())
    
    optimizer = torch.optim.Adam(all_params, lr=lr)
    
    print(f"[3/5] 预计算光线...")
    # 使用正交投影光线（所有像素共享相同方向）
    rays_o, rays_d = RaySampler2D.generate_rays_orthographic(
        H, W, scene_size=scene_size, device=device
    )
    
    # 损失权重
    w_met = 0.01   # 度量平滑
    w_vol = 0.1    # 占位耦合
    w_coh = 1.0    # 凝聚（已归一化，需要较大的权重平衡）
    
    print(f"[4/5] 开始训练 ({num_epochs} epochs)...")
    losses_log = []
    
    for epoch in range(num_epochs):
        # 随机选择一帧
        frame_idx = epoch % num_views
        target_img = images[frame_idx].reshape(-1, 3)  # (H*W, 3)
        
        # 体积渲染
        pred_color, pred_depth, pred_alpha = volume_render_2d(
            rays_o, rays_d, atoms, metric_field,
            num_samples=64, near=0.0, far=scene_size,
            scene_size=scene_size
        )
        
        # 计算各项损失
        loss_render = l1_loss(pred_color, target_img)
        loss_met = metric_smoothness_loss(metric_field) * w_met
        loss_vol = occupancy_coupling_loss(metric_field, occupancy,
                                           g_occ_target=1.0, g_bg_target=10.0) * w_vol
        
        loss = loss_render + loss_met + loss_vol
        
        # Phase 2: 加入凝聚损失
        coh_running = torch.tensor(0.0, device=device)
        if epoch >= num_epochs // 2:
            loss_coh = coherence_loss(atoms, metric_field) * w_coh
            coh_running = loss_coh.detach()
            loss += loss_coh
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses_log.append({
            'epoch': epoch,
            'total': loss.item(),
            'render': loss_render.item(),
            'met': loss_met.item(),
            'vol': loss_vol.item(),
            'coh': coh_running.item() if isinstance(coh_running, torch.Tensor) else coh_running,
        })
        
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            loss_dict = losses_log[-1]
            phase = "2" if epoch >= num_epochs // 2 else "1"
            print(f"Epoch {epoch:4d} [{phase}]: "
                  f"T={loss_dict['total']:.4f} | "
                  f"R={loss_dict['render']:.4f} | "
                  f"M={loss_dict['met']:.4f} | "
                  f"V={loss_dict['vol']:.4f} | "
                  f"C={loss_dict['coh']:.4f}")
            
            # 保存渲染对比图
            save_render_comparison(pred_color, target_img, H, W, epoch, output_path)
            
            # 保存度量场可视化
            save_metric_visualization(metric_field, H, W, epoch, output_path)
            
            # 保存原子分布
            save_atom_distribution(atoms, H, W, epoch, output_path)
    
    print(f"[5/5] 训练完成。结果保存在 {output_path}/")
    
    # 保存训练曲线
    save_loss_curves(losses_log, output_path, num_epochs)
    
    return atoms, metric_field, losses_log


def save_render_comparison(pred_color, target_img, H, W, epoch, output_path):
    """保存渲染对比图"""
    pred_img = pred_color.detach().cpu().reshape(H, W, 3).clamp(0, 1).numpy()
    target = target_img.detach().cpu().reshape(H, W, 3).numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pred_img)
    axes[0].set_title('Rendered')
    axes[0].axis('off')
    axes[1].imshow(target)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(output_path / f'render_{epoch:04d}.png', dpi=80)
    plt.close(fig)


def save_metric_visualization(metric_field, H, W, epoch, output_path):
    """保存度量场迹的可视化（仅热力图，快速模式）"""
    trace = metric_field.trace().detach().cpu().numpy()  # (H, W)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(trace, cmap='inferno')
    ax.set_title(f'Metric Trace tr(g) (epoch {epoch})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_path / f'metric_{epoch:04d}.png', dpi=80)
    plt.close(fig)


def save_atom_distribution(atoms, H, W, epoch, output_path):
    """保存原子分布图"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    for atom in atoms:
        mu = atom.position.detach().cpu().numpy()
        r = atom.radius.detach().cpu().item()
        color = atom._color.detach().cpu().numpy().clip(0.0, 1.0)
        eps = atom.existence_prob.detach().cpu().item()
        
        # 映射到图像坐标
        x = mu[0] * W
        y = mu[1] * H
        
        # 计算像素半径
        pixel_radius = max(r * W, 0.5)
        
        circle = plt.Circle((x, y), pixel_radius,
                             facecolor=color, alpha=min(eps, 0.8),
                             edgecolor='white', linewidth=0.5)
        ax.add_patch(circle)
    
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(f'Atoms (epoch {epoch})')
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / f'atoms_{epoch:04d}.png', dpi=80)
    plt.close(fig)


def save_loss_curves(losses_log, output_path, num_epochs):
    """保存训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = [d['epoch'] for d in losses_log]
    
    axes[0, 0].plot(epochs, [d['total'] for d in losses_log])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].axvline(x=num_epochs // 2, color='red', linestyle='--', label='Phase 2 start')
    axes[0, 0].legend()
    
    axes[0, 1].plot(epochs, [d['render'] for d in losses_log])
    axes[0, 1].set_title('Render Loss (L1)')
    
    axes[1, 0].plot(epochs, [d['met'] for d in losses_log])
    axes[1, 0].set_title('Metric Smoothness')
    
    axes[1, 1].plot(epochs, [d['coh'] for d in losses_log])
    axes[1, 1].set_title('Coherence Loss')
    axes[1, 1].axvline(x=num_epochs // 2, color='red', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path / 'loss_curves.png', dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 快速测试：小分辨率，少量 epoch
    atoms, metric_field, log = train_scene(
        H=64, W=64,
        num_atoms=50,
        num_epochs=300,
        num_views=4,
        lr=5e-3,
        device=device,
        output_dir='outputs/2d_test'
    )
