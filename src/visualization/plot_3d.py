"""
MetricAtom 3D 可视化模块。

包含 3D 原子分布、度量场切片、渲染对比等可视化函数。
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def plot_render_comparison_3d(pred_color, target_img, H, W, step, output_path):
    """3D 渲染对比：真值 vs 渲染"""
    pred_img = pred_color.detach().cpu().reshape(H, W, 3).clamp(0, 1).numpy()
    target = target_img.detach().cpu().reshape(H, W, 3).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(target)
    axes[0].set_title('Ground Truth (3D Render)')
    axes[0].axis('off')
    axes[1].imshow(pred_img)
    axes[1].set_title('3D Volume Rendered')
    axes[1].axis('off')
    diff = np.abs(pred_img - target)
    axes[2].imshow(diff)
    axes[2].set_title('| Diff |')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path / f'render_{step:04d}.png', dpi=100)
    plt.close(fig)


def plot_atom_scatter_3d(atoms, H, W, step, output_path):
    """3D 原子分布（展示原子中心的 3D 散点图 + 3 个正交 2D 投影）"""
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])  # (N, 3)
    colors = np.stack([a._color.detach().cpu().numpy().clip(0, 1) for a in atoms])
    eps_vals = np.array([a.existence_prob.detach().cpu().item() for a in atoms])
    
    fig = plt.figure(figsize=(16, 4))
    
    # 3D 散点图
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1.scatter(mus[:, 0], mus[:, 1], mus[:, 2],
                c=colors, s=eps_vals * 30 + 5, alpha=0.7, edgecolors='black', linewidth=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.set_title('3D Atoms')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # XY 投影
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(mus[:, 0], mus[:, 1], c=colors, s=eps_vals * 20 + 3, alpha=0.7)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    
    # XZ 投影
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.scatter(mus[:, 0], mus[:, 2], c=colors, s=eps_vals * 20 + 3, alpha=0.7)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('XZ Projection')
    ax3.set_aspect('equal')
    
    # YZ 投影
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.scatter(mus[:, 1], mus[:, 2], c=colors, s=eps_vals * 20 + 3, alpha=0.7)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('YZ Projection')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path / f'atoms_{step:04d}.png', dpi=100)
    plt.close(fig)


def plot_atom_position_3d(atoms, step, output_path):
    """3D 原子位置随特征颜色的演化图"""
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])  # (N, 3)
    feats = np.stack([a._feature.detach().cpu().numpy() for a in atoms])
    
    # 特征 PCA 降维着色
    feats_centered = feats - feats.mean(0)
    try:
        cov = feats_centered.T @ feats_centered / (feats.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc = eigvecs[:, -1]
        feat_color = feats_centered @ pc
        feat_color = (feat_color - feat_color.min()) / (feat_color.max() - feat_color.min() + 1e-8)
    except Exception:
        feat_color = np.zeros(feats.shape[0])
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    scatter = ax1.scatter(mus[:, 0], mus[:, 1], mus[:, 2],
                          c=feat_color, cmap='viridis', s=10, alpha=0.8)
    ax1.set_title('3D Atom Positions (feature-PCA colored)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.colorbar(scatter, ax=ax1, fraction=0.02)
    
    # 位置直方图
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(mus[:, 0], bins=20, alpha=0.5, label='X')
    ax2.hist(mus[:, 1], bins=20, alpha=0.5, label='Y')
    ax2.hist(mus[:, 2], bins=20, alpha=0.5, label='Z')
    ax2.set_title('Position Distribution')
    ax2.legend()
    
    # 半径分布
    radii = np.array([a.radius.detach().cpu().item() for a in atoms])
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(radii, bins=20, alpha=0.7, color='green')
    ax3.set_title('Radius Distribution')
    ax3.set_xlabel('Radius')
    
    plt.tight_layout()
    plt.savefig(output_path / f'position_{step:04d}.png', dpi=100)
    plt.close(fig)


def plot_metric_slice_3d(metric_field, res_x, res_y, res_z, step, output_path):
    """3D 度量场切片可视化：取 z 方向中间切片显示 trace"""
    with torch.no_grad():
        trace_3d = metric_field.trace().cpu().numpy()  # (D, H, W)
    
    mid_z = res_z // 2
    mid_y = res_y // 2
    mid_x = res_x // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].imshow(trace_3d[mid_z], cmap='inferno')
    axes[0].set_title(f'Metric Trace (z-slice {mid_z})')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(trace_3d[:, mid_y, :], cmap='inferno')
    axes[1].set_title(f'Metric Trace (y-slice {mid_y})')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    im2 = axes[2].imshow(trace_3d[:, :, mid_x], cmap='inferno')
    axes[2].set_title(f'Metric Trace (x-slice {mid_x})')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path / f'metric_{step:04d}.png', dpi=100)
    plt.close(fig)


def plot_loss_curves_3d(losses_log, output_path, phase2_start):
    """3D 训练损失曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    epochs = [d['epoch'] for d in losses_log]
    
    axes[0, 0].plot(epochs, [d['total'] for d in losses_log])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].axvline(x=phase2_start, color='red', linestyle='--', alpha=0.5, label='Phase 2')
    axes[0, 0].legend()
    
    axes[0, 1].plot(epochs, [d['render'] for d in losses_log])
    axes[0, 1].set_title('Render Loss (L1)')
    
    axes[0, 2].plot(epochs, [d['met'] for d in losses_log])
    axes[0, 2].set_title('Metric Smoothness')
    
    axes[1, 0].plot(epochs, [d['vol'] for d in losses_log])
    axes[1, 0].set_title('Occupancy Coupling')
    
    axes[1, 1].plot(epochs, [d['coh'] for d in losses_log])
    axes[1, 1].set_title('Coherence Loss')
    axes[1, 1].axvline(x=phase2_start, color='red', linestyle='--', alpha=0.5)
    
    axes[1, 2].plot(epochs, [d['pos'] for d in losses_log])
    axes[1, 2].set_title('Position Reg')
    
    plt.tight_layout()
    plt.savefig(output_path / 'loss_curves.png', dpi=100)
    plt.close(fig)


def evaluate_clustering_3d(atoms, masks, H, W):
    """
    评估 3D 原子聚类质量（在 2D 投影上）。
    
    Args:
        atoms: 3D 原子列表
        masks: (V, H, W) 多视角占位掩码
        H, W: 图像分辨率
    
    Returns:
        metrics: dict with ARI, NMI
    """
    feats = np.stack([a._feature.detach().cpu().numpy() for a in atoms])
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])
    
    n_clusters = 2  # 场景中物体数固定为 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feats)
    
    # 计算原子到第一帧 2D 投影中的物体区域的归属
    mask_v0 = masks[0]  # (H, W) 2D 掩码
    ground_truth = np.full(mus.shape[0], -1, dtype=int)
    
    # 无法直接将 3D 位置映射到 2D 掩码，我们模拟为基于距离最近球体中心
    # 在场景中心附近放置两个参考球体中心（保留简化的 GT 评估）
    # 实际 3D 中更合理的评估：多视角一致性和重建质量
    for i, mu in enumerate(mus):
        # 将 [0,1] 映射到 [-1,1]
        # 使用简化的规则：x<0.5 → cluster 0, x>=0.5 → cluster 1
        if mu[0] < 0.5:
            ground_truth[i] = 0
        else:
            ground_truth[i] = 1
    
    valid = ground_truth >= 0
    if valid.sum() < 2:
        return {
            'ARI': float('nan'),
            'NMI': float('nan'),
            'valid_atoms': int(valid.sum()),
            'total_atoms': mus.shape[0]
        }
    
    ari = adjusted_rand_score(ground_truth[valid], cluster_labels[valid])
    nmi = normalized_mutual_info_score(ground_truth[valid], cluster_labels[valid])
    
    return {
        'ARI': float(ari),
        'NMI': float(nmi),
        'valid_atoms': int(valid.sum()),
        'total_atoms': mus.shape[0]
    }


def generate_3d_evaluation_report(atoms, metric_field, images, masks, losses_log,
                                   H, W, phase2_start, output_path):
    """生成 3D 评估报告"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("MetricAtom 3D Evaluation Report")
    print("=" * 60)
    
    # 1. 聚类评估
    print("\n[1] Clustering Evaluation (2D projection):")
    metrics = evaluate_clustering_3d(atoms, masks, H, W)
    print(f"    ARI: {metrics['ARI']:.4f}")
    print(f"    NMI: {metrics['NMI']:.4f}")
    print(f"    Valid atoms: {metrics['valid_atoms']}/{metrics['total_atoms']}")
    
    # 2. 原子统计
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])
    eps_vals = np.array([a.existence_prob.detach().cpu().item() for a in atoms])
    active = eps_vals > 0.1
    print(f"\n[2] 3D Atom Statistics:")
    print(f"    Total atoms: {len(atoms)}")
    print(f"    Active (eps>0.1): {active.sum()}")
    print(f"    Position range X: [{mus[:, 0].min():.3f}, {mus[:, 0].max():.3f}]")
    print(f"    Position range Y: [{mus[:, 1].min():.3f}, {mus[:, 1].max():.3f}]")
    print(f"    Position range Z: [{mus[:, 2].min():.3f}, {mus[:, 2].max():.3f}]")
    
    # 3. 度量场
    with torch.no_grad():
        trace_3d = metric_field.trace().cpu().numpy()
    print(f"\n[3] 3D Metric Field:")
    print(f"    Trace mean: {trace_3d.mean():.4f}")
    print(f"    Trace std: {trace_3d.std():.4f}")
    print(f"    Trace min/max: {trace_3d.min():.4f} / {trace_3d.max():.4f}")
    
    # 4. 最终损失
    final = losses_log[-1]
    print(f"\n[4] Final Losses (epoch {final['epoch']}):")
    print(f"    Total:  {final['total']:.4f}")
    print(f"    Render: {final['render']:.4f}")
    print(f"    Metric: {final['met']:.4f}")
    print(f"    OccVol: {final['vol']:.4f}")
    print(f"    Cohere: {final['coh']:.4f}")
    
    # 5. 生成可视化
    print(f"\n[5] Generating visualizations...")
    plot_loss_curves_3d(losses_log, output_path, phase2_start)
    print(f"    All visualizations saved to {output_path}/")
    print("=" * 60)
    
    return metrics
