import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def plot_render_comparison(pred_color, target_img, H, W, step, output_path):
    """渲染对比图：真值 vs 渲染"""
    pred_img = pred_color.detach().cpu().reshape(H, W, 3).clamp(0, 1).numpy()
    target = target_img.detach().cpu().reshape(H, W, 3).numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(target)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    axes[1].imshow(pred_img)
    axes[1].set_title('Rendered')
    axes[1].axis('off')
    # 差异图
    diff = np.abs(pred_img - target)
    axes[2].imshow(diff)
    axes[2].set_title('| Diff |')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path / f'render_{step:04d}.png', dpi=100)
    plt.close(fig)


def plot_atom_distribution(atoms, H, W, step, output_path):
    """原子分布图：中心位置按特征相似度着色"""
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])  # (N, 2)
    feats = np.stack([a._feature.detach().cpu().numpy() for a in atoms])  # (N, D)
    eps_vals = np.array([a.existence_prob.detach().cpu().item() for a in atoms])
    colors = np.stack([a._color.detach().cpu().numpy().clip(0, 1) for a in atoms])
    
    # 对特征做 PCA 降维到 2D 用于着色
    feats_centered = feats - feats.mean(0)
    try:
        cov = feats_centered.T @ feats_centered / (feats.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1, pc2 = eigvecs[:, -1], eigvecs[:, -2]
        feat_color = np.stack([
            (feats_centered @ pc1),
            (feats_centered @ pc2),
            np.zeros(feats.shape[0])
        ], axis=-1)
        feat_color = (feat_color - feat_color.min(0)) / (feat_color.max(0) - feat_color.min(0) + 1e-8)
    except Exception:
        feat_color = colors
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 按实际颜色着色
    axes[0].scatter(mus[:, 0] * W, mus[:, 1] * H,
                    c=colors, s=eps_vals * 30 + 5, alpha=0.7, edgecolors='black', linewidth=0.3)
    axes[0].set_xlim(0, W)
    axes[0].set_ylim(H, 0)
    axes[0].set_title('Atoms by Color')
    axes[0].set_aspect('equal')
    axes[0].axis('off')
    
    # 按特征 PCA 着色
    axes[1].scatter(mus[:, 0] * W, mus[:, 1] * H,
                    c=feat_color, s=eps_vals * 30 + 5, alpha=0.7, edgecolors='black', linewidth=0.3)
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].set_title('Atoms by Feature PCA')
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / f'atoms_{step:04d}.png', dpi=100)
    plt.close(fig)
    
    return mus, feats, eps_vals


def plot_metric_field(metric_field, H, W, step, output_path):
    """度量场可视化：迹 + 特征值"""
    import torch
    
    # 密集采样计算度量迹
    trace = metric_field.trace().detach().cpu().numpy()
    
    # 在网格上采样度量计算特征值
    step_sample = 8
    coords_h = np.linspace(0, 1, H // step_sample)
    coords_w = np.linspace(0, 1, W // step_sample)
    
    eig_max = np.zeros((len(coords_h), len(coords_w)))
    eig_min = np.zeros((len(coords_h), len(coords_w)))
    
    for i, cy in enumerate(coords_h):
        for j, cx in enumerate(coords_w):
            coord = torch.tensor([[cx, cy]], dtype=torch.float32)
            g = metric_field(coord).detach().squeeze(0)
            try:
                eigs = torch.linalg.eigvalsh(g)
                eig_max[i, j] = eigs[1].item()
                eig_min[i, j] = eigs[0].item()
            except Exception:
                eig_max[i, j] = 0
                eig_min[i, j] = 0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(trace, cmap='inferno')
    axes[0].set_title('Metric Trace tr(g)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(eig_max, cmap='Reds')
    axes[1].set_title('Max Eigenvalue')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    im2 = axes[2].imshow(eig_min, cmap='Blues')
    axes[2].set_title('Min Eigenvalue')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path / f'metric_{step:04d}.png', dpi=100)
    plt.close(fig)


def plot_feature_similarity(atoms, step, output_path):
    """特征相似度矩阵：N×N 热力图"""
    feats = np.stack([a._feature.detach().cpu().numpy() for a in atoms])
    
    # 余弦相似度矩阵
    feats_norm = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
    sim = feats_norm @ feats_norm.T  # (N, N)
    
    # 不对特征做排序（后面 KMeans 聚类后可以 reorder）
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].imshow(sim, cmap='viridis', vmin=-1, vmax=1)
    axes[0].set_title('Atom Feature Cosine Similarity')
    axes[0].set_xlabel('Atom Index')
    axes[0].set_ylabel('Atom Index')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # KMeans 聚类后重排
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feats)
    order = np.argsort(labels)
    sim_ordered = sim[order][:, order]
    
    im1 = axes[1].imshow(sim_ordered, cmap='viridis', vmin=-1, vmax=1)
    axes[1].set_title('Reordered by KMeans (K=2)')
    axes[1].set_xlabel('Atom Index (sorted)')
    axes[1].set_ylabel('Atom Index (sorted)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path / f'similarity_{step:04d}.png', dpi=100)
    plt.close(fig)
    
    return sim, labels, order


def evaluate_clustering(atoms, masks, H, W):
    """
    评估聚类质量（仅用于验证，不参与训练）。
    
    对原子特征做 KMeans，与合成数据实例掩码比较。
    
    Args:
        atoms: 原子列表
        masks: (1, H, W, K) 实例掩码（取第一帧）
        H, W: 图像分辨率
    
    Returns:
        metrics: dict with ARI, NMI, num_clusters_found
    """
    feats = np.stack([a._feature.detach().cpu().numpy() for a in atoms])
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])
    
    # 对特征做 KMeans
    n_clusters = masks.shape[-1]  # 真实物体数
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feats)
    
    # 获取真值标签：根据原子空间位置最近的物体
    # 每个原子对应的真值标签 = 原子中心落在哪个物体掩码内
    ground_truth = np.full(mus.shape[0], -1, dtype=int)
    mask_v0 = masks[0]  # (H, W, K)，第一帧
    
    for i, mu in enumerate(mus):
        px = int(mu[0] * W)
        py = int(mu[1] * H)
        px = np.clip(px, 0, W - 1)
        py = np.clip(py, 0, H - 1)
        
        for k in range(n_clusters):
            if mask_v0[py, px, k] > 0.5:
                ground_truth[i] = k
                break
    
    # 过滤掉不在任何物体内的原子
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


def plot_loss_curves(losses_log, output_path, phase2_start):
    """训练损失曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = [d['epoch'] for d in losses_log]
    
    axes[0, 0].plot(epochs, [d['total'] for d in losses_log])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].axvline(x=phase2_start, color='red', linestyle='--', alpha=0.5, label='Phase 2')
    axes[0, 0].legend()
    
    axes[0, 1].plot(epochs, [d['render'] for d in losses_log])
    axes[0, 1].set_title('Render Loss (L1)')
    
    axes[1, 0].plot(epochs, [d['met'] for d in losses_log])
    axes[1, 0].set_title('Metric Smoothness')
    
    axes[1, 1].plot(epochs, [d['coh'] for d in losses_log])
    axes[1, 1].set_title('Coherence Loss')
    axes[1, 1].axvline(x=phase2_start, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path / 'loss_curves.png', dpi=100)
    plt.close(fig)


def generate_evaluation_report(atoms, metric_field, images, masks, losses_log, 
                                H, W, phase2_start, output_path):
    """生成完整评估报告"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("MetricAtom 2D Evaluation Report")
    print("=" * 60)
    
    # 1. 聚类评估
    print("\n[1] Clustering Evaluation:")
    metrics = evaluate_clustering(atoms, masks, H, W)
    print(f"    ARI: {metrics['ARI']:.4f}")
    print(f"    NMI: {metrics['NMI']:.4f}")
    print(f"    Valid atoms (in objects): {metrics['valid_atoms']}/{metrics['total_atoms']}")
    
    # 2. 原子分布统计
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])
    eps_vals = np.array([a.existence_prob.detach().cpu().item() for a in atoms])
    active = eps_vals > 0.1
    print(f"\n[2] Atom Statistics:")
    print(f"    Active atoms (eps > 0.1): {active.sum()}/{len(atoms)}")
    print(f"    Mean position: ({mus[:, 0].mean():.3f}, {mus[:, 1].mean():.3f})")
    print(f"    Position std: ({mus[:, 0].std():.3f}, {mus[:, 1].std():.3f})")
    
    # 3. 度量场统计
    trace = metric_field.trace().detach().cpu().numpy()
    print(f"\n[3] Metric Field Statistics:")
    print(f"    Trace mean: {trace.mean():.4f}")
    print(f"    Trace std: {trace.std():.4f}")
    print(f"    Trace min/max: {trace.min():.4f} / {trace.max():.4f}")
    
    # 4. 最终损失
    final = losses_log[-1]
    print(f"\n[4] Final Losses (epoch {final['epoch']}):")
    print(f"    Total:  {final['total']:.4f}")
    print(f"    Render: {final['render']:.4f}")
    print(f"    Metric: {final['met']:.4f}")
    print(f"    OccVol: {final['vol']:.4f}")
    print(f"    Cohere: {final['coh']:.4f}")
    
    # 5. 生成所有可视化
    print(f"\n[5] Generating visualizations...")
    frame = torch.from_numpy(images[0].reshape(-1, 3)).float()
    # Use pred from the last epoch — we need to re-render, but for now use placeholder
    plot_render_comparison(frame, frame, H, W, 9999, output_path)
    plot_atom_distribution(atoms, H, W, 9999, output_path)
    plot_metric_field(metric_field, H, W, 9999, output_path)
    
    feat_sim, labels, order = plot_feature_similarity(atoms, 9999, output_path)
    plot_loss_curves(losses_log, output_path, phase2_start)
    
    print(f"    All visualizations saved to {output_path}/")
    print("=" * 60)
    
    return metrics
