import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def plot_cluster_comparison(atoms, masks, H, W, step, output_path):
    """
    聚类对比图：左=真实掩码叠加，右=KMeans聚类分配。
    """
    feats = np.stack([a._feature.detach().cpu().numpy() for a in atoms])
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])
    eps_vals = np.array([a.existence_prob.detach().cpu().item() for a in atoms])
    
    n_clusters = masks.shape[-1]
    mask_v0 = masks[0]  # (H, W, K)
    
    # 真值标签
    gt_labels = np.full(mus.shape[0], -1, dtype=int)
    for i, mu in enumerate(mus):
        px = int(np.clip(mu[0] * W, 0, W - 1))
        py = int(np.clip(mu[1] * H, 0, H - 1))
        for k in range(n_clusters):
            if mask_v0[py, px, k] > 0.5:
                gt_labels[i] = k
                break
    
    # KMeans 聚类标签
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feats)
    
    colors_cluster = np.array([
        [1.0, 0.3, 0.3],  # red
        [0.3, 0.3, 1.0],  # blue
        [0.3, 1.0, 0.3],  # green
        [1.0, 0.8, 0.0],  # yellow
    ])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 真值标签
    for k in range(n_clusters):
        mask_k = (gt_labels == k)
        if mask_k.any():
            axes[0].scatter(mus[mask_k, 0] * W, mus[mask_k, 1] * H,
                          c=colors_cluster[k % 4].reshape(1, 3),
                          s=eps_vals[mask_k] * 30 + 5, alpha=0.7,
                          label=f'Object {k + 1}', edgecolors='black', linewidth=0.3)
    axes[0].set_xlim(0, W)
    axes[0].set_ylim(H, 0)
    axes[0].set_title('Ground Truth Assignments')
    axes[0].set_aspect('equal')
    axes[0].legend(fontsize=8)
    axes[0].axis('off')
    
    # 聚类标签
    for k in range(n_clusters):
        mask_k = (cluster_labels == k)
        if mask_k.any():
            axes[1].scatter(mus[mask_k, 0] * W, mus[mask_k, 1] * H,
                          c=colors_cluster[k % 4].reshape(1, 3),
                          s=eps_vals[mask_k] * 30 + 5, alpha=0.7,
                          label=f'Cluster {k + 1}', edgecolors='black', linewidth=0.3)
    axes[1].set_xlim(0, W)
    axes[1].set_ylim(H, 0)
    axes[1].set_title('KMeans Clustering (Features)')
    axes[1].set_aspect('equal')
    axes[1].legend(fontsize=8)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / f'clusters_{step:04d}.png', dpi=100)
    plt.close(fig)
