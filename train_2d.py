"""
MetricAtom 2D 训练脚本 — 支持 64×64 验证 + 128×128 完整训练。

目标：验证度量驱动聚类假设，BF16 混合精度 + CUDA 加速。
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from contextlib import nullcontext
from torch.amp import GradScaler
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
import warnings


def balanced_kmeans(mus_np, n_clusters, random_state=42, max_attempts=20, min_balance=0.5):
    """
    Run KMeans with multiple seeds, pick the most balanced split.
    min_balance = min(cluster_sizes) / max(cluster_sizes).
    """
    best_labels = None
    best_balance = 0.0
    for attempt in range(max_attempts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state + attempt * 7,
                            n_init=3, max_iter=100)
            labels = kmeans.fit_predict(mus_np)
        counts = np.bincount(labels, minlength=n_clusters)
        balance = counts.min() / counts.max()
        if balance >= min_balance:
            return labels, balance
        if balance > best_balance:
            best_balance = balance
            best_labels = labels
    return best_labels, best_balance

from src.geometry.metric_field import MetricField2D
from src.atoms.atom_2d import Atom2D
from src.rendering.ray_sampler import RaySampler2D
from src.rendering.volume_renderer_2d import volume_render_2d
from src.losses.reconstruction import l1_loss
from src.losses.metric_regularizer import metric_smoothness_loss
from src.losses.occupancy_coupling import occupancy_coupling_loss
from src.losses.coherence import contrastive_coherence_loss
from src.losses.direct_cluster import DirectClusterLoss
from src.losses.eco_cluster import ECOClusterLoss
from src.losses.diffusion import compute_geodesic_affinity, feature_diffusion
from src.data.synthetic_2d import generate_multi_view, get_occupancy
from src.visualization.plot_metric import (
    plot_render_comparison, plot_atom_distribution, plot_metric_field,
    plot_feature_similarity, plot_loss_curves, generate_evaluation_report
)
from src.visualization.plot_atoms import plot_atom_scatter


def create_atoms(num_atoms, device, seed=42, radius_min=0.25, radius_max=0.35, occupancy=None):
    if occupancy is not None:
        # 只在物体区域内初始化原子，确保一开始覆盖率就高
        H, W = occupancy.shape
        occ_pixels = torch.nonzero(occupancy > 0.5).float()  # (M, 2) each = (y, x)
        if occ_pixels.shape[0] > 0:
            torch.manual_seed(seed)
            atoms = []
            np.random.seed(seed)
            for i in range(num_atoms):
                idx = np.random.randint(0, occ_pixels.shape[0])
                y, x = occ_pixels[idx][0].item(), occ_pixels[idx][1].item()
                # Add small random offset for variety
                u = (x + np.random.uniform(-3, 3)) / W
                v = (y + np.random.uniform(-3, 3)) / H
                u = np.clip(u, 0.05, 0.95)
                v = np.clip(v, 0.05, 0.95)
                mu = torch.tensor([u, v], device=device, dtype=torch.float32)
                radius = radius_min + torch.rand(1, device=device, dtype=torch.float32).item() * (radius_max - radius_min)
                color = torch.rand(3, device=device, dtype=torch.float32)
                atom = Atom2D(mu, radius=radius, color=color, feature_dim=16, eps=0.5)
                atom.birth_epoch = 0
                atoms.append(atom)
            print(f"  [Init] 全部 {num_atoms} 个原子初始化在物体区域 (occupancy引导)")
            return atoms
    
    # 默认网格初始化（fallback）
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
    
    # 组合得分：高误差 × (1/原子密度) × 当前帧占用权重（强力偏好物体像素）
    score = error_smooth / (density_smooth + 1.0) * (occupancy * 10.0 + 0.2)
    
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
    new_colors = torch.stack([target_rgb[int(coord[0]), int(coord[1])] for coord in coords[idx]]).to(device)
    
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


def train_scene(H=64, W=64, num_atoms=100, num_epochs=600, num_views=8, num_objects=2,
                phase2_start=250, lr=1e-3, device='cuda', output_dir='outputs/2d_64x64',
                bf16=False, fp16=False, num_samples=128, seed_every=25, prune_every=None,
                render_chunk_size=None, quick_mode=False,
                # Hyperparameters for loss weighting
                tau=0.5, pos_thresh=0.3, neg_thresh=2.0, var_weight=0.1,
                w_met=0.01, w_vol=0.2, w_coh=2.0, w_pos=5.0,
                # Diffusion hyperparameters
                diff_K=5, diff_alpha=0.5, diff_T=2,
                # Direct cluster loss hyperparameters (Path 1+3)
                use_direct_loss=True, w_direct=2.0, sinkhorn_eps=0.1, sinkhorn_iters=50, ent_weight=0.005,
                # ECO cluster loss hyperparameters (Phase 6b)
                use_eco=False, w_eco=0.5, eco_sinkhorn_eps=0.5, eco_sinkhorn_iters=50,
                eco_ent_weight=0.005, eco_id_weight=0.1,
                # Phase 8: discriminant barrier + j-separation
                barrier_weight=0.0, sep_weight=0.0,
                # Random seed
                seed=42):
    """完整训练流程 — 支持 64x64 快速验证和 128x128 训练"""
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 混合精度策略: BF16 > FP16 > FP32
    amp_ctx = nullcontext()
    scaler_enabled = False
    if device == 'cuda' and (bf16 or fp16):
        if bf16 and torch.cuda.is_bf16_supported():
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            scaler_enabled = True
            print(f"[AMP] BF16 混合精度")
        elif fp16:
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
            scaler_enabled = True
            print(f"[AMP] FP16 混合精度")
        else:
            print(f"[AMP] BF16 不可用，使用 FP32")
    scaler = GradScaler('cuda', enabled=scaler_enabled)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scene_size = 1.0
    
    print(f"[1/5] 生成合成数据 ({H}x{W}, {num_views} 视角)...")
    images_np, masks_np, transforms = generate_multi_view(
        H=H, W=W, num_objects=num_objects, num_views=num_views, seed=seed
    )
    n_clusters = num_objects  # 每个物体的原子自成一簇
    images = torch.from_numpy(images_np).float().to(device)
    masks = torch.from_numpy(masks_np).float().to(device)
    occupancy = torch.from_numpy(get_occupancy(masks_np)).float().to(device)
    
    print(f"[2/5] 初始化度量场 ({H}x{W}) + {num_atoms} 个原子...")
    metric_field = MetricField2D(H, W, init_scale=1.0).to(device)
    
    # 预计算每帧的 occupancy（需要在创建原子之前）
    frame_occupancy = torch.zeros(num_views, H, W, device=device)
    for fv in range(num_views):
        frame_masks = masks[fv]
        frame_occupancy[fv] = (frame_masks.sum(dim=-1) > 0.5).float()
    
    atoms = create_atoms(num_atoms, device, seed=seed, radius_min=0.25, radius_max=0.35,
                         occupancy=frame_occupancy[0] if num_atoms > 0 else None)

    # ── Direct Cluster Loss 模块（Path 1+3: Sinkhorn + 直接测地距离优化） ──
    if use_eco:
        # ECO supplements DirectCluster with j-invariant identity regularization
        eco_cluster = ECOClusterLoss(
            n_clusters=n_clusters, feature_dim=16,
            sinkhorn_eps=eco_sinkhorn_eps, sinkhorn_iters=eco_sinkhorn_iters,
            ent_weight=eco_ent_weight, id_weight=eco_id_weight,
        ).to(device)
        eco_cluster.set_barrier_sep(barrier_weight, sep_weight)
        eco_initialized = False
        print(f"[ECO] ECO 正则化已启用 (id_weight={eco_id_weight}, barrier={barrier_weight}, sep={sep_weight})")
    else:
        eco_cluster = None
        eco_initialized = True

    if use_direct_loss or use_eco:
        # DirectCluster is ALWAYS the primary clustering signal when ECO is enabled
        direct_cluster = DirectClusterLoss(
            n_clusters=n_clusters, feature_dim=16,
            sinkhorn_eps=sinkhorn_eps, sinkhorn_iters=sinkhorn_iters,
            ent_weight=ent_weight
        ).to(device)
        direct_cluster_initialized = False
    else:
        direct_cluster = None
        direct_cluster_initialized = True

    atom_params = [p for a in atoms for p in a.parameters()]

    optimizer_param_groups = [
        {'params': metric_field.parameters(), 'lr': lr},
        {'params': atom_params, 'lr': lr * 3},
    ]
    if direct_cluster is not None:
        optimizer_param_groups.append(
            {'params': direct_cluster.parameters(), 'lr': lr * 3}
        )
    if eco_cluster is not None:
        optimizer_param_groups.append(
            {'params': eco_cluster.parameters(), 'lr': lr * 3}
        )

    optimizer = torch.optim.Adam(optimizer_param_groups)
    
    print(f"[3/5] 预计算光线...")
    rays_o, rays_d = RaySampler2D.generate_rays_orthographic(
        H, W, scene_size=scene_size, device=device
    )
    
    # 以下权重通过函数参数传入（默认值保持原始行为）
    # w_met, w_vol, w_coh, w_pos 来自函数参数
    
    # ── 预计算物体距离图（Position Regularization） ──
    occ_np = occupancy.cpu().numpy()
    dist_to_obj = distance_transform_edt(1 - occ_np).astype(np.float32)  # 0在物体上, >0远离
    dist_max = max(H, W)
    dist_to_obj = np.clip(dist_to_obj / dist_max, 0.0, 1.0)  # 归一化到 [0, 1]
    dist_map = torch.from_numpy(dist_to_obj).to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    losses_log = []
    atom_contrib_accum = torch.zeros(len(atoms), device=device)
    atom_birth_epochs = {id(a): 0 for a in atoms}
    diff_val = 0.0  # 扩散相关指标（Phase 2 前保持 0）
    cluster_balance_val = 0.0  # 簇平衡度（Phase 2 前保持 0）
    j_drift_val = 0.0  # ECO j-不变量漂移（Phase 2 前保持 0）
    
    # 64×64 快速验证用更激进的参数
    is_quick = (H <= 64 and num_epochs <= 600)
    prune_interval = max(num_epochs // 15, 20) if is_quick else (prune_every if prune_every else max(num_epochs // 10, 50))
    protection_epochs = 100 if is_quick else 200
    seed_freq = 15 if is_quick else seed_every
    
    # 预计算每帧的 occupancy
    frame_occupancy = torch.zeros(num_views, H, W, device=device)
    for fv in range(num_views):
        frame_occupancy[fv] = (masks[fv].sum(dim=-1) > 0.5).float()
    
    print(f"[4/5] 开始训练 ({num_epochs} epochs, Phase 2 @ epoch {phase2_start})...")
    print(f"       Prune every {prune_interval}, Seed every {seed_freq}, Protection={protection_epochs}")
    
    if prune_every is None:
        prune_every = max(num_epochs // 10, 50)
    
    for epoch in range(num_epochs):
        frame_idx = epoch % num_views
        target_img = images[frame_idx].reshape(-1, 3)
        
        do_prune = (epoch > 0 and epoch % prune_interval == 0)
        do_seed = (epoch > 0 and epoch % seed_freq == 0 and epoch >= 50 and epoch <= 2700)
        
        optimizer.zero_grad()
        
        # ── 分块渲染 + 逐块 backward（显存优化） ──
        # 原版在全批量时创建 (A, N_rays*S, D) 巨张量 → 爆显存
        # 将 16384 条光线分成块，逐块 backward 释放计算图
        N_rays = rays_o.shape[0]
        cs = render_chunk_size if render_chunk_size is not None else (4096 if N_rays > 8192 else N_rays)
        num_chunks = (N_rays + cs - 1) // cs
        loss_render_val = 0.0
        pred_color_parts = []
        
        for chunk_idx, chunk_start in enumerate(range(0, N_rays, cs)):
            chunk_end = min(chunk_start + cs, N_rays)
            n_chunk = chunk_end - chunk_start
            chunk_weight = n_chunk / N_rays
            
            with amp_ctx:
                render_result = volume_render_2d(
                    rays_o[chunk_start:chunk_end], rays_d[chunk_start:chunk_end],
                    atoms, metric_field,
                    num_samples=num_samples, near=0.0, far=scene_size, scene_size=scene_size,
                    return_per_atom=do_prune
                )
                pred_color_c, _, _ = render_result[:3]
                pred_color_parts.append(pred_color_c.detach())
                
                # 渲染损失（只有光线相关）
                loss_render_c = l1_loss(pred_color_c, target_img[chunk_start:chunk_end])
                
                if do_prune:
                    per_atom_c = render_result[3]
                    if chunk_idx == 0:
                        per_atom_frame = per_atom_c.detach()
                    else:
                        per_atom_frame += per_atom_c.detach()
            
            # 逐块 backward → 释放该块计算图
            scaler.scale(loss_render_c * chunk_weight).backward()
            loss_render_val += loss_render_c.detach().item() * chunk_weight
        
        # 整理全帧 pred_color（用于可视化/播种）
        pred_color = torch.cat(pred_color_parts, dim=0)
        
        if do_prune:
            atom_contrib_accum += per_atom_frame
        
        # ── 正则化损失（全原子/全度量场，一次性 backward） ──
        coh_val = 0.0
        loss_pos_t = torch.tensor(0.0, device=device)
        with amp_ctx:
            loss_met = metric_smoothness_loss(metric_field) * w_met
            loss_vol = occupancy_coupling_loss(metric_field, occupancy,
                                               g_occ_target=1.0, g_bg_target=10.0) * w_vol
            
            # ── Position Regularization ──
            if w_pos > 0 and len(atoms) > 0:
                atom_positions = torch.stack([a.position for a in atoms])
                grid = atom_positions.unsqueeze(0).unsqueeze(2) * 2 - 1
                pos_dist = F.grid_sample(dist_map, grid, mode='bilinear',
                                         padding_mode='border', align_corners=False)
                pos_dist = pos_dist.squeeze()
                if pos_dist.dim() == 0:
                    pos_dist = pos_dist.unsqueeze(0)
                loss_pos_t = pos_dist.mean() * w_pos
            
            loss_reg = loss_met + loss_vol + loss_pos_t
            
            if epoch >= phase2_start:
                mus = torch.stack([a.position for a in atoms])
                feats = torch.stack([a._feature for a in atoms])

                if use_eco:
                    # ═══════════════════════════════════════════════════
                    # ECO + DirectCluster 联合聚类
                    #
                    # DirectCluster: 主信号（特征-原型 Sinkhorn + 测地压缩）
                    # ECO: j-不变量身份一致性正则化（二阶稳定）
                    #
                    # 组合: L = w_direct * L_direct + w_eco * (L_eco_quad + L_id + L_ent)
                    # ═══════════════════════════════════════════════════

                    # 特征扩散（可选）
                    if diff_K > 0:
                        A = compute_geodesic_affinity(mus, metric_field, K=diff_K,
                                                       tau_max_factor=3.0, s_factor=0.1)
                        diffused_feats = feature_diffusion(feats, A, alpha=diff_alpha, T=diff_T)
                        diff_val = ((diffused_feats - feats) ** 2).mean().item()
                        cluster_feats = diffused_feats
                    else:
                        diff_val = 0.0
                        cluster_feats = feats

                    # ── KMeans init at phase2_start ──
                    if epoch == phase2_start:
                        with torch.no_grad():
                            mus_np = mus.cpu().numpy()
                            labels, balance = balanced_kmeans(mus_np, n_clusters)

                            # Init DirectCluster prototypes
                            direct_cluster.init_prototypes(cluster_feats.detach(),
                                                           torch.from_numpy(labels).to(device))
                            # Init ECO prototypes
                            eco_cluster.init_prototypes(cluster_feats.detach(),
                                                        torch.from_numpy(labels).to(device))

                            # Init atom features
                            feat_dim = cluster_feats.shape[1]
                            for c in range(n_clusters):
                                mask = labels == c
                                if mask.sum() > 0:
                                    base = direct_cluster.prototypes[c].detach()
                                    for i, a in enumerate(atoms):
                                        if mask[i]:
                                            noise = torch.randn(feat_dim, device=device) * 0.05
                                            a._feature.copy_(base + noise)

                        direct_cluster_initialized = True
                        eco_initialized = True
                        print(f"  [ECO+Direct] Balanced init: "
                              f"{np.bincount(labels).tolist()} atoms per cluster (b={balance:.2f})")

                    # ── DirectCluster: primary assignment signal ──
                    loss_direct, P, dc_metrics = direct_cluster(
                        mus, metric_field, cluster_feats
                    )
                    coh_val = loss_direct.item()
                    cluster_balance_val = dc_metrics['cluster_balance']
                    loss_primary = loss_direct * w_direct

                    # ── ECO: j-invariant identity regularization ──
                    eco_progress = min((epoch - phase2_start) / max(num_epochs - phase2_start, 1), 1.0)
                    eco_cluster.set_progress(eco_progress)
                    loss_eco, P_eco, eco_metrics = eco_cluster(
                        mus, metric_field, cluster_feats
                    )

                    # ── Phase 8: collapse detection + orthogonal mutation ──
                    if barrier_weight > 0 or sep_weight > 0:
                        if epoch < phase2_start + 20 and eco_cluster.collapse_detected(cluster_feats, threshold=0.02):
                            mutated = eco_cluster.orthogonal_mutation(noise_scale=0.3)
                            if mutated:
                                print(f"  [Phase 8] Orthogonal mutation at epoch {epoch} (feature collapse)")
                    j_drift_val = eco_metrics.get('j_drift', 0.0)
                    loss_eco_reg = loss_eco * w_eco

                    loss_reg += loss_primary + loss_eco_reg

                elif use_direct_loss:
                    # ═══════════════════════════════════════════════════
                    # Path 1+3: Direct Metric Cluster Loss
                    #
                    # 度量场 → 占位耦合（学边界）
                    #        → Sinkhorn 软分配（特征→原型 相似度）
                    #        → 直接最小化簇内测地距离
                    #
                    # 替代 InfoNCE — 消除黎曼空间中的甜区脆弱性
                    # ═══════════════════════════════════════════════════

                    # 特征扩散（可选，与直接聚类损失互补）
                    if diff_K > 0:
                        A = compute_geodesic_affinity(mus, metric_field, K=diff_K,
                                                       tau_max_factor=3.0, s_factor=0.1)
                        diffused_feats = feature_diffusion(feats, A, alpha=diff_alpha, T=diff_T)
                        diff_val = ((diffused_feats - feats) ** 2).mean().item()
                        cluster_feats = diffused_feats
                    else:
                        diff_val = 0.0
                        cluster_feats = feats

                    # ── KMeans init at phase2_start ──
                    if epoch == phase2_start:
                        with torch.no_grad():
                            mus_np = mus.cpu().numpy()
                            labels, balance = balanced_kmeans(mus_np, n_clusters)

                            # 初始化特征原型
                            direct_cluster.init_prototypes(cluster_feats.detach(),
                                                           torch.from_numpy(labels).to(device))

                            # 同时初始化原子特征（簇内相似，簇间不同）
                            feat_dim = cluster_feats.shape[1]
                            for c in range(n_clusters):
                                mask = labels == c
                                if mask.sum() > 0:
                                    base = direct_cluster.prototypes[c].detach()
                                    for i, a in enumerate(atoms):
                                        if mask[i]:
                                            noise = torch.randn(feat_dim, device=device) * 0.05
                                            a._feature.copy_(base + noise)

                        direct_cluster_initialized = True
                        print(f"  [DirectCluster] Balanced init: "
                              f"{np.bincount(labels).tolist()} atoms per cluster (b={balance:.2f})")

                    # Direct cluster loss: Sinkhorn assignment + geodesic compaction
                    loss_direct, P, dc_metrics = direct_cluster(
                        mus, metric_field, cluster_feats
                    )
                    loss_coh = loss_direct * w_direct
                    coh_val = loss_direct.item()
                    cluster_balance_val = dc_metrics['cluster_balance']
                    loss_reg += loss_coh

                else:
                    # ── 原 InfoNCE 路径（保留用于消融实验） ──
                    # 计算测地亲和矩阵
                    A = compute_geodesic_affinity(mus, metric_field, K=diff_K,
                                                   tau_max_factor=3.0, s_factor=0.1)

                    # 特征扩散
                    diffused_feats = feature_diffusion(feats, A, alpha=diff_alpha, T=diff_T)

                    # 监控：扩散前后特征差异
                    diff_val = ((diffused_feats - feats) ** 2).mean().item()

                    # InfoNCE 损失
                    loss_coh = contrastive_coherence_loss(atoms, metric_field,
                                                           tau=tau, pos_thresh=pos_thresh,
                                                           neg_thresh=neg_thresh,
                                                           var_weight=var_weight,
                                                           diffused_feats=diffused_feats) * w_coh
                    coh_val = loss_coh.item()
                    loss_reg += loss_coh

                    if epoch == phase2_start:
                        # KMeans 空间先验特征初始化
                        with torch.no_grad():
                            mus_np = mus.cpu().numpy()
                            labels, balance = balanced_kmeans(mus_np, n_clusters)

                            feat_dim = feats.shape[1]
                            cluster_centroids = []
                            for c in range(n_clusters):
                                base = torch.randn(feat_dim, device=device) * 0.5
                                cluster_centroids.append(base)

                            for i, a in enumerate(atoms):
                                c = labels[i]
                                noise_small = torch.randn(feat_dim, device=device) * 0.05
                                a._feature.copy_(cluster_centroids[c] + noise_small)

                        print(f"  [KMeans] Spatial balanced init: "
                              f"{np.bincount(labels).tolist()} atoms per cluster (b={balance:.2f})")
        
        # 正则化损失 backward（图和渲染损失的图已释放，显存充裕）
        scaler.scale(loss_reg).backward()
        
        # ── 优化器步进 ──
        scaler.unscale_(optimizer)
        all_params = [p for pg in optimizer.param_groups for p in pg['params']]
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # ── 剪枝和播种（在 autocast 外部） ──
        if do_prune:
            atoms, atom_birth_epochs = prune_atoms_contrib(
                atom_contrib_accum, atoms, atom_birth_epochs, epoch,
                threshold=0.05 if is_quick else 0.1, min_atoms=40, protection=protection_epochs
            )
            atom_contrib_accum = atom_contrib_accum[:len(atoms)]
            atom_contrib_accum.zero_()
            
            if do_seed:
                seed_count = max(12, H // 5) if is_quick else max(8, H // 8)
                frame_occ = frame_occupancy[frame_idx]
                atoms, added = seed_atoms_smart(
                    atoms, pred_color, target_img, H, W, device,
                    metric_field, frame_occ, epoch,
                    num_seeds=seed_count, radius_min=0.20, radius_max=0.30
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
                for atom in atoms:
                    for p in atom.parameters():
                        if id(p) not in existing_ids:
                            new_atom_params.append(p)
                
                if new_atom_params:
                    optimizer.add_param_group({'params': new_atom_params, 'lr': lr * 3})
        
        losses_log.append({
            'epoch': epoch,
            'total': loss_render_val + loss_reg.item(),
            'render': loss_render_val,
            'met': loss_met.item(),
            'vol': loss_vol.item(),
            'coh': coh_val,
            'diff': diff_val,
            'pos': loss_pos_t.item(),
            'balance': cluster_balance_val,
            'j_drift': j_drift_val,
        })
        
        if epoch % 100 == 0:
            feats = torch.stack([a._feature.detach() for a in atoms])
            feat_std = feats.std(dim=0).mean().item()
        
        if epoch % 200 == 0 or epoch == num_epochs - 1:
            log = losses_log[-1]
            phase = "2" if epoch >= phase2_start else "1"
            if epoch >= phase2_start:
                if use_eco:
                    bal_str = f" B={log.get('balance', 0):.2f}"
                    eco_str = f" jD={log.get('j_drift', 0):.4f}"
                elif use_direct_loss:
                    bal_str = f" B={log.get('balance', 0):.2f}"
                    eco_str = ""
                else:
                    bal_str = ""
                    eco_str = ""
                print(f"  [{epoch:4d}/{num_epochs}|P{phase}] "
                      f"T={log['total']:7.3f} R={log['render']:.3f} "
                      f"M={log['met']:.3f} V={log['vol']:.3f} "
                      f"C={log['coh']:.3f} D={log['diff']:.4f} "
                      f"P={log['pos']:.4f}{bal_str}{eco_str} A={len(atoms)} FS={feat_std:.4f}")
            else:
                print(f"  [{epoch:4d}/{num_epochs}|P{phase}] "
                      f"T={log['total']:7.3f} R={log['render']:.3f} "
                      f"M={log['met']:.3f} V={log['vol']:.3f} "
                      f"C={log['coh']:.3f} P={log['pos']:.4f} "
                      f"A={len(atoms)} FS={feat_std:.4f}")
            
            if not quick_mode:
                plot_render_comparison(pred_color, target_img, H, W, epoch, output_path)
                plot_metric_field(metric_field, H, W, epoch, output_path)
                plot_atom_scatter(atoms, H, W, epoch, output_path)
        
        if epoch >= phase2_start and epoch == num_epochs - 1:
            if not quick_mode:
                plot_atom_distribution(atoms, H, W, epoch, output_path)
                plot_feature_similarity(atoms, epoch, output_path)
    
    print(f"[5/5] 训练完成。保存模型并评估...")
    
    if not quick_mode:
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
    import argparse
    parser = argparse.ArgumentParser(description='MetricAtom 2D Training')
    parser.add_argument('--resolution', type=int, default=64, choices=[64, 128],
                        help='Resolution: 64 for validation, 128 for full training')
    parser.add_argument('--epochs', type=int, default=0,
                        help='Number of epochs (0=auto: 600 for 64x64, 3000 for 128x128)')
    parser.add_argument('--bf16', action='store_true', default=True,
                        help='Enable BF16 mixed precision training')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Enable FP16 mixed precision (fallback when BF16 unsupported)')
    parser.add_argument('--atom', type=int, default=0,
                        help='Initial atom count (0=auto)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--samples', type=int, default=0,
                        help='Ray samples per ray (0=auto)')
    parser.add_argument('--chunk-size', type=int, default=0,
                        help='Render chunk size for VRAM (0=auto: 4096 for 128×128, full for 64×64)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory override')
    # ── Hyperparameter grid search args ──
    parser.add_argument('--w-met', type=float, default=0.01,
                        help='Metric smoothness weight (default: 0.01)')
    parser.add_argument('--w-vol', type=float, default=0.2,
                        help='Occupancy coupling weight (default: 0.2)')
    parser.add_argument('--w-coh', type=float, default=2.0,
                        help='Coherence (InfoNCE) weight (default: 2.0)')
    parser.add_argument('--w-pos', type=float, default=5.0,
                        help='Position regularization weight (default: 5.0)')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='InfoNCE temperature (default: 0.5)')
    parser.add_argument('--pos-thresh', type=float, default=0.3,
                        help='InfoNCE positive threshold (default: 0.3)')
    parser.add_argument('--neg-thresh', type=float, default=2.0,
                        help='InfoNCE negative threshold (default: 2.0)')
    parser.add_argument('--var-weight', type=float, default=0.1,
                        help='InfoNCE variance weight (default: 0.1)')
    parser.add_argument('--diff-k', type=int, default=5,
                        help='Geodesic affinity K for adaptive sigma (default: 5)')
    parser.add_argument('--diff-alpha', type=float, default=0.5,
                        help='Diffusion step size alpha (default: 0.5)')
    parser.add_argument('--diff-t', type=int, default=2,
                        help='Diffusion iterations T (default: 2)')
    # ── Direct cluster loss args (Path 1+3) ──
    parser.add_argument('--use-infonce', action='store_true', default=False,
                        help='Fall back to InfoNCE instead of direct cluster loss (for ablation)')
    parser.add_argument('--w-direct', type=float, default=2.0,
                        help='Direct cluster loss weight (default: 2.0)')
    parser.add_argument('--sinkhorn-eps', type=float, default=0.1,
                        help='Sinkhorn entropy regularization (default: 0.1)')
    parser.add_argument('--sinkhorn-iters', type=int, default=50,
                        help='Sinkhorn iterations (default: 50)')
    parser.add_argument('--ent-weight', type=float, default=0.005,
                        help='Entropy penalty weight for balanced clusters (default: 0.005)')
    # ── ECO cluster loss args (Phase 6b) ──
    parser.add_argument('--use-eco', action='store_true', default=False,
                        help='Add ECO j-invariant identity regularization alongside DirectCluster')
    parser.add_argument('--w-eco', type=float, default=0.5,
                        help='ECO regularization weight for j-invariant stability (default: 0.5)')
    parser.add_argument('--eco-sinkhorn-eps', type=float, default=0.5,
                        help='ECO Sinkhorn entropy regularization (default: 0.5)')
    parser.add_argument('--eco-sinkhorn-iters', type=int, default=50,
                        help='ECO Sinkhorn iterations (default: 50)')
    parser.add_argument('--eco-ent-weight', type=float, default=0.005,
                        help='ECO entropy penalty weight (default: 0.005)')
    parser.add_argument('--eco-id-weight', type=float, default=0.1,
                        help='ECO j-invariant identity consistency weight (default: 0.1)')
    # ── Phase 8: discriminant barrier + j-separation ──
    parser.add_argument('--w-barrier', type=float, default=0.0,
                        help='Discriminant barrier loss weight (Phase 8, default: 0.0)')
    parser.add_argument('--w-separation', type=float, default=0.0,
                        help='j-space separation loss weight (Phase 8, default: 0.0)')
    parser.add_argument('--num-objects', type=int, default=2,
                        help='Number of objects in synthetic scene (default: 2)')
    parser.add_argument('--quick', action='store_true', default=False,
                        help='Force quick mode (no visualizations)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    
    H = W = args.resolution
    is_64 = (H == 64)
    
    if args.epochs > 0:
        num_epochs = args.epochs
    else:
        num_epochs = 600 if is_64 else 3000
    
    num_atoms = args.atom if args.atom > 0 else (100 if is_64 else 200)
    # Phase 2: 覆盖率验证需要更早期开始（64×64 时 ~40%）
    phase2_start = int(num_epochs * 0.4) if is_64 else (num_epochs * 2 // 5)
    num_samples = args.samples if args.samples > 0 else (64 if is_64 else 96)
    seed_every = 25 if is_64 else 25

    output_dir = args.output if args.output else f'outputs/2d_{H}x{W}{"_bf16" if args.bf16 else ""}_v2'
    render_chunk_size = args.chunk_size if args.chunk_size > 0 else (2048 if H >= 128 else W * H)
    
    # 使用默认 dtype (FP32)，通过 autocast 在 forward 时转换为 BF16
    
    n_threads = min(os.cpu_count() or 8, 8)
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)
    os.environ.setdefault('MKL_NUM_THREADS', str(n_threads))
    os.environ.setdefault('OMP_NUM_THREADS', str(n_threads))
    os.environ.setdefault('KMP_BLOCKTIME', '0')
    os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bf16_enabled = args.bf16 and device == 'cuda' and torch.cuda.is_bf16_supported()
    # 如果 BF16 不支持但请求了混合精度，自动用 FP16
    if not bf16_enabled and args.bf16 and device == 'cuda':
        args.fp16 = True
    fp16_enabled = args.fp16 and device == 'cuda'
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"CUDA: {torch.version.cuda}  |  cuDNN: {torch.backends.cudnn.version()}")
        print(f"BF16: {'Enabled' if bf16_enabled else 'Not supported, using FP32'}")
    else:
        print(f"Device: {device} (CPU fallback)")
    
    print(f"Resolution: {H}x{W}  |  Atoms: {num_atoms}  |  Epochs: {num_epochs}")
    print(f"Phase 2 start: {phase2_start}  |  Samples: {num_samples}")
    print(f"LR: {args.lr}  |  Chunk: {render_chunk_size}  |  Output: {output_dir}")
    
    quick_override = args.quick or is_64
    atoms, field, log, metrics = train_scene(
        H=H, W=W,
        num_atoms=num_atoms,
        num_epochs=num_epochs,
        num_views=8,
        num_objects=args.num_objects,
        phase2_start=phase2_start,
        lr=args.lr,
        device=device,
        output_dir=output_dir,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        num_samples=num_samples,
        seed_every=seed_every,
        render_chunk_size=render_chunk_size,
        quick_mode=quick_override,
        # Loss hyperparams
        tau=args.tau,
        pos_thresh=args.pos_thresh,
        neg_thresh=args.neg_thresh,
        var_weight=args.var_weight,
        w_met=args.w_met,
        w_vol=args.w_vol,
        w_coh=args.w_coh,
        w_pos=args.w_pos,
        # Diffusion hyperparams
        diff_K=args.diff_k,
        diff_alpha=args.diff_alpha,
        diff_T=args.diff_t,
        # Direct cluster loss hyperparams (Path 1+3)
        use_direct_loss=not args.use_infonce,
        w_direct=args.w_direct,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        ent_weight=args.ent_weight,
        # ECO cluster loss hyperparams (Phase 6b)
        use_eco=args.use_eco,
        w_eco=args.w_eco,
        eco_sinkhorn_eps=args.eco_sinkhorn_eps,
        eco_sinkhorn_iters=args.eco_sinkhorn_iters,
        eco_ent_weight=args.eco_ent_weight,
        eco_id_weight=args.eco_id_weight,
        # Phase 8: discriminant barrier + j-separation
        barrier_weight=args.w_barrier,
        sep_weight=args.w_separation,
        # Seed
        seed=args.seed,
    )
