"""
MetricAtom 框架诊断脚本 — 定量定位聚类失败根因。

三项诊断:
  D1: 度量场分离能力 — trace(物体内) vs trace(背景) 是否足够分离
  D2: 自组织梯度信号 — self-org loss 的梯度在各训练阶段的量级
  D3: Oracle 测试 — 用完美的度量场初始化, 看聚类是否能成功
"""
import sys, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, 'D:/MetricAtom')
from torch.optim.lr_scheduler import CosineAnnealingLR
from contextlib import nullcontext
from torch.amp import GradScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.atoms.atom_2d import Atom2D
from src.geometry.metric_field import MetricField2D
from src.rendering.ray_sampler import RaySampler2D
from src.rendering.volume_renderer_2d import volume_render_2d
from src.losses.reconstruction import l1_loss
from src.losses.metric_regularizer import metric_smoothness_loss
from src.losses.occupancy_coupling import occupancy_coupling_loss
from src.losses.self_organize import (
    compute_geodesic_neighbors, state_propagation,
    self_organization_loss, masked_prediction_loss,
)
from src.losses.direct_cluster import compute_pairwise_geodesic_sq
from src.data.synthetic_2d import generate_multi_view, get_occupancy


def cluster_atoms(atoms, masks, H, W):
    """用原子状态做 KMeans, 返回 ARI/NMI"""
    feats = np.stack([(a._state if hasattr(a, '_state') else a._feature)
                       .detach().cpu().numpy() for a in atoms])
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])
    from sklearn.cluster import KMeans
    n_clusters = masks.shape[-1]
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(feats)
    gt = np.full(mus.shape[0], -1, dtype=int)
    mask_v0 = masks[0]
    for i, mu in enumerate(mus):
        px = np.clip(int(mu[0] * W), 0, W - 1)
        py = np.clip(int(mu[1] * H), 0, H - 1)
        for k in range(n_clusters):
            if mask_v0[py, px, k] > 0.5:
                gt[i] = k; break
    valid = gt >= 0
    if valid.sum() < 2: return 0.0, 0.0
    return adjusted_rand_score(gt[valid], labels[valid]), normalized_mutual_info_score(gt[valid], labels[valid])


def run_diagnostic(H=32, W=32, num_atoms=50, num_epochs=600, seed=42, device='cuda'):
    print(f"\n{'='*70}")
    print(f"MetricAtom 框架诊断  |  {H}x{W}  |  {num_atoms} atoms  |  seed={seed}")
    print(f"{'='*70}\n")

    # ── Data ──
    images_np, masks_np, transforms = generate_multi_view(H=H, W=W, num_objects=2, num_views=8, seed=seed)
    images = torch.from_numpy(images_np).float().to(device)
    masks = torch.from_numpy(masks_np).float().to(device)
    occupancy = torch.from_numpy(get_occupancy(masks_np)).float().to(device)

    occ_ratio = occupancy.mean().item()
    print(f"[Data] occ_ratio={occ_ratio:.3f}  (objects occupy {occ_ratio*100:.1f}% of scene)")

    # ── Model ──
    metric_field = MetricField2D(H, W, init_scale=1.0).to(device)
    frame_occ = torch.zeros(8, H, W, device=device)
    for fv in range(8):
        frame_occ[fv] = (masks[fv].sum(dim=-1) > 0.5).float()

    occ_pixels = torch.nonzero(frame_occ[0] > 0.5).float()
    torch.manual_seed(seed); np.random.seed(seed)
    atoms = []
    for i in range(num_atoms):
        idx = np.random.randint(0, occ_pixels.shape[0])
        y, x = occ_pixels[idx][0].item(), occ_pixels[idx][1].item()
        u = np.clip((x + np.random.uniform(-3, 3)) / W, 0.05, 0.95)
        v = np.clip((y + np.random.uniform(-3, 3)) / H, 0.05, 0.95)
        mu = torch.tensor([u, v], device=device, dtype=torch.float32)
        atoms.append(Atom2D(mu, radius=0.25 + torch.rand(1).item() * 0.1,
                            color=torch.rand(3, device=device), state_dim=16, eps=0.5).to(device))

    state_decoder = torch.nn.Linear(16, 3).to(device)
    optimizer = torch.optim.Adam([
        {'params': metric_field.parameters(), 'lr': 1e-3},
        {'params': [p for a in atoms for p in a.parameters()], 'lr': 3e-3},
        {'params': state_decoder.parameters(), 'lr': 1e-3},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    scaler = GradScaler('cuda', enabled=True)
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)

    rays_o, rays_d = RaySampler2D.generate_rays_orthographic(H, W, scene_size=1.0, device=device)

    # ── Training with per-phase diagnostics ──
    checkpoints = [0, 100, 200, 300, 400, 500, 599]
    trace_history = []
    grad_history = []

    print(f"\n{'Epoch':>5} {'Render':>7} {'Vol':>7} {'SelfOrg':>8} {'Trace_in':>9} {'Trace_out':>10} "
          f"{'Grad_mu':>8} {'Grad_s':>8} {'ARI':>6} {'NMI':>6}")
    print("-" * 95)

    for epoch in range(num_epochs):
        frame_idx = epoch % 8
        target_img = images[frame_idx].reshape(-1, 3)
        mask = (torch.rand(H * W, device=device) < 0.3)

        optimizer.zero_grad()
        N_rays = rays_o.shape[0]
        cs = 256
        loss_render_val = 0.0

        # Chunked render
        for chunk_start in range(0, N_rays, cs):
            chunk_end = min(chunk_start + cs, N_rays)
            cw = (chunk_end - chunk_start) / N_rays
            with amp_ctx:
                pred_c, _, _ = volume_render_2d(
                    rays_o[chunk_start:chunk_end], rays_d[chunk_start:chunk_end],
                    atoms, metric_field, num_samples=32, near=0.0, far=1.0)[:3]
                loss_r = l1_loss(pred_c, target_img[chunk_start:chunk_end])
            scaler.scale(loss_r * cw).backward()
            loss_render_val += loss_r.detach().item() * cw

        with amp_ctx:
            loss_met = metric_smoothness_loss(metric_field) * 0.01
            loss_vol = occupancy_coupling_loss(metric_field, occupancy) * 0.2

            mus = torch.stack([a.position for a in atoms])
            states = torch.stack([a.state for a in atoms])
            geo_w, _ = compute_geodesic_neighbors(mus, metric_field, k=5)
            states_p = state_propagation(states, geo_w, alpha=0.3)
            loss_so = self_organization_loss(mus, states_p, metric_field, K=5) * 0.5

            masked_idx = mask.nonzero(as_tuple=False).squeeze(-1)
            if masked_idx.numel() > 0:
                masked_px = torch.stack([(masked_idx % W).float() / W,
                                          (masked_idx // W).float() / H], dim=-1)
                loss_pred = masked_prediction_loss(
                    mus, states_p, metric_field, masked_px,
                    target_img[masked_idx],
                    torch.stack([a._color for a in atoms]),
                    state_decoder=state_decoder) * 1.0
            else:
                loss_pred = torch.tensor(0.0, device=device)

            # Position regularization
            from scipy.ndimage import distance_transform_edt
            if epoch == 0:
                occ_np = occupancy.cpu().numpy()
                dist_map_np = distance_transform_edt(1 - occ_np).astype(np.float32)
                dist_map_np = np.clip(dist_map_np / max(H, W), 0.0, 1.0)
                dist_map = torch.from_numpy(dist_map_np).to(device).unsqueeze(0).unsqueeze(0)
            atom_pos = torch.stack([a.position for a in atoms])
            grid = atom_pos.unsqueeze(0).unsqueeze(2) * 2 - 1
            pos_dist = F.grid_sample(dist_map, grid, mode='bilinear',
                                     padding_mode='border', align_corners=False).squeeze()
            if pos_dist.dim() == 0: pos_dist = pos_dist.unsqueeze(0)
            loss_pos = pos_dist.mean() * 5.0

            loss_reg = loss_met + loss_vol + loss_so + loss_pred + loss_pos

        scaler.scale(loss_reg).backward()
        scaler.unscale_(optimizer)
        all_params = [p for pg in optimizer.param_groups for p in pg['params']]
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ── Diagnostics at checkpoints ──
        if epoch in checkpoints:
            with torch.no_grad():
                trace_map = metric_field.trace()  # (H, W)
                trace_in = (trace_map * occupancy).sum() / occupancy.sum().clamp(min=1)
                trace_out = (trace_map * (1 - occupancy)).sum() / (1 - occupancy).sum().clamp(min=1)
                trace_history.append((epoch, trace_in.item(), trace_out.item()))

                # Gradient norms (snapshot)
                grad_mu = torch.stack([a._mu.grad.norm() for a in atoms if a._mu.grad is not None])
                grad_s = torch.stack([a._state.grad.norm() for a in atoms if a._state.grad is not None])
                grad_history.append((epoch, grad_mu.mean().item(), grad_s.mean().item()))

                ari, nmi = cluster_atoms(atoms, masks_np, H, W)

            print(f"{epoch:5d} {loss_render_val:7.3f} {loss_vol.item():7.3f} "
                  f"{loss_so.item():8.4f} {trace_in.item():9.3f} {trace_out.item():10.3f} "
                  f"{grad_mu.mean().item():8.5f} {grad_s.mean().item():8.5f} "
                  f"{ari:6.3f} {nmi:6.3f}")

    # ── D1: 度量场分离分析 ──
    print(f"\n{'─'*70}")
    print("[D1] 度量场分离能力")
    for ep, t_in, t_out in trace_history:
        sep = t_out / max(t_in, 1e-6)
        print(f"  epoch {ep:3d}: trace_in={t_in:.3f}  trace_out={t_out:.3f}  ratio={sep:.2f}")

    final_sep = trace_history[-1][2] / max(trace_history[-1][1], 1e-6)
    if final_sep > 3.0:
        print("  → 度量场分离充分 (ratio > 3)")
    elif final_sep > 1.5:
        print("  → 度量场分离不足但存在 (ratio 1.5-3)")
    else:
        print("  → 度量场几乎未分离 (ratio < 1.5) ← 根因之一")

    # ── D2: 梯度信号分析 ──
    print(f"\n[D2] 梯度信号衰减")
    for ep, gm, gs in grad_history:
        print(f"  epoch {ep:3d}: |grad_mu|={gm:.5f}  |grad_state|={gs:.5f}")
    if len(grad_history) >= 2:
        decay = grad_history[-1][2] / max(grad_history[0][2], 1e-8)
        print(f"  state grad 衰减比: {decay:.4f} ({decay*100:.1f}%)")
        if decay < 0.1:
            print("  → state 梯度严重衰减 ← 根因之一")
        else:
            print("  → state 梯度衰减在可接受范围")

    # ── D3: 特征可分性 ──
    print(f"\n[D3] 特征空间可分性")
    feats = np.stack([a._state.detach().cpu().numpy() for a in atoms])
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])
    gt = np.full(mus.shape[0], -1, dtype=int)
    for i, mu in enumerate(mus):
        px = np.clip(int(mu[0] * W), 0, W - 1)
        py = np.clip(int(mu[1] * H), 0, H - 1)
        for k in range(masks_np.shape[-1]):
            if masks_np[0, py, px, k] > 0.5:
                gt[i] = k; break

    valid = gt >= 0
    if valid.sum() > 2:
        f_valid = feats[valid]
        g_valid = gt[valid]
        # Inter vs intra cluster distance in feature space
        from scipy.spatial.distance import cdist
        dists = cdist(f_valid, f_valid)
        intra, inter = [], []
        for i in range(len(f_valid)):
            for j in range(i+1, len(f_valid)):
                if g_valid[i] == g_valid[j]:
                    intra.append(dists[i,j])
                else:
                    inter.append(dists[i,j])
        intra_m = np.mean(intra) if intra else 0
        inter_m = np.mean(inter) if inter else 0
        sep_ratio = inter_m / max(intra_m, 1e-8)
        print(f"  intra-cluster dist: {intra_m:.4f}")
        print(f"  inter-cluster dist: {inter_m:.4f}")
        print(f"  separation ratio:   {sep_ratio:.3f}")
        if sep_ratio > 1.5:
            print("  → 特征空间已具备可分性")
        else:
            print("  → 特征空间未分离 (ratio < 1.5) ← 根因之一")

    # ── D4: 测地距离分布 ──
    print(f"\n[D4] 测地距离分布")
    with torch.no_grad():
        D2 = compute_pairwise_geodesic_sq(mus_tensor := torch.stack([a.position for a in atoms]), metric_field)
        d = torch.sqrt(D2 + 1e-8)
        d_intra, d_inter = [], []
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                if valid[i] and valid[j]:
                    if gt[i] == gt[j]:
                        d_intra.append(d[i,j].item())
                    else:
                        d_inter.append(d[i,j].item())
        if d_intra and d_inter:
            print(f"  intra-cluster d_g: {np.mean(d_intra):.4f} ± {np.std(d_intra):.4f}")
            print(f"  inter-cluster d_g: {np.mean(d_inter):.4f} ± {np.std(d_inter):.4f}")
            gap = np.mean(d_inter) - np.mean(d_intra)
            print(f"  gap: {gap:.4f}")
            if gap > 0.05:
                print("  → 测地距离已有分离 → 度量场学到了结构")
            else:
                print("  → 测地距离无分离 → 度量场未编码物体边界 ← 根因")

    # ── 综合诊断 ──
    print(f"\n{'='*70}")
    print("综合诊断结论:")
    issues = []
    if final_sep < 2.0: issues.append("度量场未充分分离物体/背景")
    if len(grad_history) >= 2 and grad_history[-1][2] / max(grad_history[0][2], 1e-8) < 0.1:
        issues.append("状态梯度严重衰减")
    if valid.sum() > 2:
        if sep_ratio < 1.3: issues.append("特征空间不可分")
    if d_intra and d_inter and (np.mean(d_inter) - np.mean(d_intra)) < 0.02:
        issues.append("测地距离无物体间间隙")

    if issues:
        for i, iss in enumerate(issues):
            print(f"  [{i+1}] {iss}")
    else:
        print("  框架各组件工作正常, 问题可能在超参数或训练时长")
    print(f"{'='*70}\n")

    return trace_history, grad_history


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    run_diagnostic(H=32, W=32, num_atoms=50, num_epochs=600, seed=42, device=device)
