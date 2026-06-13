"""
消融实验: 度量场分离是聚类的瓶颈吗?

Config A: w_vol=0.2 (baseline)
Config B: w_vol=1.0 (boosted)
Config C: w_vol=0.2 + trace_contrast_loss (显式推拉)
Config D: w_vol=1.0 + trace_contrast_loss + w_selforg=1.0

每个 config 跑 400 epochs, 32x32, seed 42, 比较最终 ARI + trace separation.
"""
import sys, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, 'D:/MetricAtom')
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

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
from src.data.synthetic_2d import generate_multi_view, get_occupancy
from scipy.ndimage import distance_transform_edt


def trace_contrast_loss(metric_field, occupancy, margin=3.0):
    """
    显式推拉度量场迹: 物体内 trace → 小, 背景 trace → 大.
    使用 margin-based hinge loss 确保最小分离度.
    """
    trace = metric_field.trace()  # (H, W)
    occ = occupancy
    bg = 1.0 - occupancy

    # 物体内: 鼓励 trace < 1.5
    in_obj = trace * occ
    n_obj = occ.sum().clamp(min=1)
    loss_in = ((in_obj - 1.0).clamp(min=0) ** 2).sum() / n_obj

    # 背景: 鼓励 trace > 5.0
    bg_trace = trace * bg
    n_bg = bg.sum().clamp(min=1)
    loss_bg = ((5.0 - bg_trace).clamp(min=0) ** 2).sum() / n_bg

    return loss_in + loss_bg


def run_config(H, W, num_atoms, epochs, seed, device,
               w_vol=0.2, w_tc=0.0, w_selforg=0.5, label=""):
    images_np, masks_np, _ = generate_multi_view(H=H, W=W, num_objects=2, num_views=8, seed=seed)
    images = torch.from_numpy(images_np).float().to(device)
    masks = torch.from_numpy(masks_np).float().to(device)
    occupancy = torch.from_numpy(get_occupancy(masks_np)).float().to(device)

    metric_field = MetricField2D(H, W, init_scale=1.0).to(device)
    frame_occ = (masks[0].sum(dim=-1) > 0.5).float()
    occ_pixels = torch.nonzero(frame_occ > 0.5).float()

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
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler('cuda', enabled=True)
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    rays_o, rays_d = RaySampler2D.generate_rays_orthographic(H, W, scene_size=1.0, device=device)

    occ_np = occupancy.cpu().numpy()
    dist_map_np = np.clip(distance_transform_edt(1 - occ_np).astype(np.float32) / max(H, W), 0, 1)
    dist_map = torch.from_numpy(dist_map_np).to(device).unsqueeze(0).unsqueeze(0)

    for epoch in range(epochs):
        target_img = images[epoch % 8].reshape(-1, 3)
        mask = (torch.rand(H * W, device=device) < 0.3)
        optimizer.zero_grad()

        N_rays = rays_o.shape[0]; cs = 256; loss_r_total = 0.0
        for cs0 in range(0, N_rays, cs):
            cs1 = min(cs0 + cs, N_rays); cw = (cs1 - cs0) / N_rays
            with amp_ctx:
                pc, _, _ = volume_render_2d(rays_o[cs0:cs1], rays_d[cs0:cs1],
                    atoms, metric_field, num_samples=32, near=0., far=1.)[:3]
                lr = l1_loss(pc, target_img[cs0:cs1])
            scaler.scale(lr * cw).backward()
            loss_r_total += lr.detach().item() * cw

        with amp_ctx:
            lm = metric_smoothness_loss(metric_field) * 0.01
            lv = occupancy_coupling_loss(metric_field, occupancy) * w_vol
            lt = trace_contrast_loss(metric_field, occupancy) * w_tc if w_tc > 0 else torch.tensor(0., device=device)

            mus = torch.stack([a.position for a in atoms])
            sts = torch.stack([a.state for a in atoms])
            gw, _ = compute_geodesic_neighbors(mus, metric_field, k=5)
            sp = state_propagation(sts, gw, alpha=0.3)
            ls = self_organization_loss(mus, sp, metric_field, K=5) * w_selforg

            mi = mask.nonzero(as_tuple=False).squeeze(-1)
            if mi.numel() > 0:
                mpx = torch.stack([(mi % W).float() / W, (mi // W).float() / H], dim=-1)
                lp = masked_prediction_loss(mus, sp, metric_field, mpx,
                    target_img[mi], torch.stack([a._color for a in atoms]),
                    state_decoder=state_decoder) * 1.0
            else:
                lp = torch.tensor(0., device=device)

            ap = torch.stack([a.position for a in atoms])
            g = ap.unsqueeze(0).unsqueeze(2) * 2 - 1
            pd = F.grid_sample(dist_map, g, mode='bilinear', padding_mode='border',
                               align_corners=False).squeeze()
            if pd.dim() == 0: pd = pd.unsqueeze(0)
            lpos = pd.mean() * 5.0

            total = lm + lv + lt + ls + lp + lpos

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for pg in optimizer.param_groups for p in pg['params']], 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()

    # Evaluate
    with torch.no_grad():
        tr = metric_field.trace()
        ti = (tr * occupancy).sum() / occupancy.sum().clamp(min=1)
        to = (tr * (1 - occupancy)).sum() / (1 - occupancy).sum().clamp(min=1)

    feats = np.stack([a._state.detach().cpu().numpy() for a in atoms])
    mus_np = np.stack([a.position.detach().cpu().numpy() for a in atoms])
    gt = np.full(mus_np.shape[0], -1, dtype=int)
    for i, mu in enumerate(mus_np):
        px = np.clip(int(mu[0] * W), 0, W - 1)
        py = np.clip(int(mu[1] * H), 0, H - 1)
        for k in range(masks_np.shape[-1]):
            if masks_np[0, py, px, k] > 0.5: gt[i] = k; break
    valid = gt >= 0
    ari = adjusted_rand_score(gt[valid], KMeans(2, random_state=42, n_init=10).fit_predict(feats[valid])) if valid.sum() > 2 else 0

    return {'label': label, 'ari': ari, 'trace_in': ti.item(), 'trace_out': to.item(),
            'sep_ratio': to.item() / max(ti.item(), 1e-6), 'render': loss_r_total}


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    configs = [
        dict(w_vol=0.2, w_tc=0.0, w_selforg=0.5, label="A: baseline"),
        dict(w_vol=1.0, w_tc=0.0, w_selforg=0.5, label="B: w_vol=1.0"),
        dict(w_vol=0.2, w_tc=2.0, w_selforg=0.5, label="C: +trace_contrast"),
        dict(w_vol=1.0, w_tc=2.0, w_selforg=1.0, label="D: vol+tc+so boost"),
    ]

    print(f"{'Config':<25} {'ARI':>6} {'Trace_in':>9} {'Trace_out':>10} {'Sep_ratio':>10} {'Render':>7}")
    print("-" * 72)

    results = []
    for cfg in configs:
        r = run_config(32, 32, 50, 400, 42, device, **cfg)
        results.append(r)
        print(f"{r['label']:<25} {r['ari']:6.3f} {r['trace_in']:9.3f} {r['trace_out']:10.3f} "
              f"{r['sep_ratio']:10.3f} {r['render']:7.3f}")

    best = max(results, key=lambda x: x['ari'])
    print(f"\n→ Best config: {best['label']} (ARI={best['ari']:.3f}, sep={best['sep_ratio']:.2f})")
