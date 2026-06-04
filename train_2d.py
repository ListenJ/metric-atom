"""
MetricAtom 2D — Self-Organizing Atoms (v2.0)

No Phase 2. No KMeans. No external clustering.
Atoms self-organize through:
  1. Reconstruction (pixel matching)
  2. State propagation (message passing over geodesic neighborhoods)
  3. Self-organization force (similar states attract in metric space)
  4. Masked prediction (atoms vote on occluded pixel colors)

Config: 32×32, 50 atoms, FP16, <2.5GB VRAM (3050 Ti local)
"""
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from pathlib import Path
from contextlib import nullcontext
from torch.amp import GradScaler
from scipy.ndimage import distance_transform_edt

from src.atoms.atom_2d import Atom2D
from src.geometry.metric_field import MetricField2D
from src.rendering.ray_sampler import RaySampler2D
from src.rendering.volume_renderer_2d import volume_render_2d
from src.losses.reconstruction import l1_loss
from src.losses.metric_regularizer import metric_smoothness_loss
from src.losses.occupancy_coupling import occupancy_coupling_loss
from src.losses.diffusion import compute_geodesic_affinity, feature_diffusion
from src.losses.self_organize import (
    compute_geodesic_neighbors,
    state_propagation,
    self_organization_loss,
    masked_prediction_loss,
)
from src.data.synthetic_2d import generate_multi_view, get_occupancy
from src.visualization.plot_metric import generate_evaluation_report


def create_atoms(num_atoms, device, seed=42, radius_min=0.25, radius_max=0.35, occupancy=None):
    """Create atoms initialized in object regions."""
    if occupancy is not None:
        H, W = occupancy.shape
        occ_pixels = torch.nonzero(occupancy > 0.5).float()
        if occ_pixels.shape[0] > 0:
            torch.manual_seed(seed)
            atoms = []
            np.random.seed(seed)
            for i in range(num_atoms):
                idx = np.random.randint(0, occ_pixels.shape[0])
                y, x = occ_pixels[idx][0].item(), occ_pixels[idx][1].item()
                u = (x + np.random.uniform(-3, 3)) / W
                v = (y + np.random.uniform(-3, 3)) / H
                u = np.clip(u, 0.05, 0.95)
                v = np.clip(v, 0.05, 0.95)
                mu = torch.tensor([u, v], device=device, dtype=torch.float32)
                radius = radius_min + torch.rand(1, device=device).item() * (radius_max - radius_min)
                color = torch.rand(3, device=device)
                atom = Atom2D(mu, radius=radius, color=color, state_dim=16, eps=0.5)
                atom.birth_epoch = 0
                atoms.append(atom)
            return atoms

    # Grid fallback
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
            radius = radius_min + torch.rand(1, device=device).item() * (radius_max - radius_min)
            color = torch.rand(3, device=device)
            atom = Atom2D(mu, radius=radius, color=color, state_dim=16, eps=0.5)
            atom.birth_epoch = 0
            atoms.append(atom)
        if len(atoms) >= num_atoms:
            break
    return atoms


def generate_random_mask(H, W, mask_ratio=0.3, device='cuda'):
    """Generate random binary mask for masked prediction."""
    mask = torch.rand(H * W, device=device) < mask_ratio
    return mask


def train_scene(H=32, W=32, num_atoms=50, num_epochs=600, num_views=8,
                lr=1e-3, device='cuda', output_dir='outputs/selforg',
                bf16=False, fp16=True, num_samples=32, seed=42,
                w_met=0.01, w_vol=0.2, w_pos=5.0,
                w_selforg=0.5, w_predict=1.0,
                state_alpha=0.3, mask_ratio=0.3, diff_K=5):
    """Self-organizing atom training — single phase, no external clustering."""

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Mixed precision
    amp_ctx = nullcontext()
    scaler_enabled = False
    if device == 'cuda' and (bf16 or fp16):
        if bf16 and torch.cuda.is_bf16_supported():
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            scaler_enabled = True
        elif fp16:
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
            scaler_enabled = True
    scaler = GradScaler('cuda', enabled=scaler_enabled)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scene_size = 1.0

    print(f"[1/4] Data ({H}x{W}, {num_views} views)...")
    images_np, masks_np, transforms = generate_multi_view(
        H=H, W=W, num_objects=2, num_views=num_views, seed=seed
    )
    images = torch.from_numpy(images_np).float().to(device)
    masks = torch.from_numpy(masks_np).float().to(device)
    occupancy = torch.from_numpy(get_occupancy(masks_np)).float().to(device)

    print(f"[2/4] Init metric field + {num_atoms} atoms...")
    metric_field = MetricField2D(H, W, init_scale=1.0).to(device)

    frame_occupancy = torch.zeros(num_views, H, W, device=device)
    for fv in range(num_views):
        frame_occupancy[fv] = (masks[fv].sum(dim=-1) > 0.5).float()

    atoms = create_atoms(num_atoms, device, seed=seed,
                         occupancy=frame_occupancy[0])

    # Shared state → color decoder (forces states to encode visual info)
    state_decoder = torch.nn.Linear(16, 3).to(device)

    optimizer = torch.optim.Adam([
        {'params': metric_field.parameters(), 'lr': lr},
        {'params': [p for a in atoms for p in a.parameters()], 'lr': lr * 3},
        {'params': state_decoder.parameters(), 'lr': lr},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    print(f"[3/4] Precompute rays...")
    rays_o, rays_d = RaySampler2D.generate_rays_orthographic(
        H, W, scene_size=scene_size, device=device
    )

    # Distance map for position regularization
    occ_np = occupancy.cpu().numpy()
    dist_to_obj = distance_transform_edt(1 - occ_np).astype(np.float32)
    dist_to_obj = np.clip(dist_to_obj / max(H, W), 0.0, 1.0)
    dist_map = torch.from_numpy(dist_to_obj).to(device).unsqueeze(0).unsqueeze(0)

    losses_log = []
    atom_birth_epochs = {id(a): 0 for a in atoms}
    prune_interval = max(num_epochs // 15, 20)
    seed_freq = 15

    print(f"[4/4] Training ({num_epochs} epochs)...")

    for epoch in range(num_epochs):
        frame_idx = epoch % num_views
        target_img = images[frame_idx].reshape(-1, 3)

        do_prune = (epoch > 0 and epoch % prune_interval == 0)
        do_seed = (epoch > 0 and epoch % seed_freq == 0 and epoch >= 50)

        # Generate random mask for this frame
        mask = generate_random_mask(H, W, mask_ratio, device)

        optimizer.zero_grad()

        # ── Chunked rendering ──
        N_rays = rays_o.shape[0]
        cs = 256
        num_chunks = (N_rays + cs - 1) // cs
        loss_render_val = 0.0
        pred_color_parts = []

        for chunk_start in range(0, N_rays, cs):
            chunk_end = min(chunk_start + cs, N_rays)
            n_chunk = chunk_end - chunk_start
            chunk_weight = n_chunk / N_rays

            with amp_ctx:
                render_result = volume_render_2d(
                    rays_o[chunk_start:chunk_end], rays_d[chunk_start:chunk_end],
                    atoms, metric_field,
                    num_samples=num_samples, near=0.0, far=scene_size,
                    scene_size=scene_size, return_per_atom=False
                )
                pred_color_c, _, _ = render_result[:3]
                pred_color_parts.append(pred_color_c.detach())

                loss_render_c = l1_loss(pred_color_c, target_img[chunk_start:chunk_end])

            scaler.scale(loss_render_c * chunk_weight).backward()
            loss_render_val += loss_render_c.detach().item() * chunk_weight

        pred_color = torch.cat(pred_color_parts, dim=0)

        # ── Regularization ──
        with amp_ctx:
            loss_met = metric_smoothness_loss(metric_field) * w_met
            loss_vol = occupancy_coupling_loss(metric_field, occupancy) * w_vol

            loss_pos_t = torch.tensor(0.0, device=device)
            if w_pos > 0 and len(atoms) > 0:
                atom_positions = torch.stack([a.position for a in atoms])
                grid = atom_positions.unsqueeze(0).unsqueeze(2) * 2 - 1
                pos_dist = F.grid_sample(dist_map, grid, mode='bilinear',
                                         padding_mode='border', align_corners=False)
                pos_dist = pos_dist.squeeze()
                if pos_dist.dim() == 0:
                    pos_dist = pos_dist.unsqueeze(0)
                loss_pos_t = pos_dist.mean() * w_pos

            # ── Self-organization ──
            mus = torch.stack([a.position for a in atoms])
            states = torch.stack([a.state for a in atoms])

            # State propagation (message passing)
            geo_weights, _ = compute_geodesic_neighbors(mus, metric_field, k=5)
            states_prop = state_propagation(states, geo_weights, alpha=state_alpha)

            # Self-organization loss
            loss_so = self_organization_loss(mus, states_prop, metric_field) * w_selforg

            # Masked prediction loss
            masked_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            if masked_indices.numel() > 0:
                masked_px = torch.stack([
                    (masked_indices % W).float() / W,
                    (masked_indices // W).float() / H,
                ], dim=-1).to(device)
                target_c = target_img[masked_indices]
                atom_colors = torch.stack([a._color for a in atoms])
                loss_pred = masked_prediction_loss(
                    mus, states_prop, metric_field,
                    masked_px, target_c, atom_colors,
                    state_decoder=state_decoder
                ) * w_predict
            else:
                loss_pred = torch.tensor(0.0, device=device)

            # Feature diffusion (smoothing for state visualization only)
            diff_val = 0.0
            if diff_K > 0:
                A = compute_geodesic_affinity(mus, metric_field, K=diff_K)
                states_diff = feature_diffusion(states_prop, A, alpha=0.5, T=2)
                diff_val = ((states_diff - states_prop) ** 2).mean().item()

            loss_reg = loss_met + loss_vol + loss_pos_t + loss_so + loss_pred

        scaler.scale(loss_reg).backward()

        # Optimizer step
        scaler.unscale_(optimizer)
        all_params = [p for pg in optimizer.param_groups for p in pg['params']]
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ── Pruning & seeding (stub — keep all atoms for now) ──
        # To be implemented in future iteration

        # ── Logging ──
        losses_log.append({
            'epoch': epoch,
            'total': loss_render_val + loss_reg.item(),
            'render': loss_render_val,
            'met': loss_met.item(),
            'vol': loss_vol.item(),
            'selforg': loss_so.item(),
            'predict': loss_pred.item(),
            'pos': loss_pos_t.item(),
            'diff': diff_val,
        })

        if epoch % 200 == 0 or epoch == num_epochs - 1:
            log = losses_log[-1]
            state_std = states.std(dim=0).mean().item() if len(atoms) > 0 else 0
            print(f"  [{epoch:4d}/{num_epochs}] T={log['total']:7.3f} R={log['render']:.3f} "
                  f"M={log['met']:.3f} V={log['vol']:.3f} "
                  f"O={log['selforg']:.3f} P={log['predict']:.3f} "
                  f"SS={state_std:.4f} A={len(atoms)}")

    print(f"[Done] Saving and evaluating...")
    metrics = generate_evaluation_report(
        atoms, metric_field, images_np, masks_np, losses_log,
        H, W, num_epochs // 2, output_path / 'final'
    )
    return atoms, metric_field, losses_log, metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Self-Organizing Atoms 2D')
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--atom', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--output', type=str, default='outputs/selforg_32x32')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--w-selforg', type=float, default=0.5)
    parser.add_argument('--w-predict', type=float, default=1.0)
    parser.add_argument('--state-alpha', type=float, default=0.3)
    parser.add_argument('--mask-ratio', type=float, default=0.3)
    args = parser.parse_args()

    H = W = args.resolution
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16 = args.fp16 and device == 'cuda'
    bf16 = args.bf16 and device == 'cuda' and torch.cuda.is_bf16_supported()

    print(f"Self-Organizing Atoms | {H}x{W} | {args.atom} atoms | {args.epochs} epochs")
    print(f"FP16: {fp16} | BF16: {bf16} | Device: {device}")

    train_scene(
        H=H, W=W, num_atoms=args.atom, num_epochs=args.epochs,
        lr=args.lr, device=device, output_dir=args.output,
        bf16=bf16, fp16=fp16, num_samples=args.samples, seed=args.seed,
        w_selforg=args.w_selforg, w_predict=args.w_predict,
        state_alpha=args.state_alpha, mask_ratio=args.mask_ratio,
    )
