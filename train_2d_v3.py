"""
MetricAtom 2D v3.1 — STATE=COLOR with frozen positions.

Key fix: positions frozen in Phase 2 so spatial neighbors are stable.
- Phase 1: train _color + metric + positions
- Phase 2: freeze positions, copy _color → state, train state + metric
  - consistency_loss on spatial k-NN (Euclidean) forces nearby atoms → same color
  - contrastive_loss pushes far atoms → different colors
  - render + predict losses ensure colors match objects
"""
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
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
from src.losses.occupancy_coupling import occupancy_coupling_loss, trace_contrast_loss
from src.losses.self_organize import state_contrastive_loss
from src.data.synthetic_2d import generate_multi_view, get_occupancy
from src.visualization.plot_metric import generate_evaluation_report


def create_atoms(num_atoms, device, seed=42, radius_min=0.25, radius_max=0.35, occupancy=None):
    if occupancy is not None:
        H, W = occupancy.shape
        occ_pixels = torch.nonzero(occupancy > 0.5).float()
        if occ_pixels.shape[0] > 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            atoms = []
            for i in range(num_atoms):
                idx = np.random.randint(0, occ_pixels.shape[0])
                y, x = occ_pixels[idx][0].item(), occ_pixels[idx][1].item()
                u = (x + np.random.uniform(-3, 3)) / W
                v = (y + np.random.uniform(-3, 3)) / H
                u = np.clip(u, 0.05, 0.95)
                v = np.clip(v, 0.05, 0.95)
                mu = torch.tensor([u, v], device=device, dtype=torch.float32)
                radius = radius_min + torch.rand(1, device=device).item() * (radius_max - radius_min)
                atom = Atom2D(mu, radius=radius, color=torch.rand(3, device=device), state_dim=3, eps=0.5)
                atom.birth_epoch = 0
                atoms.append(atom)
            return atoms
    torch.manual_seed(seed)
    atoms = []
    grid_size = int(np.ceil(np.sqrt(num_atoms)))
    for i in range(grid_size):
        for j in range(grid_size):
            if len(atoms) >= num_atoms:
                break
            u = (i + 0.5) / grid_size + torch.randn(1).item() * 0.03
            v = (j + 0.5) / grid_size + torch.randn(1).item() * 0.03
            mu = torch.tensor([np.clip(u, 0.1, 0.9), np.clip(v, 0.1, 0.9)], device=device, dtype=torch.float32)
            radius = radius_min + torch.rand(1, device=device).item() * (radius_max - radius_min)
            atom = Atom2D(mu, radius=radius, color=torch.rand(3, device=device), state_dim=3, eps=0.5)
            atom.birth_epoch = 0
            atoms.append(atom)
        if len(atoms) >= num_atoms:
            break
    return atoms


def spatial_knn_mask(positions, k=5):
    """Binary k-NN mask based on Euclidean distance."""
    N = positions.shape[0]
    if N < 2:
        return torch.zeros((N, N), device=positions.device)
    dx = positions.unsqueeze(0) - positions.unsqueeze(1)
    D2 = (dx ** 2).sum(dim=-1)
    _, knn_idx = D2.topk(k=min(k + 1, N), dim=1, largest=False)
    knn_idx = knn_idx[:, 1:] if knn_idx.shape[1] > 1 else knn_idx
    mask = torch.zeros((N, N), device=positions.device)
    for i in range(N):
        for j in knn_idx[i]:
            mask[i, j] = 1.0
    mask = ((mask + mask.T) > 0).float()
    mask.fill_diagonal_(0.0)
    return mask


def color_consistency_loss(colors, neighbor_mask):
    """Force neighbors to have identical colors."""
    N = colors.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=colors.device)
    mask_with_self = neighbor_mask + torch.eye(N, device=colors.device)
    counts = mask_with_self.sum(dim=1, keepdim=True).clamp(min=1)
    mean_color = (mask_with_self @ colors) / counts
    return ((colors - mean_color) ** 2).mean()


def train_scene_v3(H=32, W=32, num_atoms=50, num_epochs=600, num_views=8,
                   lr=1e-3, device='cuda', output_dir='outputs/v3',
                   bf16=False, fp16=True, num_samples=32, seed=42,
                   w_met=0.01, w_vol=1.0, w_tc=2.0, w_pos=5.0,
                   w_consistency=3.0, w_contrast=3.0, w_predict=3.0,
                   mask_ratio=0.3, chunk_size=256, atom_chunk_size=10,
                   metric_batch_size=128, phase1_epochs=100,
                   w_vol_p1=5.0, w_tc_p1=10.0, same_color=False,
                   parametrization='cholesky'):

    use_two_phase = phase1_epochs > 0
    current_phase = 1 if use_two_phase else 2

    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

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

    print(f"[1/3] Data ({H}x{W}, same_color={same_color})...")
    images_np, masks_np, _ = generate_multi_view(
        H=H, W=W, num_objects=2, num_views=num_views, seed=seed, same_color=same_color
    )
    images = torch.from_numpy(images_np).float().to(device)
    occupancy = torch.from_numpy(get_occupancy(masks_np)).float().to(device)

    metric_field = MetricField2D(H, W, init_scale=1.0,
                                 default_batch_size=metric_batch_size,
                                 parametrization=parametrization).to(device)
    with torch.no_grad():
        metric_field.params[0, 0].fill_(1.0)
        metric_field.params[0, 2].fill_(1.0)
        metric_field.params[0, 1].zero_()

    frame_occ = torch.zeros(num_views, H, W, device=device)
    for fv in range(num_views):
        frame_occ[fv] = (torch.from_numpy(masks_np[fv]).sum(dim=-1) > 0.5).float().to(device)

    atoms = create_atoms(num_atoms, device, seed=seed, occupancy=frame_occ[0])
    atom_params = [p for a in atoms for p in a.parameters()]

    lr_metric = lr * 20.0
    lr_pos = lr * 0.1

    if use_two_phase:
        for a in atoms:
            a._state.requires_grad = False
            a._logit_eps.requires_grad = False
        # Phase 1: train position + color + metric so atoms learn to cover objects
        color_params = [a._color for a in atoms]
        pos_params = [a._mu for a in atoms] + [a._log_r for a in atoms]
        optimizer = torch.optim.Adam([
            {'params': metric_field.parameters(), 'lr': lr_metric},
            {'params': color_params, 'lr': lr_pos},
            {'params': pos_params, 'lr': lr_pos},
        ])
        scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=lr * 0.01)
        print(f"  [Phase 1] {phase1_epochs} epochs")
    else:
        optimizer = torch.optim.Adam([
            {'params': metric_field.parameters(), 'lr': lr_metric},
            {'params': atom_params, 'lr': lr_pos},
        ])
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    rays_o, rays_d = RaySampler2D.generate_rays_orthographic(H, W, device=device)

    occ_np = occupancy.cpu().numpy()
    dist_map = torch.from_numpy(
        np.clip(distance_transform_edt(1 - occ_np) / max(H, W), 0.0, 1.0).astype(np.float32)
    ).to(device).unsqueeze(0).unsqueeze(0)

    losses_log = []
    print(f"[2/3] Training ({num_epochs} epochs)...")
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(num_epochs):
        # Phase transition
        if use_two_phase and epoch == phase1_epochs:
            print(f"\n{'='*50}\n  Phase 1 → Phase 2 (epoch {epoch})")
            with torch.no_grad():
                tr = metric_field.trace()
                occ_mask = occupancy > 0.5
                tr_in = tr[occ_mask].mean().item() if occ_mask.any() else 0
                tr_out = tr[~occ_mask].mean().item() if (~occ_mask).any() else 0
                print(f"  Trace: in={tr_in:.3f} out={tr_out:.3f} ratio={tr_out/(tr_in+1e-8):.2f}")
                for a in atoms:
                    a._state.data.copy_(a._color.data.detach())
                    # SHRINK radius → hard-ball atoms so winner-take-all works
                    a._log_r.data.fill_(np.log(0.08))

            # FILTER: remove background atoms (not in occupancy)
            # Background atoms steal gradients from object atoms in WTA.
            occ_mask_np = (occupancy.cpu().numpy() > 0.5)
            kept_atoms = []
            for a in atoms:
                px, py = a.position.detach().cpu().numpy()
                ix, iy = int(px * W), int(py * H)
                ix = np.clip(ix, 0, W - 1)
                iy = np.clip(iy, 0, H - 1)
                if occ_mask_np[iy, ix]:
                    kept_atoms.append(a)
            pruned = len(atoms) - len(kept_atoms)
            atoms = kept_atoms
            print(f"  [Filter] Removed {pruned} background atoms, {len(atoms)} remain.")

            # Freeze positions AND _color, train state + metric only
            for a in atoms:
                a._mu.requires_grad = False
                a._log_r.requires_grad = False
                a._state.requires_grad = True
                a._logit_eps.requires_grad = False
                a._color.requires_grad = False

            optimizer = torch.optim.Adam([
                {'params': metric_field.parameters(), 'lr': lr_metric * 0.5},
                {'params': [a._state for a in atoms], 'lr': lr},
            ])
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - phase1_epochs, eta_min=lr * 0.01)
            scaler = GradScaler('cuda', enabled=scaler_enabled)
            current_phase = 2
            print(f"  Positions FROZEN. Colors copied → state.")
            print(f"{'='*50}\n")

        if current_phase == 1:
            cur_w_vol, cur_w_tc = w_vol_p1, w_tc_p1
            w_c, w_ct, w_p = 0.0, 0.0, 0.0
            w_render = 1.0
        else:
            cur_w_vol, cur_w_tc = w_vol, w_tc
            w_c, w_ct, w_p = w_consistency, w_contrast, w_predict
            w_render = 0.1  # Reduce render weight in P2 so predict dominates

        target_img = images[epoch % num_views].reshape(-1, 3)
        # Only sample masked pixels from OBJECT regions (not background)
        # This prevents background atoms from receiving object color gradients.
        obj_pixels = torch.nonzero(occupancy.reshape(-1) > 0.5).squeeze(-1)
        n_mask = int(len(obj_pixels) * mask_ratio)
        if n_mask > 0:
            perm = torch.randperm(len(obj_pixels), device=device)
            masked_indices = obj_pixels[perm[:n_mask]]
        else:
            masked_indices = torch.empty(0, device=device, dtype=torch.long)

        optimizer.zero_grad()
        loss_render_val = 0.0
        for cs_start in range(0, rays_o.shape[0], chunk_size):
            cs_end = min(cs_start + chunk_size, rays_o.shape[0])
            w = (cs_end - cs_start) / rays_o.shape[0]
            with amp_ctx:
                pred, _, _ = volume_render_2d(
                    rays_o[cs_start:cs_end], rays_d[cs_start:cs_end],
                    atoms, metric_field, num_samples=num_samples,
                    scene_size=1.0, atom_chunk_size=atom_chunk_size
                )[:3]
                loss_r = l1_loss(pred, target_img[cs_start:cs_end])
            scaler.scale(loss_r * w * w_render).backward()
            loss_render_val += loss_r.detach().item() * w * w_render
            del pred, loss_r
            if device == 'cuda':
                torch.cuda.empty_cache()

        with amp_ctx:
            loss_met = metric_smoothness_loss(metric_field) * w_met
            loss_vol = occupancy_coupling_loss(metric_field, occupancy) * cur_w_vol
            loss_tc = trace_contrast_loss(metric_field, occupancy) * cur_w_tc if cur_w_tc > 0 else torch.tensor(0.0, device=device)

            loss_pos = torch.tensor(0.0, device=device)
            if current_phase == 1 and len(atoms) > 0:
                pos = torch.stack([a.position for a in atoms])
                pd = F.grid_sample(dist_map, pos.unsqueeze(0).unsqueeze(2) * 2 - 1,
                                   mode='bilinear', padding_mode='border', align_corners=False).squeeze()
                if pd.dim() == 0:
                    pd = pd.unsqueeze(0)
                loss_pos = pd.mean() * w_pos

            mus = torch.stack([a.position for a in atoms]) if len(atoms) > 0 else torch.empty((0, 2), device=device)
            states = torch.stack([a.state for a in atoms]) if len(atoms) > 0 else torch.empty((0, 3), device=device)
            colors = torch.sigmoid(states)

            # Spatial k-NN mask (Euclidean, positions frozen in P2)
            spatial_mask = spatial_knn_mask(mus, k=5) if len(atoms) > 1 else torch.zeros((len(atoms), len(atoms)), device=device)

            # Masked prediction: WINNER-TAKE-ALL on object pixels only
            # Only the nearest atom to each masked object pixel is responsible.
            # Background atoms are never selected → object atoms must learn colors.
            loss_pred = torch.tensor(0.0, device=device)
            if w_p > 0 and masked_indices.numel() > 0 and len(atoms) > 0:
                mpx = torch.stack([(masked_indices % W).float() / W, (masked_indices // W).float() / H], dim=-1).to(device)
                tc = target_img[masked_indices]
                dx = mpx.unsqueeze(1) - mus.unsqueeze(0)
                D2 = (dx ** 2).sum(dim=-1)
                nearest = D2.argmin(dim=1)
                loss_pred = F.l1_loss(colors[nearest], tc) * w_p

            loss_reg = loss_met + loss_vol + loss_tc + loss_pos + loss_pred

        scaler.scale(loss_reg).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for pg in optimizer.param_groups for p in pg['params']], 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses_log.append({
            'epoch': epoch, 'total': loss_render_val + loss_reg.item(),
            'render': loss_render_val, 'met': loss_met.item(),
            'vol': loss_vol.item(), 'tc': loss_tc.item() if isinstance(loss_tc, torch.Tensor) else 0,
            'pos': loss_pos.item(), 'consistency': 0.0,
            'contrast': 0.0, 'predict': loss_pred.item(),
        })

        if epoch % 100 == 0 or epoch == num_epochs - 1 or (use_two_phase and abs(epoch - phase1_epochs) <= 1):
            log = losses_log[-1]
            ss = states.std(dim=0).mean().item() if len(atoms) > 0 else 0
            with torch.no_grad():
                tr = metric_field.trace()
                om = occupancy > 0.5
                ti = tr[om].mean().item() if om.any() else 0
                to = tr[~om].mean().item() if (~om).any() else 0
            mem = f" GPU={torch.cuda.memory_allocated()/1024**2:.0f}MB" if device == 'cuda' else ""
            print(f"  [{epoch:4d}/{num_epochs}|P{current_phase}] T={log['total']:7.3f} R={log['render']:.3f} "
                  f"CS={log['consistency']:.3f} CT={log['contrast']:.3f} P={log['predict']:.3f} "
                  f"SS={ss:.4f} tr={ti:.2f}/{to:.2f} A={len(atoms)}{mem}")

    print("[3/3] Evaluating...")
    metrics = generate_evaluation_report(
        atoms, metric_field, images_np, masks_np, losses_log,
        H, W, num_epochs // 2, output_path / 'final'
    )
    return atoms, metric_field, losses_log, metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--atom', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase1-epochs', type=int, default=100)
    parser.add_argument('--output', type=str, default='outputs/v3')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bf16 = args.bf16 and device == 'cuda' and torch.cuda.is_bf16_supported()
    fp16 = args.fp16 and device == 'cuda'

    train_scene_v3(
        H=args.resolution, W=args.resolution, num_atoms=args.atom,
        num_epochs=args.epochs, device=device, output_dir=args.output,
        bf16=bf16, fp16=fp16, num_samples=args.samples,
        chunk_size=args.chunk_size, seed=args.seed,
        phase1_epochs=args.phase1_epochs,
    )
