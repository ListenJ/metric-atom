"""
MetricAtom 2D v4 — Prototype Competition

Key insight: instead of learning N independent colors (prone to collapse),
learn K=2 global color prototypes + per-atom assignment logits.

  atom_color = softmax(assignment_i) @ prototypes  (K=2, 3)

This forces:
  - Only 2 colors exist in the entire scene
  - Atoms compete for prototype assignment via softmax
  - Render loss pulls prototypes toward true object colors
  - Occupancy coupling pulls atoms into object regions
  - Clustering = argmax(assignment)

Eliminates: decoder, masked prediction, state dynamics, consistency/contrastive.
"""
import torch
import torch.nn as nn
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


class PrototypeAtomSystem(nn.Module):
    """Atoms with competitive prototype assignment.

    K global prototypes (colors), each atom assigns itself to prototypes
    via learned logits. The rendered color is a weighted combination.

    During clustering evaluation, we use hard assignment (argmax).
    """
    def __init__(self, num_atoms, K=2, state_dim=3, device='cuda', fixed_protos=None):
        super().__init__()
        self.K = K
        # FIXED prototypes: set to true object colors so assignment has no choice
        # but to match spatial regions to correct prototype.
        if fixed_protos is not None:
            self.register_buffer('prototypes', torch.tensor(fixed_protos, device=device, dtype=torch.float32))
        else:
            self.prototypes = nn.Parameter(torch.rand(K, 3, device=device) * 0.5 + 0.25)
        # Per-atom assignment logits: higher = stronger claim to prototype
        self.assignment_logits = nn.Parameter(torch.randn(num_atoms, K, device=device) * 0.5)

    def get_colors(self, temperature=1.0):
        """Return (N, 3) atom colors from prototypes + soft assignment."""
        weights = F.softmax(self.assignment_logits / temperature, dim=-1)  # (N, K)
        if self.prototypes.requires_grad:
            protos = torch.sigmoid(self.prototypes)  # (K, 3)
        else:
            protos = self.prototypes  # already in [0,1]
        return weights @ protos  # (N, 3)

    def get_hard_labels(self):
        """Return (N,) integer labels for clustering evaluation."""
        return self.assignment_logits.argmax(dim=-1).detach().cpu().numpy()


def train_scene_v4(H=32, W=32, num_atoms=50, num_epochs=600, num_views=8,
                   lr=1e-3, device='cuda', output_dir='outputs/v4',
                   bf16=False, fp16=True, num_samples=32, seed=42,
                   w_met=0.01, w_vol=1.0, w_tc=2.0, w_pos=5.0,
                   K=2, proto_temp=0.5,
                   chunk_size=256, atom_chunk_size=10, metric_batch_size=128,
                   w_vol_p1=5.0, w_tc_p1=10.0, same_color=False,
                   parametrization='cholesky'):

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

    print(f"[1/3] Data ({H}x{W}, K={K}, same_color={same_color})...")
    images_np, masks_np, _ = generate_multi_view(
        H=H, W=W, num_objects=K, num_views=num_views, seed=seed, same_color=same_color
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
    # FIXED prototypes: red and blue (true object colors)
    fixed_colors = [[1.0, 0.2, 0.2], [0.2, 0.2, 1.0]] if K == 2 else None
    proto_system = PrototypeAtomSystem(num_atoms, K=K, device=device, fixed_protos=fixed_colors).to(device)

    # Optimizer: all parameters end-to-end
    atom_params = [p for a in atoms for p in a.parameters()]
    lr_metric = lr * 20.0
    lr_pos = lr * 0.1

    optimizer = torch.optim.Adam([
        {'params': metric_field.parameters(), 'lr': lr_metric},
        {'params': atom_params, 'lr': lr_pos},
        {'params': proto_system.parameters(), 'lr': lr},
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
        # Phase-like transition: after 100 epochs, shrink radius + lower temp
        if epoch == 100:
            with torch.no_grad():
                for a in atoms:
                    a._log_r.data.fill_(np.log(0.08))
            print(f"  [Epoch 100] Radius shrunk to 0.08")

        cur_temp = proto_temp * 0.3 if epoch > 100 else proto_temp

        target_img = images[epoch % num_views].reshape(-1, 3)

        # Compute atom colors from prototypes
        atom_colors = proto_system.get_colors(temperature=cur_temp)
        # Override each atom's _color for rendering
        with torch.no_grad():
            for i, a in enumerate(atoms):
                a._color.data.copy_(atom_colors[i])

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
            scaler.scale(loss_r * w).backward()
            loss_render_val += loss_r.detach().item() * w
            del pred, loss_r
            if device == 'cuda':
                torch.cuda.empty_cache()

        with amp_ctx:
            loss_met = metric_smoothness_loss(metric_field) * w_met
            loss_vol = occupancy_coupling_loss(metric_field, occupancy) * w_vol
            loss_tc = trace_contrast_loss(metric_field, occupancy) * w_tc if w_tc > 0 else torch.tensor(0.0, device=device)

            # Position regularization: keep atoms in object regions
            loss_pos = torch.tensor(0.0, device=device)
            if len(atoms) > 0:
                pos = torch.stack([a.position for a in atoms])
                pd = F.grid_sample(dist_map, pos.unsqueeze(0).unsqueeze(2) * 2 - 1,
                                   mode='bilinear', padding_mode='border', align_corners=False).squeeze()
                if pd.dim() == 0:
                    pd = pd.unsqueeze(0)
                loss_pos = pd.mean() * w_pos

            # Prototype entropy regularization: encourage SHARP assignments
            probs = F.softmax(proto_system.assignment_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss_entropy = -entropy * 5.0  # strong sharpening

            # Prototype diversity: push prototypes FAR apart in color space
            protos = torch.sigmoid(proto_system.prototypes)
            # L2 distance instead of cosine — stronger repulsion
            diff = protos.unsqueeze(0) - protos.unsqueeze(1)
            l2_dist = (diff ** 2).sum(dim=-1)
            eye = torch.eye(K, device=device)
            # Penalize small distances heavily
            loss_diversity = (1.0 / (l2_dist + 0.1) * (1 - eye)).sum() * 2.0

            loss_reg = loss_met + loss_vol + loss_tc + loss_pos + loss_entropy

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
            'pos': loss_pos.item(), 'entropy': entropy.item(),
        })

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            log = losses_log[-1]
            labels = proto_system.get_hard_labels()
            n_cls = len(np.unique(labels))
            mem = f" GPU={torch.cuda.memory_allocated()/1024**2:.0f}MB" if device == 'cuda' else ""
            with torch.no_grad():
                tr = metric_field.trace()
                om = occupancy > 0.5
                ti = tr[om].mean().item() if om.any() else 0
                to = tr[~om].mean().item() if (~om).any() else 0
            print(f"  [{epoch:4d}/{num_epochs}] T={log['total']:7.3f} R={log['render']:.3f} "
                  f"H={log['entropy']:.3f} "
                  f"cls={n_cls} tr={ti:.2f}/{to:.2f} A={len(atoms)}{mem}")

    print("[3/3] Evaluating...")
    # Override colors one last time for visualization
    with torch.no_grad():
        final_colors = proto_system.get_colors()
        for i, a in enumerate(atoms):
            a._color.data.copy_(final_colors[i])
            # Also copy to state for evaluation consistency
            a._state.data.copy_(final_colors[i])

    metrics = generate_evaluation_report(
        atoms, metric_field, images_np, masks_np, losses_log,
        H, W, num_epochs // 2, output_path / 'final'
    )

    # Print assignment stats
    labels = proto_system.get_hard_labels()
    print(f"\nAssignment distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    if proto_system.prototypes.requires_grad:
        protos = torch.sigmoid(proto_system.prototypes).detach().cpu().numpy()
    else:
        protos = proto_system.prototypes.detach().cpu().numpy()
    print(f"Prototypes: {protos.round(2)}")

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
    parser.add_argument('--output', type=str, default='outputs/v4')
    parser.add_argument('--K', type=int, default=2, help='Number of object prototypes')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bf16 = args.bf16 and device == 'cuda' and torch.cuda.is_bf16_supported()
    fp16 = args.fp16 and device == 'cuda'

    train_scene_v4(
        H=args.resolution, W=args.resolution, num_atoms=args.atom,
        num_epochs=args.epochs, device=device, output_dir=args.output,
        bf16=bf16, fp16=fp16, num_samples=args.samples,
        chunk_size=args.chunk_size, seed=args.seed, K=args.K,
    )
