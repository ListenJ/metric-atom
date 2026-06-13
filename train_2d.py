"""
MetricAtom 2D — Two-Phase Self-Organizing Atoms (v2.1)

Phase 1 (Metric Warmup):
  Freeze atoms, train metric field only.
  Strong w_vol + w_tc force trace separation (object vs background).

Phase 2 (Clustering):
  Unfreeze atoms, enable self-organization.
  Atoms cluster via geodesic self-organization on the pre-trained metric field.

Config: 64×64, 80 atoms, FP16, <4GB VRAM (3050 Ti local)
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
from src.atoms.residual_decoder import ResidualDecoder, create_optimal_decoder
from src.geometry.metric_field import MetricField2D
from src.rendering.ray_sampler import RaySampler2D
from src.rendering.volume_renderer_2d import volume_render_2d
from src.losses.reconstruction import l1_loss
from src.losses.metric_regularizer import metric_smoothness_loss
from src.losses.occupancy_coupling import occupancy_coupling_loss, trace_contrast_loss
from src.losses.diffusion import compute_geodesic_affinity, feature_diffusion
from src.losses.self_organize import (
    compute_geodesic_neighbors,
    state_propagation,
    self_organization_loss,
    masked_prediction_loss,
    state_contrastive_loss,
)
from src.losses.homeostatic import homeostatic_loss
from src.losses.axiom_diagnostics import AxiomMonitor
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


def reproject_atoms(atoms, occupancy, seed=42, noise_scale=2.0):
    """Re-project atoms onto object pixels at phase transition.

    Uses the canonical occupancy mask to scatter atoms back into object
    regions, giving Phase 2 a clean start with the pre-trained metric field.
    """
    H, W = occupancy.shape
    occ_pixels = torch.nonzero(occupancy > 0.5).float()
    if occ_pixels.shape[0] == 0:
        return
    np.random.seed(seed + 9999)
    for a in atoms:
        idx = np.random.randint(0, occ_pixels.shape[0])
        y, x = occ_pixels[idx][0].item(), occ_pixels[idx][1].item()
        u = (x + np.random.uniform(-noise_scale, noise_scale)) / W
        v = (y + np.random.uniform(-noise_scale, noise_scale)) / H
        u = np.clip(u, 0.05, 0.95)
        v = np.clip(v, 0.05, 0.95)
        with torch.no_grad():
            a.position.data.copy_(torch.tensor([u, v], device=a.position.device))


def prune_atoms_contrib_2d(contrib, atoms, birth_epochs, epoch,
                           threshold=0.1, min_atoms=30, protection=200):
    """2D atom pruning based on rendering contribution.
    
    Removes atoms with low rendering contribution, protecting young atoms.
    
    Args:
        contrib: (A,) tensor of per-atom contribution scores
        atoms: list of Atom2D objects
        birth_epochs: dict mapping id(atom) to birth epoch
        epoch: current epoch number
        threshold: quantile threshold for pruning (prune below this percentile)
        min_atoms: minimum number of atoms to keep
        protection: number of epochs before a new atom can be pruned
    
    Returns:
        (kept_atoms, new_birth_epochs) - pruned lists
    """
    N = len(atoms)
    if N <= min_atoms:
        return atoms, birth_epochs
    
    # Build protection mask
    protect = [(i, birth_epochs.get(id(a), 0)) for i, a in enumerate(atoms)]
    protect_mask = torch.tensor([(epoch - be >= protection) for _, be in protect],
                                device=contrib.device)
    protect_mask = protect_mask.to(contrib.dtype)
    
    # If too few unprotected atoms, skip pruning
    if protect_mask.sum() < min_atoms // 2:
        return atoms, birth_epochs
    
    # Adjust contribution by protection mask (protected atoms keep full contrib)
    contrib_adjusted = contrib * protect_mask
    
    # Compute threshold on unprotected atoms
    unprotected_contrib = contrib_adjusted[protect_mask > 0]
    if unprotected_contrib.numel() > 0:
        thresh = torch.quantile(unprotected_contrib, threshold)
    else:
        thresh = 0.0
    
    # Keep atoms above threshold OR protected
    keep = (contrib_adjusted > thresh) | (protect_mask <= 0)
    
    # Filter atoms
    kept = [a for i, a in enumerate(atoms) if keep[i]]
    new_epochs = {id(a): birth_epochs.get(id(a), 0) for i, a in enumerate(atoms) if keep[i]}
    
    pruned = len(atoms) - len(kept)
    if pruned > 0:
        print(f"  [Prune] -{pruned} (prot={protection}, thresh={thresh:.4f}, contrib_m={contrib.mean():.4f})")
    
    return kept, new_epochs


def generate_random_mask(H, W, mask_ratio=0.3, device='cuda'):
    """Generate random binary mask for masked prediction."""
    mask = torch.rand(H * W, device=device) < mask_ratio
    return mask


def train_scene(H=32, W=32, num_atoms=50, num_epochs=600, num_views=8,
                lr=1e-3, device='cuda', output_dir='outputs/selforg',
                bf16=False, fp16=True, num_samples=32, seed=42,
                w_met=0.01, w_vol=1.0, w_tc=2.0, w_pos=5.0,
                w_selforg=1.0, w_predict=1.0, w_homeo=0.1, w_flat=0.0,
                w_pred_view=0.0,
                state_alpha=0.3, mask_ratio=0.3, diff_K=5,
                chunk_size=128, atom_chunk_size=None, metric_batch_size=256,
                phase1_epochs=0, w_vol_p1=5.0, w_tc_p1=10.0,
                reproject_interval=0, reproject_oracle=False,
                no_diffusion=False,
                same_color=False, parametrization='cholesky',
                homeo_mean=0.5, homeo_std=0.25,
                homeo_log_density=0.0, homeo_max_log_ratio=1.0):
    """Two-phase self-organizing atom training.

    Phase 1 (Metric Warmup): freeze atoms, train metric field with strong
    trace-separation losses.
    Phase 2 (Clustering): unfreeze atoms, enable self-organization on the
    pre-trained geodesic structure.
    """
    use_two_phase = phase1_epochs > 0
    phase2_epochs = num_epochs - phase1_epochs if use_two_phase else num_epochs
    current_phase = 1 if use_two_phase else 2

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

    print(f"[1/4] Data ({H}x{W}, {num_views} views, same_color={same_color})...")
    images_np, masks_np, transforms = generate_multi_view(
        H=H, W=W, num_objects=2, num_views=num_views, seed=seed,
        same_color=same_color
    )
    images = torch.from_numpy(images_np).float().to(device)
    masks = torch.from_numpy(masks_np).float().to(device)
    occupancy = torch.from_numpy(get_occupancy(masks_np)).float().to(device)

    print(f"[2/4] Init metric field + {num_atoms} atoms...")
    metric_field = MetricField2D(H, W, init_scale=1.0,
                                    default_batch_size=metric_batch_size,
                                    parametrization=parametrization).to(device)

    # ── Uniform metric initialization (no occupancy cheat) ──
    # Framework must learn object boundaries from scratch.
    with torch.no_grad():
        metric_field.params[0, 0].fill_(1.0)  # l11
        metric_field.params[0, 2].fill_(1.0)  # l22
        metric_field.params[0, 1].zero_()      # l21
        print(f"  [Init] Uniform metric: trace=2.0 everywhere")

    frame_occupancy = torch.zeros(num_views, H, W, device=device)
    for fv in range(num_views):
        frame_occupancy[fv] = (masks[fv].sum(dim=-1) > 0.5).float()

    atoms = create_atoms(num_atoms, device, seed=seed,
                         occupancy=frame_occupancy[0])

    # Shared state → color decoder (residual + LayerNorm + SiLU — Theorem 17+18)
    # CRITICAL FIX: linear_only=True prevents decoder from learning a
    # complex "state-collapsing" mapping. A single Linear layer forces
    # states to directly encode color — state differentiation is the
    # ONLY way to produce different colors.
    state_decoder = create_optimal_decoder(
        state_dim=16, output_dim=3, linear_only=True
    ).to(device)

    # Collect atom params for easy freeze/unfreeze
    atom_params = [p for a in atoms for p in a.parameters()]

    # ── Phase-aware optimizer with proportional learning rates (Theorem 19) ──
    # eta_s (state, via decoder+atoms) : eta_g (metric field) : eta_mu (positions)
    # = 1 : 20 : 0.005  (from singular perturbation analysis)
    lr_state = lr
    lr_metric = lr * 20.0   # Metric field learns 20× faster
    lr_position = lr * 0.1   # Positions need meaningful movement

    if use_two_phase:
        # Phase 1: freeze position/state/radius/eps, but TRAIN _color.
        # _color must learn the true object colors first, so that in Phase 2
        # state+decoder is forced to encode color (not bypassed by _color).
        for a in atoms:
            a._mu.requires_grad = False
            a._log_r.requires_grad = False
            a._state.requires_grad = False
            a._logit_eps.requires_grad = False
            # a._color remains trainable
        color_params = [a._color for a in atoms]
        other_atom_params = [p for a in atoms for n, p in a.named_parameters()
                             if n not in ('_color',)]
        optimizer = torch.optim.Adam([
            {'params': metric_field.parameters(), 'lr': lr_metric},
            {'params': color_params, 'lr': lr_position},
            {'params': state_decoder.parameters(), 'lr': lr_state, 'weight_decay': 1e-3},
        ])
        scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs,
                                      eta_min=lr * 0.01)
        print(f"  [Phase 1] Metric warmup: {phase1_epochs} epochs "
              f"(w_vol={w_vol_p1}, w_tc={w_tc_p1})")
        print(f"  [Phase 2] Clustering: {phase2_epochs} epochs "
              f"(w_vol={w_vol}, w_tc={w_tc}, w_so={w_selforg})")
        print(f"  [LR] eta_s={lr_state:.1e} eta_g={lr_metric:.1e} eta_mu={lr_position:.1e}")
    else:
        optimizer = torch.optim.Adam([
            {'params': metric_field.parameters(), 'lr': lr_metric},
            {'params': atom_params, 'lr': lr_position},
            {'params': state_decoder.parameters(), 'lr': lr_state, 'weight_decay': 1e-3},
        ])
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs,
                                      eta_min=lr * 0.01)
        print(f"  [LR] eta_s={lr_state:.1e} eta_g={lr_metric:.1e} eta_mu={lr_position:.1e}")

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
    atom_contrib_accum = torch.zeros(len(atoms), device=device)
    prune_interval = max(num_epochs // 10, 100)  # Prune every ~10% of epochs, min 100
    min_atoms = max(num_atoms // 4, 10)  # Min atoms after pruning

    # ── Axiom monitor (6-axiom verification) ──
    axiom_monitor = AxiomMonitor()

    # ── Bootstrap smoothing schedule (Theorem 20) ──
    # Higher w_met in early epochs stabilizes metric field during bootstrap
    # Then gradually reduce to allow self-organization to take over
    w_met_boot = 0.05  # Bootstrap smoothing weight
    w_met_final = w_met  # Final smoothing weight
    bootstrap_epochs = max(phase1_epochs, 100)  # Bootstrap duration

    print(f"[4/4] Training ({num_epochs} epochs)...")
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  [GPU] Allocated: {alloc:.0f}MB | Reserved: {reserved:.0f}MB | Total: {total:.0f}MB")
        pts_per_chunk = num_samples * chunk_size
        acs = atom_chunk_size or len(atoms)
        print(f"  [VRAM] Chunk: {chunk_size} rays × {num_samples} samples = {pts_per_chunk} pts")
        print(f"  [VRAM] Atom sub-batch: {acs} | Metric batch: {metric_batch_size}")
        est_peak = alloc + pts_per_chunk * acs * 4 * 3 / 1024**2
        print(f"  [VRAM] Estim peak: ~{est_peak:.0f}MB (render) + reg")
        torch.cuda.empty_cache()

    for epoch in range(num_epochs):

        # ── Phase transition ──
        if use_two_phase and epoch == phase1_epochs:
            print(f"\n{'='*50}")
            print(f"  Phase 1 → Phase 2 Transition (epoch {epoch})")
            # Log trace separation before transition
            with torch.no_grad():
                tr = metric_field.trace()
                occ_mask = occupancy > 0.5
                tr_in = tr[occ_mask].mean().item() if occ_mask.any() else 0
                tr_out = tr[~occ_mask].mean().item() if (~occ_mask).any() else 0
                print(f"  Trace: in={tr_in:.3f} out={tr_out:.3f} ratio={tr_out/(tr_in+1e-8):.2f}")

            # Re-project atoms onto object regions (oracle bias; off by default)
            if reproject_oracle:
                reproject_atoms(atoms, occupancy, seed=seed)

            # Unfreeze atoms
            for p in atom_params:
                p.requires_grad = True

            # Recreate optimizer with proportional learning rates
            # Phase 2: freeze _color, force state+decoder to carry color signal
            for a in atoms:
                a._color.requires_grad = False
                a._mu.requires_grad = True
                a._log_r.requires_grad = True
                a._state.requires_grad = True
                a._logit_eps.requires_grad = True
            state_params = [a._state for a in atoms]
            pos_params = [a._mu for a in atoms] + [a._log_r for a in atoms]
            eps_params = [a._logit_eps for a in atoms]
            optimizer = torch.optim.Adam([
                {'params': metric_field.parameters(), 'lr': lr_metric * 0.5},
                {'params': state_params, 'lr': lr_state},      # state gets lr_state
                {'params': pos_params, 'lr': lr_position},     # position
                {'params': eps_params, 'lr': lr_position},     # existence
                {'params': state_decoder.parameters(), 'lr': lr_state, 'weight_decay': 1e-3},
            ])
            scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs,
                                          eta_min=lr * 0.01)
            scaler = GradScaler('cuda', enabled=scaler_enabled)
            current_phase = 2
            print(f"  Atoms re-projected + unfrozen. Optimizer rebuilt.")
            print(f"{'='*50}\n")

        # ── Active weights for this phase ──
        if current_phase == 1:
            cur_w_vol, cur_w_tc = w_vol_p1, w_tc_p1
            cur_w_selforg, cur_w_pos = 0.0, 0.0
        else:
            cur_w_vol, cur_w_tc = w_vol, w_tc
            cur_w_selforg, cur_w_pos = w_selforg, w_pos

        frame_idx = epoch % num_views
        target_img = images[frame_idx].reshape(-1, 3)

        # Generate random mask for this frame
        mask = generate_random_mask(H, W, mask_ratio, device)

        # ── Pruning schedule ──
        do_prune = (current_phase == 2 and epoch > phase1_epochs + 100
                    and epoch % prune_interval == 0)

        optimizer.zero_grad()

        # ── Chunked rendering ──
        N_rays = rays_o.shape[0]
        cs = chunk_size
        loss_render_val = 0.0
        pred_color_parts = []
        per_atom_frame = None  # Initialize for pruning accumulation

        for chunk_start in range(0, N_rays, cs):
            chunk_end = min(chunk_start + cs, N_rays)
            n_chunk = chunk_end - chunk_start
            chunk_weight = n_chunk / N_rays

            with amp_ctx:
                # Phase 2: render through state_decoder(state) so render loss
                # DIRECTLY propagates to state → forces state differentiation.
                # Phase 1: use _color (decoder not ready yet).
                render_sd = state_decoder if current_phase == 2 else None
                render_result = volume_render_2d(
                    rays_o[chunk_start:chunk_end], rays_d[chunk_start:chunk_end],
                    atoms, metric_field,
                    num_samples=num_samples, near=0.0, far=scene_size,
                    scene_size=scene_size, return_per_atom=do_prune,
                    atom_chunk_size=atom_chunk_size,
                    state_decoder=render_sd
                )
                pred_color_c, _, _ = render_result[:3]
                pred_color_parts.append(pred_color_c.detach())

                loss_render_c = l1_loss(pred_color_c, target_img[chunk_start:chunk_end])

                # Accumulate per-atom contribution for pruning
                if do_prune:
                    per_atom_c = render_result[3]  # type: ignore
                    if per_atom_frame is None:
                        per_atom_frame = per_atom_c.detach()
                    else:
                        per_atom_frame += per_atom_c.detach()

            scaler.scale(loss_render_c * chunk_weight).backward()
            loss_render_val += loss_render_c.detach().item() * chunk_weight
            del render_result, pred_color_c, loss_render_c
            if device == 'cuda':
                torch.cuda.empty_cache()

        pred_color = torch.cat(pred_color_parts, dim=0)
        del pred_color_parts  # Free the list of chunk tensors

        # Accumulate contribution for pruning
        if do_prune and per_atom_frame is not None:
            atom_contrib_accum += per_atom_frame

        # ── Aggressive memory cleanup after rendering ──
        if device == 'cuda':
            torch.cuda.empty_cache()

        # ── Next-view prediction consistency (camera-agnostic) ──
        # Renders the SAME canonical rays against the next view's image and
        # adds an L1 reconstruction term. Because rays are canonical (no
        # affine transform applied), this is purely a self-supervised
        # cross-view consistency signal. Inactive when w_pred_view == 0.
        if w_pred_view > 0:
            next_frame_idx = (frame_idx + 1) % num_views
            target_next_img = images[next_frame_idx].reshape(-1, 3)
            loss_pred_view = torch.tensor(0.0, device=device)
            for chunk_start in range(0, N_rays, cs):
                chunk_end = min(chunk_start + cs, N_rays)
                n_chunk = chunk_end - chunk_start
                chunk_weight_pv = n_chunk / N_rays

                with amp_ctx:
                    render_sd_pv = state_decoder if current_phase == 2 else None
                    pv_result = volume_render_2d(
                        rays_o[chunk_start:chunk_end], rays_d[chunk_start:chunk_end],
                        atoms, metric_field,
                        num_samples=num_samples, near=0.0, far=scene_size,
                        scene_size=scene_size, return_per_atom=False,
                        atom_chunk_size=atom_chunk_size,
                        state_decoder=render_sd_pv
                    )
                    pred_color_pv = pv_result[0]
                    loss_pv_c = l1_loss(pred_color_pv, target_next_img[chunk_start:chunk_end])

                loss_pred_view = loss_pred_view + loss_pv_c * chunk_weight_pv
                del pv_result, pred_color_pv, loss_pv_c
                if device == 'cuda':
                    torch.cuda.empty_cache()
            del target_next_img
            if device == 'cuda':
                torch.cuda.empty_cache()
        else:
            loss_pred_view = torch.tensor(0.0, device=device)

        # ── Regularization ──
        # Bootstrap smoothing schedule (Theorem 20):
        # w_met starts high (0.05) and decays to final value over bootstrap_epochs
        if epoch < bootstrap_epochs:
            progress = epoch / bootstrap_epochs
            cur_w_met = w_met_boot + (w_met_final - w_met_boot) * progress
        else:
            cur_w_met = w_met_final

        with amp_ctx:
            loss_met = metric_smoothness_loss(metric_field) * cur_w_met
            loss_vol = occupancy_coupling_loss(metric_field, occupancy) * cur_w_vol
            loss_tc = trace_contrast_loss(metric_field, occupancy) * cur_w_tc if cur_w_tc > 0 else torch.tensor(0.0, device=device)

            loss_pos_t = torch.tensor(0.0, device=device)
            if cur_w_pos > 0 and current_phase == 2 and len(atoms) > 0:
                atom_positions = torch.stack([a.position for a in atoms])
                grid = atom_positions.unsqueeze(0).unsqueeze(2) * 2 - 1
                pos_dist = F.grid_sample(dist_map, grid, mode='bilinear',
                                         padding_mode='border', align_corners=False)
                pos_dist = pos_dist.squeeze()
                if pos_dist.dim() == 0:
                    pos_dist = pos_dist.unsqueeze(0)
                loss_pos_t = pos_dist.mean() * cur_w_pos

            # ── Self-organization (Phase 2 only) ──
            mus = torch.stack([a.position for a in atoms])
            states = torch.stack([a.state for a in atoms])

            # ── Phase 1: pre-train decoder to map state → color ──
            # Without this, decoder outputs random colors in Phase 2,
            # causing render loss to explode and never recover.
            # Phase 1 teaches: "given your current random state, output your color".
            # Phase 2 then only needs to differentiate states, not reinvent mapping.
            # ── Decoder alignment loss (ALL phases) ──
            # Forces decoder(state_i) to match atom's _color.
            # This is the critical bridge: state must encode color,
            # decoder must map state → color, _color provides the anchor.
            # High weight (5.0) prevents decoder drift when render switches
            # to state-based in Phase 2.
            if len(atoms) > 0:
                colors_target = torch.stack([a._color for a in atoms])
                colors_pred = state_decoder(states)
                loss_decode = l1_loss(colors_pred, colors_target.detach()) * 5.0
            else:
                loss_decode = torch.tensor(0.0, device=device)

            # Compute geodesic neighbors with HARD distance cutoff
            # This prevents cross-object message passing — atoms only talk
            # to geometrically nearby neighbors (within same object).
            # Without cutoff, softmax leakage causes global state collapse.
            from src.losses.direct_cluster import compute_pairwise_geodesic_sq
            D2 = compute_pairwise_geodesic_sq(mus, metric_field)
            D = D2.sqrt()
            
            # Fixed k-NN mask: each atom only connects to its k nearest
            # geodesic neighbors. This prevents cross-object contamination
            # that happens with distance-based cutoff (median*2.5 too large).
            k_cutoff = min(5, D.shape[0] - 1)
            _, knn_idx = D.topk(k=k_cutoff + 1, dim=1, largest=False)
            knn_idx = knn_idx[:, 1:]  # exclude self

            N_atoms = D.shape[0]
            geo_mask = torch.zeros_like(D)
            for i in range(N_atoms):
                geo_mask[i, knn_idx[i]] = 1.0
            # Symmetrize: if i is neighbor of j, j is neighbor of i
            geo_mask = ((geo_mask + geo_mask.T) > 0).float()
            geo_mask.fill_diagonal_(0.0)

            # Soft weights within k-NN neighborhood
            D_knn = D.topk(k=k_cutoff + 1, dim=1, largest=False)[0][:, 1:]
            sigma = D_knn[:, -1].clamp(min=1e-4)
            sigma_prod = sigma.unsqueeze(1) * sigma.unsqueeze(0)
            A = torch.exp(-D2 / (2 * sigma_prod)) * geo_mask
            row_sums = A.sum(dim=1, keepdim=True).clamp(min=1e-10)
            geo_weights = A / row_sums  # row-stochastic, masked

            states_prop = state_propagation(states, geo_weights, alpha=state_alpha)

            # Self-organization loss — uses propagated (smoothed) states
            if cur_w_selforg > 0 and current_phase == 2:
                loss_so = self_organization_loss(mus, states_prop, metric_field) * cur_w_selforg
            else:
                loss_so = torch.tensor(0.0, device=device)

            # Masked prediction loss — uses RAW states (no smoothing)
            # This gives sharp per-atom gradient that drives state differentiation
            masked_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            if masked_indices.numel() > 0:
                masked_px = torch.stack([
                    (masked_indices % W).float() / W,
                    (masked_indices // W).float() / H,
                ], dim=-1).to(device)
                target_c = target_img[masked_indices]
                atom_colors = torch.stack([a._color for a in atoms])
                loss_pred = masked_prediction_loss(
                    mus, states, metric_field,  # RAW states, NOT propagated
                    masked_px, target_c, atom_colors,
                    state_decoder=state_decoder
                ) * w_predict
            else:
                loss_pred = torch.tensor(0.0, device=device)

            # ── State dynamics step (Theorem 1 contraction mapping) ──
            # Selective mixing: only mix with neighbors that have SIMILAR states.
            # This prevents "strong state" atoms from pulling "weak state" atoms
            # toward a global mean. Instead, each cluster converges independently.
            if current_phase == 2 and len(atoms) > 0:
                with torch.no_grad():
                    mus_now = torch.stack([a.position for a in atoms])
                    states_now = torch.stack([a.state for a in atoms])
                    s_norm = F.normalize(states_now, dim=-1)
                    S = s_norm @ s_norm.T  # cosine similarity
                    
                    # Recompute geodesic neighbors with fixed k-NN
                    D2_now = compute_pairwise_geodesic_sq(mus_now, metric_field)
                    D_now = D2_now.sqrt()
                    _, knn_idx_now = D_now.topk(k=k_cutoff + 1, dim=1, largest=False)
                    knn_idx_now = knn_idx_now[:, 1:]  # exclude self

                    N_now = D_now.shape[0]
                    mask_now = torch.zeros_like(D_now)
                    for i in range(N_now):
                        mask_now[i, knn_idx_now[i]] = 1.0
                    mask_now = ((mask_now + mask_now.T) > 0).float()
                    mask_now.fill_diagonal_(0.0)
                    
                    # STATE-SIMILARITY GATING: only mix with neighbors whose
                    # cosine similarity > 0.5. This prevents cross-cluster
                    # contamination during the convergence phase.
                    sim_mask = (S > 0.5).float()
                    mask_gated = mask_now * sim_mask
                    
                    # Renormalize after gating (handle all-zero rows)
                    row_sums_gated = mask_gated.sum(dim=1, keepdim=True).clamp(min=1e-10)
                    geo_w_gated = mask_gated / row_sums_gated
                    
                    states_target = state_propagation(states_now, geo_w_gated, alpha=state_alpha)
                    
                    # VERY weak gamma: let optimizer (w_predict) drive differentiation
                    # while EMA only provides gentle intra-cluster consensus.
                    gamma = 0.005
                    for i, atom in enumerate(atoms):
                        atom._state.data = (1 - gamma) * atom._state.data + gamma * states_target[i]
            
            # Track state consistency for logging
            loss_state_con = torch.tensor(0.0, device=device)

            # Feature diffusion (smoothing for state visualization only)
            diff_val = 0.0
            if diff_K > 0 and current_phase == 2 and not no_diffusion:
                A = compute_geodesic_affinity(mus, metric_field, K=diff_K)
                states_diff = feature_diffusion(states_prop, A, alpha=0.5, T=2)
                diff_val = ((states_diff - states_prop) ** 2).mean().item()
                del A, states_diff

            # ── State contrastive loss (InfoNCE) — CRITICAL for differentiation ──
            # Forces geodesic neighbors (same object) to have similar states,
            # and non-neighbors (different objects) to have dissimilar states.
            # This is what breaks the state-collapse symmetry.
            if current_phase == 2 and len(atoms) > 0:
                loss_contrast = state_contrastive_loss(
                    states, geo_mask, temperature=0.1
                ) * 2.0  # weight = 2.0 (strong signal)
            else:
                loss_contrast = torch.tensor(0.0, device=device)

            # ── Homeostatic plasticity (Phase 2) — control seed variance ──
            if current_phase == 2 and w_homeo > 0 and len(atoms) > 0:
                eps_all = torch.stack([a.existence_prob for a in atoms])
                contrib = atom_contrib_accum if atom_contrib_accum.max() > 0 else None
                loss_homeo = homeostatic_loss(
                    atoms, per_atom_contrib=contrib,
                    target_mean=homeo_mean, target_std=homeo_std,
                    target_log_density=homeo_log_density,
                    max_log_ratio=homeo_max_log_ratio)
            else:
                loss_homeo = torch.tensor(0.0, device=device)

            # ── Metric flatness prior (grid-cell inspired) — reduce anisotropy ──
            if w_flat > 0:
                loss_flat = metric_field.metric_flatness_loss()
            else:
                loss_flat = torch.tensor(0.0, device=device)

            loss_reg = (loss_met + loss_vol + loss_tc + loss_pos_t + loss_so
                        + loss_pred + loss_contrast + loss_decode
                        + w_homeo * loss_homeo
                        + w_flat * loss_flat
                        + w_pred_view * loss_pred_view)

            # Extract values before potential deletion
            loss_met_val = loss_met.item() if isinstance(loss_met, torch.Tensor) else loss_met
            loss_vol_val = loss_vol.item() if isinstance(loss_vol, torch.Tensor) else loss_vol
            loss_tc_val = loss_tc.item() if isinstance(loss_tc, torch.Tensor) else 0.0
            loss_so_val = loss_so.item() if isinstance(loss_so, torch.Tensor) else loss_so
            loss_pred_val = loss_pred.item() if isinstance(loss_pred, torch.Tensor) else loss_pred
            loss_pos_val = loss_pos_t.item() if isinstance(loss_pos_t, torch.Tensor) else loss_pos_t
            loss_contrast_val = loss_contrast.item() if isinstance(loss_contrast, torch.Tensor) else loss_contrast
            loss_decode_val = loss_decode.item() if isinstance(loss_decode, torch.Tensor) else loss_decode
            loss_reg_val = loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg
            loss_flat_val = loss_flat.item() if isinstance(loss_flat, torch.Tensor) else 0.0
            loss_homeo_val = loss_homeo.item() if isinstance(loss_homeo, torch.Tensor) else 0.0
            loss_pred_view_val = loss_pred_view.item() if isinstance(loss_pred_view, torch.Tensor) else 0.0

        scaler.scale(loss_reg).backward()

        # ── Post-regularization cleanup ──
        del loss_met, loss_vol, loss_tc, loss_pos_t, loss_so, loss_pred, loss_contrast, loss_decode, loss_reg, loss_pred_view
        if device == 'cuda':
            torch.cuda.empty_cache()

        # Optimizer step
        scaler.unscale_(optimizer)
        all_params = [p for pg in optimizer.param_groups for p in pg['params']]
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ── Periodic atom pruning ──
        if (prune_interval > 0 and epoch % prune_interval == 0 and len(atoms) > min_atoms
                and current_phase == 2 and epoch > phase1_epochs + 100):
            # Safety: skip if no contribution accumulated (prune not ready)
            if atom_contrib_accum.max() > 0:
                atoms, atom_birth_epochs = prune_atoms_contrib_2d(
                    atom_contrib_accum, atoms, atom_birth_epochs, epoch,
                    threshold=0.1, min_atoms=min_atoms, protection=200
                )
                # Safety: reinitialize if all atoms pruned
                if len(atoms) == 0:
                    print(f"[WARN] All atoms pruned at epoch {epoch}, reinitializing...")
                    atoms = create_atoms(80, device, seed=seed + epoch,
                                         occupancy=frame_occupancy[0])
                    atom_birth_epochs = {id(a): epoch for a in atoms}
                # Reset accumulators for new atom set
                atom_contrib_accum = torch.zeros(len(atoms), device=device)
            else:
                print(f"[Prune-Skip] epoch={epoch}, contrib_accum all zero (prune not ready)")

        # ── Periodic atom re-projection (Phase 2) — oracle bias; off by default ──
        if (reproject_oracle and reproject_interval > 0 and current_phase == 2
                and epoch > phase1_epochs and epoch % reproject_interval == 0):
            reproject_atoms(atoms, occupancy, seed=seed + epoch)

        # ── Logging ──
        losses_log.append({
            'epoch': epoch,
            'total': loss_render_val + loss_reg_val,
            'render': loss_render_val,
            'met': loss_met_val,
            'vol': loss_vol_val,
            'tc': loss_tc_val,
            'selforg': loss_so_val,
            'predict': loss_pred_val,
            'pos': loss_pos_val,
            'contrast': loss_contrast_val,
            'decode': loss_decode_val,
            'homeo': loss_homeo_val,
            'flat': loss_flat_val,
            'pred_view': loss_pred_view_val,
            'diff': diff_val,
        })

        # Log every 100 epochs + first/last + phase transition
        log_interval = 100
        is_transition = use_two_phase and abs(epoch - phase1_epochs) <= 1
        if epoch % log_interval == 0 or epoch == num_epochs - 1 or is_transition:
            log = losses_log[-1]
            state_std = states.std(dim=0).mean().item() if len(atoms) > 0 else 0
            # Trace separation metric
            with torch.no_grad():
                tr = metric_field.trace()
                occ_mask = occupancy > 0.5
                tr_in = tr[occ_mask].mean().item() if occ_mask.any() else 0
                tr_out = tr[~occ_mask].mean().item() if (~occ_mask).any() else 0
                tr_ratio = tr_out / (tr_in + 1e-8)

            # ── Axiom diagnostics (every 100 epochs) ──
            axiom_summary = ""
            if epoch % 100 == 0 or is_transition or epoch == num_epochs - 1:
                axiom_monitor.step(
                    states, geo_weights, mus, metric_field, occupancy,
                    w_selforg=cur_w_selforg, w_smooth=cur_w_met,
                    state_decoder=state_decoder, target_img=target_img,
                    masked_indices=masked_indices, W=W, H=H
                )
                axiom_summary = " | " + axiom_monitor.summary()

            mem_str = ""
            if device == 'cuda':
                alloc_mb = torch.cuda.memory_allocated() / 1024**2
                max_mb = torch.cuda.max_memory_allocated() / 1024**2
                mem_str = f" GPU={alloc_mb:.0f}/{max_mb:.0f}MB"
            phase_str = f"P{current_phase}"
            print(f"  [{epoch:4d}/{num_epochs}|{phase_str}] T={log['total']:7.3f} R={log['render']:.3f} "
                  f"M={log['met']:.3f} V={log['vol']:.3f} TC={log['tc']:.3f} "
                  f"O={log['selforg']:.3f} P={log['predict']:.3f} C={log.get('contrast',0):.3f} D={log.get('decode',0):.3f} H={log.get('homeo',0):.3f} F={log.get('flat',0):.3f} PV={log.get('pred_view',0):.3f} "
                  f"SS={state_std:.4f} tr={tr_in:.2f}/{tr_out:.2f}({tr_ratio:.1f}) "
                  f"A={len(atoms)}{mem_str}{axiom_summary}")

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
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--atom-chunk-size', type=int, default=None,
                        help='Atoms per sub-batch in renderer (lower = less VRAM, None = all at once)')
    parser.add_argument('--metric-batch-size', type=int, default=256,
                        help='Max coords per metric_field query (lower = less VRAM peak)')
    parser.add_argument('--low-vram', action='store_true',
                        help='Low VRAM preset (4GB cards): sets atom_chunk=15, chunk=64, samples=16, metric_batch=128')
    parser.add_argument('--ultra-low-vram', action='store_true',
                        help='Ultra-low VRAM preset (<1GB peak): sets atom_chunk=8, chunk=32, samples=8, metric_batch=64')
    parser.add_argument('--no-diffusion', action='store_true',
                        help='Skip feature diffusion (saves VRAM, only affects visualization)')
    parser.add_argument('--output', type=str, default='outputs/selforg_32x32')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--w-vol', type=float, default=1.0)
    parser.add_argument('--w-tc', type=float, default=2.0)
    parser.add_argument('--w-selforg', type=float, default=1.0)
    parser.add_argument('--w-predict', type=float, default=1.0)
    parser.add_argument('--w-homeo', type=float, default=0.1,
                        help='Homeostatic plasticity weight (existence/contribution regularization)')
    parser.add_argument('--w-flat', type=float, default=0.0,
                        help='Metric flatness prior weight (anisotropy + spatial trace smoothness)')
    parser.add_argument('--w-pred-view', type=float, default=0.0,
                        help='Next-view prediction consistency weight (camera-agnostic)')
    parser.add_argument('--w-pos', type=float, default=5.0,
                        help='Position regularization weight')
    parser.add_argument('--state-alpha', type=float, default=0.2,
                        help='State propagation rate (0=no mixing, 1=full mix)')
    parser.add_argument('--mask-ratio', type=float, default=0.3)
    # Two-phase arguments
    parser.add_argument('--phase1-epochs', type=int, default=0,
                        help='Number of Phase 1 (metric warmup) epochs. 0 = single-phase.')
    parser.add_argument('--w-vol-p1', type=float, default=5.0,
                        help='Occupancy coupling weight in Phase 1')
    parser.add_argument('--w-tc-p1', type=float, default=10.0,
                        help='Trace contrast weight in Phase 1')
    parser.add_argument('--reproject-interval', type=int, default=0,
                        help='Re-project atoms onto objects every N epochs (0=off)')
    parser.add_argument('--reproject-oracle', action='store_true',
                        help='Allow reprojection to use ground-truth occupancy '
                             '(oracle bias; default off)')
    # Homeostatic target hyperparameters
    parser.add_argument('--homeo-mean', type=float, default=0.5,
                        help='Target mean existence probability')
    parser.add_argument('--homeo-std', type=float, default=0.25,
                        help='Target std existence probability (soft upper bound)')
    parser.add_argument('--homeo-log-density', type=float, default=0.0,
                        help='Log target for mean per-atom contribution')
    parser.add_argument('--homeo-max-log-ratio', type=float, default=1.0,
                        help='Soft upper bound on log(max/mean contribution)')
    # EXT-4 validation: same-color objects
    parser.add_argument('--same-color', action='store_true',
                        help='All objects have same color (tests if mask prediction '
                             'can distinguish objects without color cues)')
    # EXT-1 fix: SPD parametrization
    parser.add_argument('--parametrization', type=str, default='cholesky',
                        choices=['cholesky', 'matrix_exp'],
                        help="Metric field parametrization: 'cholesky' (fast, default) "
                             "or 'matrix_exp' (strictly SPD, slower)")
    args = parser.parse_args()

    H = W = args.resolution
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16 = args.fp16 and device == 'cuda'
    bf16 = args.bf16 and device == 'cuda' and torch.cuda.is_bf16_supported()

    # ── Low-VRAM preset ──
    if args.ultra_low_vram:
        args.atom_chunk_size = args.atom_chunk_size or 8
        args.chunk_size = min(args.chunk_size, 32)
        args.samples = min(args.samples, 8)
        args.metric_batch_size = min(args.metric_batch_size, 64)
        args.no_diffusion = True
        print("[Ultra-Low-VRAM] atom_chunk=8, chunk=32, samples=8, metric_batch=64, no_diffusion")
    elif args.low_vram:
        if args.atom_chunk_size is None:
            args.atom_chunk_size = 15
        if args.chunk_size == 128:  # only override if still default
            args.chunk_size = 64
        if args.samples == 32:
            args.samples = 16
        if args.metric_batch_size == 256:
            args.metric_batch_size = 128
        print("[Low-VRAM] atom_chunk=15, chunk=64, samples=16, metric_batch=128")

    phase_str = f" | Phase1: {args.phase1_epochs}ep" if args.phase1_epochs > 0 else ""
    print(f"Self-Organizing Atoms | {H}x{W} | {args.atom} atoms | {args.epochs} epochs{phase_str}")
    print(f"FP16: {fp16} | BF16: {bf16} | Device: {device}")

    train_scene(
        H=H, W=W, num_atoms=args.atom, num_epochs=args.epochs,
        lr=args.lr, device=device, output_dir=args.output,
        bf16=bf16, fp16=fp16, num_samples=args.samples, seed=args.seed,
        w_vol=args.w_vol, w_tc=args.w_tc, w_pos=args.w_pos,
        w_selforg=args.w_selforg, w_predict=args.w_predict,
        w_homeo=args.w_homeo, w_flat=args.w_flat,
        w_pred_view=args.w_pred_view,
        state_alpha=args.state_alpha, mask_ratio=args.mask_ratio,
        chunk_size=args.chunk_size, atom_chunk_size=args.atom_chunk_size,
        metric_batch_size=args.metric_batch_size,
        phase1_epochs=args.phase1_epochs,
        w_vol_p1=args.w_vol_p1, w_tc_p1=args.w_tc_p1,
        reproject_interval=args.reproject_interval,
        reproject_oracle=args.reproject_oracle,
        no_diffusion=args.no_diffusion,
        same_color=args.same_color,
        parametrization=args.parametrization,
        homeo_mean=args.homeo_mean, homeo_std=args.homeo_std,
        homeo_log_density=args.homeo_log_density,
        homeo_max_log_ratio=args.homeo_max_log_ratio,
    )
