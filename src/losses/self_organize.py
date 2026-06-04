"""
Self-organizing atom losses.

Replaces DirectClusterLoss (postmortem fc5f5af).
Atoms self-organize through:
  1. State propagation — message passing over geodesic neighborhoods
  2. Self-organization force — similar states attract in metric space
  3. Masked prediction — predict masked pixels via state-driven voting
"""

import torch
import torch.nn.functional as F
from src.losses.direct_cluster import compute_pairwise_geodesic_sq


def compute_geodesic_neighbors(mus, metric_field, k=5):
    """
    Find top-k geodesic nearest neighbors for each atom.

    Uses per-atom adaptive sigma (K-th neighbor distance) instead of a
    global median, so the affinity remains informative even when atoms
    converge and global distances shrink.

    Args:
        mus: (N, 2) atom positions
        metric_field: MetricField2D
        k: number of neighbors (excluding self)

    Returns:
        weights: (N, N) soft neighbor weights (row-stochastic, self=0)
        indices: (N, k) indices of top-k neighbors
    """
    with torch.no_grad():
        D2 = compute_pairwise_geodesic_sq(mus, metric_field)  # (N, N)
    D2_masked = D2 + torch.eye(D2.shape[0], device=D2.device) * 1e10  # exclude self
    _, indices = D2_masked.topk(k=k, dim=1, largest=False)  # (N, k)

    # Per-atom adaptive sigma: sigma_i = sqrt(D2 to k-th neighbor)
    D2_sorted = D2_masked.topk(k=k, dim=1, largest=False)[0]  # (N, k)
    sigma_i = D2_sorted[:, -1].sqrt().clamp(min=1e-4)  # (N,) k-th neighbor distance

    # Symmetric pairwise sigma: sigma_ij = sqrt(sigma_i * sigma_j)
    sigma_prod = sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)  # (N, N)

    A = torch.exp(-D2 / (2 * sigma_prod))
    A.fill_diagonal_(0.0)
    # Row normalize
    row_sums = A.sum(dim=1, keepdim=True).clamp(min=1e-10)
    weights = A / row_sums  # (N, N), row-stochastic

    return weights, indices


def state_propagation(states, weights, alpha=0.3):
    """
    Message passing: each atom's state is updated by its neighbors.

    s_i^{new} = (1 - α) * s_i + α * Σ_j w_{ij} * s_j

    This is a single diffusion step. Atoms with similar neighbors
    converge to similar states.

    Args:
        states: (N, D) current states
        weights: (N, N) row-stochastic geodesic affinity
        alpha: update rate

    Returns:
        states_new: (N, D) updated states
    """
    neighbor_mean = weights @ states  # (N, D)
    return (1 - alpha) * states + alpha * neighbor_mean


def self_organization_loss(mus, states, metric_field, K=5):
    """
    Self-organization force through the metric field.

    L_selforg = -Σ_{i,j} cos_sim(s_i, s_j) * exp(-d_g(i,j)² / (2 σ_i σ_j))

    Uses per-atom adaptive sigma (K-th geodesic neighbor distance) so the
    kernel bandwidth tracks local density instead of collapsing with the
    global distance scale.  This prevents gradient vanishing in late
    training when atoms have converged and global median(D²) → 0.

    Intuition:
      - cos_sim(s_i, s_j) > 0: similar states → attract (reduce geodesic)
      - cos_sim(s_i, s_j) < 0: different states → repel (increase geodesic)
      - The metric field adapts to make same-object atoms geodesically close.

    Args:
        mus: (N, 2) atom positions
        states: (N, D) atom states
        metric_field: MetricField2D
        K: neighbor count for adaptive sigma

    Returns:
        loss: scalar
    """
    N = states.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=states.device)

    # State similarity matrix
    s_norm = F.normalize(states, dim=-1)
    S = s_norm @ s_norm.T  # (N, N), in [-1, 1]

    # Geodesic distance (differentiable — gradients flow through D2)
    D2 = compute_pairwise_geodesic_sq(mus, metric_field)

    # ── Adaptive sigma: per-atom K-th neighbor distance ──
    # Use detached D2 for sigma computation so bandwidth selection
    # doesn't interfere with the main gradient signal.
    with torch.no_grad():
        D2_masked = D2.detach() + torch.eye(N, device=D2.device) * 1e10
        k_actual = min(K, N - 1)
        D2_knn = D2_masked.topk(k=k_actual, dim=1, largest=False)[0]  # (N, k)
        sigma_i = D2_knn[:, -1].sqrt().clamp(min=1e-4)  # (N,)

    # Symmetric pairwise bandwidth: σ_ij = √(σ_i · σ_j)
    sigma_prod = sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)  # (N, N)

    # Gaussian kernel with adaptive per-pair bandwidth
    W = torch.exp(-D2 / (2 * sigma_prod))

    # Loss: -S * W → similar states close in geodesic → low loss
    return -(S * W).sum() / (N * N)


def masked_prediction_loss(mus, states, metric_field,
                            masked_px, target_colors, atom_colors,
                            state_decoder=None, k=5):
    """
    Predict masked pixel colors via geodesic neighbor voting.

    For each masked pixel p:
      1. Find geodesic nearest atoms (soft weights via metric field)
      2. Each atom votes: pred = state_decoder(state_i) if decoder else atom._color
      3. Weighted average: pred(p) = Σ w_i * pred_i

    The decoder (nn.Linear) is shared across all atoms — states must
    learn to encode color information for prediction to work.
    This forces states to become "visually aware."

    Args:
        mus: (N, 2) atom positions
        states: (N, D) atom states — used for prediction via decoder
        metric_field: MetricField2D
        masked_px: (M, 2) masked pixel positions in [0,1]²
        target_colors: (M, 3) ground truth colors
        atom_colors: (N, 3) atom colors (fallback if no decoder)
        state_decoder: nn.Linear(D, 3) or None — shared state→color mapping
        k: number of neighbor atoms (unused, soft weights on all atoms)

    Returns:
        loss: scalar L1 prediction error
    """
    N = mus.shape[0]
    M = masked_px.shape[0]
    if N < 2 or M == 0:
        return torch.tensor(0.0, device=mus.device)

    # Compute geodesic distances from all masked pixels to all atoms
    # px: (M, 2), mus: (N, 2) → D2_px: (M, N)
    px_exp = masked_px.unsqueeze(1)  # (M, 1, 2)
    mu_exp = mus.unsqueeze(0)        # (1, N, 2)
    mids = (px_exp + mu_exp) / 2     # (M, N, 2)
    dx = px_exp - mu_exp             # (M, N, 2)

    g_mid = metric_field(mids.reshape(-1, 2)).reshape(M, N, 2, 2)
    D2_px = torch.einsum('mnd,mnde,mne->mn', dx, g_mid, dx).clamp(min=0)

    # Soft neighbor weights: w ∝ exp(-D2 / ε)
    eps = D2_px.median().clamp(min=1e-4)
    weights = torch.softmax(-D2_px / eps, dim=-1)  # (M, N)

    # Color voting:
    # With decoder: each atom votes color = decoder(state_i)
    # Without: fall back to atom's learned color
    if state_decoder is not None:
        atom_preds = state_decoder(states)  # (N, 3)
    else:
        atom_preds = atom_colors

    pred_colors = weights @ atom_preds  # (M, 3)

    return F.l1_loss(pred_colors, target_colors)
