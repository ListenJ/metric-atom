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
    D2 = D2 + torch.eye(D2.shape[0], device=D2.device) * 1e10  # exclude self
    _, indices = D2.topk(k=k, dim=1, largest=False)  # (N, k)

    # Soft weights: w_ij = exp(-D2_ij / sigma^2)
    sigma = D2[D2 < 1e9].median().sqrt().clamp(min=1e-4)
    A = torch.exp(-D2 / (2 * sigma * sigma))
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


def self_organization_loss(mus, states, metric_field):
    """
    Self-organization force through the metric field.

    L_selforg = -Σ_{i,j} cos_sim(s_i, s_j) * exp(-d_g(i,j)² / σ²)

    Intuition:
      - cos_sim(s_i, s_j) > 0: similar states → attract (reduce geodesic)
      - cos_sim(s_i, s_j) < 0: different states → repel (increase geodesic)
      - The metric field adapts to make same-object atoms geodesically close.

    Args:
        mus: (N, 2) atom positions
        states: (N, D) atom states
        metric_field: MetricField2D

    Returns:
        loss: scalar
    """
    N = states.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=states.device)

    # State similarity matrix
    s_norm = F.normalize(states, dim=-1)
    S = s_norm @ s_norm.T  # (N, N), in [-1, 1]

    # Geodesic distance
    D2 = compute_pairwise_geodesic_sq(mus, metric_field)

    # Gaussian kernel: close in geodesic → high weight
    sigma = D2.median().sqrt().clamp(min=1e-4)
    W = torch.exp(-D2 / (2 * sigma * sigma))

    # Loss: -S * W → similar states close in geodesic → low loss
    return -(S * W).sum() / (N * N)


def masked_prediction_loss(mus, states, metric_field,
                            masked_px, target_colors, atom_colors, k=5):
    """
    Predict masked pixel colors via geodesic neighbor voting.

    For each masked pixel p:
      1. Find k geodesic nearest atoms
      2. Each atom votes a color: pred = MLP(state)  (simplified: use atom._color)
      3. Weighted average: pred(p) = Σ w_i * color_i

    During early training this is inaccurate → gradient shapes the metric
    and states to improve prediction.

    Args:
        mus: (N, 2) atom positions
        states: (N, D) atom states (for future MLP decode, currently unused)
        metric_field: MetricField2D
        masked_px: (M, 2) masked pixel positions in [0,1]²
        target_colors: (M, 3) ground truth colors
        k: number of neighbor atoms to vote

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

    # Color voting: each atom's color weighted by geodesic proximity
    pred_colors = weights @ atom_colors  # (M, 3)

    return F.l1_loss(pred_colors, target_colors)
