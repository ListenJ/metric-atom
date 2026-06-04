"""
Geodesic utilities for self-organizing atom system.

[ARCHIVE] DirectClusterLoss removed (fc5f5af postmortem).
Remaining: Sinkhorn softmax (soft voting), pairwise geodesic distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_softmax(cost, epsilon=0.1, n_iters=50):
    """
    Sinkhorn-Knopp producing row-stochastic soft assignment.

    P_ik ∝ exp(-cost_ik / epsilon) * v_k
    where v_k is iteratively updated to balance cluster sizes.

    Args:
        cost: (N, K) cost matrix
        epsilon: entropy regularization strength
        n_iters: number of Sinkhorn iterations

    Returns:
        P: (N, K) soft assignment, row-stochastic
    """
    N, K = cost.shape
    K_mat = torch.exp(-cost / epsilon)

    v = torch.ones(K, device=cost.device)

    for _ in range(n_iters):
        row_sums = (K_mat * v.unsqueeze(0)).sum(dim=1, keepdim=True)
        P = K_mat * v.unsqueeze(0) / row_sums.clamp(min=1e-10)

        col_sums = P.sum(dim=0)
        target = N / K
        v = (v * target / col_sums.clamp(min=1e-10)).clamp(max=1e3)

    row_sums = (K_mat * v.unsqueeze(0)).sum(dim=1, keepdim=True)
    P = K_mat * v.unsqueeze(0) / row_sums.clamp(min=1e-10)

    return P


def compute_pairwise_geodesic_sq(mus, metric_field):
    """
    Symmetric pairwise geodesic distance-squared matrix using midpoint metric.

    d²_ij = (μ_i - μ_j)ᵀ g((μ_i + μ_j)/2) (μ_i - μ_j)

    Args:
        mus: (N, D) atom positions
        metric_field: MetricField2D instance

    Returns:
        D2: (N, N) squared geodesic distances
    """
    N, D = mus.shape
    if N < 2:
        return torch.zeros(N, N, device=mus.device)

    mids = (mus.unsqueeze(0) + mus.unsqueeze(1)) / 2
    dx = mus.unsqueeze(0) - mus.unsqueeze(1)

    g_mid = metric_field(mids.reshape(-1, D)).reshape(N, N, D, D)
    d2 = torch.einsum('ijm,ijmn,ijn->ij', dx, g_mid, dx).clamp(min=0)

    return d2
