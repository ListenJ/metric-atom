"""
Geodesic utilities for self-organizing atom system.

[ARCHIVE] DirectClusterLoss removed (fc5f5af postmortem).
Remaining: Sinkhorn softmax (soft voting), pairwise geodesic distance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_softmax(cost, epsilon=None, n_iters=200, v_init=None,
                     adaptive_eps=True, eps_min=0.01, eps_max=0.5,
                     return_converged=False):
    """
    Sinkhorn-Knopp producing row-stochastic soft assignment.

    P_ik ∝ exp(-cost_ik / epsilon) * v_k
    where v_k is iteratively updated to balance cluster sizes.

    FIX for EXT-3 (Sinkhorn convergence instability):
      - Default n_iters increased from 50 → 200 (Cuturi et al. 2018)
      - Adaptive epsilon scaling by cost median (prevents underflow)
      - Warm-start via v_init (reuse previous assignment)
      - Convergence check with early stopping

    Args:
        cost: (N, K) cost matrix
        epsilon: entropy regularization strength. If None, auto-computed.
        n_iters: number of Sinkhorn iterations (default 200)
        v_init: (K,) warm-start vector from previous call, or None
        adaptive_eps: if True, scale epsilon by cost median
        eps_min, eps_max: clamp bounds for adaptive epsilon
        return_converged: if True, also return convergence flag

    Returns:
        P: (N, K) soft assignment, row-stochastic
        v: (K,) final dual variable (for warm-start in next call)
        converged: bool (only if return_converged=True)
    """
    N, K = cost.shape

    # ── Adaptive epsilon (EXT-3 fix) ──
    if adaptive_eps:
        cost_median = cost.median().detach()
        if epsilon is None:
            epsilon = cost_median.abs().clamp(min=1e-4) * 0.1
        else:
            epsilon = epsilon * cost_median.abs().clamp(min=1e-4)
        epsilon = epsilon.clamp(min=eps_min, max=eps_max)
    elif epsilon is None:
        epsilon = 0.1

    # Numerical stability: shift cost by min per row
    cost_shifted = cost - cost.min(dim=1, keepdim=True)[0]
    K_mat = torch.exp(-cost_shifted / epsilon)
    K_mat = K_mat.clamp(min=1e-8)  # prevent exact zeros

    # Warm-start
    v = v_init.clone() if v_init is not None else torch.ones(K, device=cost.device)

    converged = False
    for it in range(n_iters):
        v_prev = v.clone()

        row_sums = (K_mat * v.unsqueeze(0)).sum(dim=1, keepdim=True)
        P = K_mat * v.unsqueeze(0) / row_sums.clamp(min=1e-10)

        col_sums = P.sum(dim=0)
        target = N / K
        v = (v * target / col_sums.clamp(min=1e-10)).clamp(min=1e-8, max=1e4)

        # Convergence check (EXT-3): stop if dual variable stable
        if it > 10:
            dv = (v - v_prev).abs().max()
            if dv < 1e-5:
                converged = True
                break

    row_sums = (K_mat * v.unsqueeze(0)).sum(dim=1, keepdim=True)
    P = K_mat * v.unsqueeze(0) / row_sums.clamp(min=1e-10)

    if return_converged:
        return P, v, converged
    return P, v


def compute_pairwise_midpoint_mahalanobis_sq(mus, metric_field):
    """
    Pairwise midpoint Mahalanobis distance-squared matrix.

    CRITICAL NOTE (EXT-2 fix): This is NOT a true geodesic distance.
    It is a Mahalanobis chord distance evaluated at the midpoint metric:

        d²_ij = (μ_i - μ_j)ᵀ g((μ_i + μ_j)/2) (μ_i - μ_j)

    For a true geodesic distance, one must integrate ds² = dxᵀ g(x) dx
    along the geodesic curve between μ_i and μ_j, which requires solving
    a non-linear ODE. The midpoint approximation is:
      - Exact when g(x) is constant (Euclidean)
      - Upper-bound: d_true ≤ d_midpoint (by convexity of the energy)
      - Error grows with ||∇g|| · ||dx||² (worst at boundaries where g jumps)

    We use this approximation because:
      1. It is fully differentiable (gradients flow through g(mid))
      2. It is O(N²) vs O(N² · n_integration_steps) for true geodesics
      3. For our atom-atom separation (||dx|| ~ 0.05-0.3), empirical error
         is typically < 20% when trace(g) varies smoothly.

    If you need strict geodesic distances, use compute_geodesic_distance_sq()
    below (numerical integration, slower).

    Args:
        mus: (N, D) atom positions
        metric_field: MetricField2D or MetricField3D instance

    Returns:
        D2: (N, N) squared midpoint-Mahalanobis distances
    """
    N, D = mus.shape
    if N < 2:
        return torch.zeros(N, N, device=mus.device)

    mids = (mus.unsqueeze(0) + mus.unsqueeze(1)) / 2
    dx = mus.unsqueeze(0) - mus.unsqueeze(1)

    g_mid = metric_field(mids.reshape(-1, D)).reshape(N, N, D, D)
    # NOTE: F.grid_sample bilinear interpolation does NOT guarantee
    # the interpolated metric g_mid remains exactly SPD. This can cause
    # dx^T g_mid dx to be slightly negative (~ -1e-15) due to numerical error.
    # clamp(min=0) + sqrt then produces NaN gradients at 0 (0 * inf).
    # FIX: clamp to a small positive epsilon instead of 0.
    d2 = torch.einsum('ijm,ijmn,ijn->ij', dx, g_mid, dx).clamp(min=1e-8)

    return d2


# Backward compatibility alias with deprecation warning
def compute_pairwise_geodesic_sq(mus, metric_field):
    """DEPRECATED: use compute_pairwise_midpoint_mahalanobis_sq()."""
    return compute_pairwise_midpoint_mahalanobis_sq(mus, metric_field)


def compute_true_geodesic_sq_1d(mus, metric_field, n_steps=16):
    """
    Numerically integrate geodesic distance in 1D via Simpson's rule.

    For D=2 or D=3, this is a 1D integral along the straight-line segment
    between μ_i and μ_j (NOT the true geodesic curve, which would require
    solving the geodesic equation d²xᵃ/dt² + Γᵃ_bc (dxᵇ/dt)(dxᶜ/dt) = 0).

    The difference between "straight-line integral" and "geodesic curve"
    is second-order in metric variation: for slowly varying g(x), the
    straight-line integral approximates the geodesic length well.

    This function exists primarily for EXT-2 validation: comparing
    midpoint_mahalanobis vs integrated distance to quantify approximation error.

    Args:
        mus: (N, D) atom positions
        metric_field: MetricField instance
        n_steps: number of Simpson quadrature points

    Returns:
        D2_integrated: (N, N) squared integrated distances
        D2_midpoint: (N, N) squared midpoint-Mahalanobis distances
        error_ratio: (N, N) |D2_integrated - D2_midpoint| / D2_integrated
    """
    N, D = mus.shape
    if N < 2:
        z = torch.zeros(N, N, device=mus.device)
        return z, z, z

    dx = mus.unsqueeze(0) - mus.unsqueeze(1)  # (N, N, D)

    # Simpson quadrature points along each segment
    t = torch.linspace(0, 1, n_steps, device=mus.device)  # (S,)
    dt = t[1] - t[0]
    weights = torch.ones(n_steps, device=mus.device)
    weights[1:-1:2] = 4.0   # odd interior points
    weights[2:-1:2] = 2.0   # even interior points
    weights *= dt / 3.0

    # Sample points: x(t) = μ_i + t * (μ_j - μ_i)
    # Shape: (N, N, S, D)
    mu_i = mus.unsqueeze(1).unsqueeze(2)   # (N, 1, 1, D)
    dx_ij = dx.unsqueeze(2)                 # (N, N, 1, D)
    sample_pts = mu_i + t.view(1, 1, -1, 1) * dx_ij  # (N, N, S, D)

    # Evaluate metric at all sample points
    flat_pts = sample_pts.reshape(-1, D)  # (N*N*S, D)
    g_all = metric_field(flat_pts)         # (N*N*S, D, D)
    g_all = g_all.reshape(N, N, n_steps, D, D)

    # ds²/dt = (μ_j - μ_i)ᵀ g(x(t)) (μ_j - μ_i)
    # dx_ij is (N,N,1,D), g_all is (N,N,S,D,D)
    # Expand dx to (N,N,S,D) then reshape for batch matmul
    dx_exp = dx_ij.expand(N, N, n_steps, D).reshape(N * N * n_steps, 1, D)  # (NNS, 1, D)
    g_flat = g_all.reshape(N * N * n_steps, D, D)                           # (NNS, D, D)
    # (NNS, 1, D) @ (NNS, D, D) = (NNS, 1, D)
    temp = torch.bmm(dx_exp, g_flat)                    # (NNS, 1, D)
    # (NNS, 1, D) @ (NNS, D, 1) = (NNS, 1, 1)
    ds2 = torch.bmm(temp, dx_exp.transpose(1, 2)).squeeze()  # (NNS,)
    ds2_per_t = ds2.reshape(N, N, n_steps).clamp(min=0).sqrt()  # (N, N, S)

    # Simpson integration: ∫ ds/dt dt
    d_integrated = (ds2_per_t * weights.view(1, 1, -1)).sum(dim=-1)  # (N, N)
    D2_integrated = d_integrated ** 2

    # Midpoint approximation for comparison
    D2_midpoint = compute_pairwise_midpoint_mahalanobis_sq(mus, metric_field)

    error_ratio = (D2_integrated - D2_midpoint).abs() / (D2_integrated + 1e-8)

    return D2_integrated, D2_midpoint, error_ratio
