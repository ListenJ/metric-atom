#!/usr/bin/env python3
"""
Diffusion Kernel PSD Verification
===================================
Tests whether the geodesic affinity kernel A_ij = exp(-d²_ij/(2σ_iσ_j))
in MetricAtom's diffusion.py is positive semi-definite under realistic conditions.

Theoretical background:
  - Standard Gaussian RBF kernel K(x,y)=exp(-||x-y||²/(2σ²)) with constant σ is PSD
  - Adaptive bandwidth σ_i·σ_j breaks the Mercer condition
  - Sigmoid mask: A *= sigmoid((τ-d)/s) modifies eigenvalues unpredictably
  - Row normalization S = D⁻¹A is not guaranteed PSD for asymmetric D⁻¹A

Tests:
  1. Constant bandwidth (baseline, should be PSD)
  2. Adaptive bandwidth σ_i·σ_j (main concern)
  3. With sigmoid mask (further PSD degradation)
  4. Row-normalized (graph Laplacian form, not necessarily PSD)
  5. Monte Carlo: random metric fields, count negative eigenvalues

Usage:
  python tasks/sim_diffusion_kernel_psd.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import sys


@dataclass
class PSDResult:
    """Result of one PSD check"""
    name: str
    min_eig: float
    max_eig: float
    cond: float  # condition number
    n_negative: int
    is_psd: bool
    fraction_negative: float  # λ_min / λ_max if negative


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def make_metric_field(points: np.ndarray) -> np.ndarray:
    """
    Construct a Cholesky parameterized 2D metric field g(x).
    g = LL^T + eps*I  at each point.
    
    Returns: (N, 1, 2, 2) metric tensor for N points
    """
    N, D = points.shape
    eps = 1e-4
    
    # Simple parameterization: metric depends on y-coordinate
    # g_11 = 1 + y², g_22 = 1, g_12 = g_21 = 0.5*y
    g = np.zeros((N, 2, 2))
    y = points[:, 1]
    g[:, 0, 0] = 1.0 + y**2 + eps
    g[:, 0, 1] = 0.5 * y
    g[:, 1, 0] = 0.5 * y
    g[:, 1, 1] = 1.0 + eps
    
    return g


def compute_geodesic_d2(pi: np.ndarray, pj: np.ndarray, g_mid: np.ndarray) -> float:
    """
    Geodesic distance squared using midpoint metric:
    d² = dx^T · g(mid) · dx
    """
    dx = pi - pj
    d2 = dx @ g_mid @ dx
    return max(d2, 0.0)


def compute_affinity_matrix(
    points: np.ndarray,
    g: np.ndarray,
    adaptive_sigma: bool = True,
    k_nearest: int = 5,
    use_sigmoid_mask: bool = True,
    tau_factor: float = 3.0,
    s_factor: float = 0.1,
) -> np.ndarray:
    """
    Build the geodesic affinity matrix A.
    Replicates diffusion.py compute_geodesic_affinity logic.
    """
    N = points.shape[0]
    A = np.zeros((N, N))
    
    # Step 1: compute all pairwise geodesic distances (symmetric)
    D2 = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            mid = (points[i] + points[j]) / 2.0
            g_mid = g[i]  # simplified: use i's metric as midpoint approx
            d2 = compute_geodesic_d2(points[i], points[j], g_mid)
            D2[i, j] = D2[j, i] = d2
    
    D = np.sqrt(D2 + 1e-8)
    
    # Step 2: adaptive sigma per point (= K-th nearest neighbor distance)
    if adaptive_sigma:
        sigmas = np.sort(D, axis=1)[:, min(k_nearest, N-1)]
        sigmas = np.maximum(sigmas, 1e-6)
    else:
        sigmas = np.full(N, np.median(D[D > 0]) if np.any(D > 0) else 1.0)
    
    # Step 3: build kernel
    for i in range(N):
        for j in range(i+1, N):
            sigma_prod = sigmas[i] * sigmas[j]
            a = np.exp(-D2[i, j] / (2.0 * sigma_prod + 1e-8))
            A[i, j] = A[j, i] = a
    
    # Step 4: sigmoid mask
    if use_sigmoid_mask:
        tau_max = tau_factor * np.mean(sigmas)
        s = s_factor * tau_max
        soft_mask = sigmoid((tau_max - D) / s)
        A = A * soft_mask
        # re-symmetrize (mask may break symmetry slightly)
        A = (A + A.T) / 2.0
    
    # Step 5: zero diagonal
    np.fill_diagonal(A, 0.0)
    
    return A, D, sigmas


def compute_row_normalized(A: np.ndarray) -> np.ndarray:
    """Row normalization: S = D⁻¹A."""
    deg = A.sum(axis=1)
    deg = np.maximum(deg, 1e-8)
    D_inv = np.diag(1.0 / deg)
    return D_inv @ A


def check_psd(M: np.ndarray, tol: float = 1e-10) -> PSDResult:
    """Compute eigenvalues and check PSD."""
    eigs = np.linalg.eigvalsh(M)
    n_negative = int(np.sum(eigs < -tol))
    min_eig = float(np.min(eigs))
    max_eig = float(np.max(eigs))
    is_psd = n_negative == 0
    cond = float(max_eig / max(abs(min_eig), 1e-15))
    frac = abs(min_eig) / max_eig if min_eig < -tol else 0.0
    
    return PSDResult(
        name="", min_eig=min_eig, max_eig=max_eig,
        cond=cond, n_negative=n_negative, is_psd=is_psd,
        fraction_negative=frac
    )


def generate_clustered_points(
    n_clusters: int = 3,
    n_per_cluster: int = 10,
    spread: float = 0.8,
    separation: float = 4.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate 2D points with cluster structure."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((n_clusters, 2))
    for k in range(n_clusters):
        angle = 2 * np.pi * k / n_clusters
        centers[k] = [separation * np.cos(angle), separation * np.sin(angle)]
    
    points = []
    for k in range(n_clusters):
        cluster_pts = centers[k] + spread * rng.randn(n_per_cluster, 2)
        points.append(cluster_pts)
    
    return np.vstack(points)


def run_single_test(
    name: str,
    points: np.ndarray,
    g: np.ndarray,
    adaptive_sigma: bool,
    k_nearest: int,
    use_sigmoid_mask: bool,
    tau_factor: float,
    s_factor: float,
    row_normalize: bool,
) -> PSDResult:
    """Run one PSD test configuration."""
    A, D, sigmas = compute_affinity_matrix(
        points, g, adaptive_sigma=adaptive_sigma,
        k_nearest=k_nearest, use_sigmoid_mask=use_sigmoid_mask,
        tau_factor=tau_factor, s_factor=s_factor,
    )
    
    if row_normalize:
        M = compute_row_normalized(A)
    else:
        M = A
    
    result = check_psd(M)
    result.name = name
    return result


def monte_carlo_psd(
    n_trials: int = 100,
    n_points: int = 30,
    n_clusters: int = 3,
    adaptive: bool = True,
    seed: int = 123,
) -> dict:
    """
    Monte Carlo: generate many random point configurations
    and count how often the kernel fails PSD.
    """
    rng = np.random.RandomState(seed)
    n_psd_fail = 0
    min_eigs = []
    
    for trial in range(n_trials):
        # Random cluster configuration
        n_c = rng.randint(2, 5)
        n_pc = max(5, n_points // n_c)
        
        points = generate_clustered_points(
            n_clusters=n_c, n_per_cluster=n_pc,
            spread=0.5 + rng.rand(), separation=2.0 + 3.0 * rng.rand(),
            seed=seed + trial * 1000,
        )
        g = make_metric_field(points)
        
        A, _, _ = compute_affinity_matrix(
            points, g, adaptive_sigma=adaptive,
            k_nearest=5, use_sigmoid_mask=True,
        )
        
        eigs = np.linalg.eigvalsh(A)
        n_neg = int(np.sum(eigs < -1e-10))
        if n_neg > 0:
            n_psd_fail += 1
        min_eigs.append(float(np.min(eigs)))
    
    return {
        "n_trials": n_trials,
        "n_psd_failures": n_psd_fail,
        "psd_fail_rate": n_psd_fail / n_trials,
        "mean_min_eig": float(np.mean(min_eigs)),
        "std_min_eig": float(np.std(min_eigs)),
        "min_of_min_eig": float(np.min(min_eigs)),
        "max_of_min_eig": float(np.max(min_eigs)),
    }


def print_separator(title: str = ""):
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")


def main():
    print_separator("Diffusion Kernel PSD Verification")
    
    # Generate test points
    points = generate_clustered_points(n_clusters=3, n_per_cluster=10, seed=42)
    g = make_metric_field(points)
    N = len(points)
    print(f"  Points: {N}  |  Metric field: g = LL^T + εI")
    
    # ──────────────────────────────────────────
    # Test 1: Constant bandwidth (baseline)
    # ──────────────────────────────────────────
    print_separator("Test 1: Constant bandwidth σ (should be PSD)")
    r1_no_norm = run_single_test(
        "1a. Constant σ, no mask, raw A", points, g,
        adaptive_sigma=False, k_nearest=5,
        use_sigmoid_mask=False, tau_factor=3.0, s_factor=0.1,
        row_normalize=False,
    )
    r1_mask = run_single_test(
        "1b. Constant σ, + sigmoid mask, raw A", points, g,
        adaptive_sigma=False, k_nearest=5,
        use_sigmoid_mask=True, tau_factor=3.0, s_factor=0.1,
        row_normalize=False,
    )
    r1_norm = run_single_test(
        "1c. Constant σ, + mask, row-normalized S", points, g,
        adaptive_sigma=False, k_nearest=5,
        use_sigmoid_mask=True, tau_factor=3.0, s_factor=0.1,
        row_normalize=True,
    )
    
    for r in [r1_no_norm, r1_mask, r1_norm]:
        status = "✓ PSD" if r.is_psd else "✗ NOT PSD"
        print(f"  {r.name:50s} λ_min={r.min_eig:+.2e}  {status}")
    
    # ──────────────────────────────────────────
    # Test 2: Adaptive bandwidth
    # ──────────────────────────────────────────
    print_separator("Test 2: Adaptive bandwidth σ_i·σ_j (MetricAtom default)")
    r2_no_mask = run_single_test(
        "2a. Adaptive σ, no mask, raw A", points, g,
        adaptive_sigma=True, k_nearest=5,
        use_sigmoid_mask=False, tau_factor=3.0, s_factor=0.1,
        row_normalize=False,
    )
    r2_mask = run_single_test(
        "2b. Adaptive σ, + sigmoid mask, raw A", points, g,
        adaptive_sigma=True, k_nearest=5,
        use_sigmoid_mask=True, tau_factor=3.0, s_factor=0.1,
        row_normalize=False,
    )
    r2_norm = run_single_test(
        "2c. Adaptive σ, + mask, row-normalized S", points, g,
        adaptive_sigma=True, k_nearest=5,
        use_sigmoid_mask=True, tau_factor=3.0, s_factor=0.1,
        row_normalize=True,
    )
    
    for r in [r2_no_mask, r2_mask, r2_norm]:
        status = "✓ PSD" if r.is_psd else "✗ NOT PSD"
        print(f"  {r.name:50s} λ_min={r.min_eig:+.2e}  {status}")
    
    # ──────────────────────────────────────────
    # Test 3: Varying K-nearest for sigma
    # ──────────────────────────────────────────
    print_separator("Test 3: Sensitivity to K-nearest (adaptive σ)")
    for k in [3, 5, 7, 10]:
        r = run_single_test(
            f"K={k:2d}, adaptive σ, + mask", points, g,
            adaptive_sigma=True, k_nearest=k,
            use_sigmoid_mask=True, tau_factor=3.0, s_factor=0.1,
            row_normalize=False,
        )
        status = "✓ PSD" if r.is_psd else "✗ NOT PSD"
        print(f"  {r.name:50s} λ_min={r.min_eig:+.2e}  {status}")
    
    # ──────────────────────────────────────────
    # Test 4: Different metric field strengths
    # ──────────────────────────────────────────
    print_separator("Test 4: Metric field anisotropy sensitivity")
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        # Scale points to change metric field variation
        pts_scaled = points * scale
        g_scaled = make_metric_field(pts_scaled)
        r = run_single_test(
            f"Scale={scale:.1f}, adaptive σ, + mask", pts_scaled, g_scaled,
            adaptive_sigma=True, k_nearest=5,
            use_sigmoid_mask=True, tau_factor=3.0, s_factor=0.1,
            row_normalize=False,
        )
        status = "✓ PSD" if r.is_psd else "✗ NOT PSD"
        print(f"  {r.name:50s} λ_min={r.min_eig:+.2e}  {status}")
    
    # ──────────────────────────────────────────
    # Test 5: Monte Carlo
    # ──────────────────────────────────────────
    print_separator("Test 5: Monte Carlo (100 random configurations)")
    mc_adaptive = monte_carlo_psd(n_trials=100, n_points=30, n_clusters=3, adaptive=True, seed=123)
    mc_constant = monte_carlo_psd(n_trials=100, n_points=30, n_clusters=3, adaptive=False, seed=123)
    
    print(f"  Adaptive σ (default):")
    print(f"    PSD failures: {mc_adaptive['n_psd_failures']}/{mc_adaptive['n_trials']} "
          f"({100*mc_adaptive['psd_fail_rate']:.1f}%)")
    print(f"    Min eig: mean={mc_adaptive['mean_min_eig']:+.2e}, "
          f"worst={mc_adaptive['min_of_min_eig']:+.2e}")
    
    print(f"\n  Constant σ:")
    print(f"    PSD failures: {mc_constant['n_psd_failures']}/{mc_constant['n_trials']} "
          f"({100*mc_constant['psd_fail_rate']:.1f}%)")
    print(f"    Min eig: mean={mc_constant['mean_min_eig']:+.2e}, "
          f"worst={mc_constant['min_of_min_eig']:+.2e}")
    
    # ──────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────
    print_separator("VERDICT")
    if mc_adaptive['psd_fail_rate'] > 0.05:
        print(f"  ⚠  Kernel is NOT reliably PSD with adaptive bandwidth!")
        print(f"     {100*mc_adaptive['psd_fail_rate']:.0f}% of random configs produce negative eigenvalues.")
        print(f"     Fix: Use constant bandwidth (global σ) for theoretical PSD guarantee.")
    else:
        print(f"  ✓ Kernel appears numerically PSD in {100*(1-mc_adaptive['psd_fail_rate']):.0f}% of cases.")
        print(f"    Adaptive bandwidth is empirically safe for these parameter ranges.")
    
    print(   "\n  Recommendation: Use constant σ = median(d_ij) for provable PSD.")
    print(f"  If adaptive is needed, apply explicit symmetrization: A ← (A + A^T)/2")
    print(f"  and clamp negative eigenvalues to 0 for downstream spectral methods.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
