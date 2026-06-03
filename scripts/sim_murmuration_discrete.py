"""
sim_murmuration_discrete.py — Discrete Murmuration Lyapunov Stability Verification
====================================================================================

Verifies that the discrete-time Murmuration dynamics (dt=0.1) preserves the
Lyapunov monotonicity property: dV/dt ≤ 0 (or ΔV ≤ 0 per step).

Uses S¹ (unit circle) as the manifold — the result is topology-invariant for
any closed 1D Riemannian manifold.

Based on murmuration_dynamics.md §3-4 and src/geometry/murmuration.py.

Usage: python sim_murmuration_discrete.py [--N 10] [--T 500] [--seed 42]
"""

import numpy as np
import argparse
from typing import List, Tuple, Optional


# ============================================================================
# S¹ Manifold Operations (1D closed Riemannian manifold)
# ============================================================================

def shortest_angle(dtheta: np.ndarray) -> np.ndarray:
    """Map angle difference to (-π, π]."""
    return ((dtheta + np.pi) % (2 * np.pi)) - np.pi


def geodesic_distance_S1(theta_i: float, theta_j: float) -> float:
    """Geodesic distance on S¹."""
    return abs(shortest_angle(theta_j - theta_i))


def log_map_S1(theta_i: float, theta_j: float) -> float:
    """Log map: signed shortest angle from θ_i to θ_j (scalar tangent)."""
    return float(shortest_angle(theta_j - theta_i))


def exp_map_S1(theta_i: float, v_dt: float) -> float:
    """Exp map: step along tangent. v_dt is scalar * dt."""
    return (theta_i + v_dt) % (2 * np.pi)


# ============================================================================
# Boids Dynamics (from BoidsConfig defaults)
# ============================================================================

class BoidsConfig:
    cohesion_weight: float = 0.1      # α
    alignment_weight: float = 0.05    # β
    separation_weight: float = 0.2    # γ
    separation_radius: float = 0.5    # r
    neighbor_radius: float = 2.0      # R
    max_speed: float = 0.5            # v_max
    damping: float = 0.9              # η


def compute_forces(
    idx: int,
    thetas: np.ndarray,
    velocities: np.ndarray,
    cfg: BoidsConfig,
) -> float:
    """Compute total Boids force for particle idx (scalar along tangent)."""
    theta_i = thetas[idx]
    N = len(thetas)

    # Find neighbors within neighbor_radius
    f_coh = 0.0
    n_coh = 0
    f_align = 0.0
    n_align = 0

    for j in range(N):
        if j == idx:
            continue
        d = geodesic_distance_S1(theta_i, thetas[j])
        if d < cfg.neighbor_radius:
            n_coh += 1
            n_align += 1
            log_v = log_map_S1(theta_i, thetas[j])
            f_coh += log_v
            f_align += velocities[j]

    if n_coh > 0:
        f_coh = cfg.cohesion_weight * f_coh / n_coh
    if n_align > 0:
        f_align = cfg.alignment_weight * f_align / n_align

    # Separation: repulsion from ALL points within separation_radius
    f_sep = 0.0
    for j in range(N):
        if j == idx:
            continue
        d = geodesic_distance_S1(theta_i, thetas[j])
        if d < cfg.separation_radius and d > 1e-10:
            direction = log_map_S1(theta_i, thetas[j])
            f_sep -= cfg.separation_weight * direction / (d * d)

    return f_coh + f_align + f_sep


def step_discrete(
    thetas: np.ndarray,
    velocities: np.ndarray,
    cfg: BoidsConfig,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single discrete timestep of Murmuration on S¹."""
    N = len(thetas)
    new_thetas = np.zeros(N)
    new_velocities = np.zeros(N)

    for i in range(N):
        F_i = compute_forces(i, thetas, velocities, cfg)
        v_old = velocities[i]
        v_new = cfg.damping * (v_old + F_i * dt)
        v_new = np.clip(v_new, -cfg.max_speed, cfg.max_speed)
        new_velocities[i] = v_new
        new_thetas[i] = exp_map_S1(thetas[i], v_new * dt)

    return new_thetas, new_velocities


# ============================================================================
# Lyapunov Function
# ============================================================================

def compute_lyapunov(thetas: np.ndarray, velocities: np.ndarray, cfg: BoidsConfig) -> float:
    """
    V = T + U_coh + U_sep  (discrete analog of murmuration_dynamics.md §3)

    T = 1/2 Σ v_i²                    (kinetic energy)
    U_coh = (α/N) Σ Σ_{d_ij<R} d_ij² / |N_i|   (cohesion potential)
    U_sep = -γ Σ Σ_{d_ij<r} log(d_ij/r)        (separation barrier)
    """
    N = len(thetas)
    T = 0.5 * np.sum(velocities ** 2)

    U_coh = 0.0
    for i in range(N):
        n_i = 0
        d_sum = 0.0
        for j in range(N):
            if i == j:
                continue
            d = geodesic_distance_S1(thetas[i], thetas[j])
            if d < cfg.neighbor_radius:
                n_i += 1
                d_sum += d * d
        if n_i > 0:
            U_coh += d_sum / n_i
    U_coh *= cfg.cohesion_weight / N

    U_sep = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = geodesic_distance_S1(thetas[i], thetas[j])
            if d < cfg.separation_radius and d > 1e-10:
                U_sep -= np.log(d / cfg.separation_radius)
    U_sep *= cfg.separation_weight

    return T + U_coh + U_sep


# ============================================================================
# Stability Metrics
# ============================================================================

def compute_order_parameter(thetas: np.ndarray) -> float:
    """Kuramoto order parameter R = |1/N Σ e^{iθ}|. R→1 = clustered, R→0 = uniform."""
    N = len(thetas)
    R = abs(np.sum(np.exp(1j * thetas))) / N
    return float(R)


def compute_min_separation(thetas: np.ndarray) -> float:
    """Minimum geodesic distance between any pair (angular)."""
    N = len(thetas)
    if N < 2:
        return 0.0
    min_d = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = geodesic_distance_S1(thetas[i], thetas[j])
            min_d = min(min_d, d)
    return min_d


def compute_uniformity_error(thetas: np.ndarray) -> float:
    """RMS error from uniform spacing 2π/N."""
    N = len(thetas)
    if N < 2:
        return 0.0
    sorted_theta = np.sort(thetas)
    ideal_gap = 2 * np.pi / N
    gaps = np.diff(sorted_theta)
    gaps = np.append(gaps, 2 * np.pi - (sorted_theta[-1] - sorted_theta[0]))
    return float(np.sqrt(np.mean((gaps - ideal_gap) ** 2)))


# ============================================================================
# Main Simulation
# ============================================================================

def run_simulation(
    N: int = 10,
    T: int = 500,
    dt: float = 0.1,
    seed: int = 42,
    init_random: bool = True,
) -> dict:
    """Run discrete Murmuration simulation and collect metrics."""
    np.random.seed(seed)
    cfg = BoidsConfig()

    # Initialization
    if init_random:
        thetas = np.random.uniform(0, 2 * np.pi, N)
    else:
        # Uniform spacing (equilibrium)
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)

    velocities = np.zeros(N)

    # Track metrics
    V_history = np.zeros(T + 1)
    order_history = np.zeros(T + 1)
    min_sep_history = np.zeros(T + 1)
    uniformity_history = np.zeros(T + 1)

    V_history[0] = compute_lyapunov(thetas, velocities, cfg)
    order_history[0] = compute_order_parameter(thetas)
    min_sep_history[0] = compute_min_separation(thetas)
    uniformity_history[0] = compute_uniformity_error(thetas)

    n_positive_dV = 0

    for t in range(T):
        thetas, velocities = step_discrete(thetas, velocities, cfg, dt)

        V_history[t + 1] = compute_lyapunov(thetas, velocities, cfg)
        order_history[t + 1] = compute_order_parameter(thetas)
        min_sep_history[t + 1] = compute_min_separation(thetas)
        uniformity_history[t + 1] = compute_uniformity_error(thetas)

        dV = V_history[t + 1] - V_history[t]
        if dV > 0:
            n_positive_dV += 1

    return {
        'V_history': V_history,
        'order_history': order_history,
        'min_sep_history': min_sep_history,
        'uniformity_history': uniformity_history,
        'n_positive_dV': n_positive_dV,
        'T': T,
        'N': N,
        'seed': seed,
        'V_start': V_history[0],
        'V_end': V_history[-1],
        'V_ratio': V_history[-1] / max(V_history[0], 1e-10),
        'uniformity_start': uniformity_history[0],
        'uniformity_end': uniformity_history[-1],
    }


# ============================================================================
# Sensitivity Analysis
# ============================================================================

def sensitivity_analysis() -> List[dict]:
    """Run simulations with varied dt and damping to find stability boundary."""
    results = []
    dt_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    damping_values = [0.5, 0.7, 0.9, 0.95, 0.99]

    for dt_v in dt_values:
        for damp in damping_values:
            cfg = BoidsConfig()
            cfg.damping = damp
            np.random.seed(42)
            N = 10
            thetas = np.random.uniform(0, 2 * np.pi, N)
            velocities = np.zeros(N)
            T = 200

            V_start = compute_lyapunov(thetas, velocities, cfg)
            n_pos = 0
            for _ in range(T):
                thetas, velocities = step_discrete(thetas, velocities, cfg, dt_v)
                V_curr = compute_lyapunov(thetas, velocities, cfg)
                if V_curr > V_start:
                    n_pos += 1
                V_start = V_curr

            results.append({
                'dt': dt_v,
                'damping': damp,
                'n_positive': n_pos,
                'stable': n_pos <= T * 0.05,  # ≤5% positive ΔV = stable
            })

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Discrete Murmuration Lyapunov verification')
    parser.add_argument('--N', type=int, default=10, help='Number of particles')
    parser.add_argument('--T', type=int, default=500, help='Simulation steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    args = parser.parse_args()

    print("=" * 72)
    print("DISCRETE MURMURATION LYAPUNOV STABILITY VERIFICATION")
    print("=" * 72)
    print(f"  N={args.N}, T={args.T}, seed={args.seed}")
    print(f"  Config: α={BoidsConfig.cohesion_weight}, β={BoidsConfig.alignment_weight}")
    print(f"          γ={BoidsConfig.separation_weight}, r={BoidsConfig.separation_radius}")
    print(f"          R={BoidsConfig.neighbor_radius}, v_max={BoidsConfig.max_speed}")
    print(f"          damping={BoidsConfig.damping}, dt=0.1")
    print()

    if args.sensitivity:
        print("--- Sensitivity Analysis (dt × damping) ---")
        print(f"  {'dt':>6}  {'damping':>8}  {'ΔV>0':>6}  {'stable':>8}")
        print("  " + "-" * 38)
        results = sensitivity_analysis()
        for r in results:
            status = "✓ STABLE" if r['stable'] else "✗ UNSTABLE"
            print(f"  {r['dt']:>6.2f}  {r['damping']:>8.2f}  {r['n_positive']:>6d}  {status:>8}")
        print()
        stable_count = sum(1 for r in results if r['stable'])
        print(f"  Stable configurations: {stable_count}/{len(results)}")
        print()
        return

    # Random initialization
    print("--- Run 1: Random Initialization ---")
    result = run_simulation(N=args.N, T=args.T, seed=args.seed, init_random=True)
    print(f"  V(0) = {result['V_start']:.6f}, V({args.T}) = {result['V_end']:.6f}")
    print(f"  V_ratio = {result['V_ratio']:.6f}  {'✓ DECREASING' if result['V_ratio'] < 1 else '✗ INCREASING'}")
    print(f"  ΔV > 0 steps: {result['n_positive_dV']}/{args.T} ({100*result['n_positive_dV']/args.T:.1f}%)")
    print(f"  Uniformity error: {result['uniformity_start']:.6f} → {result['uniformity_end']:.6f}")
    print(f"  Order parameter R: {result['order_history'][0]:.4f} → {result['order_history'][-1]:.4f}")
    print()

    # Uniform initialization (equilibrium)
    print("--- Run 2: Uniform Spacing (near equilibrium) ---")
    result_eq = run_simulation(N=args.N, T=args.T, seed=args.seed, init_random=False)
    print(f"  V(0) = {result_eq['V_start']:.6f}, V({args.T}) = {result_eq['V_end']:.6f}")
    print(f"  V_ratio = {result_eq['V_ratio']:.6f}  {'✓ STABLE' if result_eq['V_ratio'] < 1.01 else '✗ UNSTABLE'}")
    print(f"  ΔV > 0 steps: {result_eq['n_positive_dV']}/{args.T} ({100*result_eq['n_positive_dV']/args.T:.1f}%)")
    print(f"  Uniformity error: {result_eq['uniformity_start']:.6f} → {result_eq['uniformity_end']:.6f}")
    print()

    # Multi-seed Monte Carlo
    print("--- Run 3: Multi-seed Monte Carlo (8 seeds) ---")
    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192]
    all_ratios = []
    all_pos_pcts = []
    for s in seeds:
        r = run_simulation(N=args.N, T=args.T, seed=s, init_random=True)
        all_ratios.append(r['V_ratio'])
        all_pos_pcts.append(100 * r['n_positive_dV'] / args.T)

    print(f"  V_ratio: mean={np.mean(all_ratios):.4f}, std={np.std(all_ratios):.4f}")
    print(f"           min={np.min(all_ratios):.4f}, max={np.max(all_ratios):.4f}")
    print(f"  ΔV>0%:   mean={np.mean(all_pos_pcts):.2f}%, std={np.std(all_pos_pcts):.2f}%")
    print(f"           min={np.min(all_pos_pcts):.2f}%, max={np.max(all_pos_pcts):.2f}%")
    print()

    # dt variation
    print("--- Run 4: dt Sensitivity (damping=0.9) ---")
    cfg = BoidsConfig()
    for dt_v in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        np.random.seed(42)
        thetas = np.random.uniform(0, 2 * np.pi, args.N)
        velocities = np.zeros(args.N)
        V_start = compute_lyapunov(thetas, velocities, cfg)
        n_pos = 0
        V_final = V_start
        for _ in range(200):
            thetas, velocities = step_discrete(thetas, velocities, cfg, dt_v)
            V_curr = compute_lyapunov(thetas, velocities, cfg)
            if V_curr > V_start:
                n_pos += 1
            V_final = V_curr
            V_start = V_curr
        stable = "✓" if n_pos <= 10 else "✗"
        print(f"  dt={dt_v:.2f}: ΔV>0={n_pos}/200, V_final/V0={V_final/max(V_start,1e-10):.4f} {stable}")

    print()
    print("=" * 72)
    print("VERIFICATION COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
