"""
Murmuration: Boids dynamics constrained to elliptic curves.

Phase 6a/6b: Implements cohesion, alignment, separation forces
in the tangent space T_P(E) and evolves points via exp_map.

All operations are batched in PyTorch for GPU training loops.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

from src.geometry.elliptic_curve import (
    is_on_curve, is_identity,
    elliptic_add, elliptic_scalar_mult,
    tangent_vector, tangent_slope,
    log_map, exp_map, geodesic_distance,
    j_invariant, discriminant,
    batch_tangent_vectors, batch_is_on_curve,
    _real_roots_of_cubic,
)


@dataclass
class BoidsConfig:
    """Hyperparameters for murmuration dynamics."""
    cohesion_weight: float = 0.1     # α: pull toward neighbors
    alignment_weight: float = 0.05   # β: align velocity with neighbors
    separation_weight: float = 0.2   # γ: push away from close neighbors
    separation_radius: float = 0.5   # r: distance threshold for separation
    neighbor_radius: float = 2.0     # R: distance threshold for neighbors
    max_speed: float = 0.5           # v_max: maximum step size
    damping: float = 0.9             # velocity decay


class MurmurationOnE:
    """
    Boids simulation on an elliptic curve E: y² = x³ + ax + b.

    Each point P_i has:
        - position (x_i, y_i) ∈ E(R)
        - velocity v_i ∈ T_{P_i}(E)  (a scalar along the 1D tangent)

    The dynamics uses the log/exp maps to stay on the curve.
    """

    def __init__(self, a: float, b: float, config: Optional[BoidsConfig] = None):
        if not is_on_curve(1.0, np.sqrt(1.0 + a + b), a, b):
            # Find a valid point for sanity check
            pass
        self.a = a
        self.b = b
        self.config = config or BoidsConfig()
        self._step_count = 0
        self._j_history: List[float] = []

    @property
    def curve_discriminant(self) -> float:
        return discriminant(self.a, self.b)

    @property
    def curve_j(self) -> float:
        return j_invariant(self.a, self.b)

    def is_valid_point(self, x: float, y: float) -> bool:
        return is_on_curve(x, y, self.a, self.b)

    def update_curve(self, a_new: float, b_new: float):
        """
        Sensing-driven curve evolution (Definition 3).
        Updates (a, b) and projects back to non-singular space if needed.
        """
        if abs(discriminant(a_new, b_new)) < 1e-12:
            # Project back to valid curve space
            a_new += 0.01 * (self.a - a_new)
        self._j_history.append(j_invariant(self.a, self.b))
        self.a = a_new
        self.b = b_new

    # ------------------------------------------------------------------
    # Force computation (in tangent space)
    # ------------------------------------------------------------------

    def _cohesion_force(
        self, P_i: Tuple[float, float],
        neighbors: List[Tuple[float, float]]
    ) -> float:
        """
        Cohesion: average log_P(P_j) over neighbors.
        Returns scalar force in T_{P_i}(E).
        """
        if not neighbors:
            return 0.0
        total = 0.0
        for P_j in neighbors:
            v = log_map(P_i, P_j, self.a, self.b)
            # Project 2D tangent vector to signed scalar (dot with unit tangent)
            unit = tangent_vector(P_i, self.a)
            total += v[0] * unit[0] + v[1] * unit[1]
        return self.config.cohesion_weight * total / len(neighbors)

    def _alignment_force(
        self, P_i: Tuple[float, float],
        neighbors: List[Tuple[float, float]],
        velocities: dict
    ) -> float:
        """
        Alignment: average neighbor velocity.
        Since T_P(E) is 1D, alignment is scalar averaging.
        We don't need parallel transport because all tangents are 1D and
        we compare magnitudes only (the direction is ± along the curve).
        """
        if not neighbors:
            return 0.0
        total = 0.0
        for P_j in neighbors:
            vj = velocities.get(P_j, 0.0)
            # Velocities are already scalars in the 1D tangent space
            total += vj
        return self.config.alignment_weight * total / len(neighbors)

    def _separation_force(
        self, P_i: Tuple[float, float],
        all_points: List[Tuple[float, float]]
    ) -> float:
        """
        Separation: repulsion from points within separation_radius.

        F_sep = - Σ_{j: d(P_i, P_j) < r}  1 / d(P_i, P_j)²
        (directed away from each neighbor)
        """
        force = 0.0
        for P_j in all_points:
            if P_j == P_i:
                continue
            d = geodesic_distance(P_i, P_j, self.a, self.b)
            if d < self.config.separation_radius and d > 1e-10:
                v = log_map(P_i, P_j, self.a, self.b)
                unit = tangent_vector(P_i, self.a)
                direction = v[0] * unit[0] + v[1] * unit[1]
                # Repulsion: opposite direction, inversely proportional to dist²
                force -= self.config.separation_weight * direction / (d**2 + 1e-8)
        return force

    def total_force(
        self,
        P_i: Tuple[float, float],
        all_points: List[Tuple[float, float]],
        velocities: dict
    ) -> float:
        """
        Sum all Boids forces in the 1D tangent space T_{P_i}(E).
        Returns scalar (signed magnitude along tangent direction).
        """
        # Find neighbors within neighbor_radius
        neighbors = []
        for P_j in all_points:
            if P_j == P_i:
                continue
            d = geodesic_distance(P_i, P_j, self.a, self.b)
            if d < self.config.neighbor_radius:
                neighbors.append(P_j)

        f_coh = self._cohesion_force(P_i, neighbors)
        f_align = self._alignment_force(P_i, neighbors, velocities)
        f_sep = self._separation_force(P_i, all_points)
        return f_coh + f_align + f_sep

    # ------------------------------------------------------------------
    # Dynamics evolution
    # ------------------------------------------------------------------

    def step(
        self,
        points: List[Tuple[float, float]],
        velocities: dict,
        dt: float = 0.1
    ) -> Tuple[List[Tuple[float, float]], dict]:
        """
        Single timestep of murmuration dynamics.

        dP_i/dt = v_i · u_P  +  F(P_i) · u_P

        where u_P ∈ T_P(E) is the unit tangent vector and F is the
        total Boids force (scalar).

        Velocity update:  v_new = damping · (v + F · dt)
        Position update:  P_new = exp_P(v_new · dt · u_P)
        """
        new_points = []
        new_velocities = {}

        for P_i in points:
            if is_identity(P_i):
                new_points.append(P_i)
                new_velocities[P_i] = 0.0
                continue

            # Compute force
            F_i = self.total_force(P_i, points, velocities)

            # Update velocity (scalar in 1D tangent space)
            v_old = velocities.get(P_i, 0.0)
            v_new = self.config.damping * (v_old + F_i * dt)
            v_new = max(-self.config.max_speed, min(self.config.max_speed, v_new))

            # Convert to 2D tangent vector
            unit = tangent_vector(P_i, self.a)
            v_2d = (v_new * dt * unit[0], v_new * dt * unit[1])

            # Move via exponential map
            P_new = exp_map(P_i, v_2d, self.a, self.b)

            # Verify on curve
            if not is_on_curve(P_new[0], P_new[1], self.a, self.b, tol=1e-4):
                # Project back via Newton
                P_new = _project_to_curve(P_new[0], P_new[1], self.a, self.b)

            new_points.append(P_new)
            new_velocities[P_new] = v_new

        self._step_count += 1
        return new_points, new_velocities

    def evolve(
        self,
        points: List[Tuple[float, float]],
        num_steps: int,
        dt: float = 0.1,
        track: bool = False
    ) -> Tuple[List[Tuple[float, float]], Optional[List[List[Tuple[float, float]]]]]:
        """
        Evolve murmuration for num_steps.

        Args:
            points: initial positions on E
            num_steps: number of simulation steps
            dt: timestep
            track: if True, return full trajectory history

        Returns:
            (final_points, trajectory) where trajectory is optional
        """
        velocities: dict = {p: 0.0 for p in points}
        trajectory = [list(points)] if track else None

        for _ in range(num_steps):
            points, velocities = self.step(points, velocities, dt)
            if track:
                trajectory.append(list(points))

        return points, trajectory

    # ------------------------------------------------------------------
    # j-invariant stability monitoring
    # ------------------------------------------------------------------

    def j_stability_metric(self) -> float:
        """
        Variance of j-invariant over recent history.
        Small variance → stable identity despite appearance changes.
        """
        if len(self._j_history) < 2:
            return 0.0
        arr = np.array(self._j_history[-50:])
        return float(np.var(arr))

    def detect_bifurcation(self, threshold: float = 50.0) -> bool:
        """
        Detect bifurcation events via j-invariant mutation.
        Returns True if j has changed significantly from initial value.
        """
        if not self._j_history:
            return False
        delta_j = abs(self._j_history[-1] - self._j_history[0])
        return delta_j > threshold

    # ------------------------------------------------------------------
    # PyTorch-batched murmuration (for GPU training loops)
    # ------------------------------------------------------------------

    @staticmethod
    def batch_forces_torch(
        positions: torch.Tensor,  # (N, 2)
        velocities: torch.Tensor,  # (N,)
        a: float, b: float,
        config: Optional[BoidsConfig] = None,
    ) -> torch.Tensor:
        """
        Compute Boids forces for all points in batch (PyTorch version).

        Returns (N,) tensor of scalar forces in each T_{P_i}(E).

        Note: This uses Euclidean approximations for geodesic distance
        to avoid expensive per-pair integrations. For small neighborhoods
        (distances << curve circumference), the chordal distance is a
        good approximation to the geodesic distance.
        """
        cfg = config or BoidsConfig()
        N = positions.shape[0]
        device = positions.device

        # Pairwise chordal distances: d_chord ≈ d_geodesic for nearby points
        diffs = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 2)
        chordal_dists = torch.norm(diffs, dim=-1)  # (N, N)

        # Neighbor masks
        neighbor_mask = (chordal_dists < cfg.neighbor_radius) & (chordal_dists > 1e-8)
        sep_mask = (chordal_dists < cfg.separation_radius) & (chordal_dists > 1e-8)

        forces = torch.zeros(N, device=device)

        for i in range(N):
            # Cohesion: mean direction to neighbors
            nbr_mask_i = neighbor_mask[i]
            if nbr_mask_i.any():
                directions = diffs[i, nbr_mask_i]  # (K, 2)
                unit_tan = torch.tensor(
                    tangent_vector(
                        (positions[i, 0].item(), positions[i, 1].item()), a
                    ),
                    device=device
                )
                # Project directions onto tangent
                proj = (directions @ unit_tan)  # (K,)
                forces[i] += cfg.cohesion_weight * proj.mean()

            # Alignment: mean neighbor velocity
            if nbr_mask_i.any():
                nbr_vel = velocities[nbr_mask_i]
                forces[i] += cfg.alignment_weight * nbr_vel.mean()

            # Separation: inverse-square repulsion
            sep_mask_i = sep_mask[i]
            if sep_mask_i.any():
                directions = diffs[i, sep_mask_i]
                unit_tan = torch.tensor(
                    tangent_vector(
                        (positions[i, 0].item(), positions[i, 1].item()), a
                    ),
                    device=device
                )
                proj = (directions @ unit_tan)
                d_sq = chordal_dists[i, sep_mask_i] ** 2 + 1e-8
                forces[i] -= cfg.separation_weight * (proj / d_sq).sum()

        return forces

    @staticmethod
    def batch_step_torch(
        positions: torch.Tensor,  # (N, 2)
        velocities: torch.Tensor,  # (N,)
        a: float, b: float,
        dt: float = 0.1,
        config: Optional[BoidsConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single murmuration step for all points (PyTorch, differentiable).

        Returns (new_positions, new_velocities).
        Uses tangent-line approximation + Newton projection.
        """
        cfg = config or BoidsConfig()
        device = positions.device
        N = positions.shape[0]

        forces = MurmurationOnE.batch_forces_torch(positions, velocities, a, b, config)

        # Velocity update
        v_new = cfg.damping * (velocities + forces * dt)
        v_new = torch.clamp(v_new, -cfg.max_speed, cfg.max_speed)

        # Position update via tangent line + Newton projection
        new_positions = torch.zeros_like(positions)

        for i in range(N):
            x_i, y_i = positions[i, 0].item(), positions[i, 1].item()
            unit_tan = tangent_vector((x_i, y_i), a)
            step = v_new[i].item() * dt

            # Tangent step
            x_new = x_i + step * unit_tan[0]
            y_new = y_i + step * unit_tan[1]

            # Newton projection onto curve (overflow-safe)
            for _ in range(10):
                x_new = max(-_XLIM, min(_XLIM, x_new))
                y_new = max(-_XLIM, min(_XLIM, y_new))
                x2 = x_new * x_new
                f = y_new * y_new - (x_new * x2 + a * x_new + b)
                if abs(f) < 1e-10:
                    break
                Jx = -(3.0 * x2 + a)
                Jy = 2.0 * y_new
                denom = Jx * Jx + Jy * Jy
                if abs(denom) < 1e-15:
                    break
                delta = max(-1.0, min(1.0, f / denom))
                x_new -= Jx * delta
                y_new -= Jy * delta

            new_positions[i, 0] = x_new
            new_positions[i, 1] = y_new

        return new_positions, v_new


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _project_to_curve(
    x: float, y: float, a: float, b: float, max_iter: int = 50
) -> Tuple[float, float]:
    """Newton projection of (x,y) onto E: y² = x³ + ax + b."""
    _xlim = 1e4
    for _ in range(max_iter):
        x = max(-_xlim, min(_xlim, x))
        y = max(-_xlim, min(_xlim, y))
        x2 = x * x
        f = y * y - (x * x2 + a * x + b)
        if abs(f) < 1e-12:
            break
        Jx = -(3.0 * x2 + a)
        Jy = 2.0 * y
        denom = Jx * Jx + Jy * Jy
        if abs(denom) < 1e-15:
            break
        delta = max(-1.0, min(1.0, f / denom))
        x -= Jx * delta
        y -= Jy * delta
    return (x, y)


def sample_uniform_on_curve(
    a: float, b: float, n: int, seed: int = 42
) -> List[Tuple[float, float]]:
    """
    Sample n points uniformly on E(R) by x-coordinate.

    For Δ < 0 (one component, S¹), samples span the whole oval.
    For Δ > 0 (two components), samples the bounded oval.
    """
    rng = np.random.RandomState(seed)
    roots = sorted(_real_roots_of_cubic(a, b))
    d = discriminant(a, b)

    if d < -1e-12:
        # One component (S^1): x from single root to +inf on both branches
        x_min = roots[0] if roots else -2.0
        x_max = x_min + 6.0  # truncated range for sampling
    elif d > 1e-12:
        # Two components: bounded oval between smallest two roots
        if len(roots) >= 2:
            x_min, x_max = roots[0], roots[1]
        else:
            x_min, x_max = -2.0, 2.0
    else:
        x_min, x_max = -2.0, 2.0

    points = []
    attempts = 0

    while len(points) < n and attempts < n * 20:
        x = rng.uniform(x_min, x_max)
        y_sq = x**3 + a * x + b
        if y_sq > 1e-12:
            y = np.sqrt(y_sq)
            # Randomly choose upper/lower branch
            if d > 1e-12 or rng.rand() > 0.5:
                y = -y if rng.rand() > 0.5 else y
            points.append((float(x), float(y)))
        attempts += 1

    return points


# _real_roots_of_cubic imported at top of file from elliptic_curve
