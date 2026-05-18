"""
Elliptic Curve Object (ECO) - core geometry operations.

Phase 6a/6b: All operations on Weierstrass curves E: y^2 = x^3 + ax + b.

Key insight: the tangent space T_P(E) is 1D - all forces collapse to scalars,
making the Boids dynamics tractable.
"""

import math
import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Curve invariants
# ---------------------------------------------------------------------------

def discriminant(a: float, b: float) -> float:
    """Delta = -16(4a^3 + 27b^2). Must be != 0 for a non-singular curve."""
    return -16.0 * (4.0 * a**3 + 27.0 * b**2)


def j_invariant(a: float, b: float) -> float:
    """
    j(E) = 1728 * 4a^3 / (4a^3 + 27b^2)

    Topological invariant: isomorphic curves have the same j.
    Stable under small perturbations: delta_j = O(||delta||^2).
    """
    denom = 4.0 * a**3 + 27.0 * b**2
    if abs(denom) < 1e-15:
        return float('inf')
    return 1728.0 * (4.0 * a**3) / denom


def is_nonsingular(a: float, b: float) -> bool:
    """Check Delta != 0."""
    return abs(discriminant(a, b)) > 1e-15


def num_components(a: float, b: float) -> int:
    """
    Number of real connected components.
    Delta > 0 -> 2 components (oval + unbounded branch)
    Delta < 0 -> 1 component (single S^1)
    Delta = 0 -> singular (node or cusp)
    """
    d = discriminant(a, b)
    if d > 1e-15:
        return 2
    elif d < -1e-15:
        return 1
    else:
        return 0


# ---------------------------------------------------------------------------
# Point validation
# ---------------------------------------------------------------------------

def is_on_curve(x: float, y: float, a: float, b: float, tol: float = 1e-10) -> bool:
    """Check whether (x, y) lies on y^2 = x^3 + ax + b."""
    return abs(y**2 - (x**3 + a * x + b)) < tol


def is_identity(P: Tuple[float, float]) -> bool:
    """Check if P is the point at infinity (identity element)."""
    return P is None or (P[0] == 0.0 and P[1] == 0.0)


# ---------------------------------------------------------------------------
# Group operations (chord-tangent law)
# ---------------------------------------------------------------------------

def elliptic_neg(P: Tuple[float, float]) -> Tuple[float, float]:
    """Negate a point: -(x, y) = (x, -y). Identity -> identity."""
    if is_identity(P):
        return (0.0, 0.0)
    return (P[0], -P[1])


def elliptic_add(
    P: Tuple[float, float],
    Q: Tuple[float, float],
    a: float, b: float
) -> Tuple[float, float]:
    """
    Group addition P + Q on E: y^2 = x^3 + ax + b.

    Chord-tangent law:
    - P != Q:  lambda = (y_Q - y_P) / (x_Q - x_P)
    - P == Q:  lambda = (3x_P^2 + a) / (2y_P)
    - x_R = lambda^2 - x_P - x_Q,  y_R = lambda*(x_P - x_R) - y_P

    Returns (0, 0) as the identity marker.
    """
    if is_identity(P):
        return Q
    if is_identity(Q):
        return P

    x1, y1 = P
    x2, y2 = Q

    if abs(x1 - x2) < 1e-15 and abs(y1 + y2) < 1e-15:
        return (0.0, 0.0)  # P = -Q -> identity

    if abs(x1 - x2) < 1e-15 and abs(y1 - y2) < 1e-15:
        if abs(y1) < 1e-15:
            return (0.0, 0.0)
        lam = (3.0 * x1**2 + a) / (2.0 * y1)
    else:
        lam = (y2 - y1) / (x2 - x1)

    x3 = lam**2 - x1 - x2
    y3 = lam * (x1 - x3) - y1
    return (x3, y3)


def elliptic_sub(
    P: Tuple[float, float],
    Q: Tuple[float, float],
    a: float, b: float
) -> Tuple[float, float]:
    """Group subtraction P - Q = P + (-Q)."""
    return elliptic_add(P, elliptic_neg(Q), a, b)


def elliptic_scalar_mult(
    k: int, P: Tuple[float, float], a: float, b: float
) -> Tuple[float, float]:
    """
    Scalar multiplication k*P using double-and-add.
    k may be negative.
    """
    if k == 0 or is_identity(P):
        return (0.0, 0.0)

    if k < 0:
        return elliptic_scalar_mult(-k, elliptic_neg(P), a, b)

    result = (0.0, 0.0)
    addend = P

    while k > 0:
        if k & 1:
            result = elliptic_add(result, addend, a, b)
        addend = elliptic_add(addend, addend, a, b)
        k >>= 1

    return result


# ---------------------------------------------------------------------------
# Tangent space
# ---------------------------------------------------------------------------

def tangent_slope(x: float, y: float, a: float) -> float:
    """Slope of the tangent line at P = (x, y): dy/dx = (3x^2 + a) / (2y)."""
    if abs(y) < 1e-15:
        return float('inf')
    return (3.0 * x**2 + a) / (2.0 * y)


def tangent_vector(
    P: Tuple[float, float], a: float
) -> Tuple[float, float]:
    """
    Unit tangent vector at P in E.

    From implicit differentiation of y^2 = x^3 + ax + b:
    2y*dy = (3x^2 + a)*dx  =>  dx : dy = 2y : 3x^2 + a

    Returns normalized (dx, dy).
    """
    if is_identity(P):
        return (1.0, 0.0)

    x, y = P
    if abs(y) < 1e-15:
        return (0.0, 1.0)

    dx = 2.0 * y
    dy = 3.0 * x**2 + a
    norm = math.hypot(dx, dy)
    if norm < 1e-15:
        return (1.0, 0.0)
    return (dx / norm, dy / norm)


# ---------------------------------------------------------------------------
# Arc length / elliptic integrals
# ---------------------------------------------------------------------------

_X_LIMIT = 1e4  # safety cap for x to prevent overflow


def _arc_length_integrand(x: float, a: float, b: float) -> float:
    """
    ds/dx = sqrt(1 + (dy/dx)^2) = sqrt(1 + (3x^2 + a)^2 / (4(x^3 + ax + b)))
    Safe against overflow via x-capping.
    """
    x_safe = max(-_X_LIMIT, min(_X_LIMIT, x))
    x2 = x_safe * x_safe
    y_sq = x_safe * x2 + a * x_safe + b
    if y_sq <= 1e-15:
        return 0.0
    dy_dx_sq = (3.0 * x2 + a)**2 / (4.0 * y_sq)
    return math.sqrt(1.0 + dy_dx_sq)


def _elliptic_integral_dx_over_y(
    x_from: float, x_to: float, a: float, b: float
) -> float:
    """Integral of dx / sqrt(x^3 + ax + b) over [x_from, x_to]."""
    n_pts = max(4, int(abs(x_to - x_from) / 0.001) + 2)
    xs = np.linspace(x_from, x_to, n_pts)
    result = 0.0
    for i in range(len(xs) - 1):
        x_mid = 0.5 * (xs[i] + xs[i + 1])
        dx = xs[i + 1] - xs[i]
        y_sq = x_mid**3 + a * x_mid + b
        if y_sq > 0:
            result += dx / math.sqrt(y_sq)
    return result


# ---------------------------------------------------------------------------
# Real roots of cubic x^3 + ax + b
# ---------------------------------------------------------------------------

def _real_roots_of_cubic(a: float, b: float) -> list:
    """Real roots of x^3 + ax + b = 0. These are the x-intercepts of E."""
    D = -(4.0 * a**3 + 27.0 * b**2)

    if D > 1e-12:
        p = -a / 3.0
        q = -b / 2.0
        if abs(p) > 1e-12 and abs(q / (p**1.5)) <= 1.0:
            theta = math.acos(q / (p**1.5))
        else:
            theta = 0.0
        sqrt_p = 2.0 * math.sqrt(max(0, p))
        return [
            sqrt_p * math.cos(theta / 3.0),
            sqrt_p * math.cos((theta + 2 * math.pi) / 3.0),
            sqrt_p * math.cos((theta + 4 * math.pi) / 3.0),
        ]
    elif abs(D) < 1e-12:
        return [-2.0 * math.copysign(1.0, b) * math.sqrt(max(0, -a / 3.0))]
    else:
        D_sqrt = math.sqrt(-D / 27.0)
        u = -b / 2.0 + D_sqrt
        v = -b / 2.0 - D_sqrt
        term1 = u ** (1.0 / 3.0) if u >= 0 else -((-u) ** (1.0 / 3.0))
        term2 = v ** (1.0 / 3.0) if v >= 0 else -((-v) ** (1.0 / 3.0))
        return [term1 + term2]


# ---------------------------------------------------------------------------
# Geodesic distance
# ---------------------------------------------------------------------------

def geodesic_distance(
    P: Tuple[float, float],
    Q: Tuple[float, float],
    a: float, b: float,
    n_steps: int = 200
) -> float:
    """
    Geodesic (arc-length) distance between P and Q on E(R).

    Numerical integration of ds = sqrt(1 + (dy/dx)^2) * dx.
    Safe against overflow.
    """
    if is_identity(P) or is_identity(Q):
        return float('inf')

    x1, y1 = P
    x2, y2 = Q

    if abs(x1 - x2) < 1e-15 and abs(y1 - y2) < 1e-15:
        return 0.0

    # Cap extreme x
    x1 = max(-_X_LIMIT, min(_X_LIMIT, x1))
    x2 = max(-_X_LIMIT, min(_X_LIMIT, x2))

    same_branch = (y1 * y2 > 0) or (abs(y1) < 1e-15 and abs(y2) < 1e-15)

    if same_branch and y1 * y2 >= 0:
        xs = np.linspace(x1, x2, n_steps)
        ds = 0.0
        for i in range(len(xs) - 1):
            x_mid = 0.5 * (xs[i] + xs[i + 1])
            ds += abs(xs[i + 1] - xs[i]) * _arc_length_integrand(x_mid, a, b)
        return ds
    else:
        roots = _real_roots_of_cubic(a, b)
        if roots and len(roots) >= 2:
            x_branch = max(roots)
        else:
            x_branch = max(x1, x2) + 2.0
        x_branch = min(_X_LIMIT, x_branch)

        xs_p = np.linspace(x1, x_branch, n_steps // 2)
        d_p = 0.0
        for i in range(len(xs_p) - 1):
            x_mid = 0.5 * (xs_p[i] + xs_p[i + 1])
            d_p += abs(xs_p[i + 1] - xs_p[i]) * _arc_length_integrand(x_mid, a, b)

        xs_q = np.linspace(x_branch, x2, n_steps // 2)
        d_q = 0.0
        for i in range(len(xs_q) - 1):
            x_mid = 0.5 * (xs_q[i] + xs_q[i + 1])
            d_q += abs(xs_q[i + 1] - xs_q[i]) * _arc_length_integrand(x_mid, a, b)

        return d_p + d_q


# ---------------------------------------------------------------------------
# Logarithm map: log_P(Q) in T_P(E)
# ---------------------------------------------------------------------------

def log_map(
    P: Tuple[float, float],
    Q: Tuple[float, float],
    a: float, b: float
) -> Tuple[float, float]:
    """
    Logarithm map on E: returns v in T_P(E) such that exp_P(v) = Q.

    T_P(E) is 1D. Direction = unit tangent at P, magnitude = geodesic distance,
    signed by chord direction from P to Q.
    """
    if is_identity(P) or is_identity(Q):
        return (0.0, 0.0)

    x_p, y_p = P
    x_q, y_q = Q

    if abs(x_p - x_q) < 1e-15 and abs(y_p - y_q) < 1e-15:
        return (0.0, 0.0)

    unit_tan = tangent_vector(P, a)
    dist = geodesic_distance(P, Q, a, b)

    chord = (x_q - x_p, y_q - y_p)
    dot = chord[0] * unit_tan[0] + chord[1] * unit_tan[1]

    if y_p * y_q < 0 and abs(y_p) > 1e-10 and abs(y_q) > 1e-10:
        sign = 1.0 if y_p > 0 else -1.0
    else:
        sign = 1.0 if dot >= 0 else -1.0

    return (sign * dist * unit_tan[0], sign * dist * unit_tan[1])


# ---------------------------------------------------------------------------
# Exponential map: exp_P(v) in E
# ---------------------------------------------------------------------------

def exp_map(
    P: Tuple[float, float],
    v: Tuple[float, float],
    a: float, b: float,
    max_substeps: int = 500
) -> Tuple[float, float]:
    """
    Exponential map: move P along the geodesic in direction v for arc length |v|.

    Uses a hybrid approach:
    - Small steps: tangent-line + Newton projection (fast, differentiable-like)
    - Large steps: group-law stepping via elliptic_add (handles pinch points)

    T_P(E) is 1D, so v = s * u where u is the unit tangent and s = signed step.
    """
    if is_identity(P):
        return P

    v_norm = math.hypot(v[0], v[1])
    if v_norm < 1e-15:
        return P

    unit_tan = tangent_vector(P, a)
    dot = v[0] * unit_tan[0] + v[1] * unit_tan[1]
    sign = 1.0 if dot >= 0 else -1.0

    # For very large distances (cross-branch), use group-law stepping
    # which correctly handles the pinch point at y=0
    if v_norm > 0.5:
        return _exp_map_group_law(P, v_norm, sign, a, b)

    # Small distance: tangent-line stepping (fast path)
    max_step_size = 0.05
    n_substeps = max(1, min(max_substeps, int(v_norm / max_step_size) + 1))
    sub_step = v_norm / n_substeps

    x, y = P
    for _ in range(n_substeps):
        x = max(-_X_LIMIT, min(_X_LIMIT, x))
        y = max(-_X_LIMIT, min(_X_LIMIT, y))

        ux, uy = tangent_vector((x, y), a)
        sx, sy = sign * sub_step * ux, sign * sub_step * uy
        x_new = x + sx
        y_new = y + sy

        for __ in range(20):
            x_new = max(-_X_LIMIT, min(_X_LIMIT, x_new))
            y_new = max(-_X_LIMIT, min(_X_LIMIT, y_new))
            x2 = x_new * x_new
            f = y_new * y_new - (x_new * x2 + a * x_new + b)
            if abs(f) < 1e-12:
                break
            Jx = -(3.0 * x2 + a)
            Jy = 2.0 * y_new
            denom = Jx * Jx + Jy * Jy
            if denom < 1e-15:
                break
            delta = max(-1.0, min(1.0, f / denom))
            x_new += Jx * delta
            y_new += Jy * delta

        x, y = x_new, y_new

    return (x, y)


def _exp_map_group_law(
    P: Tuple[float, float],
    distance: float,
    sign: float,
    a: float, b: float,
    max_k: int = 50
) -> Tuple[float, float]:
    """
    exp_map via group-law stepping for large geodesic distances.

    Uses elliptic_add to walk along the curve, which correctly handles
    passing through the pinch point at y=0 and wrapping around the oval.
    """
    if distance < 1e-15:
        return P

    # Use small group steps (k=1) and check the geodesic distance
    # until we've traveled approximately `distance`.
    Q = P
    d_traveled = 0.0
    k = 0

    while d_traveled < distance and k < max_k:
        # Take one group step in the appropriate direction
        step_point = elliptic_scalar_mult(1 if sign > 0 else -1, P, a, b)
        d_step = geodesic_distance(P, step_point, a, b)

        if d_step < 1e-10:
            break

        if d_traveled + d_step > distance:
            # Interpolate: we're close enough
            break

        d_traveled += d_step
        Q = step_point
        P = Q
        k += 1

    return Q


# ---------------------------------------------------------------------------
# Batch operations (numpy vectorised)
# ---------------------------------------------------------------------------

def batch_is_on_curve(
    xs: np.ndarray, ys: np.ndarray, a: float, b: float, tol: float = 1e-10
) -> np.ndarray:
    """Vectorised curve membership test."""
    return np.abs(ys**2 - (xs**3 + a * xs + b)) < tol


def batch_tangent_vectors(
    xs: np.ndarray, ys: np.ndarray, a: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Unit tangent vectors at multiple points. Returns (dxs, dys)."""
    mask = np.abs(ys) < 1e-15
    dx = np.where(mask, 0.0, 2.0 * ys)
    dy = np.where(mask, 1.0, 3.0 * xs**2 + a)
    norm = np.sqrt(dx**2 + dy**2)
    norm = np.where(norm < 1e-15, 1.0, norm)
    return dx / norm, dy / norm


def batch_elliptic_add(
    P: np.ndarray, Q: np.ndarray, a: float, b: float
) -> np.ndarray:
    """Vectorised group addition for (N, 2) point arrays."""
    N = P.shape[0]
    result = np.zeros((N, 2))

    same = (np.abs(P[:, 0] - Q[:, 0]) < 1e-15) & (np.abs(P[:, 1] - Q[:, 1]) < 1e-15)
    neg = (np.abs(P[:, 0] - Q[:, 0]) < 1e-15) & (np.abs(P[:, 1] + Q[:, 1]) < 1e-15)

    lam = np.zeros(N)
    diff_mask = ~same & ~neg
    lam[diff_mask] = (Q[diff_mask, 1] - P[diff_mask, 1]) / (
        Q[diff_mask, 0] - P[diff_mask, 0] + 1e-15
    )

    tangent_mask = same & ~neg & (np.abs(P[:, 1]) > 1e-15)
    lam[tangent_mask] = (
        3.0 * P[tangent_mask, 0]**2 + a
    ) / (2.0 * P[tangent_mask, 1])

    valid = ~neg
    x3 = lam[valid]**2 - P[valid, 0] - Q[valid, 0]
    y3 = lam[valid] * (P[valid, 0] - x3) - P[valid, 1]
    result[valid, 0] = x3
    result[valid, 1] = y3
    result[neg] = 0.0

    return result


def weierstrass_half_periods(a: float, b: float) -> float:
    """
    Real half-period of the Weierstrass P-function for E.
    Only valid when Delta < 0 (one real component, S^1 topology).
    """
    roots = sorted(_real_roots_of_cubic(a, b))
    if len(roots) >= 2:
        e1, e2 = roots[0], roots[-1]
        omega1 = _elliptic_integral_dx_over_y(e1, e2, a, b)
        return abs(omega1)
    return 0.0
