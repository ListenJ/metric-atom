"""
ECO (Elliptic Curve Object) Cluster Loss — Phase 6c Unified.

Core philosophy: j-invariant IS identity.
  feature  →  sensing φ  →  curve (a,b)  →  j-invariant j

Sinkhorn cost C_ik = |j_i - j_k|  (j-invariant matching cost)
  instead of curve residual |y^2 - (x^3 + a_k*x + b_k)|.

j-invariant is second-order stable (δj = O(||δ||^2)), providing robust
identity signal for clustering without spatial position dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# Import geodesic computation (used in forward when D2 not precomputed)
try:
    from src.losses.direct_cluster import compute_pairwise_geodesic_sq
except ImportError:
    # fallback for smoke tests
    def compute_pairwise_geodesic_sq(mus, metric_field):
        return torch.cdist(mus, mus, p=2) ** 2


# ---------------------------------------------------------------------------
# j-invariant computation
# ---------------------------------------------------------------------------

def j_invariant_from_ab(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    j-invariant for elliptic curves: j = 1728 * 4a^3 / (4a^3 + 27b^2).

    Uses z-score normalization instead of tanh to preserve discriminability:
    raw j-invariant values span hundreds, but tanh(j * 0.001) compresses
    all values to [-0.3, 0.3]. Z-score normalization gives unit variance,
    ensuring the Sinkhorn cost always spans a meaningful range while
    preserving relative differences.

    Args:
        a, b: (...,) tensors of curve parameters
    Returns:
        j: (...,) j-invariants, z-score normalized (mean=0, std≈1)
    """
    a3 = 4.0 * a ** 3
    b2 = 27.0 * b ** 2
    denom = a3 + b2
    # Raw j-invariant (can be very large for near-singular curves)
    j_raw = 1728.0 * a3 / denom.clamp(min=1e-10)
    # Z-score normalize for max discriminability
    # This makes the Sinkhorn cost distribution consistently span [0, several stds]
    j_centered = j_raw - j_raw.mean()
    j = j_centered / (j_centered.std() + 1e-8)
    return j


# ---------------------------------------------------------------------------
# Sensing function: features -> curve parameters (a,b)
# ---------------------------------------------------------------------------

class SensingFunction(nn.Module):
    """
    phi: features -> (a, b) curve parameters.

    A small MLP that maps ANY feature vector (atom-level or cluster-level)
    to elliptic curve parameters (a, b). The curve parameters are then
    converted to j-invariants for identity comparison.

    Architecture: Linear(d, 32) -> ReLU -> Linear(32, 32) -> ReLU -> Linear(32, 2)
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # (a, b)
        )
        # Initialize to produce diverse outputs
        # Wider weights → more diverse (a,b) → more j-invariant diversity
        with torch.no_grad():
            self.net[0].weight.data.normal_(0, 0.15)
            self.net[2].weight.data.normal_(0, 0.15)
            # Last layer: wider to produce spread-out (a,b)
            self.net[4].weight.data.normal_(0, 0.3)
            self.net[4].bias.data[0] = -1.0   # a starts near -1
            self.net[4].bias.data[1] = 0.5    # b starts near 0.5

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (N, d) feature vectors
        Returns:
            (a, b): (N,) curve parameters, projected away from singular locus
        """
        out = self.net(features)  # (N, 2)
        a = out[:, 0]
        b = out[:, 1]

        # Project away from singular locus (Delta = -16(4a^3 + 27b^2) ≈ 0)
        delta = 4.0 * a ** 3 + 27.0 * b ** 2
        singular = torch.abs(delta) < 1e-3
        if singular.any():
            # Push away from singularity along both axes
            a = a + singular.float() * 0.1 * torch.sign(a + 1e-8)
            b = b + singular.float() * 0.1 * torch.sign(b + 1e-8)

        return a, b


# ---------------------------------------------------------------------------
# Sinkhorn soft assignment with j-invariant cost
# ---------------------------------------------------------------------------

def sinkhorn_jinv(
    j_atoms: torch.Tensor,     # (N,) per-atom j-invariants
    j_protos: torch.Tensor,    # (K,) per-prototype j-invariants
    epsilon: float = 0.1,
    n_iters: int = 50,
) -> torch.Tensor:
    """
    Sinkhorn soft assignment using j-invariant matching cost.

    Cost C_ik = |j_i - j_k|  (identity proximity in j-space)

    The j-invariant is second-order stable, so this cost is robust to
    small perturbations in curve parameters.

    Args:
        j_atoms: (N,) atom j-invariants
        j_protos: (K,) prototype j-invariants
        epsilon: Sinkhorn temperature
        n_iters: Sinkhorn iterations
    Returns:
        P: (N, K) soft assignment matrix
    """
    N = j_atoms.shape[0]
    K = j_protos.shape[0]
    device = j_atoms.device

    # Cost = |j_i - j_k| (j is z-score normalized, so cost is already in meaningful units)
    cost = torch.abs(j_atoms.unsqueeze(1) - j_protos.unsqueeze(0))  # (N, K)

    # Sinkhorn-Knopp
    K_mat = torch.exp(-cost / epsilon)  # (N, K)
    v = torch.ones(K, device=device)

    for _ in range(n_iters):
        row_sums = (K_mat * v.unsqueeze(0)).sum(dim=1, keepdim=True)
        P = K_mat * v.unsqueeze(0) / row_sums.clamp(min=1e-10)
        col_sums = P.sum(dim=0)
        target = N / K
        v = v * target / col_sums.clamp(min=1e-10)

    row_sums = (K_mat * v.unsqueeze(0)).sum(dim=1, keepdim=True)
    P = K_mat * v.unsqueeze(0) / row_sums.clamp(min=1e-10)

    return P


# ---------------------------------------------------------------------------
# Identity consistency loss
# ---------------------------------------------------------------------------

def identity_consistency_loss(
    j_current: torch.Tensor,
    j_initial: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize large deviations in j-invariant from initial values.

    L_id = mean_k tanh(|j_k - j_k_init|)

    Tanh prevents unbounded loss while retaining gradient signal.
    """
    return torch.tanh(torch.abs(j_current - j_initial).mean())


# ---------------------------------------------------------------------------
# ECO Cluster Loss (unified)
# ---------------------------------------------------------------------------

class ECOClusterLoss(nn.Module):
    """
    Elliptic Curve Object cluster loss with j-invariant matching.

    Pipeline:
        1. Sensing φ maps features → (a,b) → j-invariant (atom & prototype level)
        2. Sinkhorn cost C_ik = |j_i - j_k| (identity proximity)
        3. Intra-cluster geodesic compaction (spatial proximity)
        4. j-invariant identity consistency (stability)

    This is the unified formulation: identity (j-invariant) drives assignment,
    metric field provides spatial boundaries, sensing bridges features ↔ curves.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        feature_dim: int = 16,
        sinkhorn_eps: float = 0.1,
        sinkhorn_iters: int = 50,
        ent_weight: float = 0.005,
        id_weight: float = 0.1,
        j_diversity_min: float = 0.1,  # min |Δj| between cluster pairs
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.ent_weight = ent_weight
        self.id_weight = id_weight
        self.j_diversity_min = j_diversity_min

        # Sensing function: features → (a,b) → j-invariant
        self.sensing = SensingFunction(feature_dim)

        # Learnable per-cluster feature prototypes
        self.prototypes = nn.Parameter(
            torch.randn(n_clusters, feature_dim) * 0.1
        )

        # Stored initial j-invariants for identity tracking
        self.register_buffer('j_initial', torch.zeros(n_clusters))
        self._j_initialized = False

    def compute_j_invariants(self, features: torch.Tensor) -> torch.Tensor:
        """
        Map features to j-invariants via sensing function.

        Args:
            features: (N, d) feature vectors
        Returns:
            j: (N,) j-invariants in [-1, 1]
        """
        a, b = self.sensing(features)
        return j_invariant_from_ab(a, b)

    def init_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Initialize prototypes from KMeans hard assignments with j-diversity.

        After initializing from cluster means, we enforce j-diversity:
        if any two clusters have |j_k - j_l| < j_diversity_min, add noise to
        one of them to push them apart.

        Args:
            features: (N, d) atom features
            labels: (N,) integer cluster assignments
        """
        with torch.no_grad():
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    self.prototypes[k] = features[mask].mean(dim=0)

            # Enforce j-diversity across clusters
            j_protos = self.compute_j_invariants(self.prototypes)
            for _ in range(10):  # iterative push-apart
                needs_diversity = False
                for k1 in range(self.n_clusters):
                    for k2 in range(k1 + 1, self.n_clusters):
                        if (j_protos[k1] - j_protos[k2]).abs() < self.j_diversity_min:
                            # Push one prototype away
                            noise = torch.randn(self.feature_dim, device=self.prototypes.device) * 0.3
                            self.prototypes[k2] = self.prototypes[k2] + noise
                            needs_diversity = True
                if needs_diversity:
                    j_protos = self.compute_j_invariants(self.prototypes)
                else:
                    break

            # Store initial j-invariants
            self.j_initial = j_protos.detach()
            self._j_initialized = True

    def forward(
        self,
        mus: torch.Tensor,
        metric_field,
        features: torch.Tensor,
        D2: Optional[torch.Tensor] = None,
    ):
        """
        Compute ECO cluster loss.

        Args:
            mus: (N, 2) atom positions
            metric_field: MetricField2D
            features: (N, d) atom features
            D2: (N, N) precomputed geodesic distances (optional)
        Returns:
            loss: scalar
            P: (N, K) soft assignment
            metrics: dict with debug info
        """
        N, d = features.shape
        K = self.n_clusters
        device = mus.device

        if N < K:
            return (
                torch.tensor(0.0, device=device),
                torch.zeros(N, K, device=device),
                {},
            )

        # ── 1. Compute j-invariants ──
        j_atoms = self.compute_j_invariants(features)          # (N,)
        j_protos = self.compute_j_invariants(self.prototypes)  # (K,)

        # ── 2. Sinkhorn with j-invariant matching cost ──
        P = sinkhorn_jinv(
            j_atoms, j_protos,
            epsilon=self.sinkhorn_eps,
            n_iters=self.sinkhorn_iters,
        )  # (N, K)

        # ── 3. Intra-cluster geodesic compaction ──
        if D2 is None:
            D2 = compute_pairwise_geodesic_sq(mus, metric_field)

        loss = 0.0
        for k in range(K):
            pk = P[:, k]
            cluster_mass = pk.sum()
            if cluster_mass < 1.0:
                continue
            quad = torch.einsum('i,ij,j->', pk, D2, pk)
            loss += quad / (cluster_mass * cluster_mass)

        # ── 4. Entropy bonus (balanced assignments) ──
        row_ent = -(P * torch.log(P + 1e-10)).sum(dim=1).mean()
        loss = loss - self.ent_weight * row_ent

        # ── 5. j-invariant identity consistency ──
        if not self._j_initialized:
            self.j_initial = j_protos.detach()
            self._j_initialized = True

        loss_id = identity_consistency_loss(j_protos, self.j_initial)
        loss = loss + self.id_weight * loss_id

        # ── 6. j-diversity bonus (explicitly push clusters apart in j-space) ──
        # Penalize clusters whose j-invariants are too close
        pairwise_j = torch.abs(j_protos.unsqueeze(0) - j_protos.unsqueeze(1))  # (K, K)
        # Mask out self-distances (diagonal)
        mask = 1.0 - torch.eye(K, device=device)
        min_pairwise = (pairwise_j * mask + 1e10 * (1.0 - mask)).view(-1).min()
        # If any pair is closer than threshold, add penalty
        if min_pairwise < self.j_diversity_min:
            loss = loss + 0.05 * (self.j_diversity_min - min_pairwise)

        # ── Debug metrics ──
        cluster_sizes = P.sum(dim=0).detach()

        metrics = {
            'cluster_sizes': cluster_sizes,
            'cluster_balance': (cluster_sizes.min() / cluster_sizes.max().clamp(min=1)).item(),
            'row_entropy': row_ent.item(),
            'P_max_mean': P.max(dim=1)[0].mean().item(),
            'j_atoms': j_atoms.detach().float().cpu().numpy(),
            'j_protos': j_protos.detach().float().cpu().numpy(),
            'j_initial': self.j_initial.detach().float().cpu().numpy(),
            'j_drift': (j_protos - self.j_initial).abs().mean().item(),
            'loss_id': loss_id.item(),
        }

        return loss, P, metrics


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test():
    """Verify ECO loss runs without errors."""
    import sys
    sys.path.insert(0, '.')
    N, d, K = 10, 16, 2

    mus = torch.randn(N, 2, requires_grad=True)
    features = torch.randn(N, d)
    labels = torch.randint(0, K, (N,))

    class DummyMetric:
        def __call__(self, coords):
            N = coords.shape[0]
            g = torch.eye(2).unsqueeze(0).expand(N, 2, 2)
            return g

    eco = ECOClusterLoss(n_clusters=K, feature_dim=d)
    eco.init_prototypes(features, labels)
    loss, P, metrics = eco(mus, DummyMetric(), features)

    print(f"ECO Loss: {loss.item():.4f}")
    print(f"  Assignment P shape: {P.shape}")
    print(f"  j_protos: {metrics['j_protos']}")
    print(f"  cluster_balance: {metrics['cluster_balance']:.4f}")
    print(f"  loss_id: {metrics['loss_id']:.6f}")

    loss.backward()
    print(f"  Grad on mus: {mus.grad is not None}")
    print("ECO SMOKE TEST PASSED")


if __name__ == '__main__':
    _smoke_test()
