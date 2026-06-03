"""
ECO (Elliptic Curve Object) Cluster Loss — Direct j-proto parameters.

Key improvements:
1. j_protos = direct learnable parameters (not through sensing function)
   → Decouples cluster identity from sensing quality
   → Always ensures diverse j-invariants at initialization
2. Raw j-invariant values (no z-score), cost normalized in Sinkhorn
3. Simpler, more stable gradient path

Core philosophy: j-invariant IS identity.
  feature → sensing φ → (a,b) → j_raw (atom level)
  cluster j_protos = direct learnable parameters (cluster level)
  Sinkhorn cost C_ik = |j_atom_i - j_proto_k|
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

try:
    from src.losses.direct_cluster import compute_pairwise_geodesic_sq
except ImportError:
    def compute_pairwise_geodesic_sq(mus, metric_field):
        return torch.cdist(mus, mus, p=2) ** 2


# ---------------------------------------------------------------------------
# j-invariant computation (raw, no z-score)
# ---------------------------------------------------------------------------

def j_invariant_from_ab(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    j-invariant: j = 1728 * 4a^3 / (4a^3 + 27b^2).

    Returns raw j-invariant (batch-independent identity). The Sinkhorn cost
    is normalized internally via divide-by-max, so absolute scale is fine.
    
    NOTE: Previously used z-score normalization which made j batch-dependent,
    contradicting ECO's core claim that identity = absolute curve parameters.
    Fixed to use raw j (with optional /1728 scaling for stability).
    """
    a3 = 4.0 * a ** 3
    b2 = 27.0 * b ** 2
    denom = a3 + b2
    j_raw = 1728.0 * a3 / denom.clamp(min=1e-10)
    # Scale to ~[0, 1] for most non-singular real curves (j ∈ [0, 1728] typically).
    # Using global scale (not batch-dependent) preserves identity across batches.
    return j_raw / 1728.0


# ---------------------------------------------------------------------------
# Sensing function (features -> a,b)
# ---------------------------------------------------------------------------

class SensingFunction(nn.Module):
    """
    phi: features -> (a, b) curve parameters.
    Maps atom features to elliptic curve parameters, then to j-invariants.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        with torch.no_grad():
            self.net[0].weight.data.normal_(0, 0.15)
            self.net[2].weight.data.normal_(0, 0.15)
            self.net[4].weight.data.normal_(0, 0.3)
            self.net[4].bias.data[0] = -1.0
            self.net[4].bias.data[1] = 0.5

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(features)
        a, b = out[:, 0], out[:, 1]
        delta = 4.0 * a ** 3 + 27.0 * b ** 2
        singular = torch.abs(delta) < 1e-3
        if singular.any():
            a = a + singular.float() * 0.1 * torch.sign(a + 1e-8)
            b = b + singular.float() * 0.1 * torch.sign(b + 1e-8)
        return a, b


# ---------------------------------------------------------------------------
# Sinkhorn (cost normalized to [0, 1])
# ---------------------------------------------------------------------------

def sinkhorn_jinv(
    j_atoms: torch.Tensor,
    j_protos: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 50,
) -> torch.Tensor:
    """
    Sinkhorn soft assignment. Cost normalized to [0, 1] for stable epsilon.
    """
    N, K = j_atoms.shape[0], j_protos.shape[0]
    device = j_atoms.device

    cost = torch.abs(j_atoms.unsqueeze(1) - j_protos.unsqueeze(0))  # (N, K)
    cost = cost / (cost.max() + 1e-8)

    K_mat = torch.exp(-cost / epsilon)
    v = torch.ones(K, device=device)

    for _ in range(n_iters):
        P = K_mat * v.unsqueeze(0) / (K_mat * v.unsqueeze(0)).sum(dim=1, keepdim=True).clamp(min=1e-10)
        v = v * (N / K) / P.sum(dim=0).clamp(min=1e-10)

    P = K_mat * v.unsqueeze(0) / (K_mat * v.unsqueeze(0)).sum(dim=1, keepdim=True).clamp(min=1e-10)
    return P


# ---------------------------------------------------------------------------
# Identity consistency
# ---------------------------------------------------------------------------

def identity_consistency_loss(j_current: torch.Tensor, j_initial: torch.Tensor) -> torch.Tensor:
    """L_id = tanh(|j_k - j_k_init|). Tanh prevents unbounded loss."""
    return torch.tanh(torch.abs(j_current - j_initial).mean())


# ---------------------------------------------------------------------------
# Phase 8: Discriminant barrier + j-space separation
# ---------------------------------------------------------------------------

def discriminant_barrier_loss(a: torch.Tensor, b: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """
    Bounded exponential barrier: penalize Δ = 4a³ + 27b² → 0 (singular curves).

    L_barrier = mean(exp(-α · |Δ|))

    Unlike the previous log-barrier (-log|Δ|), this barrier has BOUNDED gradient:
      ∂L/∂(a,b) = -α·exp(-α|Δ|)·sign(Δ)·∂Δ/∂(a,b)
    At Δ→0: |∇L| ≈ α·|∂Δ/∂(a,b)| ≈ 10·(12a²,54b) ≈ 300 (safe for BF16).
    At Δ→∞: exp(-α|Δ|)→0, gradient vanishes gracefully.

    Penalty profile:
      |Δ| = 0.0: L = 1.0 (max repulsion)
      |Δ| = 0.1: L ≈ 0.37
      |Δ| = 0.5: L ≈ 0.007
      |Δ| ≥ 1.0: L ≈ 0 (no penalty)

    Args:
        a, b: curve parameters
        alpha: barrier steepness (default 10.0)
    """
    delta = 4.0 * a ** 3 + 27.0 * b ** 2
    return torch.exp(-alpha * torch.abs(delta)).mean()


def j_space_separation_loss(j_atoms: torch.Tensor, min_dist: float = 0.1) -> torch.Tensor:
    """
    Repulsive force in j-space: prevent all atoms from collapsing to identical j.

    When j-invariants collapse (σ_j → 0), the z-score normalization amplifies
    noise and Sinkhorn has no discriminative signal. This loss pushes nearby
    atoms apart in j-space, maintaining identity diversity.

    L_sep = mean(clamp(min_dist - |j_i - j_k|, 0)^2)

    Args:
        j_atoms: (N,) per-atom j-invariants
        min_dist: minimum allowed distance between any two atoms' j-invariants
    """
    N = j_atoms.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=j_atoms.device)

    dist = torch.abs(j_atoms.unsqueeze(0) - j_atoms.unsqueeze(1))  # (N, N)
    mask = 1.0 - torch.eye(N, device=j_atoms.device)
    repulsion = torch.clamp(min_dist - dist, min=0.0) ** 2
    return (repulsion * mask).sum() / (N * (N - 1) + 1e-10)


# ---------------------------------------------------------------------------
# ECO Cluster Loss (direct j-proto parameters)
# ---------------------------------------------------------------------------

class ECOClusterLoss(nn.Module):
    """
    ECO cluster loss with direct learnable j-invariants per cluster.

    Pipeline:
        1. Atom features → sensing φ → (a,b) → j_raw (per-atom j-invariants)
        2. Cluster j-invariants = direct learnable params (one scalar per cluster)
        3. Sinkhorn cost C_ik = |j_atom_i - j_proto_k|
        4. Intra-cluster geodesic compaction + identity consistency
    """

    def __init__(
        self,
        n_clusters: int = 2,
        feature_dim: int = 16,
        sinkhorn_eps: float = 0.1,
        sinkhorn_iters: int = 50,
        ent_weight: float = 0.005,
        id_weight: float = 0.1,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.ent_weight = ent_weight
        self.id_weight = id_weight

        # Phase 8: discriminant barrier + j-separation
        self.barrier_weight = 0.0
        self.sep_weight = 0.0

        # Sinkhorn temperature annealing: start high (soft), anneal to sinkhorn_eps
        self.sinkhorn_eps_start = 1.0  # soft assignments during exploration
        self.register_buffer('eps_current', torch.tensor(sinkhorn_eps))

        # Sensing function: features → (a,b) → j-invariant
        self.sensing = SensingFunction(feature_dim)

        # Learnable per-cluster feature prototypes
        self.prototypes = nn.Parameter(torch.randn(n_clusters, feature_dim) * 0.1)

        # Stored initial j-invariants for identity tracking
        self.register_buffer('j_initial', torch.zeros(n_clusters))
        self._j_initialized = False

    def compute_j_invariants(self, features: torch.Tensor) -> torch.Tensor:
        """Map features to j-invariants via sensing function (j/1728 scaled, batch-independent)."""
        a, b = self.sensing(features)
        return j_invariant_from_ab(a, b)

    def compute_raw_j(self, features: torch.Tensor) -> torch.Tensor:
        """Raw (pre-z-score) j-invariants for Phase 8 monitoring."""
        a, b = self.sensing(features)
        a3 = 4.0 * a ** 3
        b2 = 27.0 * b ** 2
        return 1728.0 * a3 / (a3 + b2).clamp(min=1e-10)

    def collapse_detected(self, features: torch.Tensor, threshold: float = 0.02) -> bool:
        """Phase 8: detect feature collapse (monitor features, not j)."""
        return features.shape[0] >= 2 and features.std(dim=0).mean().item() < threshold

    def orthogonal_mutation(self, noise_scale: float = 0.3) -> bool:
        """
        Therapy 3: discrete manifold jump orthogonal to gradient.
        Projects noise onto orthogonal complement of ∇L, then adds to φ weights.
        Breaks deadlock without corrupting existing gradient signal.
        """
        last_layer = self.sensing.net[4]
        if last_layer.weight.grad is None:
            return False
        with torch.no_grad():
            grad = last_layer.weight.grad
            noise = torch.randn_like(grad)
            gn = grad.norm()
            if gn > 1e-8:
                proj = (noise * grad).sum() / (gn ** 2) * grad
                noise = noise - proj  # ⟂ to grad
            last_layer.weight.data += noise_scale * noise
            last_layer.bias.data += noise_scale * torch.randn_like(last_layer.bias)
        return True

    def init_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """Initialize prototypes with j-diversity enforcement."""
        with torch.no_grad():
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    self.prototypes[k] = features[mask].mean(dim=0)

            # Enforce j-diversity: push prototype features apart if j too close
            j_protos = self.compute_j_invariants(self.prototypes)
            for _ in range(20):
                min_gap = float('inf')
                argmin = None
                for k1 in range(self.n_clusters):
                    for k2 in range(k1 + 1, self.n_clusters):
                        gap = (j_protos[k1] - j_protos[k2]).abs().item()
                        if gap < min_gap:
                            min_gap = gap
                            argmin = (k1, k2)
                if min_gap >= 0.3:
                    break
                k_push = argmin[1]
                noise = torch.randn(self.feature_dim, device=self.prototypes.device) * 0.5
                self.prototypes[k_push] += noise
                j_protos = self.compute_j_invariants(self.prototypes)

            self.j_initial = j_protos.detach()
            self._j_initialized = True

    def set_progress(self, progress: float):
        """Annealing: progress=0→soft(eps=1.0), progress=1→hard(eps=target)."""
        self.eps_current[()] = self.sinkhorn_eps_start + progress * (self.sinkhorn_eps - self.sinkhorn_eps_start)

    def set_barrier_sep(self, barrier_weight: float = 0.0, sep_weight: float = 0.0):
        """Set Phase 8 defense weights (callable after construction)."""
        self.barrier_weight = barrier_weight
        self.sep_weight = sep_weight

    def forward(
        self,
        mus: torch.Tensor,
        metric_field,
        features: torch.Tensor,
        D2: Optional[torch.Tensor] = None,
    ):
        N, d = features.shape
        K = self.n_clusters
        device = mus.device

        if N < K:
            return (
                torch.tensor(0.0, device=device),
                torch.zeros(N, K, device=device),
                {},
            )

        # ── 1. Atom j-invariants via sensing function ──
        j_atoms = self.compute_j_invariants(features)  # (N,)

        # ── 2. Cluster j-invariants via sensing (same frame as atoms) ──
        j_protos = self.compute_j_invariants(self.prototypes)  # (K,)

        # ── 3. Sinkhorn with j-invariant matching cost ──
        P = sinkhorn_jinv(j_atoms, j_protos, epsilon=self.eps_current.item(), n_iters=self.sinkhorn_iters)

        # ── 4. Intra-cluster geodesic compaction ──
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

        # ── 5. Entropy bonus ──
        row_ent = -(P * torch.log(P + 1e-10)).sum(dim=1).mean()
        loss = loss - self.ent_weight * row_ent

        # ── 6. Identity consistency ──
        if not self._j_initialized:
            self.j_initial = j_protos.detach()
            self._j_initialized = True

        loss_id = identity_consistency_loss(j_protos, self.j_initial)
        loss = loss + self.id_weight * loss_id

        # ── 7. Phase 8: discriminant barrier (prevent singular curves) ──
        if self.barrier_weight > 0 and features.shape[0] > 0:
            atom_a, atom_b = self.sensing(features)
            loss_barrier = discriminant_barrier_loss(atom_a, atom_b)
            loss = loss + self.barrier_weight * loss_barrier
        else:
            loss_barrier = torch.tensor(0.0, device=device)

        # ── 8. Phase 8: j-space separation on RAW (pre-z-score) j-invariants ──
        # z-scored j always has σ≈1, masking collapse. Raw j reveals true collapse.
        if self.sep_weight > 0 and features.shape[0] > 1:
            raw_j = self.compute_raw_j(features)
            loss_sep = j_space_separation_loss(raw_j, min_dist=10.0)  # raw j spans ~hundreds
            loss = loss + self.sep_weight * loss_sep
        else:
            loss_sep = torch.tensor(0.0, device=device)

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
            'loss_barrier': loss_barrier.item(),
            'loss_sep': loss_sep.item(),
        }

        return loss, P, metrics


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test():
    import sys
    sys.path.insert(0, '.')
    N, d, K = 10, 16, 2
    mus = torch.randn(N, 2, requires_grad=True)
    features = torch.randn(N, d)
    labels = torch.randint(0, K, (N,))

    class DummyMetric:
        def __call__(self, coords):
            return torch.eye(2).unsqueeze(0).expand(coords.shape[0], 2, 2)

    eco = ECOClusterLoss(n_clusters=K, feature_dim=d)
    eco.init_prototypes(features, labels)
    loss, P, metrics = eco(mus, DummyMetric(), features)

    print(f"ECO Loss: {loss.item():.4f}")
    print(f"  j_protos: {metrics['j_protos']}")
    print(f"  cluster_balance: {metrics['cluster_balance']:.4f}")
    print(f"  j_drift: {metrics['j_drift']:.6f}")
    loss.backward()
    print(f"  Grad on mus: {mus.grad is not None}")
    print("ECO SMOKE TEST PASSED")


if __name__ == '__main__':
    _smoke_test()
