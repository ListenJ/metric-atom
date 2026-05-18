"""
Direct Metric-Driven Cluster Loss (Path 1+3).

Replaces InfoNCE with:
  1. Sinkhorn soft assignment based on feature-prototype similarity (differentiable)
  2. Direct minimization of intra-cluster geodesic distances

Architecture:
    度量场 g ──→ 占位耦合（学边界）──→ g 变锐利
                                  │
                                  ▼
                            度量场 + 原子位置
                                  │
                                  ▼
                        Sinkhorn 软分配 P（可微）
                                  │
                                  ▼
                    L_direct = Σ_k Σ_{i,j} P_ik P_jk · d_g(μ_i,μ_j)²
                                  │
                                  ▼
                            梯度回传到 g 和特征
                            全链路打通！

Key insight: InfoNCE in Riemannian space has a vanishingly narrow viable region
because both features AND the metric field control the positive/negative gap.
Direct metric optimization removes this fragility — the metric field is directly
penalized for making intra-cluster distances large.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_softmax(cost, epsilon=0.1, n_iters=50):
    """
    Sinkhorn-Knopp producing row-stochastic soft assignment with
    approximately balanced column sums.

    P_ik ∝ exp(-cost_ik / epsilon) * v_k
    where v_k is iteratively updated to balance cluster sizes.

    Args:
        cost: (N, K) cost matrix (e.g., 1 - cosine_sim, scaled to [0, 1])
        epsilon: entropy regularization strength (smaller → sharper assignments)
        n_iters: number of Sinkhorn iterations

    Returns:
        P: (N, K) soft assignment, row-stochastic
    """
    N, K = cost.shape
    # Kernel matrix
    K_mat = torch.exp(-cost / epsilon)  # (N, K)

    v = torch.ones(K, device=cost.device)  # column scaling factors

    for _ in range(n_iters):
        # Row normalize with current column scaling
        row_sums = (K_mat * v.unsqueeze(0)).sum(dim=1, keepdim=True)  # (N, 1)
        P = K_mat * v.unsqueeze(0) / row_sums.clamp(min=1e-10)        # (N, K)

        # Update column scaling: v_k *= target_size / actual_size
        col_sums = P.sum(dim=0)  # (K,)
        target = N / K
        v = v * target / col_sums.clamp(min=1e-10)

    # Final row normalization
    row_sums = (K_mat * v.unsqueeze(0)).sum(dim=1, keepdim=True)
    P = K_mat * v.unsqueeze(0) / row_sums.clamp(min=1e-10)

    return P


def compute_pairwise_geodesic_sq(mus, metric_field):
    """
    Symmetric pairwise geodesic distance-squared matrix using midpoint metric.

    d²_ij = (μ_i - μ_j)ᵀ g((μ_i + μ_j)/2) (μ_i - μ_j)

    Using midpoint metric ensures d(i,j) = d(j,i), which is essential for
    well-defined intra-cluster distances.

    Args:
        mus: (N, D) atom positions
        metric_field: MetricField2D instance

    Returns:
        D2: (N, N) squared geodesic distances
    """
    N, D = mus.shape
    if N < 2:
        return torch.zeros(N, N, device=mus.device)

    mids = (mus.unsqueeze(0) + mus.unsqueeze(1)) / 2  # (N, N, D)
    dx = mus.unsqueeze(0) - mus.unsqueeze(1)           # (N, N, D)

    # Batch-evaluate metric field at all midpoint pairs
    g_mid = metric_field(mids.reshape(-1, D)).reshape(N, N, D, D)

    # d²_ij = dxᵀ g_mid dx
    d2 = torch.einsum('ijm,ijmn,ijn->ij', dx, g_mid, dx).clamp(min=0)

    return d2


class DirectClusterLoss(nn.Module):
    """
    Direct metric-driven clustering loss with Sinkhorn soft assignment.

    Pipeline:
        1. Compute feature-to-prototype similarity → cost matrix
        2. Sinkhorn soft assignment P (differentiable)
        3. Pairwise geodesic distance matrix D²_g (midpoint metric)
        4. L = Σ_k P[:,k]ᵀ @ D²_g @ P[:,k]  (intra-cluster compaction)
        5. Optional: entropy bonus to prevent degenerate assignments

    Learnable parameters:
        prototypes: (K, feature_dim) — cluster centers in feature space,
                    initialized by KMeans at Phase 2 start
    """

    def __init__(self, n_clusters=2, feature_dim=16,
                 sinkhorn_eps=0.1, sinkhorn_iters=50,
                 ent_weight=0.005):
        super().__init__()
        self.n_clusters = n_clusters
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.ent_weight = ent_weight

        # Learnable cluster prototypes in feature space
        self.prototypes = nn.Parameter(torch.randn(n_clusters, feature_dim) * 0.1)

    def init_prototypes(self, features, labels):
        """
        Initialize prototypes from KMeans hard assignments.

        Called once at Phase 2 start to provide a good initial clustering.
        After initialization, prototypes are refined by gradient descent.

        Args:
            features: (N, D) atom features
            labels: (N,) hard cluster assignments from KMeans
        """
        with torch.no_grad():
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    self.prototypes[k] = features[mask].mean(dim=0)
                else:
                    # Fallback: random near centroid
                    self.prototypes[k] = features.mean(dim=0) + \
                        torch.randn_like(self.prototypes[k]) * 0.1

    def forward(self, mus, metric_field, features, D2=None):
        """
        Compute direct metric cluster loss.

        Args:
            mus: (N, D) atom positions
            metric_field: MetricField2D
            features: (N, d) atom features
            D2: (N, N) precomputed pairwise geodesic distance-squared matrix.
                If None, computed internally.

        Returns:
            loss: scalar
            P: (N, K) soft assignment matrix (for monitoring)
            metrics: dict with debug info
        """
        N = mus.shape[0]
        K = self.n_clusters

        if N < K:
            return (torch.tensor(0.0, device=mus.device),
                    torch.zeros(N, K, device=mus.device),
                    {})

        # ── 1. Cost matrix: feature similarity to prototypes ──
        feats_norm = F.normalize(features, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim = feats_norm @ proto_norm.T          # (N, K), in [-1, 1]
        cost = (1.0 - sim) / 2.0                  # (N, K), in [0, 1]

        # ── 2. Sinkhorn soft assignment ──
        P = sinkhorn_softmax(cost, self.sinkhorn_eps, self.sinkhorn_iters)  # (N, K)

        # ── 3. Pairwise geodesic distances ──
        if D2 is None:
            D2 = compute_pairwise_geodesic_sq(mus, metric_field)  # (N, N)

        # ── 4. Intra-cluster compaction loss ──
        # L = Σ_k (P[:,k]ᵀ @ D²_g @ P[:,k]) / (Σ_i P_ik)²
        #
        # Normalized by effective cluster size, so the loss is the
        # per-pair average geodesic distance within each cluster.
        # This prevents O(N²) explosion and keeps the gradient scale stable.
        loss = 0.0
        for k in range(K):
            pk = P[:, k]  # (N,)
            cluster_mass = pk.sum()  # effective cluster size
            if cluster_mass < 1.0:
                continue
            # Quadratic form: Σ_i Σ_j pk_i · D²_ij · pk_j / (cluster_mass)²
            quad = torch.einsum('i,ij,j->', pk, D2, pk)
            loss += quad / (cluster_mass * cluster_mass)

        # ── 5. Entropy bonus: prevent trivial uniform assignments ──
        # Low row entropy = confident assignments (good)
        # We mildly penalize high entropy to encourage sharp clustering
        row_ent = -(P * torch.log(P + 1e-10)).sum(dim=1).mean()
        loss = loss - self.ent_weight * row_ent

        # ── Debug metrics ──
        cluster_sizes = P.sum(dim=0).detach()  # (K,)
        hard_pred = P.argmax(dim=1)             # (N,)
        cluster_balance = cluster_sizes.min() / cluster_sizes.max().clamp(min=1)

        metrics = {
            'cluster_sizes': cluster_sizes,
            'hard_pred': hard_pred,
            'cluster_balance': cluster_balance.item(),
            'row_entropy': row_ent.item(),
            'P_max_mean': P.max(dim=1)[0].mean().item(),  # avg confidence
        }

        return loss, P, metrics
