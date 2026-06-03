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
        v = (v * target / col_sums.clamp(min=1e-10)).clamp(max=1e3)  # prevent explosion

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
        # + ent_weight * row_ent: penalize high entropy (encourage SHARP assignments)
        # Previously was "-", which rewarded uniform assignments — a sign error
        loss = loss + self.ent_weight * row_ent

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


class MetricFeatureEncoder(nn.Module):
    """
    Source features from the metric field itself. [本源]

    Instead of per-atom learnable parameters f_i, features are
    a deterministic function of the local metric tensor:

        f_i = Φ(g(x_i))   where Φ is a lightweight MLP.

    This closes the feature-geodesic gap: when the metric changes,
    features change automatically. No alignment loss needed.
    DirectCluster gradient flows straight through Φ → metric field.

    Input: 3 unique entries of the 2×2 SPD metric tensor g(x_i)
    Output: feature_dim-dimensional feature vector
    """

    def __init__(self, feature_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim),
        )

    def forward(self, mus, metric_field):
        """
        Args:
            mus: (N, 2) atom positions
            metric_field: MetricField2D, differentiable

        Returns:
            feats: (N, feature_dim) features with gradient path to metric_field
        """
        g = metric_field(mus)                         # (N, 2, 2)
        g11 = g[:, 0, 0]                               # (N,)
        g12 = g[:, 0, 1]                               # (N,)
        g22 = g[:, 1, 1]                               # (N,)
        g_flat = torch.stack([g11, g12, g22], dim=-1)  # (N, 3)
        return self.net(g_flat)


def compute_geodesic_alignment_loss(features, mus, metric_field, epsilon=0.2):
    """
    Feature-geodesic alignment loss: KL(P_g || P_f)  [本源]

    First-principles derivation:
        The metric field g(x) is the root geometric object. It defines
        geodesic distances d_g(x_i, x_j) — "what is close" in perceptual
        space. Features f_i should embed atoms such that angular distance
        in feature space preserves geodesic neighborhood structure.

        P_g(i,j) ∝ exp(-D²_ij / ε_a)  — geodesic neighborhood distribution
        P_f(i,j) ∝ exp(cos_sim(i,j) / ε_a)  — feature similarity distribution

        L = KL(P_g || P_f)  — geodesic is teacher, feature is student

    Key design:
        - Geodesic distances are detached from the metric field gradient.
          The metric is trained for reconstruction — L_align must not
          distort the geometry. Only features receive alignment signal.
        - D² is normalized by per-batch max to match cos_sim's [-1,1] range.
        - KL is scale-invariant: only ranking matters, no need for exact
          distance matching. Neighborhood preservation is sufficient.

    Args:
        features: (N, D) atom features, learnable parameters
        mus: (N, d) atom positions
        metric_field: MetricField2D instance
        epsilon: alignment temperature (default 0.2 — softer than
                 sinkhorn_eps=0.05; broad structure, not sharp clustering)

    Returns:
        loss: scalar alignment loss
    """
    N = features.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=features.device,
                          dtype=features.dtype, requires_grad=True)

    # ── 1. Geodesic distances (DETACHED — metric is ground truth) ──
    with torch.no_grad():
        D2 = compute_pairwise_geodesic_sq(mus, metric_field)

    # Normalize to match cos_sim scale [-1, 1] ≈ [-D2_max, 0]
    # Division by batch max keeps distribution shape, avoids outlier sensitivity
    D2_norm = D2 / D2.max().clamp(min=1e-8)  # ∈ [0, 1]

    # ── 2. Feature cosine similarity ──
    feats_norm = F.normalize(features, dim=-1)
    cos_sim = feats_norm @ feats_norm.T  # ∈ [-1, 1]

    # ── 3. Soft assignments ──
    # P_g: closer in geodesic distance → higher probability
    # P_f: more similar features → higher probability
    log_P_g = torch.log_softmax(-D2_norm / epsilon, dim=-1)  # (N, N)
    log_P_f = torch.log_softmax(cos_sim / epsilon, dim=-1)   # (N, N)
    P_g = torch.exp(log_P_g)

    # ── 4. KL(P_g || P_f): Σ_i Σ_j P_g(i,j) · [log P_g(i,j) - log P_f(i,j)] ──
    loss = (P_g * (log_P_g - log_P_f)).sum(dim=-1).mean()

    return loss
