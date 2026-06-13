import torch


# ── Cholesky parameterization (fast but Euclidean-SGD-on-manifold issue: EXT-1) ──

def cholesky_to_metric(l11, l21, l22, eps=1e-4):
    """
    从Cholesky参数构建2x2正定度量矩阵的元素。

    g = L L^T + eps * I
    L = [[l11, 0], [l21, l22]]

    NOTE (EXT-1): This parameterization is fast but suffers from the fact
    that Euclidean SGD on L does NOT correspond to geodesic descent on the
    SPD manifold Sym^+(2). The mapping L -> g is non-isometric, and gradients
    in L-space can point in wrong directions on the manifold.

    Args:
        l11, l21, l22: 标量或张量
        eps: 极小偏置保证严格正定

    Returns:
        g11, g12, g22: 度量矩阵的独立元素
    """
    g11 = l11**2 + eps
    g12 = l11 * l21
    g22 = l21**2 + l22**2 + eps
    return g11, g12, g22


def cholesky_to_metric_3d(l11, l21, l22, l31, l32, l33, eps=1e-4):
    """
    从Cholesky参数构建3x3正定度量矩阵的元素。

    g = L L^T + eps * I
    L = [[l11,  0,   0 ],
         [l21, l22,  0 ],
         [l31, l32, l33]]

    NOTE (EXT-1): Same SPD-manifold optimization geometry issue as 2D.

    Args:
        l11, l21, l22, l31, l32, l33: 标量或张量
        eps: 极小偏置保证严格正定

    Returns:
        g11, g12, g13, g22, g23, g33: 度量矩阵的独立元素（上三角）
    """
    g11 = l11**2 + eps
    g12 = l11 * l21
    g13 = l11 * l31
    g22 = l21**2 + l22**2 + eps
    g23 = l21 * l31 + l22 * l32
    g33 = l31**2 + l32**2 + l33**2 + eps
    return g11, g12, g13, g22, g23, g33


# ── Matrix exponential parameterization (correct SPD geometry: EXT-1 fix) ──

def symmetric_to_metric_2d(h11, h12, h22):
    """
    Build 2x2 SPD metric via matrix exponential: g = exp(H).

    H = [[h11, h12],
         [h12, h22]]  (symmetric)

    g = exp(H) is guaranteed SPD for any real H.

    FIX for EXT-1: H lives in the tangent space (Euclidean vector space of
    symmetric matrices). Euclidean SGD on H corresponds to the "vector
    logarithm" of the SPD manifold — a reasonable approximate geodesic.

    For true geodesic SGD, one would use:
        g_{t+1} = g_t^{1/2} exp(-eta * g_t^{1/2} grad_g g_t^{1/2}) g_t^{1/2}
    (see Bonnabel 2013). Matrix exponential is the first-order approximation.

    Args:
        h11, h12, h22: scalars or tensors (any real value)

    Returns:
        g11, g12, g22: elements of g = exp(H)
    """
    # Build symmetric matrix H
    # Shape handling: inputs may be (N,) or scalars
    N = h11.numel() if hasattr(h11, 'numel') else 1
    device = h11.device if hasattr(h11, 'device') else 'cpu'
    dtype = h11.dtype if hasattr(h11, 'dtype') else torch.float32

    if N == 1 and not hasattr(h11, 'shape'):
        H = torch.tensor([[h11, h12], [h12, h22]], dtype=dtype, device=device)
        g = torch.linalg.matrix_exp(H)
        return g[0, 0], g[0, 1], g[1, 1]

    # Batch version
    H = torch.zeros(N, 2, 2, device=device, dtype=dtype)
    H[:, 0, 0] = h11.reshape(-1) if hasattr(h11, 'reshape') else h11
    H[:, 0, 1] = h12.reshape(-1) if hasattr(h12, 'reshape') else h12
    H[:, 1, 0] = h12.reshape(-1) if hasattr(h12, 'reshape') else h12
    H[:, 1, 1] = h22.reshape(-1) if hasattr(h22, 'reshape') else h22

    g = torch.linalg.matrix_exp(H)  # (N, 2, 2)
    # Reshape back to input shape
    return (
        g[:, 0, 0].reshape(h11.shape),
        g[:, 0, 1].reshape(h11.shape),
        g[:, 1, 1].reshape(h11.shape),
    )


def symmetric_to_metric_3d(h11, h12, h13, h22, h23, h33):
    """
    Build 3x3 SPD metric via matrix exponential: g = exp(H).

    H = [[h11, h12, h13],
         [h12, h22, h23],
         [h13, h23, h33]]  (symmetric)

    Args:
        h11, h12, h13, h22, h23, h33: scalars or tensors

    Returns:
        g11, g12, g13, g22, g23, g33: elements of g = exp(H)
    """
    N = h11.numel() if hasattr(h11, 'numel') else 1
    device = h11.device if hasattr(h11, 'device') else 'cpu'
    dtype = h11.dtype if hasattr(h11, 'dtype') else torch.float32

    if N == 1 and not hasattr(h11, 'shape'):
        H = torch.tensor([
            [h11, h12, h13],
            [h12, h22, h23],
            [h13, h23, h33]
        ], dtype=dtype, device=device)
        g = torch.linalg.matrix_exp(H)
        return g[0, 0], g[0, 1], g[0, 2], g[1, 1], g[1, 2], g[2, 2]

    H = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    H[:, 0, 0] = h11.reshape(-1)
    H[:, 0, 1] = h12.reshape(-1)
    H[:, 0, 2] = h13.reshape(-1)
    H[:, 1, 0] = h12.reshape(-1)
    H[:, 1, 1] = h22.reshape(-1)
    H[:, 1, 2] = h23.reshape(-1)
    H[:, 2, 0] = h13.reshape(-1)
    H[:, 2, 1] = h23.reshape(-1)
    H[:, 2, 2] = h33.reshape(-1)

    g = torch.linalg.matrix_exp(H)
    return (
        g[:, 0, 0].reshape(h11.shape),
        g[:, 0, 1].reshape(h11.shape),
        g[:, 0, 2].reshape(h11.shape),
        g[:, 1, 1].reshape(h11.shape),
        g[:, 1, 2].reshape(h11.shape),
        g[:, 2, 2].reshape(h11.shape),
    )
