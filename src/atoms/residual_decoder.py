"""
Residual LayerNorm SiLU decoder — theoretically optimal architecture.

theory_fracture_fixes.md Theorem 17 + 18:
  - Residual connections raise λ_min(J_f J_f^T) by ~2.5× (0.05 → 0.12)
  - LayerNorm eliminates degenerate Jacobian paths
  - SiLU avoids ReLU "dead neuron" problem

Architecture:
  state (d_s) → Linear → LayerNorm → SiLU → + residual ─┐
       ┌─────────────────────────────────────────────────┘
       ├→ Linear → LayerNorm → SiLU → + residual ─┐
       │   ┌───────────────────────────────────────┘
       │   └→ Linear → RGB (3)
       │
  Uses Pre-LN (LayerNorm before activation) for training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualSiLUBlock(nn.Module):
    """Residual block with Pre-LN + SiLU activation.

    h_out = h_in + SiLU(LayerNorm(Linear(h_in)))
    """

    def __init__(self, dim, expansion=1.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * expansion) if expansion >= 1.0 else dim
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Projection if hidden_dim != dim (shouldn't happen with expansion=1)
        self.proj = nn.Linear(hidden_dim, dim) if hidden_dim != dim else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = F.silu(x)
        x = self.dropout(x)
        if isinstance(self.proj, nn.Linear):
            x = self.proj(x)
        return residual + x


class ResidualDecoder(nn.Module):
    """Residual + LayerNorm + SiLU state-to-RGB decoder.

    Architecture (default: 16 → 64 → 32 → 3):
      state ──→ Linear(16→64) → LN → SiLU
        ├→ ResidualBlock(64)
        ├→ ResidualBlock(64)
        ├→ LN → Linear(64→32) → SiLU
        ├→ ResidualBlock(32)
        └→ LN → Linear(32→3) → sigmoid/tanh

    Jacobian spectral analysis (Theorem 17):
      λ_min(J_f J_f^T) ≈ 0.12 (vs 0.05 for plain MLP)
    """

    def __init__(self, state_dim=16, hidden_dims=(64, 32), output_dim=3,
                 num_res_blocks=1, activation='silu', output_activation='sigmoid',
                 dropout=0.0):
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim

        dims = [state_dim] + list(hidden_dims)
        layers = []

        # Input projection + LN
        layers.append(nn.Linear(dims[0], dims[1]))
        layers.append(nn.LayerNorm(dims[1]))
        if activation == 'silu':
            layers.append(nn.SiLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        else:
            layers.append(nn.ReLU())

        # Residual blocks at first hidden dimension
        for _ in range(num_res_blocks):
            layers.append(ResidualSiLUBlock(dims[1], dropout=dropout))

        # Hidden layer projections with LN
        for i in range(1, len(dims) - 1):
            layers.append(nn.LayerNorm(dims[i]))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if activation == 'silu':
                layers.append(nn.SiLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())

            # Residual blocks at intermediate dimensions
            for _ in range(num_res_blocks):
                layers.append(ResidualSiLUBlock(dims[i + 1], dropout=dropout))

        # Output projection + LN
        layers.append(nn.LayerNorm(dims[-1]))
        layers.append(nn.Linear(dims[-1], output_dim))

        # Output activation
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        # 'none' = no activation

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for linear layers, ones for LN weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        """Decode state vector to RGB color.

        Args:
            state: (..., state_dim) state vectors

        Returns:
            rgb: (..., 3) predicted RGB colors in [0, 1]
        """
        return self.net(state)

    def compute_jacobian_spectrum(self, s_star):
        """Compute λ_min(J_f J_f^T) at a given state (for verification).

        Args:
            s_star: (state_dim,) or (B, state_dim) state vector(s)

        Returns:
            lambda_min: minimum eigenvalue of J_f J_f^T
        """
        if s_star.dim() == 1:
            s_star = s_star.unsqueeze(0)

        results = []
        for i in range(s_star.shape[0]):
            s = s_star[i].detach().requires_grad_(True)
            out = self.forward(s)

            # Jacobian via vmap over output dimensions
            J = []
            for j in range(out.shape[-1]):
                grad = torch.autograd.grad(
                    out[..., j].sum(), s,
                    create_graph=True, retain_graph=True
                )[0]
                J.append(grad)
            J = torch.stack(J, dim=0)  # (output_dim, state_dim)

            JJT = J @ J.T  # (output_dim, output_dim)
            eigenvals = torch.linalg.eigvalsh(JJT)
            results.append(eigenvals[0].item())

        if len(results) == 1:
            return results[0]
        return results


def create_optimal_decoder(state_dim=16, output_dim=3, linear_only=False):
    """Create the theoretically optimal decoder architecture.

    Based on theory_fracture_fixes.md:
      - Residual connections: +2.5× λ_min
      - LayerNorm: eliminates degeneracy
      - SiLU: no dead neurons, smooth gradients

    Args:
        linear_only: if True, return a single Linear+Sigmoid layer.
            This prevents the decoder from learning a complex
            "state-collapsing" mapping (e.g., mapping all states
            to the same color). Forces states to be visually distinct.

    Returns:
        ResidualDecoder or nn.Sequential instance
    """
    if linear_only:
        # Single linear layer: state must encode color directly.
        # No hidden capacity to "cheat" around state collapse.
        return nn.Sequential(
            nn.Linear(state_dim, output_dim, bias=True),
            nn.Sigmoid(),
        )
    return ResidualDecoder(
        state_dim=state_dim,
        hidden_dims=(64, 32),
        output_dim=output_dim,
        num_res_blocks=1,
        activation='silu',
        output_activation='sigmoid',
        dropout=0.0,
    )
