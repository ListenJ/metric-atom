import torch
import pytest

from src.geometry.mlp_metric_field import MLPMetricField2D


class TestMLPMetricField2D:
    
    def test_shape_and_positive_definite(self):
        field = MLPMetricField2D(hidden_dim=128, num_layers=4)
        
        coords = torch.rand(100, 2)
        g = field(coords)
        
        assert g.shape == (100, 2, 2), f"Expected (100, 2, 2), got {g.shape}"
        
        assert torch.allclose(g[:, 0, 1], g[:, 1, 0]), "Metric tensor must be symmetric"
        
        eigvals = torch.linalg.eigvalsh(g)
        assert torch.all(eigvals > 0), "Metric tensor must be positive definite"
        assert torch.all(eigvals > 1e-4), "Eigenvalues must be > eps"
    
    def test_gradient_flow(self):
        field = MLPMetricField2D(hidden_dim=64, num_layers=4)
        
        coords = torch.rand(10, 2, requires_grad=False)
        g = field(coords)
        loss = g.sum()
        loss.backward()
        
        for param in field.mlp.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradient must flow to all params"
        
        first_layer_grad = field.mlp[0].weight.grad
        assert first_layer_grad is not None
        assert not torch.all(first_layer_grad == 0), "Gradient must be non-zero"
    
    def test_coordinate_bounds(self):
        field = MLPMetricField2D(hidden_dim=64, num_layers=4)
        
        coords = torch.tensor([[1.5, 0.5], [-0.1, 1.2]])
        g = field(coords)
        assert g.shape == (2, 2, 2)
        
        eigvals = torch.linalg.eigvalsh(g)
        assert torch.all(eigvals > 0), "Even clamped coords must produce positive definite g"
    
    def test_initialization(self):
        field = MLPMetricField2D(hidden_dim=64, num_layers=4, init_scale=1.0)
        
        coords = torch.tensor([[0.5, 0.5]])
        g = field(coords)
        
        expected = torch.eye(2) * (1.0 + 1e-4)
        assert torch.allclose(g[0], expected, atol=1e-2), \
            f"Expected ~identity at center, got {g[0]}"
    
    def test_trace(self):
        field = MLPMetricField2D(hidden_dim=64, num_layers=4)
        
        coords = torch.rand(20, 2)
        g = field(coords)
        trace = g[:, 0, 0] + g[:, 1, 1]
        
        trace_method = field.trace(coords)
        assert torch.allclose(trace, trace_method, atol=1e-5)
    
    def test_smoothness(self):
        field = MLPMetricField2D(hidden_dim=64, num_layers=4)
        
        coords1 = torch.tensor([[0.25, 0.25]])
        coords2 = torch.tensor([[0.2501, 0.25]])
        
        g1 = field(coords1)
        g2 = field(coords2)
        
        diff = (g1 - g2).abs().max()
        assert diff < 1e-3, f"MLP should be smooth, diff={diff}"
    
    def test_resolution_independence(self):
        field = MLPMetricField2D(hidden_dim=64, num_layers=4)
        
        coords_low = torch.rand(100, 2) * 0.5
        coords_high = coords_low
        
        g_low = field(coords_low)
        g_high = field(coords_high)
        
        assert torch.allclose(g_low, g_high, atol=1e-5), \
            "MLP metric field should be resolution-independent"
    
    def test_batch_consistency(self):
        field = MLPMetricField2D(hidden_dim=64, num_layers=4)
        
        coords = torch.rand(50, 2)
        
        g_batch = field(coords)
        
        g_individual = torch.stack([field(c.unsqueeze(0)).squeeze(0) for c in coords])
        
        assert torch.allclose(g_batch, g_individual, atol=1e-5), \
            "Batch and individual evaluation must match"