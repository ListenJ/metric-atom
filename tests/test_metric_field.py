import torch
import pytest

from src.geometry.metric_field import MetricField2D


class TestMetricField2D:
    
    def test_shape_and_positive_definite(self):
        """验证输出形状和正定性"""
        H, W = 64, 64
        field = MetricField2D(H, W)
        
        coords = torch.rand(100, 2)
        g = field(coords)
        
        assert g.shape == (100, 2, 2), f"Expected (100, 2, 2), got {g.shape}"
        
        # 验证对称性
        assert torch.allclose(g[:, 0, 1], g[:, 1, 0]), "Metric tensor must be symmetric"
        
        # 验证正定性：所有特征值 > 0
        eigvals = torch.linalg.eigvalsh(g)
        assert torch.all(eigvals > 0), "Metric tensor must be positive definite"
        assert torch.all(eigvals > 1e-4), "Eigenvalues must be > eps"
    
    def test_gradient_flow(self):
        """验证梯度可以反向传播到参数"""
        H, W = 32, 32
        field = MetricField2D(H, W)
        
        coords = torch.rand(10, 2, requires_grad=False)
        g = field(coords)
        loss = g.sum()
        loss.backward()
        
        assert field.params.grad is not None, "Gradient must flow to params"
        assert not torch.all(field.params.grad == 0), "Gradient must be non-zero"
    
    def test_coordinate_bounds(self):
        """验证坐标超出范围会报错"""
        field = MetricField2D(16, 16)
        
        with pytest.raises(ValueError):
            coords_invalid = torch.tensor([[1.5, 0.5]])
            field(coords_invalid)
        
        with pytest.raises(ValueError):
            coords_invalid = torch.tensor([[-0.1, 0.5]])
            field(coords_invalid)
    
    def test_initialization(self):
        """验证初始化产生接近单位矩阵的度量"""
        H, W = 16, 16
        field = MetricField2D(H, W, init_scale=1.0)
        
        # 在中心点采样
        coords = torch.tensor([[0.5, 0.5]])
        g = field(coords)
        
        # 初始化时 l21=0, l11=l22=1.0
        # g = [[1+eps, 0], [0, 1+eps]]
        expected = torch.eye(2) * (1.0 + 1e-4)
        assert torch.allclose(g[0], expected, atol=1e-3), \
            f"Expected ~identity, got {g[0]}"
    
    def test_trace(self):
        """验证迹的计算"""
        H, W = 16, 16
        field = MetricField2D(H, W)
        
        coords = torch.rand(20, 2)
        g = field(coords)
        trace = g[:, 0, 0] + g[:, 1, 1]
        
        trace_method = field.trace(coords)
        assert torch.allclose(trace, trace_method, atol=1e-5)
    
    def test_bilinear_interpolation(self):
        """验证双线性插值的连续性"""
        H, W = 8, 8
        field = MetricField2D(H, W)
        
        # 在两个相邻点采样
        coords1 = torch.tensor([[0.25, 0.25]])
        coords2 = torch.tensor([[0.2501, 0.25]])
        
        g1 = field(coords1)
        g2 = field(coords2)
        
        diff = (g1 - g2).abs().max()
        assert diff < 1e-2, f"Bilinear interpolation should be continuous, diff={diff}"
