import torch
import pytest

from src.atoms.atom_2d import Atom2D


def identity_metric(x):
    """恒等度量函数，用于测试"""
    g = torch.eye(2, dtype=x.dtype, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
    return g


class TestAtom2D:
    
    def test_truncation(self):
        """验证截断范围外权重为0"""
        mu = torch.tensor([0.5, 0.5])
        atom = Atom2D(mu, radius=0.1, color=torch.tensor([1.0, 0.0, 0.0]))
        
        # 在远处查询
        x_far = torch.tensor([[0.9, 0.9], [0.1, 0.1]])
        weight, density, feat = atom(x_far, identity_metric)
        
        assert torch.allclose(weight, torch.zeros_like(weight), atol=1e-5), \
            f"Far points should have zero weight, got {weight}"
    
    def test_center_weight(self):
        """验证中心点权重为最大"""
        mu = torch.tensor([0.5, 0.5])
        atom = Atom2D(mu, radius=0.1, color=torch.tensor([1.0, 0.0, 0.0]))
        
        x_center = mu.unsqueeze(0)
        weight, density, feat = atom(x_center, identity_metric)
        
        assert torch.allclose(weight, torch.ones_like(weight)), \
            f"Center should have weight=1, got {weight}"
    
    def test_gradient_flow(self):
        """验证梯度可以反向传播到所有参数"""
        mu = torch.tensor([0.5, 0.5])
        atom = Atom2D(mu, radius=0.1, color=torch.tensor([1.0, 0.0, 0.0]))
        
        x = torch.rand(10, 2)
        weight, density, feat = atom(x, identity_metric)
        # color 也参与损失，确保其梯度流
        loss = density.sum() + feat.sum() + (atom._color * density.unsqueeze(-1)).sum()
        loss.backward()
        
        assert atom._mu.grad is not None, "Gradient must flow to mu"
        assert atom._log_r.grad is not None, "Gradient must flow to log_r"
        assert atom._color.grad is not None, "Gradient must flow to color"
        assert atom._state.grad is not None, "Gradient must flow to state"
        assert atom._logit_eps.grad is not None, "Gradient must flow to logit_eps"
    
    def test_smoothstep_continuity(self):
        """验证smoothstep在边界处连续"""
        mu = torch.tensor([0.5, 0.5])
        atom = Atom2D(mu, radius=0.1, color=torch.tensor([1.0, 0.0, 0.0]))
        
        # 在截断边界附近采样
        delta = 0.2
        r = 0.1
        # t = 1.0 - delta/2 应该处于过渡区
        dist = (1.0 - delta / 2) * r
        
        x_near = torch.tensor([[0.5 + dist, 0.5]])
        weight, _, _ = atom(x_near, identity_metric)
        
        assert 0.0 < weight.item() < 1.0, \
            f"Transition region should have 0 < weight < 1, got {weight.item()}"
    
    def test_parameter_shapes(self):
        """验证参数形状正确"""
        mu = torch.tensor([0.5, 0.5])
        atom = Atom2D(mu, radius=0.1, color=torch.tensor([1.0, 0.0, 0.0]), feature_dim=32)
        
        assert atom.position.shape == (2,)
        assert atom._color.shape == (3,)
        assert atom._feature.shape == (32,)
        assert atom._log_r.numel() == 1
        assert atom._logit_eps.numel() == 1
    
    def test_density_bounded(self):
        """验证密度在 [0, eps] 范围内"""
        mu = torch.tensor([0.5, 0.5])
        atom = Atom2D(mu, radius=0.1, color=torch.tensor([1.0, 0.0, 0.0]), eps=0.5)
        
        x = torch.rand(50, 2)
        _, density, _ = atom(x, identity_metric)
        
        assert torch.all(density >= 0), "Density must be non-negative"
        assert torch.all(density <= atom.existence_prob + 1e-5), \
            f"Density must be <= existence_prob"
    
    def test_feature_contrib_scale(self):
        """验证特征贡献与密度成正比"""
        mu = torch.tensor([0.5, 0.5])
        atom = Atom2D(mu, radius=0.1, color=torch.tensor([1.0, 0.0, 0.0]))
        
        x = torch.tensor([[0.5, 0.5], [0.9, 0.9]])
        weight, density, feat = atom(x, identity_metric)
        
        # 中心点密度高，特征贡献大；远点密度为0，特征贡献为0
        assert density[1].item() < 1e-5, "Far point should have near-zero density"
        assert feat[1].abs().max().item() < 1e-5, "Far point should have near-zero feature contrib"
