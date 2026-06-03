import torch
import pytest

from src.losses.reconstruction import l1_loss, l2_loss, reconstruction_loss
from src.losses.metric_regularizer import metric_smoothness_loss
from src.losses.occupancy_coupling import occupancy_coupling_loss
from src.geometry.metric_field import MetricField2D
from src.atoms.atom_2d import Atom2D


class TestReconstructionLoss:
    
    def test_l1_loss_perfect(self):
        pred = torch.tensor([[0.5, 0.5, 0.5]])
        target = torch.tensor([[0.5, 0.5, 0.5]])
        loss = l1_loss(pred, target)
        assert loss.item() == 0.0
    
    def test_l1_loss_positive(self):
        pred = torch.tensor([[0.0, 0.5, 1.0]])
        target = torch.tensor([[1.0, 0.5, 0.0]])
        loss = l1_loss(pred, target)
        assert loss.item() > 0.0
    
    def test_l2_loss_symmetric(self):
        pred = torch.tensor([[0.3, 0.7]])
        target = torch.tensor([[0.7, 0.3]])
        loss = l2_loss(pred, target)
        expected = ((0.4) ** 2 + (0.4) ** 2) / 2
        assert torch.allclose(loss, torch.tensor(expected))
    
    def test_reconstruction_mode_l1(self):
        pred = torch.tensor([[0.5, 0.5]])
        target = torch.tensor([[1.0, 0.5]])
        loss = reconstruction_loss(pred, target, mode='l1')
        assert torch.allclose(loss, torch.tensor(0.25))
    
    def test_reconstruction_mode_l2(self):
        pred = torch.tensor([[0.5, 0.5]])
        target = torch.tensor([[1.0, 0.5]])
        loss = reconstruction_loss(pred, target, mode='l2')
        assert torch.allclose(loss, torch.tensor(0.125))
    
    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            reconstruction_loss(torch.zeros(1, 3), torch.zeros(1, 3), mode='l3')


class TestMetricSmoothnessLoss:
    
    def test_constant_field_zero_loss(self):
        """验证恒定度量场平滑损失为0"""
        field = MetricField2D(16, 16)
        loss = metric_smoothness_loss(field)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
    
    def test_varying_field_positive_loss(self):
        """验证变化度量场平滑损失为正"""
        field = MetricField2D(16, 16)
        with torch.no_grad():
            field.params[0, 0, 5, 5] += 2.0
        loss = metric_smoothness_loss(field)
        assert loss.item() > 0.0
    
    def test_gradient_flow(self):
        """验证梯度可反向传播"""
        field = MetricField2D(8, 8)
        loss = metric_smoothness_loss(field)
        loss.backward()
        assert field.params.grad is not None


class TestOccupancyCouplingLoss:
    
    def test_basic_loss(self):
        """验证占位耦合损失基本计算"""
        field = MetricField2D(8, 8)
        mask = torch.zeros(8, 8)
        mask[3:5, 3:5] = 1.0
        
        loss = occupancy_coupling_loss(field, mask)
        assert loss.item() > 0.0
    
    def test_gradient_flow(self):
        """验证梯度可反向传播"""
        field = MetricField2D(8, 8)
        mask = torch.zeros(8, 8)
        mask[3:5, 3:5] = 1.0
        
        loss = occupancy_coupling_loss(field, mask)
        loss.backward()
        assert field.params.grad is not None
    
    def test_target_values(self):
        """验证目标值对损失的影响"""
        field = MetricField2D(8, 8, init_scale=1.0)
        mask = torch.ones(8, 8)
        
        # trace 初始约为 2*(1^2+eps) ~ 2
        loss1 = occupancy_coupling_loss(field, mask, g_occ_target=2.0, g_bg_target=10.0)
        loss2 = occupancy_coupling_loss(field, mask, g_occ_target=10.0, g_bg_target=2.0)
        
        assert loss1.item() < loss2.item(), \
            "Loss should be lower when trace is close to target"
    
    def test_empty_mask(self):
        """验证空掩码（全背景）"""
        field = MetricField2D(8, 8)
        mask = torch.zeros(8, 8)
        
        loss = occupancy_coupling_loss(field, mask, g_bg_target=1.0)
        assert torch.isfinite(loss)



