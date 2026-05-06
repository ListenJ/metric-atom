import torch
import pytest

from src.losses.reconstruction import l1_loss, l2_loss, reconstruction_loss
from src.losses.metric_regularizer import metric_smoothness_loss
from src.losses.occupancy_coupling import occupancy_coupling_loss
from src.losses.coherence import coherence_loss
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


class TestCoherenceLoss:
    
    def test_single_atom(self):
        """验证单原子凝聚损失为0"""
        field = MetricField2D(8, 8)
        atom = Atom2D(
            torch.tensor([0.5, 0.5]),
            radius=0.1,
            color=torch.tensor([1.0, 0.0, 0.0])
        )
        atoms = [atom]
        
        loss = coherence_loss(atoms, field)
        assert loss.item() == 0.0
    
    def test_similar_atoms_lower_loss(self):
        """验证相似原子凝聚损失更低"""
        field = MetricField2D(8, 8)
        
        # 两个相似的原子（近距离 + 相似特征）
        atom1 = Atom2D(
            torch.tensor([0.5, 0.5]),
            radius=0.3,
            color=torch.tensor([1.0, 0.0, 0.0])
        )
        atom2 = Atom2D(
            torch.tensor([0.5, 0.5]),
            radius=0.3,
            color=torch.tensor([0.0, 0.0, 1.0])
        )
        # 设置相同特征
        with torch.no_grad():
            atom2._feature.copy_(atom1._feature)
        
        loss_close = coherence_loss([atom1, atom2], field)
        
        # 两个远离且特征不同的原子（坐标归一化到 [0, 1]）
        atom3 = Atom2D(
            torch.tensor([0.25, 0.25]),
            radius=0.05,
            color=torch.tensor([1.0, 0.0, 0.0])
        )
        atom4 = Atom2D(
            torch.tensor([0.75, 0.75]),
            radius=0.05,
            color=torch.tensor([0.0, 0.0, 1.0])
        )
        
        loss_far = coherence_loss([atom3, atom4], field)
        
        # 相近相似原子的吸引项应使损失更低（更负）
        assert loss_close.item() < loss_far.item(), \
            f"Similar atoms should have lower loss: close={loss_close}, far={loss_far}"
    
    def test_repulsion_prevents_collapse(self):
        """验证排斥项防止所有原子合并"""
        field = MetricField2D(8, 8)
        
        atoms = []
        for i in range(10):
            mu = torch.tensor([0.5, 0.5])
            atom = Atom2D(mu, radius=0.3, color=torch.rand(3))
            atoms.append(atom)
        
        loss = coherence_loss(atoms, field, repulsion_weight=0.1)
        
        # 排斥项应使损失不至于过度负值
        assert torch.isfinite(loss)
        assert loss.item() > -100.0, "Repulsion should prevent unbounded negative loss"
    
    def test_gradient_flow(self):
        """验证梯度可反向传播到原子参数"""
        field = MetricField2D(8, 8)
        
        atom1 = Atom2D(
            torch.tensor([0.5, 0.5]),
            radius=0.3,
            color=torch.tensor([1.0, 0.0, 0.0])
        )
        atom2 = Atom2D(
            torch.tensor([0.5, 0.5]),
            radius=0.15,
            color=torch.tensor([0.0, 0.0, 1.0])
        )
        atoms = [atom1, atom2]
        
        loss = coherence_loss(atoms, field)
        loss.backward()
        
        assert atom1._mu.grad is not None, "Gradient must flow to atom position"
        assert atom1._feature.grad is not None, "Gradient must flow to atom feature"
    
    def test_loss_shape(self):
        """验证损失是标量"""
        field = MetricField2D(8, 8)
        atom1 = Atom2D(torch.tensor([0.5, 0.5]), 0.3, torch.tensor([1.0, 0.0, 0.0]))
        atom2 = Atom2D(torch.tensor([0.5, 0.5]), 0.15, torch.tensor([0.0, 0.0, 1.0]))
        atoms = [atom1, atom2]
        
        loss = coherence_loss(atoms, field)
        assert loss.dim() == 0, f"Loss should be a scalar, got dim={loss.dim()}"
