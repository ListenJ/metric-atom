import torch
import pytest

from src.rendering.ray_sampler import RaySampler2D
from src.rendering.volume_renderer_2d import volume_render_2d
from src.geometry.metric_field import MetricField2D
from src.atoms.atom_2d import Atom2D


class TestRaySampler2D:
    
    def test_orthographic_shape(self):
        """验证正交投影光线形状"""
        H, W = 8, 8
        rays_o, rays_d = RaySampler2D.generate_rays_orthographic(H, W)
        
        assert rays_o.shape == (H * W, 2), f"Expected {(H*W, 2)}, got {rays_o.shape}"
        assert rays_d.shape == (H * W, 2), f"Expected {(H*W, 2)}, got {rays_d.shape}"
    
    def test_orthographic_direction_normalized(self):
        """验证正交投影光线方向归一化"""
        H, W = 8, 8
        rays_o, rays_d = RaySampler2D.generate_rays_orthographic(H, W)
        
        norms = rays_d.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms)), \
            f"Ray directions must be normalized, got norms: {norms}"
    
    def test_perspective_shape(self):
        """验证透视投影光线形状"""
        H, W = 8, 8
        camera_center = torch.tensor([1.0, 1.0])
        look_at = torch.tensor([1.0, 1.0])
        
        rays_o, rays_d = RaySampler2D.generate_rays_from_camera(
            H, W, camera_center, look_at
        )
        
        assert rays_o.shape == (H * W, 2)
        assert rays_d.shape == (H * W, 2)
    
    def test_perspective_direction_normalized(self):
        """验证透视投影光线方向归一化"""
        H, W = 8, 8
        camera_center = torch.tensor([0.5, 0.5])
        look_at = torch.tensor([1.5, 1.5])
        
        rays_o, rays_d = RaySampler2D.generate_rays_from_camera(
            H, W, camera_center, look_at
        )
        
        norms = rays_d.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Ray directions must be normalized"


class TestVolumeRenderer2D:
    
    def test_render_shape(self):
        """验证渲染输出形状"""
        H, W = 8, 8
        metric_field = MetricField2D(H, W)
        
        # 创建两个原子（坐标归一化到 [0, 1]）
        atom1 = Atom2D(
            torch.tensor([0.25, 0.25]),
            radius=0.15,
            color=torch.tensor([1.0, 0.0, 0.0])
        )
        atom2 = Atom2D(
            torch.tensor([0.75, 0.75]),
            radius=0.15,
            color=torch.tensor([0.0, 0.0, 1.0])
        )
        atoms = [atom1, atom2]
        
        rays_o, rays_d = RaySampler2D.generate_rays_orthographic(H, W, scene_size=1.0)
        
        color, depth, alpha = volume_render_2d(
            rays_o, rays_d, atoms, metric_field, num_samples=32
        )
        
        assert color.shape == (H * W, 3), f"Expected color {(H*W, 3)}, got {color.shape}"
        assert depth.shape == (H * W,), f"Expected depth {(H*W,)}, got {depth.shape}"
        assert alpha.shape == (H * W,), f"Expected alpha {(H*W,)}, got {alpha.shape}"
    
    def test_render_color_range(self):
        """验证渲染颜色在合理范围内"""
        H, W = 8, 8
        metric_field = MetricField2D(H, W)
        
        atom = Atom2D(
            torch.tensor([0.5, 0.5]),
            radius=0.25,
            color=torch.tensor([0.5, 0.5, 0.5])
        )
        atoms = [atom]
        
        rays_o, rays_d = RaySampler2D.generate_rays_orthographic(H, W, scene_size=1.0)
        
        color, depth, alpha = volume_render_2d(
            rays_o, rays_d, atoms, metric_field, num_samples=16
        )
        
        assert torch.all(color >= 0), "Color must be non-negative"
        assert torch.all(color <= 1.0 + 1e-5), "Color must be <= 1.0"
        assert torch.all(alpha >= 0), "Alpha must be non-negative"
        assert torch.all(alpha <= 1.0 + 1e-5), "Alpha must be <= 1.0"
    
    def test_gradient_flow_to_atoms(self):
        """验证梯度可以反向传播到原子参数"""
        H, W = 4, 4
        metric_field = MetricField2D(H, W)
        
        atom = Atom2D(
            torch.tensor([0.5, 0.5]),
            radius=0.25,
            color=torch.tensor([0.5, 0.5, 0.5])
        )
        atoms = [atom]
        
        rays_o, rays_d = RaySampler2D.generate_rays_orthographic(H, W, scene_size=1.0)
        
        color, depth, alpha = volume_render_2d(
            rays_o, rays_d, atoms, metric_field, num_samples=16
        )
        
        loss = color.sum() + depth.sum()
        loss.backward()
        
        assert atom._mu.grad is not None, "Gradient must flow to atom position"
        assert atom._log_r.grad is not None, "Gradient must flow to atom radius"
        assert atom._color.grad is not None, "Gradient must flow to atom color"
    
    def test_empty_scene(self):
        """验证空场景渲染为黑色"""
        H, W = 4, 4
        metric_field = MetricField2D(H, W)
        atoms = []
        
        rays_o, rays_d = RaySampler2D.generate_rays_orthographic(H, W)
        
        color, depth, alpha = volume_render_2d(
            rays_o, rays_d, atoms, metric_field, num_samples=8
        )
        
        assert torch.allclose(color, torch.zeros_like(color), atol=1e-5), \
            "Empty scene should render black"
        assert torch.allclose(alpha, torch.zeros_like(alpha), atol=1e-5), \
            "Empty scene should have zero alpha"
    
    def test_alpha_monotonic(self):
        """验证不透明度累积是单调的"""
        H, W = 4, 4
        metric_field = MetricField2D(H, W)
        
        atom = Atom2D(
            torch.tensor([0.5, 0.5]),
            radius=0.25,
            color=torch.tensor([0.5, 0.5, 0.5])
        )
        atoms = [atom]
        
        rays_o, rays_d = RaySampler2D.generate_rays_orthographic(H, W, scene_size=1.0)
        
        color, depth, alpha = volume_render_2d(
            rays_o, rays_d, atoms, metric_field, num_samples=16
        )
        
        # Alpha 应该在 [0, 1] 范围内
        assert torch.all(alpha >= 0), "Alpha must be >= 0"
        assert torch.all(alpha <= 1.0 + 1e-5), "Alpha must be <= 1"
