import torch
import torch.nn as nn


class RaySampler2D:
    """
    2D 光线采样器。
    
    从相机位置向场景发射光线，覆盖整个图像平面。
    支持正交投影和透视投影（2D简化版）。
    """
    
    @staticmethod
    def generate_rays_orthographic(H, W, scene_size=2.0, device='cpu'):
        """
        生成正交投影光线。
        
        相机位于 z=-inf（2D中对应上方），光线垂直向下照射。
        
        Args:
            H, W: 图像分辨率
            scene_size: 场景范围 [0, scene_size]
            device: 计算设备
        
        Returns:
            rays_o: (H*W, 2) 射线起点（在图像平面上）
            rays_d: (H*W, 2) 射线方向（统一向下/向内）
        """
        # 生成像素网格坐标，归一化到 [0, scene_size]
        y, x = torch.meshgrid(
            torch.linspace(0, scene_size, H, device=device),
            torch.linspace(0, scene_size, W, device=device),
            indexing='ij'
        )
        
        # 射线起点就是像素位置
        rays_o = torch.stack([x.flatten(), y.flatten()], dim=-1)  # (H*W, 2)
        
        # 正交投影：所有光线方向相同（沿 x 轴正方向，简化 2D 场景）
        # 实际上在 2D 中，"光线"是从像素位置出发的线段
        # 为了简化，我们让光线沿对角线方向穿过场景
        rays_d = torch.ones_like(rays_o)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # 归一化
        
        return rays_o, rays_d
    
    @staticmethod
    def generate_rays_from_camera(H, W, camera_center, look_at, scene_size=2.0, device='cpu'):
        """
        从相机位置向场景发射光线（2D 简化透视投影）。
        
        Args:
            H, W: 图像分辨率
            camera_center: (2,) 相机中心位置
            look_at: (2,) 相机看向的点
            scene_size: 场景范围
            device: 计算设备
        
        Returns:
            rays_o: (H*W, 2) 射线起点
            rays_d: (H*W, 2) 射线方向（已归一化）
        """
        # 生成像素网格
        y, x = torch.meshgrid(
            torch.linspace(0, scene_size, H, device=device),
            torch.linspace(0, scene_size, W, device=device),
            indexing='ij'
        )
        
        pixels = torch.stack([x.flatten(), y.flatten()], dim=-1)  # (H*W, 2)
        
        # 光线从相机中心指向像素
        rays_o = camera_center.unsqueeze(0).expand(H * W, -1)  # (H*W, 2)
        rays_d = pixels - rays_o  # (H*W, 2)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # 归一化
        
        return rays_o, rays_d
