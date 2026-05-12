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


class RaySampler3D:
    """
    3D 光线采样器。
    生成从相机出发穿过像素平面的光线。
    """

    @staticmethod
    def generate_rays(H, W, fx, fy, cam_pos, cam_rot, near=0.1, far=2.0, device='cpu'):
        """
        从相机生成 3D 光线（针孔相机模型）。

        Args:
            H, W: 图像分辨率
            fx, fy: 焦距（像素单位）
            cam_pos: (3,) 相机位置
            cam_rot: (3, 3) 旋转矩阵（世界→相机）
            near, far: 近远平面
            device: 计算设备

        Returns:
            rays_o: (H*W, 3) 射线起点
            rays_d: (H*W, 3) 射线方向（已归一化）
        """
        # 生成像素坐标
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        # 归一化到相机坐标系
        x_cam = (xs - W / 2.0) / fx
        y_cam = (ys - H / 2.0) / fy
        z_cam = torch.ones_like(x_cam)  # 前向

        # 相机空间光线方向
        dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (H, W, 3)
        dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)

        # 转到世界空间
        dirs_world = (cam_rot @ dirs_cam.unsqueeze(-1)).squeeze(-1)  # (H, W, 3)

        rays_o = cam_pos.unsqueeze(0).unsqueeze(0).expand(H, W, -1)
        rays_d = dirs_world

        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    @staticmethod
    def look_at(eye, target, up=None):
        """
        构建 look-at 旋转矩阵（世界→相机）。

        Args:
            eye: (3,) 相机位置
            target: (3,) 目标点
            up: (3,) 上方向，默认 (0,1,0)

        Returns:
            rot: (3, 3) 旋转矩阵
        """
        eye = torch.as_tensor(eye, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        if up is None:
            up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

        z = (eye - target) / (eye - target).norm()
        x = torch.cross(up, z)
        x = x / x.norm()
        y = torch.cross(z, x)

        rot = torch.stack([x, y, z], dim=0)  # (3, 3), 从世界到相机
        return rot
