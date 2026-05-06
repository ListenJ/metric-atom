import numpy as np
import cv2


def generate_scene(H=128, W=128, num_objects=2, seed=None):
    """
    生成包含几何形状的 2D 场景。
    
    场景范围 [0, 1] x [0, 1]。
    随机生成圆、矩形。
    
    Args:
        H, W: 图像分辨率
        num_objects: 物体数量 (2-4)
        seed: 随机种子
    
    Returns:
        image: (H, W, 3) float32, RGB in [0, 1]
        masks: (H, W, K) float32, K 个实例掩码
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 背景颜色（浅灰）
    bg_color = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    image = np.ones((H, W, 3), dtype=np.float32) * bg_color
    masks = []
    
    colors_pool = [
        [1.0, 0.2, 0.2],  # 红
        [0.2, 0.2, 1.0],  # 蓝
        [0.2, 1.0, 0.2],  # 绿
        [1.0, 0.8, 0.0],  # 黄
        [1.0, 0.0, 1.0],  # 紫
    ]
    
    for k in range(num_objects):
        color = np.array(colors_pool[k % len(colors_pool)], dtype=np.float32)
        mask = np.zeros((H, W), dtype=np.float32)
        
        shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
        
        if shape_type == 'circle':
            cx = int(np.random.uniform(0.15 * W, 0.85 * W))
            cy = int(np.random.uniform(0.15 * H, 0.85 * H))
            radius = int(np.random.uniform(0.08 * min(H, W), 0.25 * min(H, W)))
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
            
        elif shape_type == 'rectangle':
            x1 = int(np.random.uniform(0.1 * W, 0.5 * W))
            y1 = int(np.random.uniform(0.1 * H, 0.5 * H))
            x2 = int(x1 + np.random.uniform(0.2 * W, 0.4 * W))
            y2 = int(y1 + np.random.uniform(0.2 * H, 0.4 * H))
            x2 = min(x2, W - 1)
            y2 = min(y2, H - 1)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
            
        elif shape_type == 'triangle':
            pts = []
            cx = np.random.uniform(0.2 * W, 0.8 * W)
            cy = np.random.uniform(0.2 * H, 0.8 * H)
            size = np.random.uniform(0.1 * min(H, W), 0.25 * min(H, W))
            for _ in range(3):
                angle = np.random.uniform(0, 2 * np.pi)
                px = int(cx + size * np.cos(angle))
                py = int(cy + size * np.sin(angle))
                px = np.clip(px, 0, W - 1)
                py = np.clip(py, 0, H - 1)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1.0)
        
        image[mask > 0.5] = color
        masks.append(mask)
    
    masks = np.stack(masks, axis=-1).astype(np.float32)  # (H, W, K)
    
    return image, masks


def generate_multi_view(H=128, W=128, num_objects=2, num_views=8, seed=None):
    """
    生成多视角数据，通过随机仿射变换模拟相机运动。
    
    Args:
        H, W: 图像分辨率
        num_objects: 物体数量
        num_views: 视角数量
        seed: 随机种子
    
    Returns:
        images: (V, H, W, 3) float32, RGB in [0, 1]
        masks: (V, H, W, K) float32, K 个实例掩码
        transforms: list of (2, 3) 仿射变换矩阵
    """
    if seed is not None:
        np.random.seed(seed)
    
    images = []
    all_masks = []
    transforms = []
    
    for v in range(num_views):
        # 生成基础场景
        image, masks_v = generate_scene(H, W, num_objects, seed=seed + v if seed is not None else None)
        
        # 随机仿射变换（轻微旋转/缩放/平移）
        angle = np.random.uniform(-15, 15) * np.pi / 180.0
        scale = np.random.uniform(0.85, 1.15)
        tx = np.random.uniform(-0.05, 0.05) * W
        ty = np.random.uniform(-0.05, 0.05) * H
        
        M = np.array([
            [scale * np.cos(angle), -scale * np.sin(angle), tx],
            [scale * np.sin(angle),  scale * np.cos(angle), ty]
        ], dtype=np.float32)
        transforms.append(M)
        
        # 对图像和每个掩码通道应用变换
        image_v = cv2.warpAffine(image, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(0.9, 0.9, 0.9))
        
        masks_v_transformed = []
        for k in range(masks_v.shape[-1]):
            mask_k = cv2.warpAffine(masks_v[..., k], M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            masks_v_transformed.append(mask_k)
        
        masks_v = np.stack(masks_v_transformed, axis=-1)
        
        images.append(image_v)
        all_masks.append(masks_v)
    
    images = np.stack(images, axis=0).astype(np.float32)  # (V, H, W, 3)
    all_masks = np.stack(all_masks, axis=0).astype(np.float32)  # (V, H, W, K)
    
    return images, all_masks, transforms


def get_occupancy(masks):
    """
    从实例掩码计算占位掩码（所有物体区域的并集）。
    
    Args:
        masks: (V, H, W, K)
    
    Returns:
        occupancy: (H, W) 0/1 掩码（取第一帧的占位区域）
    """
    # 使用第一帧，对所有实例取或
    mask_v0 = masks[0]  # (H, W, K)
    occupancy = (mask_v0.sum(axis=-1) > 0.5).astype(np.float32)
    return occupancy
