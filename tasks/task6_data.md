# Task 6: 2D合成数据生成器

## 描述
生成多个带有多物体的2D场景的多视角图像和对应实例掩码（仅用于评估，不参与训练）。

## 约束词
- 使用OpenCV绘制基础形状（圆、矩形、三角形）
- 生成多视图通过随机仿射变换模拟相机运动
- 输出归一化到[0,1]的RGB图像和0/1实例掩码
- 不包含任何真实图像或3D模型
- 场景必须包含至少2个不同颜色的物体

## 输出文件
- src/data/synthetic_2d.py: 合成数据生成器
- src/data/transforms.py: 几何/颜色变换
- src/data/data_loader.py: 数据加载器

## 接口定义
```python
def generate_scene(H=128, W=128, num_objects=2, num_views=8):
    """
    Returns:
        images: (V, H, W, 3) float32 in [0,1]
        masks: (V, H, W, K) uint8, K个实例掩码
        camera_poses: (V, 3, 3) affine变换矩阵
    """
```

## 测试要求
1. 生成数据并可视化，确认物体边界清晰
2. 验证多视角间的一致性（同一物体在不同视角可见）
3. 验证掩码与图像完全对齐
