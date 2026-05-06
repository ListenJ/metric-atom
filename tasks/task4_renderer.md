# Task 4: 2D体积渲染器

## 描述
实现2D体积渲染器，对每条光线采样并积分，计算像素颜色和深度。

## 约束词
- 纯PyTorch实现，不使用任何光栅化库
- 光线采样步长固定为2.0/num_samples（初期）
- 在每条光线上仅计算被激活原子的贡献（利用空间索引）
- 体积积分使用alpha合成公式：
  - alpha = 1 - exp(-sigma * dt)
  - T = cumprod(1 - alpha)
  - weight = T * alpha
- 输出渲染颜色和深度
- 单元测试验证梯度可反向传播到原子参数

## 输出文件
- src/rendering/ray_sampler.py: 光线生成
- src/rendering/volume_renderer_2d.py: VolumeRenderer2D
- tests/test_renderer.py: 单元测试

## 测试要求
1. 固定原子和度量，渲染已知像素值，检查数值正确性
2. 验证梯度可反向传播到原子位置和颜色
3. 验证深度估计在简单场景中的准确性
