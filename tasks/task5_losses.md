# Task 5: 损失函数模块

## 描述
实现四个核心损失函数：重建损失、度量平滑损失、占位耦合损失、凝聚损失。

## 约束词
- 所有损失函数输入输出均为torch.Tensor
- 凝聚损失必须包含吸引项和排斥项，防止特征坍缩
- 度量平滑损失用有限差分计算梯度（空间导数）
- 不使用任何预先训练的语义模型
- 所有损失函数必须可微

## 输出文件
- src/losses/reconstruction.py: L1/L2重建损失
- src/losses/metric_regularizer.py: 度量平滑损失
- src/losses/occupancy_coupling.py: 占位耦合损失
- src/losses/coherence.py: 凝聚损失
- tests/test_losses.py: 单元测试

## 各损失公式

### Reconstruction Loss
`L_render = |pred - target|.mean()`

### Metric Smoothness
`L_met = (g(x+1,y) - g(x,y))^2 + (g(x,y+1) - g(x,y))^2`

### Occupancy Coupling
`L_vol = ((trace(g) - g_occ)^2 * mask).mean() + ((trace(g) - g_bg)^2 * (1-mask)).mean()`

### Coherence Loss
```
feat_affinity = exp(-||fi - fj||^2 / (2*sigma^2))
geodesic_dist = (mi - mj)^T * g * (mi - mj)
adjacency = geodesic_dist < (ri + rj)^2
C = sum(eps_i * eps_j * affinity * adjacency)
L_coh = -C + 0.01 * C^2 / N
```

## 测试要求
1. 构造简单场景，验证每个损失函数对参数的梯度不为零
2. 验证凝聚损失在特征相似时降低，特征不同时升高
3. 验证占位耦合损失在物体内部和背景的正确引导
