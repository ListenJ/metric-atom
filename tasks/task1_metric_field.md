# Task 1: Cholesky参数化的度量场

## 描述
实现2D黎曼度量场 g(x)，将坐标映射到2×2正定矩阵。使用Cholesky分解 g = LL^T + εI 保证正定性。

## 约束词
- 必须使用纯PyTorch实现，继承nn.Module
- 正定性由Cholesky分解保证，不允许使用任何其他参数化
- 输入坐标归一化到[0,1]范围
- 初期实现为可学习像素网格（nn.Parameter），每个像素存储l11,l21,l22三个值
- 必须包含双线性插值采样功能（grid_sample）
- 不依赖任何第三方库除了PyTorch和numpy
- 单元测试必须验证正定性和梯度流

## 输出文件
- src/geometry/cholesky_param.py: Cholesky参数化工具函数
- src/geometry/metric_field.py: MetricField2D类
- tests/test_metric_field.py: 单元测试

## 测试要求
1. 随机采样度量矩阵，验证所有特征值 > 0
2. 验证梯度可反向传播到参数
3. 验证双线性插值在边界处的连续性
