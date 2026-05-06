# Task 8: 可视化与评估

## 描述
实现训练过程的可视化工具，将度量场、原子分布、聚类结果绘制出来。评估聚类质量。

## 约束词
- 使用matplotlib绘制，不依赖任何GUI库（使用Agg后端）
- 度量场可视化采用椭圆场图（主轴代表特征向量和特征值）
- 聚类评估使用sklearn的ARI和NMI（仅用于验证，不参与训练）
- 所有可视化保存为PNG，可组成GIF（可选）
- 原子分布图必须显示：中心点、半径圆、存在概率（透明度）

## 输出文件
- src/visualization/plot_metric.py: 度量场可视化
- src/visualization/plot_atoms.py: 原子分布可视化
- src/visualization/plot_clusters.py: 聚类结果可视化
- notebooks/04_evaluation.ipynb: 评估笔记本

## 可视化内容
1. **度量场热力图**: trace(g)的空间分布
2. **椭圆场图**: 在每个像素绘制小椭圆表示局部度量
3. **原子分布图**: 原子中心、半径、颜色叠加在渲染图上
4. **聚类对比图**: 左=真实掩码，右=原子聚类结果
5. **训练曲线**: 各损失随epoch变化

## 评估指标
- **ARI** (Adjusted Rand Index): 聚类与真实掩码的相似度
- **NMI** (Normalized Mutual Information): 信息论相似度
- **IoU**: 簇边界与物体边界的重叠度
- **PSNR/SSIM**: 渲染质量

## 测试要求
1. 生成模拟数据，验证可视化函数不报错
2. 验证ARI/NMI计算正确性（已知聚类应得1.0）
3. 验证椭圆场图的方向和大小与度量特征值一致
