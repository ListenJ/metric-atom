# MetricAtom

**黎曼度量驱动的感知原子：从多视图图像中无监督涌现物体实例**

MetricAtom 是一个基于黎曼度量场和有界感知原子的三维场景理解系统。核心假设是：通过联合优化外观重建和度量结构，原子能自然地凝聚为与真实物体边界对齐的簇，无需任何预训练语义模型（SAM/CLIP）或外部 2D 分割引导。

## 核心思想

```
多视角图像
     │
     ▼
黎曼度量场 g(x) ←── 联合优化
     │
感知原子（紧支撑、特征向量、存在概率）
     │
可微体积渲染 ─── 重建损失
     │
凝聚损失 ───── 自发聚类
     │
物体实例（簇）
```

## 系统架构

```
src/
├── geometry/           # 黎曼度量场（2D/3D）、Cholesky 参数化
│   ├── cholesky_param.py    # 2D/3D Cholesky 分解 → 正定度量矩阵
│   └── metric_field.py      # MetricField2D / MetricField3D（三线性插值）
├── atoms/              # 感知原子定义
│   ├── base_atom.py         # 原子基类
│   ├── atom_2d.py           # 2D 原子（smoothstep 截断，马氏距离）
│   └── atom_3d.py           # 3D 原子
├── rendering/          # 体积渲染器
│   ├── ray_sampler.py        # RaySampler2D / RaySampler3D（针孔相机模型）
│   └── volume_renderer_2d.py # 2D/3D 可微体积渲染（向量化，17x 加速）
├── losses/             # 损失函数
│   ├── reconstruction.py     # L1/L2 重建损失
│   ├── coherence.py          # 凝聚损失（吸引 + 方差排斥）
│   ├── metric_regularizer.py # 度量平滑损失（2D/3D）
│   └── occupancy_coupling.py # 占位耦合损失
├── data/               # 合成数据生成
│   ├── synthetic_2d.py       # 2D 多形状多视角场景
│   └── synthetic_3d.py       # 3D 多球体场景（多视角渲染）
├── visualization/      # 可视化
│   ├── plot_metric.py        # 度量场迹 / 特征值可视化
│   ├── plot_atoms.py         # 原子分布散点图
│   └── plot_3d.py            # 3D 评估报告
└── training/           # 训练循环（规划中）
```

## 当前进度

### ✅ 已验证的功能

| 功能 | 状态 | 说明 |
|---|---|---|
| 度量场 Cholesky 参数化 | ✅ 完成 | 2D 网格/3D 体素，保证正定性，双线性/三线性插值 |
| 感知原子 smoothstep 截断 | ✅ 完成 | C² 连续，马氏距离空间支持 |
| 体积渲染器（向量化） | ✅ 完成 | 2D/3D 渲染管线，17x 加速 |
| 重建损失 | ✅ 完成 | L1 收敛至 0.04-0.08 |
| 度量平滑损失 | ✅ 完成 | 2D/3D 空间连续性正则化 |
| 占位耦合损失 | ✅ 完成 | 物体内 trace < 1，背景 trace > 9 |
| 凝聚损失（吸引+方差排斥） | ✅ 完成 | 特征多样性保持 (feat_std > 0.3) |
| 合成数据生成 | ✅ 完成 | 2D 多形状 + 3D 多球体场景 |
| 动态原子管理（播种/剪枝） | ✅ 完成 | 渲染贡献剪枝 + 误差驱动播种 |
| 原子位置正则化 | ✅ 完成 | 阻止原子漂离物体区域 |
| BF16 混合精度训练 | ✅ 完成 | CUDA 加速，显存优化 |
| Checkpoint 评估 | ✅ 完成 | 覆盖率 + ARI 快速验证 |
| MKL CPU 优化 | ✅ 完成 | 6 线程，4x 加速 |

### 🚧 进行中

| 工作 | 进度 | 说明 |
|---|---|---|
| 3D 场景理解扩展 | 🔄 开发中 | 3D 度量场 + 3D 原子 + 3D 渲染 + 训练脚本完成，正在调试验证 |
| 空间覆盖率提升 | 🔄 优化中 | 从 22% 目标提升至 40%+，掩码引导初始化已实现 |
| 论文撰写 | 📝 初稿中 | 私有仓库 |

### 📋 近期计划

- 覆盖率达标后：使用测地邻接改进特征-空间耦合
- 在真实图像上验证度量场分离能力
- 多物体 3D 场景聚类评估
- 时间维度扩展 g(x,t)

## 训练配置

| 参数 | 2D 验证 | 2D 完整 | 3D |
|---|---|---|---|
| 分辨率 | 64×64 | 128×128 | 64×64×64 |
| 原子数 | 100 | 200 | 200 |
| 训练步数 | 600 | 3000 | 2000 |
| Phase 2 开始 | epoch 250 | epoch 1200 | epoch 800 |
| 采样数/光线 | 64 | 128 | 128 |
| 学习率 | 1e-3 | 1e-3 | 5e-4 |
| 精度 | BF16 | BF16 | BF16 |

## 快速开始

```bash
# 安装依赖
pip install torch numpy scipy matplotlib opencv-python

# 2D 快速验证（64×64, BF16, ~5min on CUDA）
python train_2d.py --resolution 64 --epochs 600

# 2D 完整训练（128×128, ~30min on CUDA）
python train_2d.py --resolution 128 --epochs 3000

# 3D 训练
python train_3d.py

# 评估 checkpoint
python tasks/eval_checkpoint.py
```

## 环境要求

- **Python** 3.10+
- **PyTorch** 2.1+（推荐 CUDA 12+）
- **CUDA** 可选（支持 BF16 的 GPU 最佳）
- **CPU** MKL 加速（Intel CPU）

## 约束

本项目遵循严格的零外部先验原则：
- ❌ 禁止使用 COLMAP、SAM、CLIP、高斯泼溅
- ❌ 禁止使用预训练语义/分割模型
- ✅ 仅使用多视图图像 + 可微渲染 + 几何正则化

## 分支说明

| 分支 | 用途 |
|---|---|
| `main` | 稳定发布版本 |
| `feat/coverage-boost` | 空间覆盖率优化（最新实验） |
| `exp/repulsion-redesign` | 方差排斥 + 特征多样性实验 |
| `feat/atom-coverage` | 覆盖机制初版 |

## 许可

MIT
