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
├── geometry/           # 黎曼度量场（2D/3D）+ 椭圆曲线几何 (Phase 6b)
│   ├── cholesky_param.py
│   ├── metric_field.py
│   ├── elliptic_curve.py    # ECO: 群运算 / log/exp 映射 / j-不变量
│   └── murmuration.py       # Boids 在椭圆曲线上的动力学
├── atoms/              # 感知原子定义
├── rendering/          # 体积渲染器
├── losses/             # 损失函数
│   ├── direct_cluster.py    # Direct Loss (Path 1+3, ARI 0.755)
│   └── eco_cluster.py       # ECO Cluster Loss (Phase 6c, j-不变量正则化)
├── data/               # 合成数据生成
├── visualization/      # 可视化
└── training/           # 训练循环
```

## 历史成果

> 完整实验历史（Phase 1 → Phase 7）请见 [docs/history.md](docs/history.md)。
> 以下仅列出最近三次关键结果。

### Phase 6b: ECO 代码实现（2026-05-18）

| 模块 | 功能 |
|------|------|
| `elliptic_curve.py` | 群运算（加法/标量乘法）、log/exp 映射、测地距离数值积分、j-不变量、分岔检测 |
| `murmuration.py` | Boids 在椭圆曲线上的动力学（凝聚/对齐/分离力在 1D 切空间中），PyTorch 批处理版本 |
| `eco_cluster.py` | ECO 版 Direct Loss：感知函数 φ: features→(a,b)，曲线残差成本矩阵，j-不变量身份一致性正则项 |

### Phase 7: Landscape 系统性扫描（2026-05-19）

**目标**：量化 DirectCluster 对随机初始化的敏感度，验证 Phase 6a "尖锐悬崖"理论。

在 Tesla T4 云端 GPU 上运行 8 个不同 seed（100-107），每个 200 epochs，111 atoms，64×64 分辨率。

**核心发现**：
- 存在**完美解**（seed 107: ARI=1.0, NMI=1.0, 簇平衡 B=1.00），证明 DirectCluster 在有利初始化下可以完全解决 2D 双物体场景
- 但方差极大（σ=0.39），约 50% seed 未能有效聚类——确认了尖锐悬崖现象
- 完美聚类仅需 50/111 原子落在物体内，说明聚类质量 ≠ 覆盖数量

### Phase 6c: ECO 统一公式 + 精细调参（2026-05-20）

**核心统一公式**：`feature → 感知函数 φ → (a,b) → j-不变量 → Sinkhorn cost C_ik = |j_i - j_k|`

#### 调参历程

| 迭代 | 配置变化 | 均值 ARI | 标准差 | ≥0.5 |
|------|---------|---------|--------|------|
| 基线 DC | — | 0.421 | 0.390 | 4/8 |
| + z-score ECO | --use-eco, w=0.5 | 0.548 | 0.278 | 6/8 |
| + 平衡 KMeans | Balanced init | 0.548 | 0.278 | 6/8 |
| + **温度 eps=0.05** | --sinkhorn-eps 0.05 | **0.637** | 0.308 | **7/8** |
| eps=0.02 | 更冷 | 0.621 | 0.321 | 6/8 |
| eps=0.10 | 更暖 | 0.509 | 0.342 | 5/8 |

#### 最优配置

```bash
python train_2d.py --bf16 --use-eco --w-eco 0.5 --eco-id-weight 0.1 \
                   --sinkhorn-eps 0.05
```

| 指标 | 值 |
|------|-----|
| 均值 ARI | **0.637** |
| 标准差 | 0.308 |
| ≥0.5 | 7/8 |
| 最佳种子 | seed107=0.922, seed105=0.833, seed103=0.908 |
| 失败种子 | seed106=0.003 |

#### 被排除的方向

| 方向 | 结论 |
|------|------|
| 特征扩散 | ❌ 扩散与 ECO j-不变量匹配互斥，全部退化为更差结果 |
| 直接 j_protos 参数 | ❌ 脱离 φ 统一参考系后框架失调，seed101 从 0.639→0.009 |
| Sinkhorn 退火 | 🔶 边际收益微小，不优先投入 |
| 曲线残差 cost | ❌ J-不变量匹配显著更优 |


## 约束

本项目遵循严格的零外部先验原则：
- 禁止使用 COLMAP、SAM、CLIP、高斯泼溅
- 禁止使用预训练语义/分割模型
- 仅使用多视图图像 + 可微渲染 + 几何正则化

## 分支说明

| 分支 | 用途 |
|------|------|
| `main` | 稳定发布版本 |
| `feat/clustering-breakthrough` | 无监督聚类突破（Direct Loss + ECO 理论框架） |

## 许可

MIT
