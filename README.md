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
│   └── eco_cluster.py       # ECO Cluster Loss (Phase 6b, j-不变量稳定)
├── data/               # 合成数据生成
├── visualization/      # 可视化
└── training/           # 训练循环
```

## 当前进度

### 实验结果（2026-05-18）

| 实验 | ARI | NMI | Valid | 结论 |
|------|-----|-----|-------|------|
| seed 123 (64) | -0.0292 | 0.0049 | 58/82 | 失败 — 初始化接近分岔 |
| **seed 456 (64)** | **0.9375** | 0.8992 | 64/82 | **优秀** — 远离奇异点 |
| 3-object (64) | 0.1755 | 0.1928 | 66/82 | 差 — j-不变量冲突 |
| 128×128 (seed 42) | 0.0000 | 0.0000 | 45/90 | 失败 — 高分辨率尖锐性加剧 |

**核心诊断**：Direct Loss 的 loss landscape 存在尖锐悬崖。Sinkhorn 软分配 $P_{ij}=e^{-C_{ij}/\varepsilon} / \sum_k e^{-C_{ik}/\varepsilon}$ 对成本矩阵一阶敏感，特征初始化的微小差异被指数放大。4 个 seed 中仅 1 个找到好盆地（$\sigma_{ARI}\approx 0.35$）。

### 理论框架：ECO (Elliptic Curve Object)

物体 = 椭圆曲线上的概率流形 $O = (E, \mu_E, v)$，身份由 j-不变量编码：

$$j(E) = 1728 \cdot \frac{4a^3}{4a^3+27b^2}$$

- **二阶稳定**：$\delta j = O(\|\delta\|^2)$ vs 特征空间的一阶敏感 $\delta C = O(\|\delta\|)$
- **紧致流形**：$E(\mathbb{R}) \cong S^1$，Boids 不崩溃
- **分岔检测**：$\Delta \to 0$ 意味着物体分裂/融合

ECO 将 landscape 从尖锐悬崖（$\sigma_{ARI}\approx 0.35$）变为平坦山谷（预期 $\sigma_{ARI}\approx 0.08$）。

详见 [`docs/phase6a_eco_theory.md`](docs/phase6a_eco_theory.md)。

### 数学补充路线图

| 优先级 | 数学领域 | 状态 | 目的 |
|--------|---------|------|------|
| P0-1 | 微分几何：EC 上的 exp/log 映射 | ✅ 完成 | Boids 能在 E 上跑 |
| P0-2 | 动力系统：Lyapunov 稳定性 | 📝 待完成 | 证明不崩溃 |
| P1-1 | 流形 Sinkhorn | ✅ 完成 | ECO 版 Direct Loss |
| P1-2 | 分岔理论 | 📝 待完成 | 形式化非稳态检测 |
| P2 | 代数几何/模空间 | 📝 待完成 | 理论深度 |

## 约束

本项目遵循严格的零外部先验原则：
- ❌ 禁止使用 COLMAP、SAM、CLIP、高斯泼溅
- ❌ 禁止使用预训练语义/分割模型
- ✅ 仅使用多视图图像 + 可微渲染 + 几何正则化

## 分支说明

| 分支 | 用途 |
|------|------|
| `main` | 稳定发布版本 |
| `feat/clustering-breakthrough` | 无监督聚类突破（Direct Loss + ECO 理论框架） |

## 许可

MIT
