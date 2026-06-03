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

## 数学框架

### 1. 黎曼度量场

定义：每点 $x$ 赋予一个正定对称矩阵 $g(x) \in \mathrm{Sym}^+(d)$，定义测地距离 $d_g(P,Q)$。

Cholesky 参数化保证正定性：$g = LL^\top + \epsilon I$。

- **2D**：3 自由度/点，$l_{11}, l_{21}, l_{22}$
- **3D**：6 自由度/点，$l_{11}, l_{21}, l_{22}, l_{31}, l_{32}, l_{33}$

度量场与**占位耦合损失**协同学边界：
- 物体内部 $\mathrm{tr}(g) < 1$，背景 $\mathrm{tr}(g) > 9$
- 度量场在物体边界处锐利跳变 → 自然定义聚类边界

### 2. 直接测地聚类损失 (Direct Cluster)

用 Sinkhorn 可微软分配替代 InfoNCE，消除黎曼空间逻辑循环：

$$\mathcal{L}_{\text{direct}} = \sum_k \frac{P[:,k]^\top D_g^2\, P[:,k]}{(\text{cluster\_mass})^2}$$

- **Sinkhorn 软分配** $P$：基于特征-原型余弦相似度，可微，梯度流连续
- 度量场 $g$ 直接最小化簇内测地距离，不再通过特征间接优化
- 训练稳定性：InfoNCE 的"甜区宽度极窄"（$w_{\text{vol}} = 0.1 \pm 0.025$）已被 Direct Cluster 替代，后者在 ε=0.05 下 ARI=0.93

### 3. Murmuration 动力学 [HISTORICAL]

> 2026-06-03: Murmuration 代码（murmuration.py, elliptic_curve.py）已随 ECO 路径移除。Lyapunov 稳定性分析（murmuration_dynamics.md）作为理论成果保留，数值验证已通过（S¹ 离散 Murmuration Lyapunov 单调递减）。

> 📐 **详细数学文档**
> - [docs/framework_audit.md](docs/framework_audit.md)：**框架系统审计** — 34 项命题/实验/假设的数学严格性评估（17.6% 已证明，8 项阻塞级缺陷）
> - [docs/gradient_flow_analysis.md](docs/gradient_flow_analysis.md)：Direct Cluster vs InfoNCE 梯度流分析（甜区宽度理论、Sinkhorn 最优 ε 推导、Phase 7 landscape 双稳态解释）
> - [docs/convergence_rate_analysis.md](docs/convergence_rate_analysis.md)：收敛速率严格分析（Lipschitz 常数推导、O(1/t) 次线性收敛证明、PL 条件线性收敛、ε-条件数关系、vs InfoNCE 收敛对比）
> - [docs/remaining_proofs.md](docs/remaining_proofs.md)：**三大遗留问题完整证明** — PL 条件严格证明（定理 1,2）、K > 2 簇泛化（命题 3,4）、ECO 协同收敛（命题 5,6,7）
> - [docs/phase6a_eco_theory.md](docs/phase6a_eco_theory.md)：ECO 完整形式化（椭圆曲线群运算、j-不变量稳定性定理证明、Sinkhorn 兼容性定理、传感函数 φ、分岔检测、模空间优先级矩阵）
> - [docs/math_analysis.md](docs/math_analysis.md)：3D 黎曼度量场的数学可行性分析（Cholesky 推广、测地截断 smoothstep 公式、InfoNCE 超参数学解释、占位耦合与位置正则的权重推导、3D Murmuration 接口）
> - [docs/murmuration_dynamics.md](docs/murmuration_dynamics.md)：**Murmuration 动力学严格分析** — Lyapunov 函数存在性证明（V=T+U, dV/dt ≤ 0 当 η>β）、Hartman-Grobman 局部稳定性（Fourier 谱分析）、吸引域估计（能量水平集方法）、Cucker-Smale 联系、训练 Phase 对应
> - [docs/theoretical_extensions.md](docs/theoretical_extensions.md)：**四大理论扩展** — Phase 2 最优切换控制（Pontryagin 视角 + 间隙条件）、K 自适应选择（Sinkhorn 有效秩 + Silhouette 扫描 + 特征谱间隙）、泛化误差界（PAC-Bayes + Rademacher 复杂度估计）、超参数敏感性（Hessian 谱分析 + ε 主导的谱分离器 + 阻尼 Lyapunov 阈值 + 随机矩阵视角）

---

## 系统架构

```
src/
├── geometry/           # 黎曼度量场（2D/3D）
│   ├── cholesky_param.py
│   └── metric_field.py
├── atoms/              # 感知原子定义
├── rendering/          # 体积渲染器
├── losses/             # 损失函数
│   └── direct_cluster.py    # Direct Cluster Loss (ε=0.05, ARI=0.93)
├── data/               # 合成数据生成
├── visualization/      # 可视化
└── training/           # 训练循环
```

## 历史成果

> 完整实验历史（Phase 1 → Phase 7）请见 [docs/history.md](docs/history.md)。
> 以下仅列出最近三次关键结果。

### Phase 6b: ECO 代码实现（2026-05-18）[HISTORICAL]

> ⚠️ 2026-06-03: ECO 路径已放弃（ARI=0.30 vs DirectCluster=0.93），所有相关代码已移除。保留作为理论探索记录。

| 模块 | 功能 |
|------|------|
| `elliptic_curve.py` (已删除) | 群运算（加法/标量乘法）、log/exp 映射、测地距离数值积分、j-不变量、分岔检测 |
| `murmuration.py` (已删除) | Boids 在椭圆曲线上的动力学（凝聚/对齐/分离力在 1D 切空间中），PyTorch 批处理版本 |
| `eco_cluster.py` (已删除) | ECO 版 Direct Loss：感知函数 φ: features→(a,b)，曲线残差成本矩阵，j-不变量身份一致性正则项 |

### Phase 7: Landscape 系统性扫描（2026-05-19）

**目标**：量化 DirectCluster 对随机初始化的敏感度，验证 Phase 6a "尖锐悬崖"理论。

在 Tesla T4 云端 GPU 上运行 8 个不同 seed（100-107），每个 200 epochs，111 atoms，64×64 分辨率。

**核心发现**：
- 存在**完美解**（seed 107: ARI=1.0, NMI=1.0, 簇平衡 B=1.00），证明 DirectCluster 在有利初始化下可以完全解决 2D 双物体场景
- 但方差极大（σ=0.39），约 50% seed 未能有效聚类——确认了尖锐悬崖现象
- 完美聚类仅需 50/111 原子落在物体内，说明聚类质量 ≠ 覆盖数量

### ⚠️ Phase 6c: ECO 统一公式 + 精细调参 [HISTORICAL]（2026-05-20，2026-06-03 废止）

> **ECO 路径已于 2026-06-03 完全废止**：j 是模函数映射到 ℂ∪{∞}，非欧几里得空间，j 空间聚类本质病态。实验确认 ECO 严重恶化聚类（ARI 0.30），而 Direct Cluster ε=0.05 + CosineLR 达到 ARI 0.93。ECOClusterLoss、elliptic_curve、murmuration 代码已全部删除。保留下方调参记录仅作为历史参考。

**核心统一公式**：`feature → 感知函数 φ → (a,b) → j-不变量 → Sinkhorn cost C_ik = |j_i - j_k|`

#### 调参历程（历史记录）

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


## 当前进度

### ✅ 已验证的功能

| 功能 | 状态 | 说明 |
|---|---|---|
| 度量场 Cholesky 参数化 | ✅ 完成 | 2D 网格/3D 体素，保证正定性 |
| 感知原子 smoothstep 截断 | ✅ 完成 | C² 连续，马氏距离空间支持 |
| 体积渲染器（向量化） | ✅ 完成 | 2D/3D 渲染管线，17x 加速 |
| 合成数据生成 | ✅ 完成 | 2D 多形状 + 3D 多球体场景 |
| 重建 + 度量平滑 + 占位耦合损失 | ✅ 完成 | L1 < 0.08，trace 分离明显 |
| 凝聚损失（InfoNCE） | ✅ 完成 | KMeans 特征初始化 + 对比学习 |
| 特征扩散（测地亲和矩阵） | ✅ 完成 | 零额外参数，全可微 |
| 直接测地聚类损失（Sinkhorn + L_direct） | ✅ **突破** | ARI 0.440→**0.755**→**0.931**（ε=0.05 + CosineLR） |
| ~~椭圆曲线几何 + Murmuration 动力学 (Phase 6b)~~ | ⚠️ 已废弃 | ECO 路径已删除（代码移除 2026-06-03） |
| ~~ECO 聚类损失 (Phase 6c)~~ | ⚠️ 已废弃 | ECO 路径已删除（代码移除 2026-06-03） |
| 超参网格搜索框架 | ✅ 完成 | Phase 4/5：w_direct × sinkhorn_eps |
| 数值稳定性优化 | ✅ 完成 | overflow-safe Newton + group-law exp_map |
| 动态原子管理 | ✅ 完成 | 剪枝 + 播种 + 位置正则化 |
| 3D 扩展 | ✅ 完成 | 3D 度量场 + 3D 原子 + 3D 渲染 + 训练脚本 |
| BF16 混合精度 + MKL CPU 优化 | ✅ 完成 | CUDA 加速，显存优化 |

### 🚧 进行中

| 工作 | 进度 | 说明 |
|---|---|---|
| Landscape 系统性扫描 | ✅ Phase 7 完成 | 8 seed, σ=0.39, 确认尖锐悬崖 |
| ECO 精细调参 (Phase 6c) | ✅ 最优配置确认 | 均值 ARI 0.637, 7/8 seed ≥0.5 |
| 高分辨率 (128×128) 验证 | 🔄 规划中 | 验证最佳超参泛化性 |
| 多物体场景 (3-4 个) 聚类评估 | 🔄 规划中 | 验证聚类上限 |
| 3D 场景聚类验证 | 🔄 规划中 | 3D 度量场 + 原子 + 渲染 |
| 论文撰写 | 📝 初稿进行中 | 私有仓库 |

### 🏆 ARI 进化历程

| 阶段 | 方法 | ARI | 日期 |
|---|---|---|---|
| v1 基线 | InfoNCE + KMeans init | 0.175 | 2026-05-13 |
| v2 调参 | 优化 tau/thresh/var_weight | ~0.2 | 2026-05-15 |
| v3 特征扩散 | 测地亲和矩阵 + 扩散平滑 | 0.440 | 2026-05-16 |
| **v4 突破** | **直接测地聚类损失** | **0.755** | **2026-05-18** |
| Phase 6c | ECO + eps=0.05 | 0.637 ± 0.308 (8 seed) | 2026-05-20 |
| Phase 7 | Landscape 扫描最佳 | 1.000 (seed 107) | 2026-05-19 |

---

## 约束

本项目遵循严格的零外部先验原则：
- 禁止使用 COLMAP、SAM、CLIP、高斯泼溅
- 禁止使用预训练语义/分割模型
- 仅使用多视图图像 + 可微渲染 + 几何正则化

## 快速开始

```bash
# 安装依赖
pip install torch numpy scipy matplotlib opencv-python

# 2D 快速验证（64×64, BF16, ~5min on CUDA）
python train_2d.py --resolution 64 --epochs 600

# 2D ECO 聚类训练（推荐配置）
python train_2d.py --bf16 --use-eco --w-eco 0.5 --eco-id-weight 0.1 --sinkhorn-eps 0.05

# 3D 训练
python train_3d.py

# 超参网格搜索
python tasks/sweep_hyperparams.py
```

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

## 环境要求

- **Python** 3.10+
- **PyTorch** 2.1+（推荐 CUDA 12+）
- **CUDA** 可选（支持 BF16 的 GPU 最佳）
- **CPU** MKL 加速（Intel CPU）

## 分支说明

| 分支 | 用途 |
|------|------|
| `master` | 主线稳定版本 |
| `feat/clustering-breakthrough` | 无监督聚类突破（Direct Loss + ECO 理论框架） |

## 许可

MIT
