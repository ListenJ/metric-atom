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

定义：每点 $x$ 赋予一个正定对称矩阵 $g(x) \in \mathrm{Sym}^+(d)$。

**参数化方式（可切换）**：
- **Cholesky**：$g = LL^\top + \epsilon I$（快速，但欧几里得SGD ≠ SPD流形测地下降）
- **矩阵指数**：$g = \exp(H)$，$H$ 对称（严格SPD，切空间优化）

- **2D**：3 自由度/点
- **3D**：6 自由度/点

**距离度量**：使用**中点马氏距离**（midpoint Mahalanobis chord distance）作为测地距离的近似：
$$d^2_{ij} \approx (\mu_i - \mu_j)^\top g\left(\frac{\mu_i + \mu_j}{2}\right) (\mu_i - \mu_j)$$

> ⚠️ 这不是严格的测地距离。在度量变化剧烈区域（如边界），误差可能显著。详见 `src/losses/direct_cluster.py` 中的 `compute_true_geodesic_sq_1d` 验证函数。

度量场与**占位耦合损失**协同学边界：
- 物体内部 $\mathrm{tr}(g) < 1$，背景 $\mathrm{tr}(g) > 9$
- 度量场在物体边界处跳变 → 提供聚类边界的几何信号

### 2. 直接测地聚类损失 (Direct Cluster)

用 Sinkhorn 可微软分配替代 InfoNCE，消除黎曼空间逻辑循环：

$$\mathcal{L}_{\text{direct}} = \sum_k \frac{P[:,k]^\top D_g^2\, P[:,k]}{(\text{cluster\_mass})^2}$$

- **Sinkhorn 软分配** $P$：基于特征-原型余弦相似度，可微，梯度流连续
- 度量场 $g$ 直接最小化簇内测地距离，不再通过特征间接优化
- 训练稳定性：InfoNCE 的"甜区宽度极窄"（$w_{\text{vol}} = 0.1 \pm 0.025$）已被 Direct Cluster 替代，后者在 ε=0.05 下 ARI=0.93

### 3. Murmuration 动力学 [HISTORICAL]

> 2026-06-03: Murmuration 代码（murmuration.py, elliptic_curve.py）已随 ECO 路径移除。Lyapunov 稳定性分析（murmuration_dynamics.md）作为理论成果保留，数值验证已通过（S¹ 离散 Murmuration Lyapunov 单调递减）。

> 📐 **详细数学文档**
>
> 建议先读 [docs/theory_index.md](docs/theory_index.md) 获取完整导航图、阅读路径与可信度评级，再按主题深入以下文档。
>
> - [docs/framework_audit.md](docs/framework_audit.md)：**框架系统审计** — 34 项命题/实验/假设的数学严格性评估（17.6% 已证明，8 项阻塞级缺陷）
> - [docs/gradient_flow_analysis.md](docs/gradient_flow_analysis.md)：Direct Cluster vs InfoNCE 梯度流分析（甜区宽度理论、Sinkhorn 最优 ε 推导、Phase 7 landscape 双稳态解释）
> - [docs/convergence_rate_analysis.md](docs/convergence_rate_analysis.md)：收敛速率严格分析（Lipschitz 常数推导、O(1/t) 次线性收敛证明、PL 条件线性收敛、ε-条件数关系、vs InfoNCE 收敛对比）
> - [docs/remaining_proofs.md](docs/remaining_proofs.md)：**三大遗留问题完整证明** — PL 条件严格证明（定理 1,2）、K > 2 簇泛化（命题 3,4）、ECO 协同收敛（命题 5,6,7）
> - [docs/phase6a_eco_theory.md](docs/phase6a_eco_theory.md)：ECO 完整形式化（椭圆曲线群运算、j-不变量稳定性定理证明、Sinkhorn 兼容性定理、传感函数 φ、分岔检测、模空间优先级矩阵）
> - [docs/math_analysis.md](docs/math_analysis.md)：3D 黎曼度量场的数学可行性分析（Cholesky 推广、测地截断 smoothstep 公式、InfoNCE 超参数学解释、占位耦合与位置正则的权重推导、3D Murmuration 接口）
> - [docs/murmuration_dynamics.md](docs/murmuration_dynamics.md)：**Murmuration 动力学严格分析** — Lyapunov 函数存在性证明（V=T+U, dV/dt ≤ 0 当 η>β）、Hartman-Grobman 局部稳定性（Fourier 谱分析）、吸引域估计（能量水平集方法）、Cucker-Smale 联系、训练 Phase 对应
> - [docs/theoretical_extensions.md](docs/theoretical_extensions.md)：**四大理论扩展** — Phase 2 最优切换控制（Pontryagin 视角 + 间隙条件）、K 自适应选择（Sinkhorn 有效秩 + Silhouette 扫描 + 特征谱间隙）、泛化误差界（PAC-Bayes + Rademacher 复杂度估计）、超参数敏感性（Hessian 谱分析 + ε 主导的谱分离器 + 阻尼 Lyapunov 阈值 + 随机矩阵视角）
> - [docs/numerical_verification.md](docs/numerical_verification.md)：**数值验证报告** — Murmuration Lyapunov 离散验证（8/8 种子收敛，终态 V/V₀=0.047）、测地高斯核 PSD 验证（100/100 非 PSD，确认为缺陷）
> - [docs/blocker_verification.md](docs/blocker_verification.md)：**阻塞级缺陷严格数学验证** — j-不变量梯度精确计算（|∇j|≈31000 在初始化点）、z-score 归一化矛盾（batch-dependent 身份）、Δ≠0 的理论不可保证性
> - [docs/postmortem_direct_cluster.md](docs/postmortem_direct_cluster.md)：**DirectCluster 事后分析** — 6 条路线完整记录：三个致命假设被证伪（特征≠物体感知、测地≠聚类 teacher、种子敏感性 ≠ 初始化问题）
> - [docs/atom_selforg_redesign.md](docs/atom_selforg_redesign.md)：**自组织原子系统重新设计** — 掩码多视图预测新任务、状态动力学（图注意力消息传递）、自组织力、涌现聚类
> - [docs/theory_selforg.md](docs/theory_selforg.md)：**自组织原子理论基础** — 状态动力学收敛（收缩映射，几何速率）、涌现聚类 Landau 理论（序参量 + 自由能）、Lyapunov 稳定性（总损失 = Lyapunov 函数，均匀解不稳定）、信息论解释（信息瓶颈 → 聚类 = 最优压缩）、泛化误差界、7 条可检验数值预测
> - [docs/theory_selforg_2.md](docs/theory_selforg_2.md)：**自组织理论深化 II** — 5 定理 + 4 命题 + 5 推论：掩码预测形式分析（命题 12,13）、联合 Hessian 谱分析与 PL 条件（定理 8,9）、状态动力学收敛加速（定理 10）、信息瓶颈 β_c 量化（定理 11, 命题 14,15）、自适应温度 τ 调度（命题 16）、6 条可检验预测
> - [docs/theory_selforg_3.md](docs/theory_selforg_3.md)：**自组织理论深化 III** — 5 定理 + 7 命题 + 3 推论 + 1 引理：解码器 Jacobian 谱下界（定理 13, 引理 16.1）、多物体 β_c 定量预测（定理 14, 命题 16）、真实图像收缩性（定理 15, 命题 17）、自适应 τ PI 控制（命题 18）、多时间尺度奇异摄动（命题 19）、分岔理论深化（命题 20-22）、测地-状态对偶性（定理 16）、8 条可检验预测
> - [docs/theory_selforg_4.md](docs/theory_selforg_4.md)：**自组织理论深化 IV** — 8 定理 + 10 命题 + 4 推论：残差解码器谱优化（定理 17,18, 命题 23）、自适应时间尺度分离（定理 19, 命题 24,25）、有限 N 分岔效应（定理 20, 命题 26, 推论 22.1）、双曲状态流形（定理 21,22, 命题 27）、跨视角一致性（定理 23, 命题 28,29）、非刚性形变（定理 24, 命题 30, 推论 23.1）、9 条可检验预测
> - [docs/theory_audit_and_roadmap.md](docs/theory_audit_and_roadmap.md)：**理论审计与发展路线** — 61 条理论陈述的 R/H/S 严格性分级、4 条不可再简化数学公理、3 个关键断裂点、P0-P3 优化路线图
> - [docs/theory_fracture_fixes.md](docs/theory_fracture_fixes.md)：**断裂点修复与理论重整** — 废除 8 条 IB 伪命题、新增 9 条 R 级严格命题（引理 1-2、定理 17-22、命题 23）、六公理完备体系（全部 R 级）、修订后 R22(36%)·H27(44%)·S12(20%)

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
| 自组织原子理论基础 | ✅ 2026-06-06 | theory_selforg.md（3 定理 + 8 命题 + 2 推论） |
| 自组织理论深化 II | ✅ 2026-06-06 | theory_selforg_2.md（5 定理 + 4 命题 + 5 推论） |
| 自组织理论深化 III | ✅ 2026-06-07 | theory_selforg_3.md（5 定理 + 7 命题 + 3 推论 + 1 引理） |
| 自组织理论深化 IV | ✅ 2026-06-08 | theory_selforg_4.md（8 定理 + 10 命题 + 4 推论） |
| 理论审计与发展路线 | ✅ 2026-06-08 | theory_audit_and_roadmap.md（61 陈述严格性分级 + 4 公理 + 优化路线图） |
| 论文撰写 | 📝 初稿进行中 | 私有仓库 |
| 低偏置训练选项 | ✅ 2026-06-13 | reproject-oracle 默认关闭、homeostatic 参数显式化、度量平坦/视角预测可选 |

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

本项目的**设计目标**是最小化外部先验，但以下归纳偏置是不可避免的：
- 场景由局部、紧支撑的球体（原子）组成
- 度量场通过 Cholesky 分解或矩阵指数参数化为低维正定场
- 状态传播假设局部测地连通性
- 使用体积渲染方程作为光传输模型

外部模型限制（设计选择）：
- 不使用 COLMAP、SAM、CLIP、高斯泼溅
- 不使用预训练语义/分割模型
- 仅使用多视图图像 + 可微渲染 + 几何正则化

**可选先验（默认关闭/权重为 0，可显式开启）**：
- `--w-flat`：度量平坦先验（降低各向异性）
- `--w-homeo`：稳态可塑性（约束存在概率分布）
- `--w-pred-view`：下一视角预测一致性
- `--reproject-oracle`：使用真实 occupancy 引导原子位置（默认关闭）

这些开关的目的不是引入隐藏的 oracle，而是让实验者能显式地研究每个偏置的影响。默认配置下不依赖任何 ground-truth 分割。

## 快速开始

```bash
# 安装依赖
pip install torch numpy scipy matplotlib opencv-python

# 2D 快速验证（64×64, BF16, ~5min on CUDA）
python train_2d.py --resolution 64 --epochs 600

# 2D 完整训练（掩码预测 + 自组织）
python train_2d.py --resolution 64 --epochs 600 --bf16 \
    --w-predict 1.0 --w-selforg 1.0

# 3D 训练
python train_3d.py

# 超参网格搜索
python tasks/sweep_hyperparams.py
```

### 可选正则化/约束开关

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--w-homeo` | 0.1 | 稳态可塑性：约束原子存在概率的均值/方差与密度 |
| `--homeo-mean` | 0.5 | 存在概率目标均值 |
| `--homeo-std` | 0.25 | 存在概率目标标准差 |
| `--homeo-log-density` | 0.0 | 期望 log 原子密度 |
| `--homeo-max-log-ratio` | 1.0 | 密度正则上限 |
| `--w-flat` | 0.0 | 度量平坦先验：惩罚各向异性与 trace 空间不平滑 |
| `--w-pred-view` | 0.0 | 下一视角预测一致性（相机无关） |
| `--reproject-oracle` | False | 使用真实 occupancy 重投影原子（默认关闭，避免 oracle 偏置） |

> 以上均为**可选**正则化项，默认关闭或权重为 0。开启前请参考“约束”一节。

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

## 理论文档链

```
README.md (§数学框架)
├── math_analysis.md               [3D 可行性 + 超参数推导]
├── phase6a_eco_theory.md          [ECO 理论 · 已废止]
├── gradient_flow_analysis.md      [Direct Cluster vs InfoNCE 梯度流]
├── framework_audit.md             [34 项缺陷审计]
├── blocker_verification.md        [3 阻塞级严格验证 · ECO 已废止]
├── convergence_rate_analysis.md   [Lipschitz + O(1/t) + PL + ε 分析]
├── remaining_proofs.md            [PL 严格证明 + K>2 + ECO 协同]
├── murmuration_dynamics.md        [Lyapunov + HG + 吸引域 · ECO 相关]
├── theoretical_extensions.md      [Phase 2 控制 + 自适应 K + 泛化 + 敏感性]
├── numerical_verification.md      [数值验证：Lyapunov ✓、扩散核 ✗]
├── postmortem_direct_cluster.md   [事后分析：6 条路线记录 + 5 条教训]
├── atom_selforg_redesign.md       [新架构设计：掩码预测 + 自组织原子]
├── theory_selforg.md              [新架构基础：3 定理 + 8 命题 + 2 推论]
├── theory_selforg_2.md            [新架构深化：5 定理 + 4 命题 + 5 推论]
├── theory_selforg_3.md            [新架构深化 III：5 定理 + 7 命题 + 3 推论 + 1 引理]
└── theory_selforg_4.md            [新架构深化 IV：8 定理 + 10 命题 + 4 推论]
    ├── Part 1-6: (见上)
├── theory_audit_and_roadmap.md    [数学审计：R21%·H57%·S21% + 4 公理]
└── theory_fracture_fixes.md       [断裂点修复：废除8条IB + 新增9条R + 6公理(全R)]
```

**总计**：18 篇理论/设计文档。其中 ECO 相关 3 篇已废止，DirectCluster 相关 9 篇部分适用。自组织原子 7 篇（1 设计 + 4 理论 + 1 审计 + 1 修复）。修复后严格性：22R(36%) + 27H(44%) + 12S(20%)。30 条可检验数值预测。

## 分支说明

| 分支 | 用途 |
|------|------|
| `master` | 主线稳定版本 |
| `feat/clustering-breakthrough` | 无监督聚类突破（Direct Loss + ECO 理论框架） |
| `feat/selforg` | 自组织原子架构（掩码预测 + 涌现聚类） |

## 许可

MIT
