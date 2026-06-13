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

> 2026-06-13 重构（分支 `feat/clustering-breakthrough`）。
> 本节整合六公理体系、核心定理链与外部审计矛盾。
> 导航：[docs/theory_index.md](docs/theory_index.md) 给出完整阅读路径、可信度评级（R/H/S）与依赖关系。

### 0. 最小公理体系（全部 R 级 · 来源 theory_fracture_fixes.md §6.1）

框架的最终数学基础由六条公理构成，全部为 R 级（严格证明，假设明确）：

| 公理 | 内容 | 级别 | 来源 |
|------|------|------|------|
| **A1** | 状态传播收缩性：$\|\mathcal{T}_W(S) - \mathcal{T}_W(S')\| \leq (1 - \alpha\lambda_2)\|S - S'\|$ | **R** | theory_selforg 定理 1 |
| **A2** | 掩码预测强制物体推理：$\mathcal{L}_{\text{predict}}$ 在度量场不编码物体边界时有非零下界 | **R** ⚠ | theory_selforg_2 命题 13（**受 EXT-4 质疑**） |
| **A3** | 自组织力符号正确性：$\nabla_g \mathcal{L}_{\text{selforg}}$ 推同簇原子靠近、跨簇原子远离 | **R** | theory_selforg §2.2 |
| **A4** | 均匀解不稳定性：$H_{ss}$ 在物体区分方向上有负特征值（鞍点） | **R** ⚠ | 定理 22（**自承证明含符号跳变**，详见原文） |
| **A5** | 梯度局部性 + 度量场自修正性：$\partial\mathcal{L}_{\text{predict}}/\partial g(x)$ 仅在原子-像素中点附近非零；偏离最优的 $g$ 被梯度推回 | **R** | 引理 2 + 定理 17 |
| **A6** | Bootstrap 收敛：重建驱动度量场在颜色边缘指数收敛到非零稳态 $\Delta g^* = \eta_{\text{recon}} G_{\text{edge}} / (\eta_s \lambda_2)$ | **R** | 命题 23 + 定理 20 |

**整体严格率（自我审计）**：六公理 + 九条新 R 命题（引理 1–2、定理 17–22、命题 23）。理论陈述统计为 **22R(36%) · 27H(44%) · 12S(20%)**，废除 8 条信息瓶颈伪命题后修订为 14/53 = 26%。
**外部审计重估**：[theory_defect_report_external_audit.md](docs/theory_defect_report_external_audit.md) 估计真实 R 率约 **18%**（口径差异详见 §3 主要矛盾）。

### 1. 核心定理链（公理 → 聚类涌现）

从公理到聚类涌现的逻辑链（来源 theory_index.md §2.2）：

```
A4 (均匀解不稳定) → SGD 必然离开均匀解 → 状态开始按物体分化
        ↓
A6 (Bootstrap) → 重建驱动在颜色边缘产生初始度量场结构
        ↓
A2 (掩码预测) → 预测误差迫使度量场不跨越物体边界
        ↓
A1 (状态收缩) → 物体内部状态通过消息传递坍缩 → 物体间通过度量场隔离
        ↓
A3 + A5 (自组织 + 自修正) → 度量场在边界处锐化 → 状态坍缩与边界形成正反馈
        ↓
聚类涌现（K 个簇）
```

**主线定理编号速查**：

1. **状态动力学**：[theory_selforg.md](docs/theory_selforg.md) 定理 1–2 → 推论 1.1–1.2
2. **涌现条件**：theory_selforg 定理 5（条件 C1+C2+C3）
3. **Lyapunov 稳定性**：theory_selforg 命题 6–8
4. **联合 Hessian / PL 条件**：[theory_selforg_2.md](docs/theory_selforg_2.md) 定理 8–9
5. **解码器谱**：[theory_selforg_3.md](docs/theory_selforg_3.md) 定理 13 + 引理 16.1
6. **残差解码器谱优化**：[theory_selforg_4.md](docs/theory_selforg_4.md) 定理 17–18
7. **有限 N 分岔**：theory_selforg_4 定理 20
8. **跨视角一致性**：theory_selforg_4 定理 23（β_c 降低 30%）

### 2. 黎曼度量场参数化

定义：每点 $x$ 赋予一个正定对称矩阵 $g(x) \in \mathrm{Sym}^+(d)$。

**参数化方式（可切换）**：
- **Cholesky**：$g = LL^\top + \epsilon I$（快速；⚠ **EXT-1**：欧几里得 SGD ≠ SPD 流形测地下降）
- **矩阵指数**：$g = \exp(H)$，$H$ 对称（严格 SPD，切空间优化 — 推荐替代）

2D 为 3 自由度/点，3D 为 6 自由度/点。

**距离度量**：中点马氏距离（midpoint Mahalanobis chord distance）作为测地距离的近似：
$$d^2_{ij} \approx (\mu_i - \mu_j)^\top g\!\left(\frac{\mu_i + \mu_j}{2}\right) (\mu_i - \mu_j)$$

> ⚠️ **EXT-2**：中点近似在强各向异性区误差可达 100%+，无理论界。代码 `src/losses/direct_cluster.py::compute_true_geodesic_sq_1d` 仅 1D 验证。

**占位耦合**：物体内部 $\mathrm{tr}(g) \u003c 1$，背景 $\mathrm{tr}(g) \u003e 9$。度量场在边界跳变 → 提供聚类边界的几何信号。

### 3. 直接测地聚类损失 (Direct Cluster) [历史突破保留]

用 Sinkhorn 可微软分配替代 InfoNCE：
$$\mathcal{L}_{\text{direct}} = \sum_k \frac{P[:,k]^\top D_g^2\, P[:, k]}{(\text{cluster\_mass})^2}$$

- Sinkhorn 软分配 $P$ 可微，梯度流连续
- 度量场 $g$ 直接最小化簇内测地距离
- 数值结果：DirectCluster 在 ε=0.05 下 ARI 0.440 → **0.755** → **0.931**

⚠ **EXT-3**：Sinkhorn ε=0.05 处于收敛不稳定区；代码 `n_iters=50` 可能不足。P0：自适应 ε 或 ≥200 次迭代。

> **路线状态（2026-06-04 弃用）**：DirectCluster 6 条路线全失败（σ=0.39）；保留作为历史突破。当前主路线为自组织原子架构（公理 A1–A6）。详见 [postmortem_direct_cluster.md](docs/postmortem_direct_cluster.md)。

### 4. 外部审计：未解决的 8 项主要矛盾（EXT-1–EXT-8）

| # | 缺陷 | 级别 | 状态 / 优先级 |
|---|---|---|---|
| **EXT-1** | Cholesky + 欧氏 SGD ≠ SPD 流形优化 | 🔴 blocking | P1：矩阵指数 vs 自然梯度对比 |
| **EXT-2** | 中点度量近似无测地距离误差界 | 🔴 blocking | P0：在解析度量场上量化 |
| **EXT-3** | Sinkhorn ε=0.05 在不稳定区，迭代次数不足 | 🔴 blocking | P0：自适应 ε 或 ≥200 次迭代 |
| EXT-4 | 掩码预测不必然强制物体推理 | 🟡 major | P1：同色多物体 ARI 验证 |
| EXT-5 | Lojasiewicz θ 可能接近 1/2，收敛极慢 | 🟡 major | P2：建议定理 18–19 降级为 H |
| EXT-6 | "零外部先验"声称不实 | 🟡 major | P0：README 删除该声称 |
| EXT-7 | 3D 测地邻接稀疏性被严重低估 | 🟡 major | P1：3D 可行性验证 |
| EXT-8 | 自组织力热力学类比缺乏严格性 | 🟢 minor | P2 |

### 5. 历史路径（已弃用）

- **Murmuration 动力学 [HISTORICAL]**：代码 2026-06-03 随 ECO 路径移除；Lyapunov 分析保留（[murmuration_dynamics.md](docs/murmuration_dynamics.md)）。
- **ECO / 椭圆曲线 / j-不变量 [DEPRECATED]**：ARI 0.30 vs DirectCluster 0.93；j-空间聚类本质病态。
- **InfoNCE + 特征扩散 [HISTORICAL]**：被 Direct Cluster 替代；测地高斯核 100/100 非 PSD（[numerical_verification.md](docs/numerical_verification.md)）。

> 📐 **详细数学文档**
>
> 建议先读 [docs/theory_index.md](docs/theory_index.md) 获取完整导航图、阅读路径与可信度评级，再按路径深入：
>
> - 路径 A（30 分钟）：[README §核心思想](#核心思想) → theory_index → [atom_selforg_redesign.md](docs/atom_selforg_redesign.md)
> - 路径 B（系统学习）：theory_selforg → v2 → v3 → v4 → [theory_fracture_fixes.md](docs/theory_fracture_fixes.md)
> - 路径 C（审计与可信度）：[theory_audit_and_roadmap.md](docs/theory_audit_and_roadmap.md) → theory_defect_report_external_audit → theory_fracture_fixes → [framework_audit.md](docs/framework_audit.md)
> - 路径 D（实现）：atom_selforg_redesign → [neuroscience_informed_roadmap.md](docs/neuroscience_informed_roadmap.md) → [theoretical_extensions.md](docs/theoretical_extensions.md) → numerical_verification
> - 路径 E（历史与失败）：[history.md](docs/history.md) → postmortem_direct_cluster → phase6a_eco_theory
>
> 完整文档列表与 R/H/S 评级见 [theory_index.md §1 文档地图](docs/theory_index.md#1-文档地图)。

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
