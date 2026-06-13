# MetricAtom 理论与代码差距分析

> 2026-06-13 | 分支 `feat/clustering-breakthrough`
> 范围：`docs/theory_fracture_fixes.md`、`docs/atom_selforg_redesign.md`、`docs/neuroscience_informed_roadmap.md`、`docs/theory_selforg_4.md`
> 对照代码：`src/losses/*.py`、`src/atoms/*.py`、`src/geometry/*.py`、`train_2d.py`

---

## 0. 阅读方法

每个理论组件按以下格式记录：

| 字段 | 含义 |
|------|------|
| **来源** | 理论文档节号 + 定理/命题编号（如有） |
| **状态** | `✅ 实现` / `🟡 部分实现` / `❌ 缺失` |
| **代码位置** | 文件:行号 + 函数名（如实现） |
| **偏离说明** | 与理论描述的具体差异（默认配置下尤其重要） |

状态判定严格基于代码实际内容；只读取源码，不依赖 README 或 docstring 的声明。

---

## 1. 组件差距表

### 1.1 稳态可塑性（Homeostatic Plasticity）

| 字段 | 内容 |
|------|------|
| **来源** | `neuroscience_informed_roadmap.md` Phase 0；`atom_selforg_redesign.md` §5 |
| **状态** | ✅ 实现 |
| **代码位置** | `src/losses/homeostatic.py:18-44`（`existence_homeostasis`）、`src/losses/homeostatic.py:47-77`（`contribution_homeostasis`）、`src/losses/homeostatic.py:80-107`（`homeostatic_loss`）；调用点 `train_2d.py:674-683` |
| **偏离说明** | 实现完整覆盖了路线图 Phase 0 的两个最小实现版本（existence + contribution）。目标均值/标准差/对数密度/最大对数比均显式为 CLI 参数（`--homeo-mean/--homeo-std/--homeo-log-density/--homeo-max-log-ratio`，`train_2d.py:866-873`）。默认 `w_homeo=0.1` 与路线图推荐范围（0.01~0.05）上限略高，可能压制信号；建议在实验时降低至 0.03 以更接近神经科学初衷。 |

### 1.2 掩码像素预测（Masked Prediction）

| 字段 | 内容 |
|------|------|
| **来源** | `atom_selforg_redesign.md` Part 1 + Part 2；`theory_fracture_fixes.md` 推论 13.1（掩码预测 ⇒ 物体感知度量场）；`theory_selforg_2.md` 命题 12,13 |
| **状态** | ✅ 实现 |
| **代码位置** | `src/losses/self_organize.py:184-241`（`masked_prediction_loss`）；`train_2d.py:171-174`（`generate_random_mask`）；调用点 `train_2d.py:594-607` |
| **偏离说明** | mask 比例 0.3、状态→颜色预测结构、测地近邻投票全部按设计文档实现。但 `masked_prediction_loss` 中对所有原子使用 `softmax(-D²_px / ε)`（`self_organize.py:228-229`），未限制为 top-k；理论上应当限定为 `k=5` 近邻以避免远距原子拉低信号。`atom_selforg_redesign.md` Part 2 §预测机制 仅说"找到测地近邻"，并未硬性约束 k，但隐含"局部"。 |

### 1.3 度量平坦先验（Metric Flatness / Grid-Cell Prior）

| 字段 | 内容 |
|------|------|
| **来源** | `neuroscience_informed_roadmap.md` Phase 2；`theory_selforg_4.md` 命题 23 + 定理 17,18（谱保证） |
| **状态** | ✅ 实现（2D + 3D） |
| **代码位置** | `src/geometry/metric_field.py:157-193`（`MetricField2D.metric_flatness_loss`）；`src/geometry/metric_field.py:332-365`（`MetricField3D.metric_flatness_loss`）；调用点 `train_2d.py:686-689` |
| **偏离说明** | 实际公式与路线图建议公式**不同**：路线图给出 `((trace/d) - det^(1/d)).mean()`（鼓励 trace/d ≈ det^(1/d)，即各向同性），而代码实现的是 `mean(|g12|/(g11+g22))` + trace 空间 L1 平滑（`metric_field.py:181-191`）。两者物理动机类似但数学形式不同。代码版本的"aniso"分母是 `g11+g22+eps`（对角元之和），而路线图版本是 `det^(1/d)`（几何平均）。在 SPD 矩阵上后者更接近"几何平均"，前者更接近"算术平均的归一化"。两者对实际训练效果可能相近，但偏离理论原文值得在文档中说明。 |

### 1.4 下一视角预测（Next-View Prediction）

| 字段 | 内容 |
|------|------|
| **来源** | `neuroscience_informed_roadmap.md` Phase 1；`theory_selforg_4.md` 命题 28 + 定理 23（跨视角一致性降低 β_c 30%） |
| **状态** | 🟡 部分实现 |
| **代码位置** | `train_2d.py:472-503`（`loss_pred_view` 块）；CLI `train_2d.py:846-847`（`--w-pred-view`） |
| **偏离说明** | 当前实现使用 **canonical rays**（无相机外参变换）渲染下一帧 + L1 重建误差，与文档建议的 "状态转换" / "跨视角一致性" 不一致：<br>① **没有 `state_transition` 模块**：路线图要求 `state_transition(states_t, camera_delta)` 用小 MLP 预测 `states_{t+1}`，再解码成颜色 — 代码完全没有这个网络。<br>② **没有 3D→2D 投影一致性**：理论要求 epipolar 几何或原子投影匹配（命题 28，公式 427）。当前代码仅渲染相邻帧并比较颜色，相机无实际变化（`rays_o/rays_d` 在 473-490 行完全相同）。<br>③ **没有定理 23 的 30% β_c 收益**：因 (i) 和 (ii) 未实现，无法触发定理 23 所需的"跨视角互信息增益 γ_cross"。<br>权重 `w_pred_view` 默认 0，未生效。 |

### 1.5 残差解码器（Residual + LayerNorm + SiLU Decoder）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_selforg_4.md` 第一部分：定理 17（残差 + 2.5× λ_min）+ 定理 18（LayerNorm 谱正则化）+ 命题 23（联合谱保证） |
| **状态** | 🟡 部分实现（架构已就位但训练默认未启用） |
| **代码位置** | `src/atoms/residual_decoder.py:24-49`（`ResidualSiLUBlock`）；`src/atoms/residual_decoder.py:51-176`（`ResidualDecoder` + `compute_jacobian_spectrum`）；`src/atoms/residual_decoder.py:179-211`（`create_optimal_decoder` 含 `linear_only=True` 切换） |
| **偏离说明** | `ResidualDecoder` 架构完整实现（Pre-LN + SiLU + 残差块 + Xavier 初始化）。但 `train_2d.py:256` 调用 `create_optimal_decoder(..., linear_only=True)`，**默认配置使用单层 Linear+Sigmoid 而不是 ResidualDecoder**。理由在 `train_2d.py:251-254` 注释中说明：线性层强制状态直接编码颜色，防止解码器学习"绕过"映射导致状态坍缩。但结果是：<br>① 训练中没有使用理论最优架构，定理 17/18 的 2.5× 谱下界提升**未被实测**。<br>② `compute_jacobian_spectrum`（`residual_decoder.py:143-176`）存在但未接入 `AxiomMonitor`，无法在训练中验证谱保证。<br>③ 建议引入开关 `--decoder {linear|residual}`，至少在消融实验中验证理论预期。 |

### 1.6 自适应时间尺度（Adaptive Timescale via Lanczos）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_selforg_4.md` 第二部分：算法 1（随机 Lanczos 谱估计）+ 命题 24（Lanczos 收敛）+ 定理 19（自适应学习率 η_k^(t)=η_0·λ̂_s/λ̂_k）+ 命题 25（EMA + 钳位鲁棒性） |
| **状态** | 🟡 部分实现（固定比例已生效，在线估计缺失） |
| **代码位置** | 固定比例：`train_2d.py:265-267`（`lr_state:lr_metric:lr_position = 1:20:0.1`）。在线 Lanczos：**未实现**。 |
| **偏离说明** | 比例 `1 : 20 : 0.005` 在 `train_2d.py:262-264` 注释中被引用为"奇摄动分析的理论值"，但实际代码采用 `1 : 20 : 0.1`（位置 lr 不是 0.005 而是 0.1，差 20×）。代码完全没有：<br>① `Hessian-vector product` 的双重 autograd 工具函数；<br>② Lanczos 三对角化迭代；<br>③ 三块 Hessian 的独立估计（`λ̂_s/λ̂_g/λ̂_μ`）；<br>④ EMA 平滑（β=0.9）；<br>⑤ η_min=1e-6、η_max=0.1 钳位。<br>当前为开环固定比例；理论闭环反馈未落地。 |

### 1.7 Bootstrap 收敛（Bootstrap Smoothing Schedule）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_fracture_fixes.md` 第二部分：定理 20（Bootstrap 收敛）+ 实践建议 `η_s^boot=0.05` → `η_s^final=0.01` |
| **状态** | ✅ 实现（schedule） + ✅ 实现（监控） |
| **代码位置** | Schedule：`train_2d.py:325-329`（`w_met_boot=0.05`, `bootstrap_epochs=max(phase1_epochs,100)`），`train_2d.py:506-513`（线性退火）。监控：`src/losses/axiom_diagnostics.py:265-299`（`compute_bootstrap_rate` 返回 `Δg, G_edge`）。 |
| **偏离说明** | 退火 schedule 与理论推荐一致（η_s^boot=0.05）。Bootstrap 监控量 `Δg = tr_out - tr_in` 和 `G_edge`（基于 `scipy.ndimage.distance_transform_edt`）均已实现。但 `compute_bootstrap_rate` 使用 `tr_in/tr_out` 像素均值估计"边缘梯度强度"，与理论 G_edge（命题 23）所描述的 "重建梯度在颜色边缘处的边界分量" 不严格等价——后者需要重建损失对度量场的梯度范数，而当前实现是 trace 场的空间梯度。**功能接近，定义不同**。 |

### 1.8 涌现检测器 R(t)（Theorem 21 / §3.3）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_fracture_fixes.md` 定理 21（基于梯度比的涌现条件）+ §3.3 公式 285；代理 `R̃(t)=η_selforg/η_s · Var(cos(s_i,s_j) · 1[d_g<r])` |
| **状态** | ✅ 实现（精确梯度比 + 诊断） |
| **代码位置** | `src/losses/axiom_diagnostics.py:224-262`（`compute_gradient_ratio`）；`src/losses/axiom_diagnostics.py:393-397`（`get_emergence_epoch`）；`train_2d.py:786-793`（每 100 epoch 调用并打印 `R_emergence`） |
| **偏离说明** | 实现是**精确梯度比** R(t) = (η_selforg·‖∇_g L_selforg‖) / (η_s·‖∇_g L_smooth‖)，比代理 `R̃` 更严格（理论两个等价：精确比更准，代理更便宜）。`AxiomMonitor.summary()` 在 `R>1` 时打印 `EMERGENCEOK` 标签（`axiom_diagnostics.py:385-388`）。完全满足定理 21。 |

### 1.9 交叉视图一致性（Cross-View Consistency）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_selforg_4.md` 第五部分：命题 28（跨视角预测一致性 ⇒ 3D 物体理解）+ 定理 23（β_c 降低 30%）+ 命题 29（多视角位姿正则） |
| **状态** | 🟡 部分实现（仅帧间一致性，无 epipolar / 3D 投影） |
| **代码位置** | 见 1.4（`w_pred_view`）。Epipolar / 投影匹配：**未实现**（`grep` 搜索 `epipolar|cross_view` 仅在 docs 中出现）。 |
| **偏离说明** | 当前 `w_pred_view` 是"渲染下一帧 + L1"的相机无关自监督，不是真正的跨视角一致性（需要相机外参、3D→2D 投影、epipolar 约束）。`src/data/synthetic_2d.py:99` 的 `generate_multi_view` 生成的是 **2D 形状的多次扰动**，不是 3D 多视角（无相机模型）。因此：<br>① 命题 28 的 `p^corr`（epipolar 搜索）不存在；<br>② 定理 23 的 γ_cross 不可估计；<br>③ 命题 29 的 `H_μμ` epipolar 修正不存在。<br>理论需要切换到 3D 训练管线（`train_3d.py`）+ 真实相机位姿才能闭环。 |

### 1.10 Poincaré 状态流形（Poincaré Ball / 双曲状态）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_selforg_4.md` 第四部分：定理 21（双曲空间自发层次化）+ 定理 22（乘积流形 S^d × B^d）+ 命题 27（Lyapunov 稳定性） |
| **状态** | ❌ 缺失 |
| **代码位置** | `grep poincaré|hyperbolic|Möbius|gyrovector` 在 `src/` 命中 0；`theory_selforg_4.md` 是唯一详细文档 |
| **偏离说明** | 完全没有双曲几何实现。状态 `s_i ∈ R^{16}` 是标准欧氏向量（`src/atoms/atom_2d.py:41`）。状态相似度使用 `cos(s_i, s_j) = s_norm @ s_norm.T`（`src/losses/self_organize.py:105-106`），即单位球面度量。理论推荐的：<br>① `‖s_i‖ < 1` 球内约束；<br>② Möbius 加法 `⊕` 与标量乘法 `⊙`；<br>③ Einstein 中点（Karcher 均值）；<br>④ 乘积流形 `S^d_flat × B^d_hier`；<br>⑤ arcosh 测地距离。<br>全部缺失。这在 README 与状态维度注释中均无迹象。 |

### 1.11 非刚性形变（Non-Rigid Deformation）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_selforg_4.md` 第六部分：定理 24（度量场双因子分解 g = h·g_obj + (1-h)·g_def）+ 命题 30（形变容忍界 ε_def < 13.9）+ 推论 23.1（形变-度量协同学习）+ §6.6 三阶段方案 |
| **状态** | ❌ 缺失 |
| **代码位置** | `grep deflection|deformation|non.?rigid|nonrigid` 在整个 repo 命中 0 |
| **偏离说明** | 完全没有任何形变场、形变-度量协同、形变-残差正则项。理论要求的：<br>① 微分同胚 `φ_vw(x) = x + u_vw(x)` 形变场；<br>② 双因子 g(x) = h(‖u‖)·g_obj + (1-h)·g_def；<br>③ u(x) 的微型预测网络；<br>④ 形变感知正则项 `L_metric-obj = L_selforg - η_def·‖∇g·∇u‖²`；<br>⑤ 训练阶段 I/II/III 切换。<br>全部缺失。 |

### 1.12 测地距离近似（Midpoint Mahalanobis vs True Geodesic）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_fracture_fixes.md` 引理 2（梯度局部性，使用中点度量 `d² = (μ_i-p)ᵀ g((μ_i+p)/2) (μ_i-p)`） |
| **状态** | ✅ 实现（中点近似） + ✅ 实现（验证工具） |
| **代码位置** | `src/losses/direct_cluster.py:89-136`（`compute_pairwise_midpoint_mahalanobis_sq`）；`src/losses/direct_cluster.py:145-216`（`compute_true_geodesic_sq_1d` Simpson 积分验证）；`src/losses/direct_cluster.py:140-142`（`compute_pairwise_geodesic_sq` 弃用别名） |
| **偏离说明** | 与理论中点度量公式完全一致。文档明确说明这是"非真正测地距离"的 chord 近似（`direct_cluster.py:94-112` 详细注释）。`compute_true_geodesic_sq_1d` 提供数值积分版本用于误差验证（文档 EXT-2 验证）。理论要求 d_true ≤ d_midpoint 的凸性已注释但未实现严格测地距离积分训练（速度成本）。当前训练使用中点近似，是理论与工程的合理权衡。 |

### 1.13 矩阵指数参数化（matrix_exp 严格 SPD）

| 字段 | 内容 |
|------|------|
| **来源** | `README.md` §数学框架 + `docs/math_analysis.md` EXT-1 修复；`cholesky_param.py:58-156` 中的 `symmetric_to_metric_2d/3d` 函数 |
| **状态** | ✅ 实现（2D + 3D） |
| **代码位置** | `src/geometry/cholesky_param.py:60-107`（2D `symmetric_to_metric_2d` 使用 `torch.linalg.matrix_exp`）；`src/geometry/cholesky_param.py:110-156`（3D 同理）；`src/geometry/metric_field.py:110-115, 149-155, 283-292`（按 `parametrization` 字符串切换） |
| **偏离说明** | 实现完整支持两种参数化：`cholesky`（快速，欧氏 SGD）与 `matrix_exp`（严格 SPD，切空间近似测地下降）。CLI `--parametrization {cholesky|matrix_exp}`（`train_2d.py:879-882`）允许切换。默认仍是 `cholesky`，因为 matrix_exp 慢且 §"注意这是首次阶近似"的注解（`cholesky_param.py:73-75`）说明对真正测地 SGD 仍需 Bonnabel 2013 风格 retraction。当前实现符合 EXT-1 修复要求。 |

### 1.14 度量场平滑（Metric Smoothness）

| 字段 | 内容 |
|------|------|
| **来源** | `framework_audit.md` / `math_analysis.md`（度量场正则化） |
| **状态** | ✅ 实现 |
| **代码位置** | `src/losses/metric_regularizer.py:4-35`（`metric_smoothness_loss`，2D）；`src/losses/metric_regularizer.py:38-67`（`metric_smoothness_loss_3d`） |
| **偏离说明** | 标准的有限差分 L2 空间梯度平滑。完整实现，无偏离。 |

### 1.15 占位耦合 + Trace 对比（Occupancy Coupling + Trace Contrast）

| 字段 | 内容 |
|------|------|
| **来源** | `README.md` §数学框架 + `math_analysis.md`（占位耦合损失） |
| **状态** | ✅ 实现（含 hinge 改进版） |
| **代码位置** | `src/losses/occupancy_coupling.py:4-31`（`occupancy_coupling_loss`）；`src/losses/occupancy_coupling.py:34-68`（`trace_contrast_loss` hinge 改进） |
| **偏离说明** | 实现完整，包含 hinge 版（避免 MSE 在接近目标时梯度消失）。`trace_contrast_loss` 是对 `occupancy_coupling_loss` 的改进（`occupancy_coupling.py:38-45` 自述）。 |

### 1.16 自组织损失（Self-Organization Force）

| 字段 | 内容 |
|------|------|
| **来源** | `atom_selforg_redesign.md` Part 2；`theory_selforg_2.md` §3 |
| **状态** | ✅ 实现（含自适应 sigma 改进） |
| **代码位置** | `src/losses/self_organize.py:75-128`（`self_organization_loss`）；调用点 `train_2d.py:586-589` |
| **偏离说明** | 公式 `-Σ_{i,j} cos_sim(s_i, s_j) · exp(-d_g(i,j)² / (2σ_iσ_j))` 完全按文档实现。**自适应 σ**（每原子 K-th 近邻距离，`self_organize.py:113-119`）是 EXT-3 修复（解决 global median(D²)→0 后期梯度消失问题），是相对理论的工程改进。 |

### 1.17 状态传播（State Propagation / GAT-like Message Passing）

| 字段 | 内容 |
|------|------|
| **来源** | `atom_selforg_redesign.md` Part 2（"类似于 Graph Attention Network"）；`theory_selforg.md` 定理 1（状态传播收缩） |
| **状态** | 🟡 部分实现（线性加权平均 + state-similarity gating，非 softmax attention） |
| **代码位置** | `src/losses/self_organize.py:54-73`（`state_propagation` 基础函数 — 形式正确：`(1-α)s_i + α·Σ w_{ij} s_j`）；但 `train_2d.py:613-649` 的训练循环使用 **kNN mask + 状态相似度硬门控**（cos > 0.5），而非 softmax attention `w_{i→j} = softmax(cos_sim/τ)`。 |
| **偏离说明** | `state_propagation` 本身是线性加权平均形式（与文档一致：`s_i^{t+1} = (1-α)s_i + α·Σ w_{ij}s_j`）。但 `train_2d.py` 使用 **二值掩码 + 硬门控**（`sim_mask = (S > 0.5).float()`，`mask_gated = mask_now * sim_mask`，`train_2d.py:636-641`），实际生效的传播规则是 "在 kNN 内、且 cos_sim > 0.5 的邻居平均"，不是文档设计的 softmax attention。EMA 系数 γ=0.005（`train_2d.py:647`）。偏差对理论"软注意力"实现差异较大，建议引入开关或保留实验记录。 |

### 1.18 状态对比 InfoNCE（State Contrastive）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_selforg_2.md` 命题 11（InfoNCE 自监督对比）；`theory_selforg_3.md` 命题 17 |
| **状态** | ✅ 实现 |
| **代码位置** | `src/losses/self_organize.py:131-181`（`state_contrastive_loss`，InfoNCE 公式 `-log(pos/neg)`）；调用点 `train_2d.py:666-671`（权重 2.0） |
| **偏离说明** | 实现采用标准的 InfoNCE `-log(pos_sum/neg_sum)` 形式（`self_organize.py:180`）。温度默认 0.1（`self_organize.py:150`）。weight=2.0 是相对理论默认 1.0 的实验性加强（`train_2d.py:669` 注释："strong signal"）。完整实现。 |

### 1.19 状态谱监控（AxiomMonitor 与 6 公理）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_fracture_fixes.md` §6.1 六公理 + 定理 22（A4 鞍点性质） |
| **状态** | 🟡 部分实现（A1, A2, A3, A4, A6 监控存在；A5 未独立监控；J(F) 谱未监控） |
| **代码位置** | `src/losses/axiom_diagnostics.py:22-47`（A1 Fiedler）；`:50-102`（A2 测地分离比）；`:105-156`（A3 符号对齐 + 解析保证返回 0.95）；`:159-221`（A4 Hessian Rayleigh quotient）；`:265-299`（A6 bootstrap）；`:302-391`（`AxiomMonitor` 类，每 100 epoch 汇总） |
| **偏离说明** | A1-A4, A6 的诊断函数都已实现。但：<br>① **A5（度量场自修正）没有独立诊断**。当前仅在 `verify_axioms.py:162-181`（测试代码）做了对比 loss/grad 大小的检查，没有集成进 `AxiomMonitor`。<br>② **`ResidualDecoder.compute_jacobian_spectrum`**（`residual_decoder.py:143`）没有接入 `AxiomMonitor`，无法在训练中验证定理 17 的 2.5× 谱保证。<br>③ A4 的 `compute_uniform_instability_rayleigh`（`axiom_diagnostics.py:159-221`）实现完整，但 `train_2d.py:345-351` 注释说"expensive — compute periodically"且仅当 `state_decoder is not None` 才调用 — 在默认 `linear_only=True` 配置下，解码器是单层 Linear，仍可调用，但实际每 50 epoch 计算。 |

### 1.20 特征扩散（Feature Diffusion）

| 字段 | 内容 |
|------|------|
| **来源** | `docs/feature-diffusion-v0.3.md` + `theory_selforg.md` 历史背景 |
| **状态** | ✅ 实现（仅用于可视化） |
| **代码位置** | `src/losses/diffusion.py:24-76`（`compute_geodesic_affinity`）；`src/losses/diffusion.py:79-103`（`feature_diffusion`）；调用点 `train_2d.py:654-660` |
| **偏离说明** | 实现完整测地亲和矩阵 + 自适应 sigma + sigmoid 软掩码 + 强制对称。但 `train_2d.py:656` 注释"smoothing for state visualization only"，仅用于计算 `diff_val` 指标，没有作为可微损失驱动训练。这是历史保留路径（README 标记 `v0.4`），与新架构无强耦合。 |

### 1.21 动态原子管理（Pruning + Reprojection）

| 字段 | 内容 |
|------|------|
| **来源** | `README.md` 已验证功能列表；`atom_selforg_redesign.md` §"对扰动鲁棒" |
| **状态** | ✅ 实现（pruning） + ✅ 实现（reprojection，默认关闭） |
| **代码位置** | Pruning：`train_2d.py:115-168`（`prune_atoms_contrib_2d`）；`train_2d.py:727-744`（调度）。Reprojection：`train_2d.py:93-112`（`reproject_atoms`）；`train_2d.py:361-362`（Phase 1 → Phase 2 触发）；`train_2d.py:746-749`（周期性）；CLI `train_2d.py:862-864`（`--reproject-oracle`，默认关闭） |
| **偏离说明** | 完整实现。`reproject_oracle` 默认关闭（README 与 CLI 均声明），符合低偏置原则。 |

### 1.22 状态动力学收缩（Contraction Mapping）

| 字段 | 内容 |
|------|------|
| **来源** | `theory_selforg.md` 定理 1（状态传播收缩性） + `theory_fracture_fixes.md` 公理 A1 |
| **状态** | ✅ 实现（基础传播） + ✅ 实现（λ₂ 监控） |
| **代码位置** | 基础：`src/losses/self_organize.py:54-73`（`state_propagation`）。监控：`src/losses/axiom_diagnostics.py:22-47`（`compute_state_laplacian_eigenvalues`）。 |
| **偏离说明** | 收缩性在公式层面成立：`(1-α) s_i + α Σ w_{ij} s_j` 是加权平均，‖Δs‖ 在 α < 1, w 行随机时单步收缩。`compute_state_laplacian_eigenvalues` 通过对称化 W 后计算 `I - W` 的特征值监控 λ₂。`verify_axioms.py:72-99`（test_A1）通过迭代 `state_propagation` 验证单调收敛。完整实现。 |

### 1.23 Sinkhorn 软分配

| 字段 | 内容 |
|------|------|
| **来源** | `docs/gradient_flow_analysis.md`（Sinkhorn ε=0.05 推导）；历史 DirectCluster 已废弃但工具保留 |
| **状态** | ✅ 实现（工具） + ❌ 缺失（作为训练损失使用） |
| **代码位置** | `src/losses/direct_cluster.py:13-86`（`sinkhorn_softmax`，含 EXT-3 修复：自适应 ε、warm-start、收敛检测） |
| **偏离说明** | `DirectClusterLoss` 在 `direct_cluster.py:1-6` 自述中标记 `[ARCHIVE] DirectClusterLoss removed (fc5f5af postmortem)`。`sinkhorn_softmax` 保留为工具函数，但**没有在任何训练路径中调用**（`grep sinkhorn_softmax` 在 `train_2d.py` 命中 0）。`diffusion.py:24-76` 的 `compute_geodesic_affinity` 是行随机 affinity，不依赖 Sinkhorn。当前实现是后 DirectCluster 时代的合理选择，但 `sinkhorn_softmax` 现为死代码。 |

### 1.24 多种子鲁棒性测试（Seed-Variance Validation）

| 字段 | 内容 |
|------|------|
| **来源** | `neuroscience_informed_roadmap.md` 验证指标"跑 8 seed，看 ARI 均值和标准差 σ 是否下降"；`docs/history.md` Phase 7 Landscape 扫描 |
| **状态** | 🟡 部分实现（脚本存在但未集成到 `train_2d.py`） |
| **代码位置** | `tasks/sweep_hyperparams.py`、`tasks/ablation_trace_sep.py`（这些是超参扫描脚本） |
| **偏离说明** | 路线图明确建议"跑 8 seed，看 σ 从 0.39 → 0.25"。`train_2d.py` 只支持 `--seed` 单种子运行，没有内置多种子扫描。`tasks/sweep_hyperparams.py` 存在但需要确认是否针对 `w_homeo/w_flat/w_pred_view` 路由图 Phase 0-2 的开关进行验证。建议在 `tasks/` 下新增 `seed_sweep_homeo.py` 或类似脚本直接验证 σ 改善。 |

---

## 2. 优先级矩阵

按 (理论严格性 × 当前缺口严重性 × 实现成本) 综合排序。

### P0 — 必立即修复（理论核心闭环）

| # | 组件 | 当前状态 | 推荐动作 |
|---|------|---------|---------|
| 1 | **Residual Decoder 默认启用 + 谱验证闭环** | ResidualDecoder 已实现但 `linear_only=True` 绕开；`compute_jacobian_spectrum` 未接入训练 | (a) 引入 `--decoder {linear,residual}` 开关；(b) 在 `AxiomMonitor.step()` 中每 N epoch 调用 `compute_jacobian_spectrum` 验证 λ_min ≥ 0.08；(c) 与 `linear_only` 做消融对比 |
| 2 | **真正跨视角一致性（命题 28 / 定理 23）** | 仅相机无关的"渲染下一帧"，缺 epipolar / 3D 投影 | (a) 在 `src/data/synthetic_2d.py` 添加 3D 相机参数（外参、内参）；(b) 在 `src/atoms/atom_2d.py` 添加 `μᵢ ∈ R² → R³` 反投影；(c) 在 `self_organize.py` 加 `cross_view_consistency_loss(mus, states, views, decoder)` 用极线搜索 |
| 3 | **自适应时间尺度（Lanczos 在线估计，定理 19）** | 固定比例 1:20:0.1，缺在线闭环 | (a) 在 `src/training/` 添加 `hessian_estimator.py`：实现随机 Lanczos 三对角化（双重 autograd）；(b) 在 `AxiomMonitor` 或独立 `AdaptiveLR` 类中根据 λ̂_s/λ̂_g/λ̂_μ 调整学习率；(c) 与当前固定比例做收敛速度消融 |

### P1 — 中期实现（提升理论严谨性）

| # | 组件 | 当前状态 | 推荐动作 |
|---|------|---------|---------|
| 4 | **Poincaré 状态流形（定理 21）** | 缺失 | (a) 在 `src/atoms/` 新增 `state_manifold.py`：Möbius 加法 ⊕、标量乘法 ⊙、Einstein 中点、artanh/arcosh；(b) 在 `Atom2D` 加 `project_to_ball(s)` 约束 ‖s‖ < 1；(c) 在 `self_organize.py` 添加双曲版本的 `state_propagation` 与 `self_organization_loss`；(d) 在 3+ 物体场景做 ARI 消融对比 |
| 5 | **真实状态转换 `state_transition`**（路线图 Phase 1 状态版） | 仅渲染下一帧，无状态预测 | (a) 在 `src/atoms/` 新增 `state_transition.py`：MLP `(s_t, camera_delta) → s_{t+1}`；(b) 在 `self_organize.py` 加 `predictive_next_state_loss`；(c) `--w-pred-state` 开关；(d) 与 `w_pred_view`（像素版本）做对比 |
| 6 | **A5 度量场自修正显式诊断** | 测试代码中存在，`AxiomMonitor` 未集成 | (a) 在 `axiom_diagnostics.py` 加 `compute_metric_self_correction`：在边界处取 `∂L_selforg/∂g` 与 `∂L_recon/∂g` 的符号对比；(b) 在 `AxiomMonitor.step()` 中调用 |
| 7 | **度量平坦公式与路线图对齐** | 实际公式与 `neuroscience_informed_roadmap.md` Phase 2 给出的 `((trace/d) - det^(1/d)).mean()` 不同 | (a) 在 `metric_field.py` 新增 `metric_conformal_flatness_loss`（det^(1/d) 版本）；(b) 保留旧 `metric_flatness_loss`；(c) 加 `--flat-mode {aniso,conformal}` 开关 |

### P2 — 远景 / 实验性

| # | 组件 | 当前状态 | 推荐动作 |
|---|------|---------|---------|
| 8 | **非刚性形变（定理 24 + 命题 30 + 推论 23.1）** | 完全缺失 | (a) 在 `src/data/synthetic_2d.py` 加 `generate_nonrigid_scene`；(b) 在 `src/atoms/` 加 `deformation_field.py`：微型网络预测 `u(x)`；(c) 在 `src/geometry/` 加双因子 `g(x) = h·g_obj + (1-h)·g_def`；(d) 三阶段训练方案 Phase I/II/III |
| 9 | **Hebbian WTA / STDP 风格稀疏竞争（路线图 Phase 3）** | 缺失 | 在 `state_propagation` 中加硬 WTA：只让 top-k 状态维度的原子更新；与软注意力做对比 |
| 10 | **柱状预测编码（路线图 Phase 4）** | 文档标记"暂不实现" | 暂不投入；前 9 项完成后评估 |
| 11 | **有限 N β_c 偏移（定理 20）经验验证** | 理论结论未被实测 | (a) 在 `tasks/` 加 `beta_c_sweep.py`：N=25, 50, 100, 200, 400 扫描；(b) 拟合 β_c^(N) = β_c^(∞) + A/N；(c) 与理论预测的 2.2× 偏移对比 |
| 12 | **Sinkhorn 工具函数的去留** | 死代码 | 删除 `src/losses/direct_cluster.py` 的 `sinkhorn_softmax`，或迁移到 `tasks/` 作为分析工具 |
| 13 | **多种子扫描脚本（路线图 Phase 0 验证）** | 缺失（`tasks/sweep_hyperparams.py` 未针对 homeostatic） | 新增 `tasks/seed_sweep_homeo.py`：固定超参，扫 `--w-homeo ∈ {0, 0.05, 0.1, 0.2}` × 8 seed，记录 ARI 均值与 σ |

---

## 3. 前 5 个最高影响力缺口（Top-5）

### Top-1：Residual Decoder 默认配置绕开理论最优架构

- **影响**：所有依赖谱下界 λ_min 的理论保证（定理 17, 18, 命题 23）**未被实测**。`linear_only=True` 是为防止"decoder 学习 state-collapse 捷径"而设的实验性 hack，但它让项目丧失了残差架构的核心收益。
- **理论缺口**：22 条 R 级定理中至少 3 条（定理 17, 18, 命题 23）依赖残差 + LayerNorm + SiLU 联合架构。当前配置直接放弃。
- **修复成本**：低。`ResidualDecoder` 已完整实现（211 行），只需切换 `linear_only=True → False` 并加入 λ_min 监控。

### Top-2：跨视角一致性未实质实现

- **影响**：定理 23 承诺 β_c 降低 30%，是 `theory_selforg_4.md` Part 5 的核心收益。当前 `w_pred_view` 是相机无关的占位实现，定理 23 的 γ_cross 不可观测。
- **理论缺口**：命题 28 + 定理 23 + 命题 29 形成完整"3D 几何一致性"论证链，完全没有对应代码。
- **修复成本**：中。需要：
  ① `synthetic_2d.py` 添加真实相机外参生成 3D→2D 多视角数据；
  ② `atom_2d.py` 添加 3D→2D 投影；
  ③ `self_organize.py` 添加 epipolar 损失。
  
### Top-3：Poincaré 状态流形完全缺失

- **影响**：定理 21 的"层次化聚类涌现"与 3+ 物体场景直接相关。当前欧氏/球面度量无法编码层次结构，是 `atom_selforg_redesign.md` 列举的扩展性瓶颈之一。
- **理论缺口**：定理 21, 22 + 命题 27 是 Part 4 全部内容，4 条新理论陈述。完全无代码。
- **修复成本**：中高。需要 Möbius 算子 + 自微分 + 球内约束。预计 200-400 行新代码。

### Top-4：自适应时间尺度（Lanczos）未闭环

- **影响**：理论自适应学习率 η_k = η_0·λ̂_s/λ̂_k 应在训练中动态平衡三块（状态、度量场、位置）。当前固定比例 `1:20:0.1` 是开环近似，无法应对非平稳 Hessia 谱（特别是 bootstrap → emergence 相变点附近）。
- **理论缺口**：定理 19 + 命题 24, 25 构成完整自适应方案，3 条新陈述。完全无对应工具函数。
- **修复成本**：中。需要 Hessian-vector product 工具（双重 autograd，约 100 行）+ Lanczos 三对角化（约 80 行）+ EMA 调度（约 50 行）。

### Top-5：状态传播与文档设计偏离（硬门控 vs 软注意力）

- **影响**：虽然不是"缺失"，但 `train_2d.py:613-649` 的硬门控（kNN + cos > 0.5）与文档设计的 softmax 注意力差异较大。理论优势（梯度的连续性、对温度 τ 的可调性）**未实现**。
- **理论缺口**：状态传播是 `atom_selforg_redesign.md` Part 2 的核心机制，硬门控是其简化实现。
- **修复成本**：低。保留 `state_propagation` 函数（已实现），仅修改 `train_2d.py:613-649` 的传播规则为 softmax 注意力（`w_{i→j} = softmax(cos(s_i, s_j) / τ)`）。

---

## 4. 总结

### 4.1 数字一览

| 类别 | 数量 |
|------|------|
| 已完全实现的组件 | **11**（§1.1, 1.2, 1.7, 1.8, 1.12, 1.13, 1.14, 1.15, 1.16, 1.18, 1.20, 1.21, 1.22 — 实际 13 项） |
| 部分实现的组件 | **7**（§1.4, 1.5, 1.6, 1.9, 1.17, 1.19, 1.24） |
| 完全缺失的组件 | **3**（§1.10 Poincaré, 1.11 非刚性形变, 1.23 Sinkhorn 训练路径） |

### 4.2 理论公理覆盖率

| 公理 | 来源 | 监控 | 训练集成 | 评价 |
|------|------|------|----------|------|
| A1 收缩性 | theory_selforg.md 定理 1 | ✅ Fiedler λ₂ | ✅ 通过 `state_propagation` 显式使用 | 完整 |
| A2 掩码预测 ⇒ 物体推理 | theory_selforg_2.md 命题 13 | ✅ r_sep | ✅ `masked_prediction_loss` | 完整 |
| A3 自组织符号 | theory_selforg.md §2.2 | ✅ 解析返回 0.95 | ✅ `self_organization_loss` | 完整（监控是解析保证） |
| A4 均匀解不稳定 | theory_fracture_fixes.md 定理 22 | ✅ Rayleigh quotient | ❌ 未作为显式目标 | 监控存在，未强制 |
| A5 度量自修正 | theory_fracture_fixes.md 定理 17 | ❌ 测试代码存在 | ❌ 未集成 | 缺口 |
| A6 Bootstrap 收敛 | theory_fracture_fixes.md 定理 20 | ✅ Δg, G_edge | ✅ schedule | 完整 |

### 4.3 最大杠杆点

| 杠杆 | 预期收益 | 实施难度 |
|------|---------|---------|
| 启用 ResidualDecoder 并实测 λ_min | 验证 3 条 R 级定理 | 低 |
| 引入真正跨视角一致性 | 30% β_c 降低（理论预期） | 中 |
| Lanczos 自适应学习率 | 25-40% 收敛加速（理论预测 P24） | 中 |
| 软注意力状态传播 | 与路线图"软 Hebbian"一致 | 低 |

### 4.4 死代码 / 风险代码

| 代码 | 状态 | 建议 |
|------|------|------|
| `src/losses/direct_cluster.py:sinkhorn_softmax` | 死代码（无训练路径调用） | 删除或迁移到 `tasks/` |
| `src/geometry/geodesic.py` | 空文件 | 删除 |
| `src/atoms/atom_collection.py` | 空文件 | 删除 |
| `src/training/{trainer,optimizer,validator,__init__}.py` | 全部空（实际训练在根目录 `train_2d.py`） | 删除或迁移训练逻辑到此处 |
| `train_2d.py` 默认 `linear_only=True` | 与理论架构相悖 | 引入开关，默认 `residual`，附 ARI 消融 |

---

*所有结论基于对 4 篇理论文档与 `src/losses/`、`src/atoms/`、`src/geometry/`、`train_2d.py`、`src/losses/axiom_diagnostics.py` 的逐行阅读；`grep` 仅用于交叉验证声明位置。任何"理论已声明但代码无对应"的结论均通过反向 grep 确认（关键词：`poincaré|hyperbolic|Möbius|gyrovector|epipolar|cross_view|state_transition|predictive_next_view|deformation|nonrigid|Lanczos`）。*