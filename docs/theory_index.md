# MetricAtom 理论文档索引

> 最后更新：2026-06-13 · 分支：`feat/clustering-breakthrough`
>
> 本索引是 `docs/` 目录下 26 篇理论/分析/路线图文档的统一导航图，按主题、可信度与阅读依赖组织。若你是第一次阅读，建议先查看 [5. 推荐阅读路径](#5-推荐阅读路径)。

---

## 1. 文档地图

### 1.1 核心理论链：自组织原子系统

| 文档 | 标题 | 状态 | 定位 | 核心贡献 | 可信度 |
|---|---|---|---|---|---|
| [theory_selforg.md](theory_selforg.md) | 自组织原子系统：理论基础 | current | 奠基 | 状态动力学、涌现条件、Lyapunov、信息论、泛化界；7 条预测 P1–P7 | 3R / 7H / 4S |
| [theory_selforg_2.md](theory_selforg_2.md) | 自组织原子系统：理论深化 II | current | 深化 I | 解决 v1 的 5 个开放问题；掩码预测、联合 Hessian、IB 量化、τ 调度、K>2；6 条预测 P8–P13 | 3R / 6H / 2S |
| [theory_selforg_3.md](theory_selforg_3.md) | 自组织原子系统：理论深化 III | current | 深化 II | 解码器 Jacobian 下界、多物体 β_c、真实图像收缩性、自适应 τ、分岔理论、测地-状态对偶；8 条预测 P14–P21 | 3R / 10H / 1S |
| [theory_selforg_4.md](theory_selforg_4.md) | 自组织原子系统：理论深化 IV | current | 深化 III | 残差解码器、自适应时间尺度、有限 N 分岔、Poincare 状态流形、跨视角一致性、非刚性物体；9 条预测 P22–P30 | 4R / 12H / 6S |

按文档自述，v1 → v2 → v3 → v4 为**叠加完善**：后篇解决前篇的开放问题，而非取代。

### 1.2 审计、修复与缺陷

| 文档 | 标题 | 状态 | 定位 | 核心内容 |
|---|---|---|---|---|
| [theory_audit_and_roadmap.md](theory_audit_and_roadmap.md) | 自组织原子理论：数学可行性审计与发展方向重思考 | active | 全景审计 | 61 条陈述的 R/H/S 分级；4 条公理；FP1–FP4；P0–P3 路线图 |
| [theory_fracture_fixes.md](theory_fracture_fixes.md) | 自组织原子框架：断裂点修复与理论重整 | active | 修复提案 | FP1–FP3 + AD 的严格修复；8 条 IB 伪命题降级；6 公理体系 |
| [theory_defect_report_external_audit.md](theory_defect_report_external_audit.md) | 理论缺陷深度研究报告 — 外部交叉验证 | active | 外部审计 | EXT-1–EXT-8；真实严格率约 18%；与 Slot Attention 等对比 |
| [blocker_verification.md](blocker_verification.md) | 阻塞级缺陷的严格数学验证 | historical | ECO 审计 | ECO 三大阻塞级缺陷验证（ECO 路径已弃用） |
| [postmortem_direct_cluster.md](postmortem_direct_cluster.md) | DirectCluster 路线事后分析 | active | 失败记录 | 6 条路线全失败；3 个致命假设被证伪；推动 redesign |

### 1.3 数学分析（Direct Cluster / 收敛 / 旧框架）

| 文档 | 标题 | 状态 | 定位 | 核心内容 |
|---|---|---|---|---|
| [convergence_rate_analysis.md](convergence_rate_analysis.md) | Direct Cluster 收敛速率分析 | active | 收敛理论 | Lipschitz 界、PL 条件、线性收敛、ε 调度 |
| [gradient_flow_analysis.md](gradient_flow_analysis.md) | Direct Cluster vs InfoNCE：梯度流分析 | active | 梯度分析 | 甜区宽度、Dirac δ 非光滑性、最优 ε |
| [remaining_proofs.md](remaining_proofs.md) | 收敛理论三大遗留问题：完整证明 | active | 证明补完 | PL 条件、K>2 泛化、ECO 协同收敛（ECO 部分历史保留） |
| [math_analysis.md](math_analysis.md) | 数学分析：3D 黎曼度量场与当前超参设计 | active | 超参分析 | 3D 扩展、smoothstep C1 修正、权重标定 |
| [murmuration_dynamics.md](murmuration_dynamics.md) | Murmuration 动力学严格分析 | historical | ECO 理论 | Lyapunov 存在性、稳定性、吸引域；代码已删除 |

### 1.4 框架、历史、路线图与扩展

| 文档 | 标题 | 状态 | 定位 | 核心内容 |
|---|---|---|---|---|
| [framework_audit.md](framework_audit.md) | 框架缺陷与未证明部分：系统审计 | active | 系统审计 | 34 项命题/假设审计；8 项阻塞级缺陷 |
| [numerical_verification.md](numerical_verification.md) | 数值验证报告 | active | 验证 | Murmuration Lyapunov 8/8 收敛；测地高斯核 PSD 100/100 非 PSD |
| [theoretical_extensions.md](theoretical_extensions.md) | 理论扩展 | active | 扩展 | Phase 2 切换、自适应 K、PAC-Bayes 泛化界、敏感性 |
| [atom_selforg_redesign.md](atom_selforg_redesign.md) | 自组织原子系统：重新设计 | active | 架构设计 | 掩码多视图预测、状态动力学、涌现聚类验证 |
| [neuroscience_informed_roadmap.md](neuroscience_informed_roadmap.md) | 神经科学启发路线图 | active | 实现路线 | Homeostatic、下一视角预测、度量平坦性等可选先验 |
| [history.md](history.md) | MetricAtom 完整实验历史 | historical | 实验档案 | Phase 1–7 编年史；ARI 0.175 → 0.755 → 1.0 |
| [phase6a_eco_theory.md](phase6a_eco_theory.md) | Phase 6a: Murmuration-Elliptic Curve 物体表示理论框架 | deprecated | ECO 理论 | 椭圆曲线/j-不变量理论；已弃用 |
| [dev_plan.md](dev_plan.md) | MetricAtom 发展计划总纲 | historical | 旧路线图 | 2026-05-18 快照；已被后续 doc 覆盖 |
| [feature-diffusion-v0.3.md](feature-diffusion-v0.3.md) | 特征扩散模块设计文档 v0.3 | historical | 旧设计 | InfoNCE 时代的扩散模块；PSD 缺陷 |

---

## 2. 核心理论骨架

### 2.1 最小公理体系（源自 theory_fracture_fixes.md）

| 公理 | 内容 | 状态 |
|---|---|---|
| A1 | 状态收缩性：图注意力传播在适当条件下是收缩映射 | R（待验证） |
| A2 | 掩码预测要求物体推理：仅纹理记忆无法完成掩码预测 | R（受 EXT-4 质疑） |
| A3 | 符号正确性：自组织力方向与度量场协同 | R |
| A4 | 对称破缺：均匀解是不稳定鞍点 | R（Theorem 22 自承证明有符号跳变） |
| A5 | 梯度局部性（新增） | R |
| A6 | 跨对象分解（新增） | R |

### 2.2 主线定理链

1. **状态动力学**：[theory_selforg.md](theory_selforg.md) 定理 1–2 → 推论 1.1–1.2
2. **涌现条件**：[theory_selforg.md](theory_selforg.md) 定理 5（C1+C2+C3）
3. **Lyapunov**：[theory_selforg.md](theory_selforg.md) 命题 6–8
4. **联合 Hessian / PL**：[theory_selforg_2.md](theory_selforg_2.md) 定理 8–9
5. **解码器谱**：[theory_selforg_3.md](theory_selforg_3.md) 定理 13 → [theory_selforg_4.md](theory_selforg_4.md) 定理 17–18
6. **有限 N 分岔**：[theory_selforg_4.md](theory_selforg_4.md) 定理 20
7. **跨视角一致性**：[theory_selforg_4.md](theory_selforg_4.md) 定理 23

### 2.3 被降级的旧命题

以下原 R/H 级声明在 [theory_fracture_fixes.md](theory_fracture_fixes.md) 中被降级为启发式（H）或弃用：

- 命题 9：预测损失 ↔ 互信息对偶
- 命题 10：聚类作为最优压缩
- 命题 11：Rademacher 泛化界
- 定理 11：IB β_c 量化
- 命题 14：β_c ↔ SNR_min 反相关（定量部分移除）
- 命题 15：β_c 预测涌现 epoch
- 定理 14：多物体 β_c 精确公式
- 定理 23：跨视角 β_c 降低 30%

---

## 3. 可信度评级与内部矛盾

### 3.1 R / H / S 定义

| 评级 | 含义 |
|---|---|
| **R**（Rigorous） | 有完整数学证明或严格推导，且假设明确 |
| **H**（Heuristic） | 有分析直觉、数值支持或部分推导，但缺少完整证明 |
| **S**（Speculative） | 假设性强、尚未验证或仅为研究方向 |

### 3.2 不同文档给出的严格率

| 来源 | 统计口径 | R 率 |
|---|---|---|
| [theory_audit_and_roadmap.md](theory_audit_and_roadmap.md) | 61 条陈述总计 | 13/61 = **21%** |
| [theory_fracture_fixes.md](theory_fracture_fixes.md) §5.3 | 移除 8 条 IB 伪命题后 | 14/53 = **26%** |
| [theory_fracture_fixes.md](theory_fracture_fixes.md) 附录 A | 原 61 条 + 9 条新 R | 22/61 = **36%** |
| [theory_defect_report_external_audit.md](theory_defect_report_external_audit.md) | 外部审计估计 | ≈ **18%** |

> **注意**：上述数字来自不同统计口径，彼此不完全可比。README 的“22R(36%) + 27H(44%) + 12S(20%)”分母为 61，但附加了 11 条新声明，因此与纯 v1–v4 审计的 21% 有差异。

### 3.3 主要跨文档矛盾（未解决）

1. **R 率口径**：如上表，不同文档给出 18%–36% 的 R 率估计。
2. **Lojasiewicz 收敛**：[theory_fracture_fixes.md](theory_fracture_fixes.md) 定理 18 标为 R；外部审计 EXT-5 认为 θ 可能接近 1，实际保证不可行。
3. **掩码预测是否强制物体推理**：[theory_fracture_fixes.md](theory_fracture_fixes.md) 公理 A2 标为 R；外部审计 EXT-4 引用 MAE 文献指出仅当颜色/纹理边界与物体边界对齐时成立。
4. **测地距离有效性**：[theory_fracture_fixes.md](theory_fracture_fixes.md) 定理 17–19 假设真实测地距离；外部审计 EXT-2 指出中点近似无误差界，高各向异性区可能偏差 100%+。
5. **ECO 路径状态**：README 与 [theory_audit_and_roadmap.md](theory_audit_and_roadmap.md) 称 2026-06-03 已弃用并删除代码；[blocker_verification.md](blocker_verification.md) 仍在验证 ECO 代码路径。

---

## 4. 断裂点、缺陷与弃用路径

### 4.1 内部断裂点（FP）

| ID | 描述 | 严重度 | 状态 |
|---|---|---|---|
| FP1 | 度量场 SGD 是否收敛到 g* | 单点故障 | 修复提案：soft min-cut + Lojasiewicz（未验证） |
| FP2 | 状态-度量场 bootstrap 冷启动 | 高 | 修复提案：颜色边缘梯度分析（未验证） |
| FP3 | 信息瓶颈伪形式化 | 中 | 已降级 8 条 IB 声明 |
| FP4 | 公理到实验的 5 个 gaps | — | 未解决 |

### 4.2 外部审计缺陷（EXT）

| ID | 描述 | 严重度 | 状态 |
|---|---|---|---|
| EXT-1 | Cholesky + 欧氏 SGD ≠ SPD 流形优化 | blocking | P1：比较 Cholesky / 矩阵指数 / 自然梯度 |
| EXT-2 | 中点度量近似无测地距离保证 | blocking | P0：与数值积分对比 |
| EXT-3 | Sinkhorn ε=0.05 在不稳定区，迭代太少 | blocking | P0：自适应 ε 或 ≥200 次迭代 |
| EXT-4 | 掩码预测不必然强制物体推理 | major | P1：同色多物体 ARI 验证 |
| EXT-5 | Lojasiewicz θ 可能接近 1 | major | P2：建议降级为 H |
| EXT-6 | “零外部先验”声明不成立 | major | P0：README 删除该声明 |
| EXT-7 | 3D 测地邻接稀疏性被低估 | major | P1：3D 可行性验证 |
| EXT-8 | 自组织力热力学类比缺乏严格性 | minor | P2 |

### 4.3 弃用路径

| 路径 | 弃用日期 | 原因 | 保留文档 |
|---|---|---|---|
| ECO / Murmuration / j-不变量框架 | 2026-06-03 | jDist 2.3e11，ARI 0.30 vs DirectCluster 0.93 | [phase6a_eco_theory.md](phase6a_eco_theory.md), [murmuration_dynamics.md](murmuration_dynamics.md), [blocker_verification.md](blocker_verification.md) |
| DirectCluster 路线 | 2026-06-04 | 6 条路线全失败；种子敏感性 | [postmortem_direct_cluster.md](postmortem_direct_cluster.md) |
| InfoNCE + 特征扩散 | 2026-06-03 | Direct Loss 替代；扩散核 PSD 缺陷 | [feature-diffusion-v0.3.md](feature-diffusion-v0.3.md) |

---

## 5. 推荐阅读路径

### 路径 A：30 分钟快速了解
1. [README.md](../README.md) 核心思想 + 数学框架
2. [theory_index.md](theory_index.md)（本文档）
3. [atom_selforg_redesign.md](atom_selforg_redesign.md) 新架构设计

### 路径 B：系统学习自组织理论
1. [theory_selforg.md](theory_selforg.md) 基础
2. [theory_selforg_2.md](theory_selforg_2.md) 深化 I
3. [theory_selforg_3.md](theory_selforg_3.md) 深化 II
4. [theory_selforg_4.md](theory_selforg_4.md) 深化 III
5. [theory_fracture_fixes.md](theory_fracture_fixes.md) 断裂点修复

### 路径 C：审计与可信度
1. [theory_audit_and_roadmap.md](theory_audit_and_roadmap.md)
2. [theory_defect_report_external_audit.md](theory_defect_report_external_audit.md)
3. [theory_fracture_fixes.md](theory_fracture_fixes.md)
4. [framework_audit.md](framework_audit.md)

### 路径 D：实现与路线图
1. [atom_selforg_redesign.md](atom_selforg_redesign.md)
2. [neuroscience_informed_roadmap.md](neuroscience_informed_roadmap.md)
3. [theoretical_extensions.md](theoretical_extensions.md)
4. [numerical_verification.md](numerical_verification.md)

### 路径 E：历史与失败教训
1. [history.md](history.md)
2. [postmortem_direct_cluster.md](postmortem_direct_cluster.md)
3. [phase6a_eco_theory.md](phase6a_eco_theory.md)（仅作历史参考）

---

## 6. 可检验预测与实验 ID 速查

### 6.1 自组织理论链预测（P1–P30）

| 预测范围 | 编号 | 来源 | 示例 |
|---|---|---|---|
| v1 | P1–P7 | [theory_selforg.md](theory_selforg.md) | 前 K 个状态 PCA 主成分捕获 >90% 方差 |
| v2 | P8–P13 | [theory_selforg_2.md](theory_selforg_2.md) | τ 对数冷却避免过冷陷阱 |
| v3 | P14–P21 | [theory_selforg_3.md](theory_selforg_3.md) | 纹理梯度 >0.3 时 α 需 <0.05 |
| v4 | P22–P30 | [theory_selforg_4.md](theory_selforg_4.md) | 跨视角一致性使 β_c 降低 30% |

### 6.2 关键实验 ID

| ID | 描述 | 状态 |
|---|---|---|
| Seeds 100–107 | Phase 7 8-seed landscape | 已完成；seed-107 ARI=1.0，σ=0.39 |
| Seeds 42/123/99/77 | 自组织架构多种子验证 | 待完成 |
| E1–E5 | 3D 聚类、K>2、128×128、形状变化、真实图像 | 未验证 |
| H1–H7 | 工程假设（双线性插值、中点度量、smoothstep C1 等） | 未形式化 |

---

## 7. 待解决问题与下一步

### 7.1 当前最高优先级（来自 active 文档）

| 优先级 | 问题 | 来源 |
|---|---|---|
| P0 | 测地高斯核 A 非 PSD：[numerical_verification.md](numerical_verification.md) 100/100 失败 | 需修改理论/代码 |
| P0 | EXT-2 中点度量近似无误差界 | 需数值验证 |
| P0 | EXT-3 Sinkhorn ε=0.05 稳定性 | 需自适应 ε 或增加迭代 |
| P0 | 外部审计建议 README 删除“零外部先验”声明 | 文档 |
| P1 | 实现 [neuroscience_informed_roadmap.md](neuroscience_informed_roadmap.md) Phase 0 homeostatic | 代码 |
| P1 | 实现 [neuroscience_informed_roadmap.md](neuroscience_informed_roadmap.md) Phase 1 下一视角预测 | 代码 |
| P1 | 验证 EXT-4（同色多物体 ARI） | 实验 |

### 7.2 理论开放问题（来自 theory_selforg_4.md §7.4）

1. 广义残差架构的谱理论（DenseNet、Highway）
2. 曲率感知状态流形的维度分配
3. 时序一致性对 β_c 的影响
4. 3D 原子位置的 2D-3D 提升
5. 动态原子数量的涌现（birth-death 过程）
6. 显式背景原子的统计理论

---

## 8. 维护说明

- 本文档根据 `docs/` 全部理论文档的元数据梳理生成。
- 当新增/修改理论文档时，应同步更新本索引的 [文档地图](#1-文档地图)、[理论骨架](#2-核心理论骨架) 与 [待解决问题](#7-待解决问题)。
- 任何对 R/H/S 评级的调整，应在此索引中说明并标注来源文档。
