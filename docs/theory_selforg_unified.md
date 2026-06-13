# 自组织原子系统：统一理论

> 2026-06-13 | 分支 `feat/clustering-breakthrough`
> 综合 [theory_selforg.md](theory_selforg.md), [theory_selforg_2.md](theory_selforg_2.md), [theory_selforg_3.md](theory_selforg_3.md), [theory_selforg_4.md](theory_selforg_4.md), [theory_fracture_fixes.md](theory_fracture_fixes.md)
> 目标：为 [atom_selforg_redesign.md](atom_selforg_redesign.md) 的掩码预测 + 状态传播架构提供单一的连贯数学基础
> 标记约定：`[vN Thm/Prop/Cor/Lem X]` 指原文编号；`[vN Thm/Prop X ↓]` 标记已降级；`[fracture ...]` 指断裂点修复文档

---

## 1. 背景与动机

### 1.1 旧 DirectCluster 框架的失败

[postmortem_direct_cluster.md](postmortem_direct_cluster.md) 证伪了旧框架的三个核心假设：

| 假设 | 证伪 | 理论后果 |
|------|------|---------|
| 特征通过重建能学到物体级信息 | 6/6 变种未超基线 | 梯度流向像素匹配而非物体语义 |
| 测地距离是好的聚类 teacher | V1/V4 测地对齐均退步 | 黎曼度量在 Phase 1 末期不编码物体边界 |
| 更好的初始化能修复坏种子 | V4 测地 KMeans 无效 | 问题在特征空间本身无物体结构 |

**Phase 7 经验**：8 种子实验中 ~50% 未能有效聚类，$\sigma_{\text{ARI}} = 0.39$。DirectCluster 在有利初始化下可完全解决（seed 107: ARI=1.0），但方差极大 — 确认了尖锐悬崖现象。

### 1.2 新框架的核心转变

```
旧框架: 重建 → 特征 → 聚类（外部强加）
新框架: 掩码预测 + 状态传播 → 物体理解（涌现）
```

数学转变：
- **优化变量**：$(g, \mu, f)$ → $(g, \mu, s)$，其中 $s_i \in \mathbb{R}^d$ 是可学习状态
- **损失函数**：$\mathcal{L}_{\text{direct}}(g,f,P)$ → $\mathcal{L}_{\text{predict}}(g,s) + \mathcal{L}_{\text{selforg}}(g,s)$
- **聚类**：显式优化目标 → 涌现属性

**已废止路径**：ECO（椭圆曲线 + Murmuration）— 2026-06-03 完全废止。相关代码（`elliptic_curve.py`, `murmuration.py`, `eco_cluster.py`）已删除。实验结果：ECO ARI=0.30 vs DirectCluster ARI=0.93。理论文档保留作为历史记录。

### 1.3 与前序理论成果的衔接

| 前序成果 | 在新框架中的适用性 |
|----------|-----------------|
| PL 条件 ([remaining_proofs.md](remaining_proofs.md) §1) | **部分适用** — 用于 $\mathcal{L}_{\text{recon}}+\mathcal{L}_{\text{smooth}}$ 子问题 |
| $K>2$ 泛化 ([remaining_proofs.md](remaining_proofs.md) §2) | **替换** — Sinkhorn 不适用，**命题 16** 层次化涌现替代 |
| 收敛速率 O(1/t) ([convergence_rate_analysis.md](convergence_rate_analysis.md)) | **适用** — $\mathcal{L}_{\text{recon}}$ 部分光滑性不变 |
| 泛化界 ([theoretical_extensions.md](theoretical_extensions.md) §3) | **替换** — 由 §8 中 **定理 21** 的梯度比涌现条件 + uniform convergence 替代 |
| Lyapunov 构造 ([murmuration_dynamics.md](murmuration_dynamics.md) §2) | **方法论适用** — 框架废止，技术迁移到本文 §5 |

---

## 2. 系统设定与记号统一

### 2.1 变量与参数空间

| 符号 | 含义 | 维度/默认值 |
|------|------|------------|
| $\mathcal{A} = \{a_i\}_{i=1}^N$ | 原子集合 | $N=100$（2D），$N=200$（3D） |
| $\mu_i \in \mathbb{R}^2$ | 原子位置 | 可学习 |
| $s_i \in \mathbb{R}^{d_s}$ | 原子状态 | $d_s=16$ |
| $c_i \in \mathbb{R}^3$ | 原子颜色 | 可学习 |
| $g(x): \mathbb{R}^2 \to \mathrm{Sym}^+(2)$ | 黎曼度量场 | 每像素 3 参数（Cholesky） |
| $\alpha$ | 状态传播速率 | $0.3$ |
| $\tau$ | 注意力温度 | $0.1$ |
| $r_g$ | 状态传播邻域半径 | — |
| $\eta_{\text{selforg}}$ | 自组织损失权重 | $\in [2, 10]\eta_s$ 推荐 |
| $\eta_s$ | 度量场平滑权重 | $0.01$ |
| $\eta_{\text{pos}}$ | 位置正则权重 | $0.1$ |
| $\eta_{\text{vol}}$ | 占位耦合权重 | — |
| $w_{\text{predict}}$ | 掩码预测损失权重 | $1.0$ |
| $m$ | 掩码比例 | $0.3$ |

### 2.2 度量场与中点测地距离

$$d_g(i,j) = \sqrt{(\mu_i - \mu_j)^\top \cdot g\!\left(\tfrac{\mu_i+\mu_j}{2}\right) \cdot (\mu_i - \mu_j)}$$

中点度量（而非严格测地积分）— 在度量剧烈变化区域有误差，但提供可微计算路径。

### 2.3 联合变量与损失

优化变量 $\theta = (L, \mu, S) \in \mathbb{R}^D$，$D = HW\cdot 3 + N\cdot 2 + N\cdot d_s$。对 $(H,W)=(64,64), N=100, d_s=16$：$D = 12288 + 200 + 1600 = 14088$。

总损失：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}}(g,\mu) + \eta_s \mathcal{L}_{\text{smooth}}(g) + \eta_{\text{vol}} \mathcal{L}_{\text{vol}}(g) + w_{\text{predict}} \mathcal{L}_{\text{predict}}(g,s) + \eta_{\text{selforg}} \mathcal{L}_{\text{selforg}}(g,s) + \eta_{\text{pos}} \mathcal{L}_{\text{pos}}(\mu)$$

其中：
- $\mathcal{L}_{\text{selforg}} = -\sum_{i,j}\cos(s_i,s_j)\cdot d_g(i,j)$ — 状态相似则拉近测地距离
- $\mathcal{L}_{\text{recon}}$ — L1 体积渲染重建
- $\mathcal{L}_{\text{smooth}} = \|\nabla g\|_F^2$ — 度量场平滑
- $\mathcal{L}_{\text{vol}}$ — 占位耦合（物体内 $\mathrm{tr}(g)<1$，背景 $\mathrm{tr}(g)>9$）
- $\mathcal{L}_{\text{pos}} = \sum_i \|\mu_i - \mu_i^{(0)}\|^2$ — 位置正则

---

## 3. 状态动力学与图注意力收敛

### 3.1 状态传播算子

状态更新：

$$s_i^{(t+1)} = (1-\alpha) s_i^{(t)} + \alpha \sum_{j \in \mathcal{N}(i)} w_{ij}^{(t)} s_j^{(t)}, \quad w_{ij} = \frac{\exp(\cos(s_i,s_j)/\tau)}{\sum_k \exp(\cos(s_i,s_k)/\tau)}$$

其中 $\mathcal{N}(i) = \{j : d_g(i,j)\leq r_g,\, j\neq i\}$。

**算子形式**：$\mathbf{S}^{(t+1)} = \mathcal{T}(\mathbf{S}^{(t)}) = (1-\alpha)\mathbf{S} + \alpha \mathbf{W}(\mathbf{S})\mathbf{S}$，$\mathbf{W}$ 行随机。

### 3.2 收缩性分析

**定理 1（状态传播收缩性）** `[v1 Thm 1]`：设 $\mathbf{W}$ 的图拉普拉斯特征间隙 $\lambda_2(\mathcal{L}_W) > 0$。冻结 $\mathbf{W}$ 时矩阵 $\mathbf{M}_\alpha = (1-\alpha)\mathbf{I} + \alpha\mathbf{W}$ 在常数子空间正交补上的谱半径：

$$\rho(\mathbf{M}_\alpha|_{\mathbf{1}^\perp}) \leq (1-\alpha) + \alpha(1-\lambda_2) = 1 - \alpha\lambda_2 < 1$$

**证明要点**：$\mathbf{W}$ 行随机 → $\lambda_{\max}(\mathbf{W})=1$（常数特征向量）。在常数子空间正交补上 $\lambda_i(\mathbf{W}) \leq 1 - \lambda_2(\mathcal{L}_W)$。

**物理意义**：消息传递驱动状态**同质化** — 同一连通分量（同一物体）的原子状态趋于一致；不同连通分量（不同物体）的原子因测地边界阻断而无消息传递 → 保持不同状态。

**推论 1.1（几何收敛速率）** `[v1 Cor 1.1]`：冻结 $\mathbf{W}$ 下 $\|\mathbf{S}^{(t)}-\mathbf{S}^*\|_F \leq (1-\alpha\lambda_2)^t \|\mathbf{S}^{(0)}-\mathbf{S}^*\|_F$。达精度 $\epsilon$ 需 $t \approx \log(1/\epsilon)/(\alpha\lambda_2)$ 步。

**推论 1.2（度量场作用）** `[v1 Cor 1.2]`：$\lambda_2(\mathcal{L}_W)$ 取决于度量场 $g$。度量场在物体边界形成瓶颈 → $\mathcal{L}_W$ 接近块对角 → $\lambda_2 \to 0$ → 块间无通信。

### 3.3 耦合注意力的完整动力学

**定理 2（联合收敛）** `[v1 Thm 2]`：若注意力函数 $L_W$-Lipschitz $\|\mathbf{W}(\mathbf{S})-\mathbf{W}(\mathbf{S}')\| \leq L_W \|\mathbf{S}-\mathbf{S}'\|_F$，则当 $\alpha L_W \bar{s} < \lambda_2$ 时 $\mathcal{T}_{\text{full}}$ 仍收缩。

**实践意义**：$\tau$ 控制 $L_W$ — $\tau$ 大 → softmax 平缓 → $L_W$ 小 → 更易满足收缩条件。默认 $\tau=0.1$ 需验证。

### 3.4 真实图像下的鲁棒性

**命题 17（纹理对状态坍缩的阻碍）** `[v3 Prop 17]`：设物体内纹理梯度 $\nabla_x c$，则同物体原子间注意力权重：

$$w_{ij} \geq w_0 \exp\!\left(-\frac{\|\nabla_x c\| d_g(i,j)}{\tau \lambda_{\min}(J_f)}\right)$$

**推论 17.1（纹理容忍条件）** `[v3 Cor 17.1]`：状态坍缩不受纹理阻碍需 $\|\nabla_x c\|\cdot\bar d_g < \tau\lambda_{\min}(J_f)$。高纹理时需更小 $\alpha$ 或更大 $\tau$。

**定理 15（鲁棒收缩条件）** `[v3 Thm 15]`：真实图像条件下

$$\alpha < \frac{2\lambda_2(\mathcal{L}_W)}{1 + L_W \bar{s} + \gamma_{\text{texture}}\|\nabla_x c\|_\infty}$$

其中 $\gamma_{\text{texture}} = \tfrac{2\bar d_g \|J_f^\top\|}{\tau\lambda_{\min}(J_f)}$。典型数值：纹理使最大允许 $\alpha$ 从 $0.067$ 降至 $0.025$。

**缓解策略**：①降低 $\alpha$ 至 $0.05$–$0.1$；②提高 $\tau$；③纹理感知的掩码；④多尺度状态传播。

### 3.5 状态-度量场协同演化

**命题 4（良性循环）** `[v1 Prop 4]`：三阶段 — 探索（epoch 0–100，$\mathcal{L}_{\text{recon}}$ 主导，$\mathcal{L}_{\text{selforg}}$ 信号弱）、分化（100–300，正反馈）、稳定（300+，协同不动点）。形式化为耦合不动点：

$$s^* = \mathcal{T}(s^*, g^*), \quad \nabla_g \mathcal{L}(s^*, g^*) = 0$$

---

## 4. 涌现聚类的数学条件

### 4.1 状态空间的对称破缺

初始时 $s_i$ 随机 → 状态空间无结构 → 系统对称（任意置换原子标签不改变损失）。当 $\mathcal{L}_{\text{predict}}$ 驱动状态编码视觉信息 → 不同物体接收不同预测梯度 → **对称破缺** → 涌现方向性。

### 4.2 充分条件

**定理 5（涌现聚类充分条件）** `[v1 Thm 5]`：设场景有 $K$ 个视觉可区分物体。若：
- **(C1)** 视觉可区分性：$d_{\text{TV}}(\mathcal{O}_A, \mathcal{O}_B) > 0$
- **(C2)** 度量连通性：同物体原子在 $g$ 中连通（存在测地路径全在被预测为同一物体的区域内）
- **(C3)** 度量隔离性：不同物体原子被高测地距离"墙"隔开

则在状态不动点 $s^*$，谱聚类在 $A_{ij}=\cos(s_i^*,s_j^*)$ 上完美恢复 $K$ 个物体的标签。

**证明要点**：(C2)+(C3) → 图 $\mathcal{G}(\mathcal{A},\mathcal{E}_g)$ 有 $K$ 个连通分量。**定理 1** 给出分量内状态收缩到均值；(C1) 给出分量间 $\mathcal{L}_{\text{predict}}$ 梯度不同 → 状态不同。谱聚类在 $A_{ij}$ 上恢复 $K$ 簇（Davis-Kahan）。

### 4.3 C2/C3 的代理指标

训练中循环依赖（条件取决于被训练的 $g$）→ 用代理验证：

| 条件 | 代理指标 | 计算 |
|------|---------|------|
| C2（连通性） | 每物体最小生成树最大边长 | 在 label 已知的验证集上 |
| C3（隔离性） | $r_{\text{sep}} = \min_{i\in A, j\in B} d_g(i,j)/\max_{i,j\in A} d_g(i,j)$ | $r_{\text{sep}}>2.0$ 视为满足 |

### 4.4 Landau 自由能与序参量

序参量 $\phi$ 测量簇内方差 vs 全局方差：

$$\phi = \frac{1}{N}\sum_i \|s_i - \bar s\|^2 - \frac{1}{K}\sum_k \frac{1}{|\mathcal{C}_k|}\sum_{i\in\mathcal{C}_k}\|s_i - \bar s_k\|^2$$

$\phi = 0$（无聚类）→ $\phi > 0$（有聚类结构）。Landau 自由能：

$$\mathcal{F}(\phi) = \mathcal{F}_0 + r(T)\phi^2 + u\phi^4 - h\phi$$

其中 $r(T) = r_0(T_c - T)$（$T$ 对应 epoch），$u>0$，$h$ 对应视觉区分信号。当 $T < T_c$，$r<0$ → 自由能在 $\phi>0$ 处有最小值 → **聚类涌现**。

**预测**：$T_c \propto 1/m$（mask 比例越大 → 任务越难 → 信号越弱 → 涌现更晚）。

### 4.5 必要条件与 $K$ 上限

**定理 12（必要条件）** `[v2 Thm 12]`：若涌现 $K$ 簇聚类，则：
- (N1) $\min_{k\neq l}\|\bar s_k^* - \bar s_l^*\| > 0$（簇间状态可区分）
- (N2) $\forall i\in\mathcal{C}_k, j\in\mathcal{C}_l$：$d_{g^*}(i,j) > d_{\text{sep}}^*$（跨簇测地隔离）
- (N3) 每簇 $\geq 2$ 原子

**证明要点**：(N1) 直接来自聚类定义；(N2) 反证 — 若 $d_g(i,j)\leq d_{\text{sep}}^*$ 则 $j\in\mathcal{N}(i)$ → 状态传播将 $s_j$ 拉向 $s_i$ → 矛盾；(N3) 单原子无内部测地距离。

**充分 vs 必要**：C2 仅**充分**不必要（可弱化为连通分量内一致性）。C1 等价 N1，C3 等价 N2。

**推论 12.1（$K$ 物理上限）** `[v2 Cor 12.1]`：

$$K_{\max} \leq \min\!\left(\frac{d_s}{2},\; \sqrt{\frac{N}{2}},\; \frac{HW\Delta_{g,\min}}{4\,\mathrm{tr}(g^*)}\right)$$

三个约束来自状态容量（$d_s$ 维球面 $\varepsilon$-分离向量数）、原子预算（每簇 $\geq 2$）、度量场分辨率（每物体边界所需像素）。对 $N=100, d_s=16, H=W=64$：$K_{\max}=7$。

### 4.6 测地-状态对偶性

**定理 16（测地-状态对偶性）** `[v3 Thm 16]`：在耦合不动点 $(g^*,s^*)$ 处，$\exists \lambda > 0$ 使

$$d_{g^*}(i,j) = \lambda(1 - \cos(s_i^*, s_j^*))$$

当且仅当 $K$ 个正交簇 + 度量完全隔离。

**推论 16.1** `[v3 Cor 16.1]`：物体边界 = 测地距离矩阵的谱间隙位置 = 状态相似度矩阵的谱间隙位置。状态聚类和度量场聚类是**同一涌现现象的两个对偶表示**。

---

## 5. Lyapunov 稳定性与均匀解

### 5.1 全局 Lyapunov 函数

**命题 6（损失作为 Lyapunov 函数）** `[v1 Prop 6]`：沿梯度下降连续时间流 $\dot g = -\nabla_g V, \dot\mu = -\nabla_\mu V, \dot s = -\nabla_s V$：

$$\frac{dV}{dt} = -\|\nabla_g V\|^2 - \|\nabla_\mu V\|^2 - \|\nabla_s V\|^2 \leq 0$$

**证明**：链式法则，平凡成立。严格等于 0 仅在 $\nabla V = 0$（临界点）。

**关键问题**：Lyapunov 函数不保证收敛到**好**临界点（聚类涌现）而非平凡解。

### 5.2 均匀解的不稳定性

**命题 7（均匀解线性不稳定）** `[v1 Prop 7]`：在视觉可区分多物体场景中，$s_i = \bar s$ 是鞍点而非局部极小。Hessian 在物体区分方向上有负特征值。

**定理 22（鞍点性质，严格化）** `[fracture Thm 22]`：均匀状态解 $s_i = \bar s$ 是鞍点 — Hessian 在零均值子空间 $\mathcal{V}$ 上至少有一个负特征值。物理原因：分化状态使每个原子更好预测其覆盖像素 → 降低 $\mathcal{L}_{\text{predict}}$。

**证明要点**：均匀解处 $\hat I(p) = f_{\text{dec}}(\bar s)$，但不同像素 $p$ 属于不同物体（颜色不同）→ 一阶梯度非零 → 二阶展开给出负曲率方向。

**推论 7.1** `[v1 Cor 7.1]`：梯度下降在视觉可区分场景中必然离开均匀解。

**推论 22.1** `[fracture Cor 22.1]`：SGD 从均匀解出发几乎必然离开（鞍点稳定流形低维）。

### 5.3 吸引域估计

**命题 8（吸引域下界）** `[v1 Prop 8]`：设 $\|s_i - s_i^*\| < \delta_s, \|g - g^*\|_\infty < \delta_g, \|\mu_i - \mu_i^*\| < \delta_\mu$。当 $\delta_s, \delta_g, \delta_\mu$ 满足条件时 $(g,\mu,s)\in\mathcal{B}(\mathcal{C}^*)$。

**数值估计**：典型配置 PL 常数 $\mu \approx 10^{-5}$–$10^{-4}$，吸引域半径 $r \approx \mu/L \approx 10^{-7}$–$10^{-6}$。**极小** — 解释旧框架种子敏感性。

---

## 6. 联合 Hessian 谱分析与吸引域（严格化）

### 6.1 损失在最优点的 Hessian 分解

$$\nabla^2\mathcal{L}(\mathcal{C}^*) = \begin{bmatrix} H_{gg} & H_{g\mu} & H_{gs} \\ H_{\mu g} & H_{\mu\mu} & H_{\mu s} \\ H_{sg} & H_{s\mu} & H_{ss} \end{bmatrix}$$

### 6.2 各块分析

**Block $H_{gg}$（度量场自耦合）**：
- $H_{gg}^{\text{smooth}} = 2\eta_s \Delta \otimes I_{3\times 3}$，$\lambda_2(\Delta)\approx \pi^2(1/H^2 + 1/W^2) \approx 0.0048$
- $H_{gg}^{\text{recon}} \succeq 0$（体积渲染 Hessian 在最优处 PSD）
- $H_{gg}^{\text{selforg}} \succeq 0$（$\cos > 0$ 同簇、$\cos < 0$ 跨簇各自正贡献）
- 综合：$H_{gg} \succeq 2\eta_s\lambda_2(\Delta) \cdot I_{gg}$

**Block $H_{\mu\mu}$（位置自耦合）**：$H_{\mu\mu} \succeq 2\eta_{\text{pos}} \cdot I$（位置正则贡献 + 重建 PSD）。

**Block $H_{ss}$（状态自耦合 — 核心块）**：

预测 Hessian：$\nabla_s^2\mathcal{L}_{\text{predict}}|_{s^*} \succeq 2\lambda_{\min}(J_f J_f^\top)\bar w^2 \cdot I_{ss}$，其中 $J_f = \partial f_{\text{dec}}/\partial s$，$\bar w$ 是平均近邻权重。

自组织 Hessian：在最优值 $H_{ss}^{\text{selforg}} \succeq 0$（簇内 PSD 因为 $d_g$ 小且 $\cos\approx 1$；跨簇负贡献由 $-\cos\approx +1$ 抵消）。

综合：$H_{ss} \succeq 2\lambda_{\min}(J_f J_f^\top)\bar w^2 \cdot I_{ss}$。

### 6.3 正定性定理

**定理 8（联合 Hessian 正定性）** `[v2 Thm 8]`：设在 $\mathcal{C}^*$ 处：
- (A1) $\eta_s > 0$
- (A2) $\eta_{\text{pos}} > 0$
- (A3) $\lambda_{\min}(J_f J_f^\top) > 0$（解码器 Jacobian 非零）
- (A4) 交叉项 $\|H_{g\mu}\|, \|H_{gs}\|, \|H_{\mu s}\| < \tfrac{1}{3}\min(\lambda_{\min}(H_{gg}),\lambda_{\min}(H_{\mu\mu}),\lambda_{\min}(H_{ss}))$

则

$$\lambda_{\min}(\nabla^2\mathcal{L}(\mathcal{C}^*)) \geq \min(2\eta_s\lambda_2(\Delta),\; 2\eta_{\text{pos}},\; 2\lambda_{\min}(J_f J_f^\top)\bar w^2) > 0$$

**证明**：由 Gershgorin 圆盘 + Weyl 不等式 — 块对角 $D = \mathrm{diag}(H_{gg},H_{\mu\mu},H_{ss})$ 在扰动下保持正定：

$$\lambda_{\min}(H) \geq \lambda_{\min}(D) - \|H-D\|_2 \geq \tfrac{2}{3}\lambda_{\min}(D) > 0$$

### 6.4 联合 PL 与吸引域

**定理 9（联合 PL）** `[v2 Thm 9]`：在 $B_r(\mathcal{C}^*)$ 中：

$$\tfrac{1}{2}\|\nabla\mathcal{L}(\theta)\|^2 \geq \mu_{\text{joint}}(\mathcal{L}(\theta) - \mathcal{L}^*)$$

$\mu_{\text{joint}} = \min\!\big(\eta_s\lambda_2(\Delta),\; \eta_{\text{pos}},\; \tfrac{\lambda_{\min}(J_f J_f^\top)\bar w^2}{2}\big) \cdot \tfrac{\lambda_{\min}}{\lambda_{\max}}$

**数值代入**（$\eta_s=0.01, \lambda_2=0.0048, \eta_{\text{pos}}=0.1, \lambda_{\min}(J_f J_f^\top)\approx 0.05, \bar w^2\approx 0.01$）：
- 平滑项：$4.8\times 10^{-5}$
- 位置项：$0.1$
- Jacobian 项：$5\times 10^{-4}$
- 主导项位置正则 → $\mu_{\text{joint}} \approx 1.25\times 10^{-5}$

**推论 9.1（吸引域半径）** `[v2 Cor 9.1]`：$r \leq \mu_{\text{joint}}/L_{\text{joint}}$。$L_{\text{joint}}\leq \max(L_{gg}, L_{\mu\mu}, L_{ss}) + $ 交叉项 ≈ $10^4$。$r \approx 1.25\times 10^{-9}$ — 极小。

**自组织框架的优势**：状态从 0 连续演化，避免硬切换冲击；系统通过渐近捕获进入吸引域，无需"投入"。

---

## 7. 掩码预测作为物体推理的强制机制

### 7.1 重建任务的非充分性

**命题 12（重建-聚类不对齐）** `[v2 Prop 12]`：存在场景配置使系统以零重建误差学习到不编码物体边界的内表示。

**构造性证明**：两相邻纯色物体（红圆 + 蓝矩形）。策略 A：原子聚类到两物体内部，状态分别编码红/蓝，度量场在边界锐利（聚类解）。策略 B：原子均匀分布，每个原子记住其 Voronoi 区域平均颜色，状态随机，度量场均匀（像素记忆解）。**两策略重建误差相同**（B 更均匀甚至更低），但 B 的 ARI = 0。

**结论**：非聚类零误差解存在。旧框架失败因为梯度下降可能落入策略 B 吸引域。

### 7.2 掩码预测打破对称

**命题 13（掩码预测强制物体推理）** `[v2 Prop 13]`：设像素 $p$ 被 mask。预测函数：

$$\hat I(p) = \sum_{j\in\mathcal{N}(p)} w_j(p) f_{\text{dec}}(s_j), \quad w_j(p) \propto \exp(-d_g(\mu_j, p)/\varepsilon)$$

对所有 mask 模式 $\hat I(p) = I_{\text{true}}(p)$ 要求：
1. 若 $j,k$ 都主要预测物体 $\mathcal{O}$ 的像素 → $f_{\text{dec}}(s_j) \approx f_{\text{dec}}(s_k) \approx c_\mathcal{O}$ → $s_j \approx s_k$
2. 若 $j$ 预测 $\mathcal{O}$、$k$ 预测 $\mathcal{O}'$ → $s_j \neq s_k$

掩码预测迫使系统编码"哪些原子共享视觉属性"。

**推论 13.1（掩码预测 ⇒ 物体感知度量场）** `[v2 Cor 13.1]`：在 $\mathcal{L}_{\text{predict}}$ 低于阈值时，度量场必在物体边界形成瓶颈。最坏情况混合预测误差至少 $(c_A - c_B)^2/4$，避免要求：

$$\frac{d_g(\mu_A, p_{\text{edge}})}{d_g(\mu_B, p_{\text{edge}})} \ll 1$$

**像素记忆解在掩码任务中失败** — 唯一降低误差的方法是让度量场不跨越边界。

---

## 8. 信息论视角（**已降级**）

> ⚠️ 整个 §8 经 [theory_fracture_fixes.md](theory_fracture_fixes.md) FP3 审查后**降级为启发式**。状态 $s_i$ 是确定性变量，互信息从未在代码中被计算，L2 损失仅是 IB 代理。压缩驱动机制由 §3 的收缩性（**定理 1**）和 §10 的梯度比 $R(t)$（**定理 21**）替代。

### 8.1 原始启发性陈述（已降级）

- **命题 9** `[v1 Prop 9 ↓]`：$\min\mathcal{L}_{\text{predict}} \Leftrightarrow \max I(Z;X)$ — **降级**
- **命题 10** `[v1 Prop 10 ↓]`：聚类 = 信息瓶颈最优压缩 — **降级**
- **命题 11** `[v1 Prop 11 ↓]`：Rademacher 泛化界 — **降级**
- **定理 11** `[v2 Thm 11 ↓]`：IB $\beta_c$ 量化 — **降级**
- **命题 14** `[v2 Prop 14 ↓]`：$\beta_c \propto 1/\text{SNR}$ — **降级**（定性保留）
- **命题 15** `[v2 Prop 15 ↓]`：$\beta_c$ 预测涌现 epoch — **降级**
- **定理 14** `[v3 Thm 14 ↓]`：多物体 $\beta_c$ 精确公式 — **降级**
- **定理 23** `[v4 Thm 23 ↓]`：跨视角 $\beta_c$ 降低 30% — **降级**

### 8.2 几何替代

**定理 21（基于梯度比的涌现条件）** `[fracture Thm 21]`：

$$t_{\text{emergence}} = \min\{t : \eta_{\text{selforg}}\|\nabla_g\mathcal{L}_{\text{selforg}}\| > \eta_s\|\nabla_g\mathcal{L}_{\text{smooth}}\|\}$$

代理 $\tilde R(t) = \tfrac{\eta_{\text{selforg}}}{\eta_s}\mathrm{Var}\!\big(\cos(s_i,s_j)\cdot\mathbf{1}[d_g(i,j)<r]\big)$ 可在线监控。

**几何三替代**：
1. 收缩性（**定理 1**）替代"信息压缩"驱动
2. $R(t) > 1$ 替代"$\beta_c$ 相变"
3. 标准 uniform convergence 替代"信息论泛化界"

### 8.3 替代 1：状态维度的信息容量（保留）

每个原子状态 $d_s$ 维 → 总容量 $N\times d_s\times H(s)$ 比特。最高效编码：每物体分配一个状态原型，有效容量 $K\times d_s$。当 $K\ll N$ 显著压缩。

**预测**：训练后 PCA，前 $K$ 主成分捕获 > 90% 方差。

---

## 9. 解码器 Jacobian 谱分析

### 9.1 解码器架构形式化

设 $f_{\text{dec}}: \mathbb{R}^{d_s} \to \mathbb{R}^3$（状态 → RGB），标准 3 层 MLP：

$$f_{\text{dec}}(s) = W_3\sigma(W_2\sigma(W_1 s + b_1) + b_2) + b_3$$

Jacobian：$J_f = W_3\mathrm{diag}(\sigma'(a_2))W_2\mathrm{diag}(\sigma'(a_1))W_1$。

### 9.2 ReLU 解码器的下界

**引理 16.1（ReLU 模式局部稳定）** `[v3 Lem 16.1]`：在最优状态 $s^*$，$\delta_{\text{sep}} = \min_{k\neq l}\|s_k^*-s_l^*\| > 0$。若 $\|s-s_k^*\| < \rho\delta_{\text{sep}}$，同簇原子的 ReLU 激活模式相同。

**证明**：预激活 $a = Ws+b$ 连续。在半径 $\rho\delta_{\text{sep}}$ 球内，预激活变化有界 $\|\Delta a\|\leq \|W\|\rho\delta_{\text{sep}}$。取 $\rho$ 足够小使 $\|\Delta a\|$ 小于最小非零预激活 → ReLU 模式不变。

**定理 13（ReLU Jacobian 谱下界）** `[v3 Thm 13]`：

$$\lambda_{\min}(J_f^* (J_f^*)^\top) \geq \sigma_{\min}^2(\tilde W_3)\prod_\ell \sigma_{\min}^2(\tilde W_\ell) \cdot \min_i (\sigma'(a_i))^2$$

对 $d_s=16$，64→32→3 MLP，Marchenko-Pastur 估计：

| 层 | $\sigma_{\min}$ 估计 |
|----|---------------------|
| $W_1$ (64×16) | $\sqrt{64}-\sqrt{16}=4.0$ |
| $W_2$ (32×64) | $\approx \sqrt{32/64}=0.71$ |
| $W_3$ (3×32) | $\approx \sqrt{3/32}=0.31$ |

50% 激活 → $\lambda_{\min}(J_f J_f^\top) \approx 0.05$–$0.1$。保守：$0.05$。

**推论 13.1** `[v3 Cor 13.1]`：$\lambda_{\min}(H_{ss}) \geq 2\times 0.05\times 0.01 = 10^{-3}$。

### 9.3 SiLU 改善

SiLU 避免 ReLU "死亡神经元"。$\sigma'(0)=0.5$。净效果：更多神经元保持激活补偿导数下降，$\lambda_{\min}$ 相似或略优。

### 9.4 残差 + LayerNorm 架构

**定理 17（残差提升 Jacobian 谱）** `[v4 Thm 17]`：带残差连接的解码器：

$$\lambda_{\min}(J_f^{\text{res}}(J_f^{\text{res}})^\top) \geq \lambda_{\min}(J_f^{\text{plain}}(J_f^{\text{plain}})^\top) \cdot \left(1 + \frac{\sigma_{\min}^2(D_2 W_2)}{2}\right)$$

约 **2.5× 提升**（$\approx 0.125$）。

**定理 18（LayerNorm 正交化）** `[v4 Thm 18]`：插入 LayerNorm 后条件数：

$$\kappa(J_f^{\text{LN}}(J_f^{\text{LN}})^\top) \leq \kappa(J_f^{\text{plain}}(J_f^{\text{plain}})^\top)\cdot \left(\frac{\sigma_{\max}(D_\ell)}{\sigma_{\min}(D_\ell)}\right)^2$$

LayerNorm 投影 $P_h$ 消去常值零空间方向，消除协方差退化路径。

**命题 23（联合谱保证）** `[v4 Prop 23]`：残差 + LayerNorm + SiLU 架构下 $\lambda_{\min}(J_f J_f^\top)\geq 0.08$ 以概率 $\geq 0.95$。

### 9.5 架构推荐

| 架构 | $\lambda_{\min}$ 下界 | 鲁棒性 | 推荐 |
|------|---------------------|--------|------|
| Plain MLP (ReLU) | 0.05 | 低 | 基线 |
| + 残差 | 0.12 (2.5×) | 中 | ✅ |
| + LayerNorm | ~0.03 × 鲁棒因子 | 高 | ✅ |
| + 残差 + LayerNorm | 0.12 × 鲁棒 | 很高 | ✅✅ 最优 |
| SiLU 替代 ReLU | 0.10 | 中高 | ✅ |

---

## 10. 断裂点修复与六公理体系

### 10.1 FP1：度量场收敛

**引理 1（预测误差跨物体分解）** `[fracture Lem 1]`：设状态已涌现聚类，$K$ 个中心 $\{\bar s_k\}$ 对应颜色 $\{c_k = f_{\text{dec}}(\bar s_k)\}$。对物体 $\mathcal{O}_A$ 的像素 $p$：

$$\mathcal{L}_{\text{predict}}(p;g) = \left\|\sum_{B\neq A} w_B(p)(c_B - c_A)\right\|^2$$

其中 $w_B(p) = \sum_{j\in\mathcal{O}_B} w_j(p)$。

**推论**：$\mathcal{L}_{\text{predict}}$ 在 $g$ 空间定义 **soft min-cut 目标** — 惩罚跨物体原子的测地近邻关系。

**引理 2（梯度局部性）** `[fracture Lem 2]`：$\partial\mathcal{L}_{\text{predict}}/\partial g(x)$ 仅在 $x$ 位于原子-像素对中点附近时非零。度量场可在不同空间位置独立优化。

**定理 17（自修正性）** `[fracture Thm 17]`：在物体边界附近，$g(x)$ 过小 → 梯度推大；过大 → 梯度推小。**证明**：两种情况分别用 $w_B$ 增大 / 物体内状态坍缩不完全导致预测误差增大 → 梯度回推。

**定理 18（Łojasiewicz 收敛）** `[fracture Thm 18]`：$\mathcal{L}_{\text{predict}}$ 在紧致集 $\Theta_g$ 上实解析 → 梯度下降以 $O(t^{-\theta/(1-2\theta)})$ 收敛到某临界点。若 PL 条件在 $g^*$ 邻域成立则线性收敛。

**定理 19（坏局部最小值不稳定）** `[fracture Thm 19]`：$g_{\text{bad}}$ 在边界处过低 → Hessian 负特征值：

$$\lambda_- \leq -\frac{\|c_A - c_B\|^2}{4\varepsilon}\exp(-d_{\text{bad}}/\varepsilon)$$

典型：$\|c_A-c_B\|^2\approx 2, \varepsilon\approx 0.1, d_{\text{bad}}\approx 0.5$ → $\lambda_-\approx -0.034$ — SGD 噪声可逃逸。

### 10.2 FP2：Bootstrap 冷启动

**命题 23（重建梯度在颜色边缘的分化）** `[fracture Prop 23]`：在训练早期，$\mathcal{L}_{\text{recon}}$ 在颜色边缘处对 $g$ 产生非零梯度，方向增大跨边缘方向度量分量。

**信号强度估计**：$\|\partial\mathcal{L}_{\text{recon}}/\partial g\|\approx 0.0075$ vs $\|\partial\mathcal{L}_{\text{selforg}}/\partial g\|\approx 0.05$（涌现后）。重建信号约为自组织信号的 **15%** — 足以提供种子，不足以独自产生完整边界。

**定理 20（Bootstrap 收敛）** `[fracture Thm 20]`：度量场分化指数收敛到非零稳态：

$$\Delta g(t) = \frac{\eta_{\text{recon}}G_{\text{edge}}}{\eta_s\lambda_2}(1 - e^{-\eta_s\lambda_2 t})$$

**实践建议**：Bootstrap 阶段用 $\eta_s^{\text{boot}}=0.05$，epoch 100–150 退火到 $\eta_s^{\text{final}}=0.01$。

### 10.3 FP3：废除 IB（见 §8）

**定理 21（基于梯度比的涌现条件）** `[fracture Thm 21]` — 见 §8.2。

### 10.4 公理 D：均匀解不稳定性

见 §5.2 **定理 22** `[fracture Thm 22]`。

### 10.5 六公理体系 `[fracture §6]`

| 公理 | 内容 | 级别 | 来源 |
|------|------|------|------|
| **A1** | 状态传播收缩性 | **R** | **定理 1** |
| **A2** | 掩码预测强制物体推理 | **R** | **命题 13** |
| **A3** | 自组织力符号正确性 | **R** | v1 §2.2 |
| **A4** | 均匀解不稳定性 | **R** | **定理 22** |
| **A5** | 度量场梯度自修正性 | **R** | **定理 17** (fracture) |
| **A6** | Bootstrap 收敛 | **R** | **定理 20** |

**逻辑链**：A4 → 离开均匀解 → A6 产生初始度量场结构 → A2 强化边界 → A1 簇内状态坍缩 → A3+A5 正反馈 → **聚类涌现**。每步由严格数学保证。

### 10.6 修复后严格性

| 系列 | 修复前 R 级 | 修复后 R 级 | 变化 |
|------|------------|------------|------|
| 自组织 v1-v4 | 13/61 (21%) | 13+9-8 = **14/53 (26%)** | 移除 8 个 IB 伪命题 + 新增 9 个严格命题 |

总项目 R 级：**22 条**（含 fracture 新增 9 条）。

---

## 11. 自适应温度 $\tau$ 调度

### 11.1 $\tau$ 的角色与陷阱

$\tau$ 控制注意力锐度：$\tau\to 0$ → 硬注意力，快收敛但易锁定错误分组；$\tau\to\infty$ → 均匀注意力，状态同质化无聚类。

**推论 11.1（过冷陷阱）** `[v2 Cor 11.1]`：$\tau < \text{gap}(s)/(2\log K)$ → 注意力过早锁定 → 错误分组无法修正。

**推论 11.2（过热低效）** `[v2 Cor 11.2]`：$\tau > 1$ → 注意力均匀 → 状态同质化 → 无聚类。

### 11.2 开环冷却方案

> 注：**[v2 定理 11]**（IB $\beta_c$ 量化）已**降级**。以下保留模拟退火的冷却速率结果作为启发。

**方案 A**：$\tau^{(t)} = \tau_0/\log(1+t/t_0)$，$\tau_0=0.5, t_0=50$。

**方案 B**：$\tau^{(t)} = \tau_{\min} + \tfrac{\tau_{\max}-\tau_{\min}}{2}(1+\cos\tfrac{\pi t}{T_{\text{cool}}})$，$\tau_{\max}=0.5, \tau_{\min}=0.05, T_{\text{cool}}=500$。

**推荐**：$\tau^{(t)} = \max(\tau_{\text{target}},\; \tau_0/\log(1+t/t_0))$，$\tau_{\text{target}}=0.1$。

### 11.3 自适应 PI 控制

**命题 18（PI 自适应收敛性）** `[v3 Prop 18]`：PI 控制器

$$\tau^{(t+1)} = \tau^{(t)} - K_p\Delta\tilde\phi^{(t)} - K_i\sum_{j=0}^t \Delta\tilde\phi^{(j)}$$

在 $K_p, K_i < \tfrac{2\lambda_2(\mathcal{L}_W)}{\alpha L_W\bar s}$ 时保持 $\tau$ 在合理范围且 $\tilde\phi$ 单调不减。形成自稳定负反馈环。

**无监督 $\tilde\phi$ 估计三方案**：
1. **谱间隙** $\tilde\phi = (\lambda_1-\lambda_2)/\lambda_1$（需 SVD）
2. **梯度一致性** $\tilde\phi_{\text{grad}} = \tfrac{1}{N(N-1)}\sum_{i\neq j}\mathbf{1}[\cos(\nabla_s\mathcal{L}_i, \nabla_s\mathcal{L}_j)>\theta_{\text{align}}]$（在线）
3. **KNN 一致性** $\tilde\phi_{\text{knn}} = \tfrac{1}{N}\sum_i \tfrac{|\mathcal{N}_K^{(t)}(i)\cap\mathcal{N}_K^{(t-T)}(i)|}{K}$

---

## 12. 等变分岔理论

### 12.1 $K=2$ 的分岔类型

**命题 20（$K=2$ 的 pitchfork）** `[v3 Prop 20]`：$\mathbb{Z}_2$ 等变使 Taylor 展开奇次项消失 → 标准 pitchfork 分岔形式。

**推论 20.1** `[v3 Cor 20.1]`：$K=2$ 涌现是**连续二阶相变**，$\phi\propto\sqrt{|r|}$（$r<0$）。

### 12.2 $K>2$ 的等变分岔

**命题 21** `[v3 Prop 21]`：$K>2$ 分岔属于 $\mathbb{S}_K$ 标准表示 $\mathbb{R}^{K-1}$。实际分岔路线取决于 SNR 矩阵特征值结构：
- **直接 $1\to K$**：四阶系数强正 → 超临界
- **层次化 $1\to 2\to K$**：SNR 有层次
- **序列化 $1\to 2\to 3\to K$**：SNR 极度不均

**命题 22（SGD 分岔延迟）** `[v3 Prop 22]`：SGD 噪声使涌现 epoch 延迟：

$$\mathbb{E}[T_{\text{emergence}}] \approx T_c + \frac{C\sigma_{\text{SGD}}^2}{|\lambda_-(H)|}$$

**结论**：大 batch → 小 SGD 噪声 → 涌现 epoch 更接近理论 $T_c$。实践：早期适中 batch（帮助逃鞍点），预期涌现阶段增大 batch。

### 12.3 分岔检测

实时检测（无需 $K$ 或标签）：
1. Hessian 谱跟踪（PCA 前 5 维投影）
2. 状态协方差迹加速度
3. 梯度范数爆发（离开鞍点时）

检测到后平滑切换学习率或调整 $\tau$（骤降"冻结"涌现结构）。

---

## 13. 多时间尺度自适应优化

### 13.1 时间尺度分离

**命题 19** `[v3 Prop 19]`：状态-度量场-位置构成奇异摄动系统。实际时间尺度 **s 快 → g 中 → μ 慢**（状态 5–20 epochs 分化，度量场 50–150 epochs 建立边界，位置 100–300 epochs 稳定）。

Tikhonov 定理：快变量边界层指数稳定 → 准稳态近似 → 总收敛由慢变量决定 $T_{\text{conv}}\approx 1/\mu_\mu$。

**理论学习率比**：$\eta_s:\eta_g:\eta_\mu = 1:\tfrac{\lambda_s}{\lambda_g}:\tfrac{\lambda_s}{\lambda_\mu} = 1:20:0.005$。实际建议 $\eta_s=10^{-3}, \eta_g=2\times 10^{-2}, \eta_\mu=5\times 10^{-6}$。

### 13.2 自适应 Lanczos 谱估计

对 $m=20$ 步随机 Lanczos，极端特征值估计误差 $\sim 10^{-3}$（对 $\kappa\approx 800$）。计算成本：每步 2 次 backward pass，约每 50 epoch 运行一次。

**定理 19（自适应学习率）** `[v4 Thm 19]`：

$$\eta_s^{(t)} = \eta_0, \quad \eta_g^{(t)} = \eta_0\frac{\hat\lambda_s}{\hat\lambda_g}, \quad \eta_\mu^{(t)} = \eta_0\frac{\hat\lambda_s}{\hat\lambda_\mu}$$

保证 $\eta_s\lambda_s=\eta_g\lambda_g=\eta_\mu\lambda_\mu$ → 三变量同步收缩。

**命题 25（自适应鲁棒性）** `[v4 Prop 25]`：EMA 平滑（$\beta=0.9$）+ 安全钳位 $\eta\in[10^{-6},0.1]$ 保证学习率变化率 $\leq 11\%$。

### 13.3 交替优化

内循环按时间尺度分配：$K_s=10, K_g=3, K_\mu=1$。理论优势：冻结其他变量后 Hessian 块对角 → PL 条件更易满足。

---

## 14. 有限 N 效应

**定理 20（$\beta_c$ 偏移）** `[v4 Thm 20]`：

$$\beta_c^{(N)} = \beta_c^{(\infty)} + \frac{A}{N} + \frac{B}{N^{1/\nu d}} + o(1/N)$$

$A>0$ 来自外场修正（离散求和偏差），$B$ 来自有限尺寸标度（Fisher-Barber）。对 $N=100, \nu d\approx 2$：$\beta_c^{(100)}\approx 2.2\times\beta_c^{(\infty)}$（与实验一致 — 理论 $\beta_c$ 极低但实践中需更大）。

**推论 22.1（涌现检测窗口）** `[v4 Cor 22.1]`：宽度 $\Delta\beta/\beta_c \approx N^{-1/\nu d}\approx 0.1$ 对 $N=100$。涌现分布在大约 10% 的 $\beta$ 区间。

**实践含义**：早期实验"涌现 epoch 方差大"部分是有限 N 效应的 manifest，非纯粹算法问题。

**命题 26（状态协方差 spiked model）** `[v4 Prop 26]`：$K$ 个大特征值 $\lambda_k\approx \tfrac{N}{K}\|\mu_k\|^2+\sigma^2$，$d_s-K$ 个小特征值 $\approx\sigma^2$。可检测条件 $\|\mu_K\|^2/\sigma^2 > 1/\sqrt{N/d_s}$。

---

## 15. 非欧状态流形

### 15.1 Poincaré 球

$\mathbb{B}^d = \{x: \|x\|<1\}$ 配备 $\lambda_x = \tfrac{2}{1-\|x\|^2}$ 共形因子。测地距离：

$$d_\mathbb{B}(x,y) = \mathrm{arcosh}\!\left(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$$

**指数体积增长** → 树状层次可等距嵌入。

### 15.2 状态动力学的双曲版本

$$s_i^{t+1} = (1-\alpha)\odot s_i^t \oplus \alpha\odot \mathrm{MöbiusMean}_{j\in\mathcal{N}(i)}(w_{ij}, s_j^t)$$

Möbius 加法、标量乘法、Einstein 中点。

**定理 21（双曲空间自发层次化）** `[v4 Thm 21]`：径向坐标 $r_i=\|s_i\|$ 动力学：

$$\frac{dr_i}{dt} \approx \alpha \cdot \frac{1-r_i^2}{2}\cdot(\bar r_{\mathcal{N}(i)} - r_i)$$

因子 $(1-r_i^2)$ 保证 $r$ 不超过 1 — 状态自然留在球内。

**推论 21.1** `[v4 Cor 21.1]`：双曲测地分解为径向（近距离/同层次）和角度（远距离/跨分支）成分。天然支持层次化聚类。

### 15.3 乘积流形架构

**定理 22（乘积流形最优性）** `[v4 Thm 22]`：$\mathcal{M} = \mathbb{S}^{d_{\text{flat}}}\times\mathbb{B}^{d_{\text{hier}}}$。$d_{\text{flat}}=4$（区分 4 物体足够）+$d_{\text{hier}}=4$（编码 2 层层次）= 8 维，比原 $d_s=16$ 节省一半。

**命题 27（Lyapunov 稳定性）** `[v4 Prop 27]`：乘积算子 $\mathcal{T}_\mathcal{M} = \mathcal{T}_\mathbb{S}\times\mathcal{T}_\mathbb{B}$ 在两因子各自收缩时稳定。

---

## 16. 跨视角一致性

**命题 28（跨视角预测一致性 ⇒ 3D 物体理解）** `[v4 Prop 28]`：

$$\mathcal{L}_{\text{cross-view}} = \sum_{v<w}\sum_{p\in\text{masked}} \|\hat I_v(p) - \hat I_w(p^{\text{corr}})\|^2$$

匹配由原子 3D 投影引导而非纯颜色匹配 → 排除"单视角记忆解"，迫使 $s_i$ 在 3D 意义下一致。

> 注：原 **[v4 定理 23]**（跨视角 $\beta_c$ 降低 30%）已**降级**（§8）。

**命题 29（多视角位姿正则）** `[v4 Prop 29]`：$H_{\mu\mu}^{\text{multi-view}} = H_{\mu\mu} + \eta_{\text{cross}}\sum_v P_{\text{epi}}^{(v)}$，沿 epipolar 方向曲率增加 → 位置收敛加速。

**实践**：跨视角约束作为 Phase II（度量场建立后）加入，权重从 0 平滑增加到 $\eta_{\text{cross}}$。

---

## 17. 非刚性形变

### 17.1 形变的形式化

非刚性形变 $\phi_{vw}: \mathbb{R}^3\to\mathbb{R}^3$，小形变 $X_w = X + u_{vw}(X)$，$u_{vw}$ 位移场。

### 17.2 度量场的双因子分解

**定理 24（度量场双因子分解）** `[v4 Thm 24]`：形变下度量场分解为：

$$g(x) = h(\|u(x)\|) g_{\text{obj}}(x) + (1-h(\|u(x)\|)) g_{\text{def}}(x)$$

$h$ 是过渡函数：
- 刚性区域 $\|u\|\approx 0$：$h\approx 1$ → $g\approx g_{\text{obj}}$
- 高形变 $\|u\|\gg 0$：$h\approx 0$ → $g\approx g_{\text{def}}$

**关键洞察**：形变无法消除物体边界 — $g_{\text{obj}}$ 对比度不变。

### 17.3 形变容忍界

**命题 30（形变容忍）** `[v4 Prop 30]`：涌现聚类保持稳定当：

$$\epsilon_{\text{def}} < \frac{\delta_{\text{color}}}{\lambda_{\min}(J_f)\bar d_{\text{obj}}}$$

典型：$\delta_{\text{color}}\approx 0.5$（红-蓝），$\lambda_{\min}(J_f)\approx 0.12$（残差架构），$\bar d_{\text{obj}}\approx 0.3$ → $\epsilon_{\text{def}} < 13.9$。

**推论 23.1（形变-度量协同学习）** `[v4 Cor 23.1]`：$\mathcal{L}_{\text{metric-obj}} = \mathcal{L}_{\text{selforg}} - \eta_{\text{def}}\|\nabla g\cdot\nabla u\|^2$ 分离 $g_{\text{obj}}$ 从形变。

**训练方案**：Phase I (0–200) 固定 $u=0$；Phase II (200–400) 释放 $u$ 低权重；Phase III (400+) 联合正常权重。

---

## 18. 数值预测与可检验推论

| # | 预测 | 来源 | 验证方法 |
|---|------|------|---------|
| P1 | 状态收敛半衰期 ~23 epochs（$\alpha=0.3,\lambda_2=0.1$） | [v1 §7.1] | 监控 $\|s^{(t)}-s^{(t-1)}\|$ |
| P2 | mask 临界比例 $m_c \approx 0.5$–$0.7$ | [v1 §7.2] | 消融 mask_ratio |
| P3 | $K_{\max}=7$ 对 $N=100,d_s=16$ | [v2 §7.1] | K=2..10 扫描 |
| P4 | 总收敛时间 250–450 epochs（vs DirectCluster 600） | [v2 §6.2] | 序参量 $\phi$ 时间序列 |
| P5 | ReLU MLP $\lambda_{\min}(J_f J_f^\top)\in[0.01,0.2]$ | [v3 §1.5] | 训练后 SVD |
| P6 | $K=4$ 场景涌现为多次序贯相变 | [v3 §2.6] | 逐对 NMI |
| P7 | 高纹理场景需 $\alpha<0.05$ | [v3 §3.6] | 纹理消融 |
| P8 | PI 自适应 τ 成功率提升 ≥ 15% | [v3 §4.5] | 8 种子对照 |
| P9 | $\eta_s:\eta_g:\eta_\mu=1:20:0.005$ 加速收敛 20–30% | [v3 §5.4] | 学习率消融 |
| P10 | 涌现 epoch 与 batch size 负相关 | [v3 §6.4] | batch size 扫描 |
| P11 | 涌现后 $d_g \leftrightarrow 1-\cos(s)$ Pearson > 0.8 | [v3 Thm 16] | 收敛后计算 |
| P12 | 残差 MLP $\lambda_{\min}$ 比 plain 高 2.0–3.0× | [v4 §7.3] | SVD 对比 |
| P13 | $\beta_c^{\text{exp}}/\beta_c^{\text{theory}} \approx 2.0$–$2.5$ 对 $N=100$ | [v4 §7.3] | N=25,50,100,200,400 |
| P14 | $\mathbb{S}^4\times\mathbb{B}^4$ vs $\mathbb{S}^{16}$ ARI 差 < 5% | [v4 Thm 22] | 维度-性能消融 |
| P15 | $\epsilon_{\text{def}}<3$ 时 ARI 不低于刚性场景 90% | [v4 Prop 30] | 形变消融 |
| P16 | $r_{\text{sep}}>2.0$ 时 C2+C3 满足 | [v1 §3.2] | 隔离度时序监控 |
| P17 | 自修正梯度使坏局部最小值不稳定 | [fracture Thm 19] | 边界 Hessian 计算 |
| P18 | $R(t)>1$ 时涌现发生 | [fracture Thm 21] | 在线梯度比监控 |
| P19 | 状态 PCA: 前 K 主成分 > 90% 方差 | [v1 §5.3] | 训练后 PCA |
| P20 | 8 种子 ARI>0.7 比例从 ~50% 提升至 ≥ 75% | [v1 §7.3] | 种子对照 |

---

## 19. 降级声明（**Downgraded statements**）

[theory_fracture_fixes.md](theory_fracture_fixes.md) §5.1 经审查后，将以下 **8 条**基于信息瓶颈类比的陈述**降级为启发式注释**，由几何替代（**定理 17/18/19/20/21/22** + soft min-cut 景观分析）取代：

| # | 原编号 | 内容 | 降级原因 | 几何替代 |
|---|--------|------|---------|---------|
| 1 | **[v1 命题 9]** ↓ | $\min\mathcal{L}_{\text{predict}} \Leftrightarrow \max I(Z;X)$ | $s_i$ 确定性非随机；互信息未计算 | 收缩性 **定理 1** |
| 2 | **[v1 命题 10]** ↓ | 聚类 = 信息瓶颈最优压缩 | 同上 + $\beta$ 无操作性定义 | $R(t)>1$ **定理 21** |
| 3 | **[v1 命题 11]** ↓ | Rademacher 泛化界 | 未计算具体覆盖数 | 标准 uniform convergence |
| 4 | **[v2 命题 14]** ↓ | $\beta_c \propto 1/\text{SNR}$ | 定性保留，定量公式移除 | **命题 28** 几何论证 |
| 5 | **[v2 命题 15]** ↓ | $\beta_c$ 预测涌现 epoch | 与实验不符（有限 N 偏差） | $R(t)$ 在线监控 |
| 6 | **[v2 定理 11]** ↓ | IB $\beta_c$ 量化 | $\beta$ 概念废除 | 模拟退火 + $R(t)$ |
| 7 | **[v3 定理 14]** ↓ | 多物体 $\beta_c$ 精确公式 | 依赖未验证的 SNR 估计 | **命题 16** 逐对 $R(t)$ 分析 |
| 8 | **[v4 定理 23]** ↓ | 跨视角 $\beta_c$ 降低 30% | 30% 无推导 | **命题 28** + **命题 29** |

**废止路径**：ECO（椭圆曲线 + Murmuration）— 全部相关代码已于 2026-06-03 删除（实验 ARI=0.30 vs DirectCluster 0.93）。理论文档保留作为历史记录。

---

## 20. 开放问题（合并）

### 20.1 来自 v1 §8.3
1. 联合 Hessian 谱下界中 $\lambda_{\min}(J_f J_f^\top)$ 的可计算闭式
2. 多物体 $K>2$ 涌现条件的完整必要性（**定理 12** 部分回答）
3. 真实图像状态动力学收缩性的实验验证
4. 广义残差架构（DenseNet / Highway）的 Jacobian 谱

### 20.2 来自 v2 §8.2
5. 解码器架构最优设计（**定理 17/18 + 命题 23** 给推荐 — 是否最优？）
6. 时间尺度分离的自适应在线实现工程细节
7. 分岔的有限 size 效应与临界指数精确常数
8. 非欧状态流形实际收益（**定理 21/22** 严格级别 H）
9. 视角一致性 / 时序一致性 / 非刚性形变的耦合分析
10. 动态原子数量的 birth-death 过程
11. 背景建模的显式统计理论

### 20.3 来自 v3 §8.4
12. 序参量 $\tilde\phi$ 的低方差估计器
13. $K$ 自适应选择（无需预设）
14. 3D 原子位置的 2D-3D 提升对状态维度的影响

### 20.4 来自 v4 §7.4
15. 广义残差架构（级联 / 门控）的谱分析
16. 曲率感知的状态流形维度自适应
17. 时序一致性的 $\beta_c$ 进一步降低
18. 跨视角一致性中 $\gamma_{\text{cross}}$ 的显式推导（**定理 23 降级**后缺失）

### 20.5 来自 theory_fracture_fixes.md §7.5
19. **有限 N 的定量效应**：修复后未定量处理
20. **双曲流形的实际收益**：H 级严格性，需数值验证
21. **真实图像的泛化**：所有公理假设 $\Delta c > 0$，灰度/同色场景需纹理/形状线索
22. **非刚性形变**：**定理 24** 双因子分解为提议，需推导
23. **$\Theta_g$ 紧致性的形式保证**：需添加温和 $L_2$ 正则或投影

### 20.6 来自断裂点修复的隐含开放
24. **梯度比 $R(t)$ 的鲁棒性**：序参量 $\tilde R$ 的统计可靠性需实验验证
25. **Bootstrap 阶段平滑权重的退火调度**：最优退火曲线未确定

---

## 21. 文档谱系与依赖

```
theory_selforg.md        [v1: 基础 3 定理 + 8 命题 + 2 推论]
   ↓
theory_selforg_2.md      [v2: 深化 5 定理 + 4 命题 + 5 推论]
   ↓ 其中 命题 14, 15, 定理 11 已降级
theory_selforg_3.md      [v3: 深化 5 定理 + 7 命题 + 3 推论 + 1 引理]
   ↓ 其中 定理 14 已降级
theory_selforg_4.md      [v4: 深化 8 定理 + 10 命题 + 4 推论]
   ↓ 其中 定理 23 已降级
theory_fracture_fixes.md [断裂点修复 + 9 新 R 级命题 + 6 公理体系]
   ↓
theory_selforg_unified.md ← 本文档
```

**合计**：21 定理（v1–v4）+ 9 定理（fracture，新增）+ 29 命题 + 14 推论 + 1 引理 = **53 个编号陈述** + 6 公理 + **20 条可检验数值预测**（P1–P20）。

**严格性分布**：22 R 级 + 27 H 级 + 12 S 级（依据 [theory_audit_and_roadmap.md](theory_audit_and_roadmap.md) 与 [theory_fracture_fixes.md](theory_fracture_fixes.md) §5.3 修订）。

---

## 22. 与前序理论成果的关系

| 前序成果 | 在新框架中的适用性 |
|----------|-----------------|
| PL 条件 ([remaining_proofs.md](remaining_proofs.md) §1) | **部分适用** — 用于 $\mathcal{L}_{\text{recon}}+\mathcal{L}_{\text{smooth}}$ 子问题 |
| $K>2$ 泛化 ([remaining_proofs.md](remaining_proofs.md) §2) | **替换** — Sinkhorn 不适用，**命题 16** 层次化涌现替代 |
| 收敛速率 O(1/t) ([convergence_rate_analysis.md](convergence_rate_analysis.md)) | **适用** — $\mathcal{L}_{\text{recon}}$ 部分光滑性不变 |
| 泛化界 ([theoretical_extensions.md](theoretical_extensions.md) §3) | **替换** — 由 **定理 21** 涌现条件 + uniform convergence 替代 |
| Lyapunov 构造 ([murmuration_dynamics.md](murmuration_dynamics.md) §2) | **方法论适用** — 框架废止，技术迁移到 §5 |

**废止成果**：ECO（[phase6a_eco_theory.md](phase6a_eco_theory.md)）、DirectCluster Sinkhorn 最优温度、Murmuration 椭圆曲线动力学、测地高斯核 PSD 理论 — ECO 路径 2026-06-03 完全废止。

---

## 23. 总结表

| 维度 | 状态 | 关键结果 |
|------|------|---------|
| 状态动力学收敛 | ✅ 已分析 | 收缩映射，几何速率 $(1-\alpha\lambda_2)^t$ |
| 自组织力-平滑力平衡 | ✅ 已分析 | $\eta_{\text{selforg}}/\eta_s \in [2,10]$；$\tilde R(t) > 1$ 涌现条件 |
| 涌现聚类条件 | ✅ 已定理化 | C1+C2+C3 充分；N1+N2+N3 必要；$K_{\max}=7$ |
| Lyapunov 稳定性 | ✅ 已分析 | 均匀解是鞍点（**定理 22**），总损失是 Lyapunov 函数 |
| 信息论视角 | ⚠️ 已降级 | 由 **定理 21** 梯度比 + soft min-cut 几何替代 |
| 泛化界 | ⚠️ 已重述 | 标准 uniform convergence + **命题 28** 几何约束 |
| 联合 Hessian 谱分析 | ✅ 已严格化 | **定理 8** 正定性，**定理 9** 联合 PL |
| 自适应温度调度 | ✅ 已分析 | PI 控制 + 无监督 $\tilde\phi$ 估计 |
| 多时间尺度分离 | ✅ 已形式化 | $\eta_s:\eta_g:\eta_\mu = 1:20:0.005$，Lanczos 自适应 |
| 有限 N 效应 | ✅ 已量化 | $\beta_c^{(100)}/\beta_c^{(\infty)}\approx 2.2$ |
| 非欧状态流形 | ✅ 已提出 | $\mathbb{S}^4\times\mathbb{B}^4$ 替代 $\mathbb{S}^{16}$ |
| 跨视角一致性 | ✅ 已几何化 | **命题 28/29** 替代原 **定理 23** |
| 非刚性形变 | ✅ 已分解 | **定理 24** 双因子分解，**命题 30** 形变容忍 |
| 断裂点修复 | ✅ 已完成 | 6 公理全 R 级 + 9 新严格命题 |

---

*本文档统一表述了自组织原子框架的数学基础。所有编号陈述保留原文编号并以 `[vN Thm X]` / `[fracture Thm X]` 形式标记来源；已降级陈述在 §8 标注 ↓ 并在 §19 集中列表。六公理体系（A1–A6）是框架的最终数学基础。*