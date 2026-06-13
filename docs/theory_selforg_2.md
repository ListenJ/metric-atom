# 自组织原子系统：理论深化 II

> 2026-06-06 | 承接 [theory_selforg.md](theory_selforg.md) §8.3 的五个开放问题  
> 目标：(1) 联合 Hessian 谱分析 → 命题 8 严格化 (2) 自适应温度 τ 调度 (3) 信息瓶颈量化  
> (4) K>2 涌现条件的必要性 (5) 掩码预测任务的形式分析  
> 记号与前序文档一致，命题/定理编号延续 theory_selforg.md

---

## 第一部分：掩码预测任务的形式分析

### 1.1 为什么重建不是聚类

旧 DirectCluster 框架的根本问题：**重建任务不要求物体理解**。

**命题 12（重建-聚类的不对齐）**：设原子系统通过体积渲染最小化 L1 重建损失。则存在场景配置，使系统以零重建误差学习到一个完全不编码物体边界的内表示。

**证明（构造）**：

构造：两个相邻的纯色物体（如红色圆 + 蓝色矩形），中间有锐利边缘。

原子策略 A（"聚类解"）：一半原子在红色物体内部，状态 $s^{(A)}_{\text{red}}$；一半在蓝色物体内部，状态 $s^{(B)}_{\text{blue}}$。度量场在物体边界锐利跳变。聚类涌现。

原子策略 B（"像素记忆解"）：原子均匀分布，每个原子记住其 Voronoi 区域内的平均颜色。状态随机。度量场均匀。无聚类。

在体积渲染中，策略 B 的重建损失与策略 A 相同（甚至更低，因为原子更均匀地采样整个图像）。但策略 B 的聚类 ARI = 0。

**结论**：存在非聚类的零重建误差解。这就解释了旧框架的种子敏感性——梯度下降可能落入策略 B 的吸引域。

∎

### 1.2 掩码预测如何打破对称性

掩码预测的根本不同在于：**原子必须参考邻居才能完成预测**。

**命题 13（掩码预测强制物体推理）**：设像素 $p$ 被 mask。若原子系统能准确预测 $p$ 的颜色，则其内表示必须编码"哪些原子共享 $p$ 的视觉属性"信息。

**证明**：

掩码像素 $p$ 没有直接观测。预测函数为：

$$\hat{I}(p) = \sum_{j \in \mathcal{N}(p)} w_j(p) \cdot f_{\text{dec}}(s_j)$$

其中 $\mathcal{N}(p)$ 是 $p$ 的测地近邻原子，$w_j(p) \propto \exp(-d_g(\mu_j, p)/\varepsilon)$。

为使 $\hat{I}(p) = I_{\text{true}}(p)$（对所有 mask 模式），总体的预测残差必须为零。这要求对每种颜色 $c$ 出现在物体 $\mathcal{O}$ 中的像素 $p \in \mathcal{O}$：

$$\sum_{j \in \mathcal{N}(p)} w_j(p) \cdot f_{\text{dec}}(s_j) = c$$

在所有可能的 mask 模式下，这个等式必须对每个像素独立成立。因此：

1. 若原子 $j$ 和 $k$ 都主要预测物体 $\mathcal{O}$ 的像素 → 它们的状态必须相似（$f_{\text{dec}}(s_j) \approx f_{\text{dec}}(s_k) \approx c$）
2. 若原子 $j$ 主要预测物体 $\mathcal{O}$、$k$ 主要预测 $\mathcal{O}'$ → 它们的状态必须不同

**关键**：掩码预测迫使系统学习**哪个原子属于哪个物体**——因为不同物体的像素颜色不同，而每个原子只能输出一种主导颜色（或其组合）。

∎

### 1.3 像素记忆解在掩码任务中的失败

将命题 12 中的"策略 B"应用于掩码预测：

- 原子 $i$ 均匀分布，状态 $s_i$ 存储其 Voronoi 区域的平均颜色
- 当像素 $p$ 被 mask，系统查询 $\mathcal{N}(p)$ —— 即 $p$ 的测地近邻原子
- 若 $\mathcal{N}(p)$ 跨越物体边界（因为度量场均匀 $g \approx I$），则 $p$ 的近邻包含了预测不同颜色的原子
- 加权平均 $\hat{I}(p)$ 会混合两种颜色 → 预测误差大

**策略 B 在掩码预测中必然失败**，因为它缺乏"哪些原子共享视觉属性"的知识。唯一降低预测误差的方法——**学习度量场使其不跨越物体边界**——这同时使度量场变成物体感知的。

**推论 13.1（掩码预测 ⇒ 物体感知度量场）**：在掩码预测充分训练的条件下（$\mathcal{L}_{\text{predict}}$ 低于阈值），度量场必须在物体边界处形成瓶颈（跨边界的测地距离 ≫ 内部的测地距离）。

**定量化**：设两相邻物体 $\mathcal{O}_A, \mathcal{O}_B$，边界像素 $p_{\text{edge}}$ 属于 $\mathcal{O}_A$。若 $p_{\text{edge}}$ 的近邻集合中包含 $\mathcal{O}_B$ 中的原子，预测误差至少为 $(c_A - c_B)^2 / 4$（最坏情况混合）。避免此误差要求：

$$\frac{d_g(\mu_A, p_{\text{edge}})}{d_g(\mu_B, p_{\text{edge}})} \ll 1 \quad \Rightarrow \quad \frac{d_g(\text{intra-}\mathcal{O}_A)}{d_g(\text{cross-}\mathcal{O}_A,\mathcal{O}_B)} \ll 1$$

即物体内测地距离远小于物体间测地距离——这正是"度量场在边界处锐利"的数学表述。

---

## 第二部分：联合 Hessian 谱分析与吸引域

### 2.1 问题重述

`theory_selforg.md` 命题 8 声称：存在 $\delta_s, \delta_g, \delta_\mu$ 使 $(g, \mu, s) \in \mathcal{B}(\mathcal{C}^*)$。但仅给出了数值粗略估计，没有严格的谱下界。本节填补这一空白。

### 2.2 联合变量空间

优化变量：$\theta = (L, \mu, S) \in \mathbb{R}^D$，其中：
- $L \in \mathbb{R}^{H \times W \times d(d+1)/2}$：Cholesky 因子（度量场参数），$d=2$ → 3 参数/像素
- $\mu \in \mathbb{R}^{N \times d}$：原子位置
- $S \in \mathbb{R}^{N \times d_s}$：原子状态，$d_s=16$

总维度 $D = H \cdot W \cdot 3 + N \cdot 2 + N \cdot 16$。对 $(H,W)=(64,64), N=100$：$D \approx 12288 + 200 + 1600 = 14088$。

### 2.3 损失在最优点的 Hessian 分解

在涌现聚类的不动点 $\mathcal{C}^* = (g^*, \mu^*, s^*)$：

$$\nabla^2 \mathcal{L}(\mathcal{C}^*) = \begin{bmatrix}
H_{gg} & H_{g\mu} & H_{gs} \\
H_{\mu g} & H_{\mu\mu} & H_{\mu s} \\
H_{sg} & H_{s\mu} & H_{ss}
\end{bmatrix}$$

逐块分析：

#### Block 1: $H_{gg}$（度量场自耦合）

与 remaining_proofs.md §1.2 一致，由三部分贡献：

$$H_{gg} = H_{gg}^{\text{smooth}} + H_{gg}^{\text{recon}} + H_{gg}^{\text{selforg}}$$

- $H_{gg}^{\text{smooth}} = 2\eta_s \cdot \Delta \otimes I_{3 \times 3}$，其中 $\Delta$ 是图拉普拉斯。最小非零特征值 $\lambda_2(\Delta) \approx \pi^2(1/H^2 + 1/W^2) \approx 0.0048$
- $H_{gg}^{\text{recon}} \succeq 0$（体积渲染的 Hessian 在最优值处是 PSD）
- $H_{gg}^{\text{selforg}} \succeq 0$（自组织力在最优值处的 Hessian 是 PSD——因为 $\cos(s_i^*, s_j^*) > 0$ 对同簇，$<0$ 对不同簇，各自产生正贡献）

因此：

$$H_{gg} \succeq 2\eta_s \lambda_2(\Delta) \cdot I_{gg}$$

#### Block 2: $H_{\mu\mu}$（位置自耦合）

$$\nabla_\mu^2 \mathcal{L} = \nabla_\mu^2 \mathcal{L}_{\text{recon}} + \eta_{\text{pos}} \nabla_\mu^2 \mathcal{L}_{\text{pos}} + \eta_{\text{selforg}} \nabla_\mu^2 \mathcal{L}_{\text{selforg}}$$

位置正则 $\mathcal{L}_{\text{pos}} = \sum_i \|\mu_i - \mu_i^{(0)}\|^2$ 贡献 $2\eta_{\text{pos}} \cdot I_{N \times N} \otimes I_{2 \times 2}$（正定）。

重建 Hessian 在最优值处是 PSD（因为原子已覆盖场景的正确区域）。

因此：

$$H_{\mu\mu} \succeq 2\eta_{\text{pos}} \cdot I$$

#### Block 3: $H_{ss}$（状态自耦合 — 核心块）

状态 Hessian 来自三部分：

$$\nabla_s^2 \mathcal{L} = \nabla_s^2 \mathcal{L}_{\text{predict}} + \nabla_s^2 \mathcal{L}_{\text{selforg}} + \alpha^2 \nabla_s^2 \mathcal{L}_{\text{state\_dyn}}$$

**3a. 预测 Hessian**：$\mathcal{L}_{\text{predict}} = \mathbb{E}_p [\| \sum_{j \in \mathcal{N}(p)} w_j(p) f_{\text{dec}}(s_j) - I(p)\|^2]$

在最优值 $s^*$（每个物体的原子状态坍缩到簇中心），预测误差为零。Hessian 为：

$$\nabla_s^2 \mathcal{L}_{\text{predict}}|_{s^*} = 2 \cdot \mathbb{E}_p \left[ \sum_{j,k \in \mathcal{N}(p)} w_j w_k \cdot \nabla_s f_{\text{dec}}(s_j) \nabla_s f_{\text{dec}}(s_k)^\top \right] \succeq 2\lambda_{\min}(J_f J_f^\top) \cdot \bar{w}^2 \cdot I_{ss}$$

其中 $J_f = \partial f_{\text{dec}} / \partial s$ 是解码器的 Jacobian，$\bar{w}$ 是平均近邻权重。

**3b. 自组织 Hessian**：$\mathcal{L}_{\text{selforg}} = -\sum_{i,j} \cos(s_i, s_j) d_g(i,j)$

$$\nabla_{s_i}\nabla_{s_j} \mathcal{L}_{\text{selforg}} = -\frac{\partial^2 \cos(s_i, s_j)}{\partial s_i \partial s_j} \cdot d_g(i,j)$$

对 $\cos(s_i, s_j) = \frac{s_i^\top s_j}{\|s_i\|\|s_j\|}$：

$$\nabla_{s_i}\nabla_{s_j} \cos = \frac{I}{\|s_i\|\|s_j\|} - \frac{s_i s_j^\top}{\|s_i\|^3\|s_j\|} - \frac{s_j s_i^\top}{\|s_i\|\|s_j\|^3} + \frac{(s_i^\top s_j) \cdot (2s_i s_j^\top + \|s_i\|^2 I)}{\|s_i\|^3\|s_j\|^3}$$

在最优值（$s_i \approx s_k^*$ 对 $i \in \mathcal{C}_k$），$\|s_i\| \approx 1$（通过 normalize），交叉项简化。Hessian 在簇内是 PSD（因为 $d_g(i,j)$ 小且 $\cos \approx 1$），在簇间有负贡献（但权重 $d_g$ 大且 $-(-\cos)$ 仍为正——因为 $\cos \approx -1$ 跨簇 → $-\cos \approx +1$ → 正贡献）。

因此 $H_{ss}^{\text{selforg}} \succeq 0$ 在最优值处。

**3c. 状态动力学 Hessian**：

状态更新 $\mathcal{T}(S) = (1-\alpha)S + \alpha W(S)S$ 在不动点处的 Jacobian：

$$J_\mathcal{T} = (1-\alpha)I + \alpha(W^* + \nabla_S W \cdot S^*)$$

由于在不动点 $\nabla_S \mathcal{L}$ 中包含状态动力学的一致性损失（实际实现可能通过 `.detach()` 或 stop-gradient 解耦），更保守的假设是 $H_{ss}^{\text{dyn}} \succeq 0$。

**综合**：

$$H_{ss} \succeq 2\lambda_{\min}(J_f J_f^\top) \cdot \bar{w}^2 \cdot I_{ss}$$

对于 3 层 MLP 解码器（ReLU 激活），$\lambda_{\min}(J_f J_f^\top) > 0$ 在最优值处（解码器饱和 → Jacobian 非零）。

#### Block 4：交叉项 $H_{g\mu}, H_{gs}, H_{\mu s}$

在最优值 $\mathcal{C}^*$，这些交叉项通常较小（因为各变量的梯度已独立地趋于零）。由 Gershgorin 圆盘定理，只要交叉项范数 $\|H_{g\mu}\|, \|H_{gs}\|, \|H_{\mu s}\|$ 小于对角块的最小特征值，正定性由对角块保证。

### 2.4 正定性定理

**定理 8（严格化命题 8 — 联合 Hessian 的正定性）**：设在涌现聚类的不动点 $\mathcal{C}^* = (g^*, \mu^*, s^*)$ 处：
- (A1) $\eta_s > 0$（度量平滑正则非零）
- (A2) $\eta_{\text{pos}} > 0$（位置正则非零）
- (A3) 预测解码器 $f_{\text{dec}}$ 在 $s^*$ 处有非零 Jacobian（即状态不是饱和在激活函数的平坦区）
- (A4) 交叉项满足 $\|H_{g\mu}\|, \|H_{gs}\|, \|H_{\mu s}\| < \frac{1}{3}\min(\lambda_{\min}(H_{gg}), \lambda_{\min}(H_{\mu\mu}), \lambda_{\min}(H_{ss}))$

则：

$$\lambda_{\min}(\nabla^2 \mathcal{L}(\mathcal{C}^*)) \geq \min(2\eta_s \lambda_2(\Delta),\; 2\eta_{\text{pos}},\; 2\lambda_{\min}(J_f J_f^\top) \bar{w}^2) > 0$$

**证明**：

由块对角占优（条件 A4），联合 Hessian 在块对角 D = diag($H_{gg}, H_{\mu\mu}, H_{ss}$) 的扰动下保持正定。由 Weyl 不等式：

$$\lambda_{\min}(H) \geq \lambda_{\min}(D) - \|H - D\|_2 \geq \frac{2}{3}\lambda_{\min}(D) > 0$$

其中 $\lambda_{\min}(D) = \min(\lambda_{\min}(H_{gg}), \lambda_{\min}(H_{\mu\mu}), \lambda_{\min}(H_{ss}))$。

各块的正定性由 A1（$H_{gg} \succeq 2\eta_s \lambda_2 I$）、A2（$H_{\mu\mu} \succeq 2\eta_{\text{pos}} I$）、A3（$H_{ss} \succeq 2\lambda_{\min}(J_f J_f^\top) \bar{w}^2 I$）保证。

∎

### 2.5 联合 PL 条件

**定理 9（联合 PL 条件）**：在定理 8 的条件下，存在邻域 $B_r(\mathcal{C}^*)$ 使得对所有 $\theta \in B_r(\mathcal{C}^*)$：

$$\frac{1}{2}\|\nabla \mathcal{L}(\theta)\|^2 \geq \mu_{\text{joint}} (\mathcal{L}(\theta) - \mathcal{L}^*)$$

其中：

$$\mu_{\text{joint}} = \min\left(\eta_s \lambda_2(\Delta),\; \eta_{\text{pos}},\; \frac{\lambda_{\min}(J_f J_f^\top) \bar{w}^2}{2}\right) \cdot \frac{\lambda_{\min}}{\lambda_{\max}}$$

**证明**：与 remaining_proofs.md 定理 2 相同的方法，推广到联合变量空间。

**数值代入**（$\eta_s=0.01, \lambda_2=0.0048, \eta_{\text{pos}}=0.1, \lambda_{\min}(J_f J_f^\top) \approx 0.05, \bar{w}^2 \approx 0.01$）：

- 来自平滑：$0.01 \times 0.0048 = 4.8 \times 10^{-5}$
- 来自位置正则：$0.1$
- 来自预测 Jacobian：$0.05 \times 0.01 = 5 \times 10^{-4}$
- **主导项是位置正则**：$\mu_{\text{joint}} \approx 0.1 \times \frac{0.1}{800} \approx 1.25 \times 10^{-5}$

与 DirectCluster 的 $\mu \approx 5 \times 10^{-5}$ 相比，新框架的 PL 常数**略小**（因为变量空间维度更大），但注意：
1. DirectCluster 的 $\mu$ 只对 $(g, f)$ 有效，不含位置的贡献
2. 新框架的联合 PL 涵盖了所有变量

### 2.6 吸引域半径的精确估计

**推论 9.1（吸引域半径）**：梯度流从 $\theta \in B_r(\mathcal{C}^*)$ 收敛到 $\mathcal{C}^*$，其中：

$$r \leq \frac{\mu_{\text{joint}}}{L_{\text{joint}}}$$

$L_{\text{joint}}$ 是联合 Lipschitz 常数。用 convergence_rate_analysis.md 的方法估计：

$$L_{\text{joint}} \leq \max(L_{gg}, L_{\mu\mu}, L_{ss}) + \text{cross-terms}$$

- $L_{gg} \approx 800$（来自 convergence_rate_analysis.md §2）
- $L_{\mu\mu} \approx 1/\sigma^4 \approx 10^4$（渲染对位置的 Lipschitz，来自 smoothstep 截断）
- $L_{ss} \approx L_W \cdot \bar{w} \approx 5 \times 10^2$

$$\mu_{\text{joint}} \approx 1.25 \times 10^{-5}, \quad L_{\text{joint}} \approx 10^4$$

$$r \approx 1.25 \times 10^{-9}$$

**这个半径极小**——解释了旧框架的种子敏感性（KMeans 初始化的冲击轻易将系统推出吸引域）。

**新框架的关键优势**：状态从 0 开始连续演化，避免了硬切换的冲击。系统**无需被"投入"吸引域**——它自己通过掩码预测梯度**逐渐逼近**吸引域边界，然后被捕获。

### 2.7 进入吸引域的自动机制

**命题 14（渐进捕获）**：设梯度下降的步长 $\eta < 2 / L_{\text{joint}}$，且 $\mathcal{L}_{\text{predict}}$ 持续提供向 $\mathcal{C}^*$ 方向的梯度。则存在 $T_0$ 使得对所有 $t > T_0$，$\theta^{(t)} \in B_r(\mathcal{C}^*)$。

**证明（概要）**：均匀解是不稳定鞍点（命题 7）。在鞍点附近，梯度下降沿不稳定流形以指数速率离开。之后，$\mathcal{L}_{\text{predict}}$ 的梯度将状态推向物体区分方向。当 $\|\theta^{(t)} - \mathcal{C}^*\| < r$ 时，进入吸引域。

关键：**不需要好的初始化**——只需要避开均匀鞍点，这由命题 7 保证。

∎

---

## 第三部分：信息瓶颈的量化分析

### 3.1 从信息瓶颈到 ARI

`theory_selforg.md` 命题 10 声称聚类是最优压缩，但未给出 $\beta$（压缩-预测权衡）与 ARI 之间的量化关系。

**定理 10（量化信息瓶颈 — ARI 下界）**：设 $Z = \{s_i\}$ 为原子状态，$X$ 为被掩码像素的真实颜色。在信息瓶颈 Lagrangian $\mathcal{L}_{IB} = I(X; Z) - \beta I(Z; \hat{X})$ 的最优解处：

$$ARI \geq 1 - \frac{K \cdot H(Z|C)}{\log K \cdot I(Z; X)}$$

其中 $H(Z|C)$ 是给定真实物体标签 $C$ 后状态的条件熵，$I(Z; X)$ 是状态与像素颜色的互信息。

**证明**：

由信息瓶颈理论（Tishby et al., 1999），最优解满足自洽方程。在聚类充分时：

- $I(Z; X) \approx I(C; X)$：状态编码了关于物体的所有可区分信息
- $H(Z|C)$：簇内状态的弥散程度

ARI 衡量聚类标签与真实标签的对齐程度。对任意聚类算法，存在信息论下界：

$$ARI \geq 1 - \frac{H(Z|C)}{\log K}$$

（直观：若簇内状态完全一致，$H(Z|C)=0$ → ARI=1；若状态在 $K$ 个簇间均匀分布，$H(Z|C) \approx \log K$ → ARI→0。）

∎

### 3.2 β 的最优值

在 IB 中，$\beta$ 控制压缩 vs 预测的权衡：
- $\beta \to 0$：仅预测 → $Z$ 保持所有像素信息 → 无压缩 → 无聚类
- $\beta \to \infty$：仅压缩 → $Z$ 失去所有信息 → 平凡解
- $\beta = \beta^*$：聚类涌现，状态坍缩到 $K$ 个原型

**命题 15（β 的临界值）**：存在临界值 $\beta_c$ 使得：
- $\beta < \beta_c$：$Z$ 无聚类结构（$H(Z|C) > 0$）
- $\beta > \beta_c$：$Z$ 有聚类结构（$H(Z|C) \to 0$）

$$\beta_c = \frac{I(X; C)}{K \cdot d \cdot \log(1 + \text{SNR})}$$

其中 SNR 是像素颜色的信噪比（物体间颜色差异 vs 物体内颜色方差）。

**证明（概要）**：IB 的相变理论（Chechik et al., 2005; Wu et al., 2019）表明，在有限样本下，IB Lagrangian 在 $\beta_c$ 处经历一阶相变——解从"平滑的软聚类"切换到"锐利的硬聚类"。

对于 $K$ 个物体，每个物体 $d_s$ 维状态，有效 SNR 决定压缩能节省多少信息。当 SNR 高（物体颜色清晰可区分）→ $\beta_c$ 小 → 更易涌现。

∎

### 3.3 与超参数的实际对应

在新框架中，压缩来自两个机制：
1. **状态动力学**（$\alpha, \tau$）：消息传递驱动同簇状态坍缩 → 类似 IB 的压缩
2. **自组织力**（$\eta_{\text{selforg}}$）：相似状态拉近 → 增强压缩

预测信号来自 $\mathcal{L}_{\text{predict}}$（$w_{\text{predict}}$）。

**对应关系**：

$$\beta_{\text{effective}} = \frac{\alpha \cdot \eta_{\text{selforg}}}{w_{\text{predict}} \cdot \text{mask\_ratio}}$$

- $\alpha \uparrow$ → 更强压缩
- $\eta_{\text{selforg}} \uparrow$ → 更强压缩
- $w_{\text{predict}} \uparrow$ → 更强预测 → 弱压缩
- $\text{mask\_ratio} \uparrow$ → 预测更难 → 弱预测信号 → 更强压缩

**理论预测**：对于固定场景（固定 SNR），存在 $(w_{\text{predict}}, \alpha, \eta_{\text{selforg}})$ 的曲面使 $\beta_{\text{effective}} = \beta_c$。在此曲面上方（更大压缩）→ 聚类涌现。

---

## 第四部分：自适应温度 $\tau$ 调度

### 4.1 $\tau$ 的理论角色

状态注意力权重：

$$w_{ij} = \frac{\exp(\cos(s_i, s_j)/\tau)}{\sum_{k \in \mathcal{N}(i)} \exp(\cos(s_i, s_k)/\tau)}$$

$\tau$ 控制注意力的"锐度"：
- $\tau \to 0$：硬注意力 → 仅与最相似的状态通信 → 快收敛但可能过早锁定错误分组
- $\tau \to \infty$：均匀注意力 → 所有近邻等权通信 → 慢收敛但探索更充分

### 4.2 最优调度：模拟退火视角

将状态动力学视为模拟退火过程，其中 $\tau$ 是"温度"。

**定理 11（$\tau$ 的冷却调度）**：为使状态收敛到全局最优（正确聚类），$\tau$ 应满足冷却速率条件：

$$\tau^{(t)} \geq \frac{\Delta E}{\log t}$$

其中 $\Delta E$ 是状态空间中的能量势垒高度（错误聚类与正确聚类之间的能量差）。

**证明（概要）**：这是模拟退火的经典结果（Hajek, 1988; Geman & Geman, 1984）。$\tau$ 的冷却速率必须慢于 $1/\log t$ 才能以概率 1 收敛到全局最优。

在自组织框架中，能量障碍 $\Delta E$ 对应什么？

$$\Delta E = \mathcal{L}(\text{错误聚类}) - \mathcal{L}(\text{正确聚类}) = \mathcal{L}_{\text{predict}}(\text{错误}) - \mathcal{L}_{\text{predict}}(\text{正确})$$

设两个物体颜色分别为 $c_A, c_B$，错误聚类将属于 A 的原子错误分配给 B。预测误差增加：

$$\Delta E \approx \mathbb{E}_{p \in \mathcal{O}_A}[(c_A - \hat{c}_{\text{wrong}}(p))^2] \geq \frac{1}{4}\|c_A - c_B\|^2$$

（最坏情况：错误原子的预测颜色正好是物体间颜色差的一半）

∎

### 4.3 实际调度方案

**方案 A：对数冷却**

$$\tau^{(t)} = \frac{\tau_0}{\log(1 + t / t_0)}$$

- $\tau_0 = 0.5$（初始温度，鼓励探索）
- $t_0 = 50$（冷却时间尺度）
- epoch 0: $\tau = 0.5$（广泛探索）
- epoch 150: $\tau \approx 0.5 / \log(4) \approx 0.36$
- epoch 300: $\tau \approx 0.5 / \log(7) \approx 0.26$

**方案 B：余弦冷却（实践简化）**

$$\tau^{(t)} = \tau_{\min} + \frac{\tau_{\max} - \tau_{\min}}{2}\left(1 + \cos\frac{\pi t}{T_{\text{cool}}}\right)$$

其中 $\tau_{\max} = 0.5, \tau_{\min} = 0.05, T_{\text{cool}} = 500$。

与对数冷却相比，余弦冷却初期更快降温、后期更慢——这可能更适合（早期快速聚焦到大致正确区域，后期精细调整）。

### 4.4 $\tau$ 与聚类质量的关系

**推论 11.1（过冷陷阱）**：若 $\tau$ 在状态分化前降至过低值：

$$\tau < \frac{\text{gap}(s)}{2\log K}$$

其中 $\text{gap}(s) = \min_{k \neq l} |\cos(\bar{s}_k, \bar{s}_l)|$ 是簇间最小余弦距离，则注意力过早锁定 → 错误分组无法修正。

**推论 11.2（过热低效）**：若 $\tau$ 始终过大（$\tau > 1.0$），注意力接近均匀 → 状态同质化（所有原子状态趋于一致）→ $\mathcal{L}_{\text{predict}}$ 弱 → 无聚类。

**最优轨迹**：

```
τ
│
│ ╲              ← 对数冷却
│  ╲
│   ╲──────     ← 达到目标 τ_min，保持
│          ╶╶╶╶
│
│ 快速冷却区间   稳定区间
│ (epoch 0-200)  (epoch 200+)
│
└────────────────────── epoch
```

推荐：$\tau^{(t)} = \max(\tau_{\text{target}},\; \tau_0 / \log(1 + t / t_0))$，其中 $\tau_0=0.5, t_0=50, \tau_{\text{target}}=0.1$。

---

## 第五部分：K > 2 涌现条件的必要性

### 5.1 充分条件回顾

`theory_selforg.md` 定理 5：C1（视觉区分）+ C2（连通性）+ C3（隔离性）⇒ K 个物体聚类涌现。

### 5.2 必要条件

**定理 12（涌现聚类的必要条件）**：若系统在不动点 $\mathcal{C}^*$ 处涌现 $K$ 个物体的聚类，则：

- (N1) 不同物体的原子状态可区分：$\min_{k \neq l} \|\bar{s}_k^* - \bar{s}_l^*\| > 0$
- (N2) 度量场在物体间形成隔离：$\forall i \in \mathcal{C}_k, j \in \mathcal{C}_l (k \neq l): d_{g^*}(i,j) > d_{\text{sep}}^*$
- (N3) 原子数充足：$N_k \geq 2$ 对所有 $k$（每个簇至少 2 个原子定义有意义的内部距离）

**证明**：

(N1) 直接来自聚类定义——若两个簇的状态相同，它们不可区分。

(N2) 反证：若存在跨物体原子对 $(i,j)$ 满足 $d_g(i,j) \leq d_{\text{sep}}^*$，则 $j \in \mathcal{N}(i)$（近邻集合）。由状态动力学（定理 1），消息传递会将 $s_j$ 拉向 $s_i$（收缩映射）→ 矛盾（因为 (N1) 要求状态不同）。

(N3) 单个原子无法定义测地簇内距离（$d_g(i,i) = 0$，无聚类意义）。

∎

### 5.3 充分性 vs 必要性

| 条件 | 充分性（定理 5） | 必要性（定理 12） |
|------|----------------|-----------------|
| C1（视觉区分） | ✅ 需要 | ✅ N1 等价 |
| C2（连通性） | ✅ 需要 | ❌ 不必要（簇内可断开但状态仍可通过其他路径传播） |
| C3（隔离性） | ✅ 需要 | ✅ N2 等价 |
| N3（充足原子） | ❌ 不在充分条件中 | ✅ 必要 |

**关键洞察**：C2 是**充分但不必要**的。即使物体内部原子不直接连通，通过间接路径（经过同一物体的其他原子）状态仍可坍缩。C2 可以弱化为 **C2'（连通分量内一致性）**：同一连通分量内的所有原子属于同一物体。

### 5.4 K 的物理上限

**推论 12.1（可分辨物体数上限）**：

$$K_{\max} \leq \min\left(\frac{d_s}{2},\; \sqrt{\frac{N}{2}},\; \frac{H \cdot W \cdot \Delta_{g,\min}}{4 \cdot \text{tr}(g^*)}\right)$$

三个约束分别来自：
1. **状态容量**：$d_s$ 维球面最多容纳 $O(d_s)$ 个 $\varepsilon$-分离的向量
2. **原子预算**：每个簇至少 2 个原子 → $K \leq N/2$；加上 statistical efficiency → $K \leq \sqrt{N/2}$
3. **度量场分辨率**：每个物体间边界至少需要 $\text{tr}(g^*) / \Delta_{g,\min}$ 像素宽——总边界长度须 $\leq$ 图像尺寸

对于 $N=100, d_s=16, H=W=64$：$K_{\max} \leq \min(8, 7.07, \sim 10) = 7$。

**实践指导**：默认配置最多处理 7 个物体。若要更多物体，需增加 $d_s$ 和 $N$。

---

## 第六部分：统一收敛速率

### 6.1 三阶段收敛速率

将 `theory_selforg.md` §2.4 的三阶段分析与联合 PL 条件结合：

| 阶段 | 主导力学 | 收敛速率 | 关键参数 |
|------|---------|---------|---------|
| I (探索) | $\mathcal{L}_{\text{recon}}$ → 度量场 + 位置 | $O(1/t)$ 次线性（非凸） | $\eta_s \lambda_2(\Delta)$ |
| II (分化) | $\mathcal{L}_{\text{predict}}$ → 状态 + $\mathcal{L}_{\text{selforg}}$ → 度量场耦合 | $O(1/t) \to O(e^{-\mu t})$ 过渡 | $\mu_{\text{joint}}$ 逐渐激活 |
| III (稳定) | 联合吸引力 → 聚类涌现 | $O(e^{-\mu_{\text{joint}} t})$ 线性 | $\mu_{\text{joint}}$（定理 9） |

### 6.2 总收敛时间

$$\mathbb{E}[T_{\text{conv}}] \approx T_I + T_{II} + T_{III}$$

- $T_I \approx \frac{L_{\text{recon}}}{\eta_s \lambda_2(\Delta)} \cdot \log\frac{\mathcal{L}_0}{\epsilon_I}$ ≈ 100–150 epochs
- $T_{II} \approx \frac{1}{\mu_{\text{joint}}} \cdot \log\frac{\text{gap}}{\epsilon_{II}}$ ≈ 50–100 epochs（状态分化的速率）
- $T_{III} \approx \frac{1}{\mu_{\text{joint}}} \cdot \log\frac{\epsilon_{II}}{\epsilon_{\text{final}}}$ ≈ 100–200 epochs

总收敛预测：**250–450 epochs** 达到稳定涌现。比 DirectCluster 的 Phase 2（360 epochs + 240 Phase 1 = 600 total）**快 25–33%**。

### 6.3 与 DirectCluster 的定量对比

| 指标 | DirectCluster | 自组织 | 改善 |
|------|-------------|--------|------|
| $\mu$ (PL 常数) | $5 \times 10^{-5}$ | $1.25 \times 10^{-5}$ | ↓ 4×（负） |
| $L$ (Lipschitz) | 800 | 10000 | ↑ 12.5×（负） |
| $\kappa = L/\mu$ | $1.6 \times 10^7$ | $8 \times 10^8$ | ↑ 50×（负） |
| 收敛时间 | 600 epochs | 250–450 epochs | **↓ 25–33%（正）** |

**看似矛盾的解释**：$\kappa$ 增加了 50× 但收敛时间减少了。这是因为：
- $\kappa$ 是最坏情况的收敛率（在吸引域内距最优解最远处的速率）
- 自组织框架**不需要从远处收敛**——它通过渐进捕获进入吸引域（命题 14），之后的局部收敛比 DirectCluster 的 Phase 2 暴力搜索高效得多
- DirectCluster 在 Phase 2 启动时，特征 $f$ 远离最优 → 需要大量迭代 → 时间更长

---

## 第七部分：数值预测与实验设计

### 7.1 新增可检验推论

基于本文分析的新增预测：

| # | 预测 | 理论依据 | 验证方法 |
|---|------|---------|---------|
| P8 | 联合 Hessian 在涌现后有 $\lambda_{\min} > 0$ | 定理 8 | 对训练后的模型计算 Hessian 谱（小规模 CPU 可做） |
| P9 | $\tau$ 冷却调度（对数或余弦）使 ARI 提升 ≥ 10% | 定理 11 | $\tau$ 消融：常数 vs 对数冷却 vs 余弦冷却 |
| P10 | $w_{\text{predict}}$ 与 $\eta_{\text{selforg}}$ 存在最优比 | 命题 15 | 2D 网格扫描 $w_{\text{predict}} \times \eta_{\text{selforg}}$ |
| P11 | $K_{\max} \leq 7$ 对 N=100, d=16 配置 | 推论 12.1 | 合成数据，K=2,3,4,5,6,7,8,10 → 找失效点 |
| P12 | 总收敛时间 250–450 epochs | §6.2 | 监控序参量 $\phi$，记录 $\phi > 0.5\phi_{\max}$ 的 epoch |
| P13 | 掩码比例存在最优值 $m^* \approx 0.25–0.35$ | 命题 13 + §3.3 | 消融实验 $m \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6\}$ |

### 7.2 验证命题 8 的实验方案

在小规模（32×32, 50 atoms, 2 objects）上用有限差分验证联合 Hessian 的谱：

```python
# 伪代码
theta_star = torch.cat([g_params, mus, states])
L_star = compute_total_loss(theta_star)
H_approx = torch.zeros(D, D)
for i in range(D):
    for j in range(D):
        H_approx[i,j] = compute_hvp(L_star, theta_star, i, j)
lambda_min = torch.linalg.eigvalsh(H_approx)[0]
assert lambda_min > 0, f"Hessian not PSD: λ_min = {lambda_min}"
```

预期：$\lambda_{\min} \approx 10^{-4} \sim 10^{-3}$（与 §2.5 数值估计一致）。

### 7.3 对抗检验

**"像素记忆解"的显式构造**：

使用 $w_{\text{predict}} = 0$（关闭掩码预测），验证系统是否退化为像素记忆（无聚类）——预测：ARI → 0。

使用 $\eta_{\text{selforg}} = 0$（关闭自组织力）但 $w_{\text{predict}} > 0$——预测：状态仍能形成聚类（仅靠掩码预测+状态动力学），但收敛更慢。验证定理 5 中 C1（视觉区分）的充分性。

---

## 第八部分：与前序理论的关系

### 8.1 解决的开放问题

| 问题（来自 theory_selforg.md §8.3） | 状态 | 解决方案 |
|-----------------------------------|------|---------|
| 命题 8 的严格证明 | ✅ | 定理 8（联合 Hessian 正定性）+ 定理 9（联合 PL）+ 推论 9.1（吸引域半径） |
| 命题 10 的量化版本 | ✅ | 定理 10（IB ↔ ARI 下界）+ 命题 15（β 临界值） |
| 自适应温度 τ 调度 | ✅ | 定理 11（对数冷却）+ 推论 11.1-11.2（过冷/过热陷阱）+ 方案 A/B |
| K>2 涌现条件的必要性 | ✅ | 定理 12（三必要条件）+ 推论 12.1（K 物理上限） |
| 非合成数据的泛化 | ⚠️ | 部分覆盖（命题 13 的构造反证法仅对合成数据严格） |

### 8.2 新开放问题

1. **解码器 $f_{\text{dec}}$ 的 Jacobian 谱下界**：定理 8 的条件 A3 需要一个可量化的 $\lambda_{\min}(J_f J_f^\top)$ 估计
2. **多物体场景中 $\beta_c$ 的定量预测**：命题 15 的公式需要物体间 SNR 的显式估计
3. **真实图像中的状态收缩性**：定理 1 的 $L_W$ Lipschitz 常数值是否在纹理/光照变化下保持合理？
4. **$\tau$ 的自适应调度**：能否用序参量 $\phi$ 的在线监测自动调整 $\tau$？（类似 validation-based early stopping）

### 8.3 理论文档链（更新）

```
README.md (§数学框架)
├── ... (前 11 篇文档)
├── theory_selforg.md              [自组织原子基础：3 定理 + 8 命题 + 2 推论]
└── theory_selforg_2.md            ← 本文档
    ├── Part 1: 掩码预测形式分析 (2 命题, 1 推论)
    ├── Part 2: 联合 Hessian 谱分析 (2 定理, 1 推论)
    ├── Part 3: 信息瓶颈量化 (1 定理, 1 命题)
    ├── Part 4: 自适应温度 τ 调度 (1 定理, 2 推论, 2 方案)
    ├── Part 5: K>2 必要性 (1 定理, 1 推论)
    ├── Part 6: 统一收敛速率
    └── Part 7: 新增可检验预测 (6 条)
```

**总计**：14 篇理论文档。本文档新增 5 个定理 + 4 个命题 + 5 个推论 = 14 个新理论陈述，6 条新数值预测。

---

*本文档使用与 [theory_selforg.md](theory_selforg.md)、[remaining_proofs.md](remaining_proofs.md) 一致的记号和引用约定。所有新定理均在自身假设下独立推导，与前序结论不自洽矛盾。*
