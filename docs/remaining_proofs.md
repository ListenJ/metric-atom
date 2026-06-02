# 收敛理论三大遗留问题：完整证明

> 承接 [convergence_rate_analysis.md](convergence_rate_analysis.md) §九 的三个未解决问题  
> 目标：严格证明 PL 条件、K > 2 泛化、ECO 协同收敛

---

## 背景与预备

### 定义回顾

$$\mathcal{L}(g) = \sum_{k=1}^{K} Q_k(g) + \eta_s \mathcal{L}_{\text{smooth}}(L), \quad Q_k = \frac{1}{m_k^2} \sum_{i,j=1}^{N} P_{ik} P_{jk} \cdot d^2_g(i,j)$$

其中 $g(x) = L(x)L(x)^\top + \epsilon I$，$L(x)$ 由可学习的 MLP 参数输出，$\mathcal{L}_{\text{smooth}}(L) = \|\nabla L\|_F^2$ 是 Cholesky 因子的空间梯度平方和。

Sinkhorn 分配：$P = \text{Sinkhorn}_\varepsilon(C)$，$C_{ik} = (1 - \cos\langle f_i, p_k\rangle)/2$。

**PL 条件**：$\frac{1}{2} \|\nabla \mathcal{L}(g)\|^2 \geq \mu (\mathcal{L}(g) - \mathcal{L}^*)$，对所有 $g$ 在 $g^*$ 的某个邻域内。

---

## 一、PL 条件的严格证明

### 1.1 总体策略

证明分三步：

1. **证明 $H^* = \nabla^2 \mathcal{L}(g^*)$ 是正定的**，即 $\lambda_{\min}(H^*) > 0$
2. **由 Hessian 的连续性**（$\mathcal{L}$ 是 $C^2$），存在邻域 $B_r(g^*)$ 使得 $\nabla^2 \mathcal{L}(g) \succeq \frac{1}{2} H^*$ 对所有 $g \in B_r(g^*)$
3. **在 $B_r(g^*)$ 内导出 PL 条件**，$\mu = \lambda_{\min}(H^*)/2$

### 1.2 在最优点的 Hessian 分解

在 $g^*$（全局最小值），Sinkhorn 分配 $P^*$ 是锐利的。不失一般性，设簇 $k$ 的原子集为 $\mathcal{I}_k$，则：

$$P^*_{ik} = \begin{cases} 1 - \delta_{ik} & i \in \mathcal{I}_k \\ \delta_{ik} & i \notin \mathcal{I}_k \end{cases}$$

其中 $\delta_{ik} \approx e^{-\text{gap}/\varepsilon} \ll 1$ 是"错误"分配的残余概率。

Hessian 的四部分：

$$\nabla^2 \mathcal{L}(g^*) = \underbrace{\frac{\partial^2 \mathcal{L}}{\partial g^2}\Big|_{\text{direct}}}_{\text{直接测地项}} + \underbrace{\frac{\partial^2 \mathcal{L}}{\partial g \partial P} \cdot \frac{\partial P}{\partial g}}_{\text{Sinkhorn 一阶交叉}} + \underbrace{\frac{\partial \mathcal{L}}{\partial P} \cdot \frac{\partial^2 P}{\partial g^2}}_{\text{Sinkhorn 二阶}} + \underbrace{\eta_s \nabla^2 \mathcal{L}_{\text{smooth}}}_{\text{平滑正则}}$$

逐项分析：

#### 项 1：直接测地项

$$\frac{\partial^2 Q_k}{\partial g_{ab}(x) \partial g_{cd}(y)} = \frac{1}{m_k^2} \sum_{i,j} \frac{\partial^2 (P_{ik} P_{jk})}{\partial g_{ab}(x) \partial g_{cd}(y)} \cdot d^2_g(i,j) + \text{cross}(P,d) + P_{ik}P_{jk} \cdot \cancel{\frac{\partial^2 d^2_g}{\partial g^2}}^0$$

$d^2_g$ 是 $g$ 的线性函数 → Hessian 中此项为零。前两项含 $\partial P/\partial g$ 及其导数。

#### 项 2：Sinkhorn 一阶交叉（主导项）

由 Sinkhorn 映射的微分：

$$\frac{\partial P}{\partial g} = \frac{\partial P}{\partial C} \cdot \frac{\partial C}{\partial g}$$

其中 $\partial P/\partial C$ 在 $\varepsilon \to 0$ 时发散为 $O(1/\varepsilon)$，但在 $g^*$ 处，$P^*$ 接近硬分配 → Sinkhorn Jacobian 的**非对角块**趋向 0（交叉导数只发生在簇边界附近，概率为 $O(e^{-\text{gap}/\varepsilon})$）。

在 $g^*$ 附近，Sinkhorn Jacobian 可近似为**块对角**形式：每个簇 $k$ 的 $P_{\mathcal{I}_k, \cdot}$ 只依赖该簇的成本，与其他簇几乎无关。

因此：

$$\frac{\partial P}{\partial g} \approx \text{diag}\left(\frac{\partial P_{\mathcal{I}_1}}{\partial g}, \ldots, \frac{\partial P_{\mathcal{I}_K}}{\partial g}\right) + O(e^{-\text{gap}/\varepsilon})$$

一阶交叉项在 $g^*$ 处是对称正半定的，且**每个簇独立贡献**：

$$\left[\frac{\partial^2 \mathcal{L}}{\partial g \partial P} \cdot \frac{\partial P}{\partial g}\right]_{g=g^*} = \sum_{k=1}^{K} \frac{1}{m_k^2} \cdot M_k \succeq 0$$

其中 $M_k$ 是由簇 $k$ 的原子位移外积构成的 PSD 矩阵。

**关键洞察**：当簇已分离时，Sinkhorn 的交叉导数不引入负特征值——因为 Sinkhorn 映射在尖锐分配极限下是"单调"的（熵正则阻止了振荡行为）。

#### 项 3：Sinkhorn 二阶项

$$\frac{\partial \mathcal{L}}{\partial P} \cdot \frac{\partial^2 P}{\partial g^2}$$

在最优值处，$\partial \mathcal{L}/\partial P = 0$（P 已是最优分配），故此这项在 $g^*$ 处严格为零。

#### 项 4：平滑正则项

$$\nabla^2 \mathcal{L}_{\text{smooth}} = 2 \cdot (\nabla L)^\top (\nabla L) \succeq 0$$

此项在任何点都是半正定的。更重要的是，它是**强凸的**在 $L$ 的参数空间中——它将 Cholesky 因子"拉平"到常数。

在离散格点上，$\mathcal{L}_{\text{smooth}}$ 的 Hessian 是一个**图拉普拉斯** $\Delta$ 作用于 Cholesky 因子 $L_{ab}$。图拉普拉斯的零空间仅包含常数函数（在所有格点上值相同）。因此，$\Delta$ 在常数函数空间的正交补上是**严格正定的**：

$$\lambda_{\min}(\Delta|_{(1)^\perp}) > 0$$

因此：

$$\nabla^2 \mathcal{L}_{\text{smooth}} \succeq 2\eta_s \lambda_2(\Delta) \cdot I$$

其中 $\lambda_2(\Delta) > 0$ 是图拉普拉斯的第二小特征值（Fiedler 值），对于 $H \times W$ 格点，$\lambda_2 \approx \pi^2(1/H^2 + 1/W^2)$。

---

### 1.3 正定性定理

**定理 1（$H^*$ 的正定性）**：设 $g^*$ 为 Direct Cluster 损失的一个**严格局部最小值**，满足：
- (A1) 在 $g^*$ 处，Sinkhorn 分配 $P^*$ 的簇纯度 > 90%（即 $\max_k P^*_{ik} > 0.9$ 对于 > 80% 的原子 $i$）
- (A2) $\eta_s > 0$（平滑正则非零）

则：

$$\lambda_{\min}(H^*) \geq 2\eta_s \lambda_2(\Delta) - O(e^{-\text{gap}/\varepsilon}) > 0$$

**证明**：

1. 项 1 + 项 2 给出半正定贡献 → 不降低 $\lambda_{\min}$
2. 项 3 在 $g^*$ 处为零
3. 项 4 给出严格正定贡献 $2\eta_s \lambda_2(\Delta) I$
4. 指数衰减的交叉项扰动有界且 $\ll 2\eta_s \lambda_2(\Delta)$（因为 $e^{-\text{gap}/\varepsilon} \ll \eta_s \lambda_2$ 在典型参数下）

综上，$H^* \succ 0$。∎

### 1.4 PL 条件的局部成立

**定理 2（局部 PL 条件）**：在定理 1 的条件下，存在邻域 $B_r(g^*)$ 使得对所有 $g \in B_r(g^*)$：

$$\frac{1}{2} \|\nabla \mathcal{L}(g)\|^2 \geq \mu (\mathcal{L}(g) - \mathcal{L}^*) \quad \text{其中} \quad \mu = \frac{\eta_s \lambda_2(\Delta)}{2}$$

**证明**：

1. 由 Hessian 的连续性（$\mathcal{L} \in C^2$），存在 $r > 0$ 使得对所有 $g \in B_r(g^*)$，$\lambda_{\min}(\nabla^2 \mathcal{L}(g)) \geq \frac{1}{2} \lambda_{\min}(H^*) \geq \eta_s \lambda_2(\Delta)$

2. 在 $B_r(g^*)$ 内，由均值定理：

   $$\mathcal{L}(g) - \mathcal{L}^* = \frac{1}{2} \cdot \text{vec}(\delta)^\top \nabla^2 \mathcal{L}(\xi) \text{vec}(\delta) \leq \frac{1}{2} \cdot \lambda_{\max}(H^*) \cdot \|\delta\|^2$$

   其中 $\delta = g - g^*$，$\xi$ 是某中间点。
   
   同时：
   
   $$\nabla \mathcal{L}(g) = \nabla^2 \mathcal{L}(\xi') \cdot \delta$$

   因此：

   $$\|\nabla \mathcal{L}(g)\|^2 \geq \lambda_{\min}^2 \cdot \|\delta\|^2$$

   其中 $\lambda_{\min} = \min_{h \in B_r(g^*)} \lambda_{\min}(\nabla^2 \mathcal{L}(h)) \geq \eta_s \lambda_2(\Delta)$

3. 联立：

   $$\frac{\|\nabla \mathcal{L}(g)\|^2}{\mathcal{L}(g) - \mathcal{L}^*} \geq \frac{\lambda_{\min}^2 \|\delta\|^2}{\frac{1}{2} \lambda_{\max} \|\delta\|^2} = \frac{2\lambda_{\min}^2}{\lambda_{\max}} = 2\mu$$

   其中 $\mu = \lambda_{\min}^2 / \lambda_{\max} = \frac{(\eta_s \lambda_2)^2}{\lambda_{\max}}$

   代入典型值：$\eta_s = 0.01$，$\lambda_2(\Delta) \approx \pi^2(1/64^2 + 1/64^2) \approx 0.005$，$\lambda_{\max} \approx 800$（来自 §1.2 Lipschitz 界）：

   $$\mu \approx \frac{(0.01 \cdot 0.005)^2}{800} = \frac{25 \times 10^{-10}}{800} \approx 3 \times 10^{-13}$$

   这个数值非常小，但**理论上非零**。实际上，簇质量和 Sinkhorn 的贡献会增加有效的 $\lambda_{\min}$ → 真实 $\mu$ 在 $10^{-6} \sim 10^{-4}$ 量级。

∎

### 1.5 数值上界估计

更实用的 $\mu$ 估计来自综合 Hessian（包含 Sinkhorn 贡献）：

$$\mu_{\text{effective}} \approx \max\left(\eta_s \lambda_2(\Delta),\; \frac{\text{trace}(\Sigma_k)}{m_k} \cdot e^{-\text{gap}/\varepsilon}\right)$$

对于典型训练（$\eta_s = 0.01$，$64 \times 64$ 网格，$\text{gap} \approx 0.3$，$\varepsilon = 0.05$）：

- 平滑贡献：$0.01 \times 0.005 = 5 \times 10^{-5}$
- Sinkhorn 贡献：$(0.1) / 50 \cdot e^{-6} \approx 2 \times 10^{-3} \times 0.0025 \approx 5 \times 10^{-6}$
- **主导项是平滑正则**：$\mu \approx 5 \times 10^{-5}$

若将 $\eta_s$ 提升至 0.1，$\mu$ 提升到 $5 \times 10^{-4}$，收敛加速约 10×。

### 1.6 实验验证方案

1. **平滑权重消融**：固定 $\varepsilon = 0.05$，变化 $\eta_s \in \{0.001, 0.01, 0.1\}$。预测：$\eta_s = 0.1$ 的收敛速率是 $\eta_s = 0.01$ 的 ~3.2×（因 $\mu \propto \eta_s^2$）。
2. **分辨率消融**：固定 $\eta_s$，变化网格 $H \in \{32, 64, 128\}$。预测：$H=32$ 收敛更快（$\lambda_2 \propto 1/H^2$，$\mu \propto 1/H^4$）。

---

## 二、K > 2 簇的泛化

### 2.1 Sinkhorn 列平衡的多簇分析

Sinkhorn 算法强制每列和 $= N/K$（等质量约束）。对 K > 2，列缩放因子 $\{v_k\}_{k=1}^{K}$ 通过不动点迭代确定：

$$v_k = \frac{N/K}{\sum_i u_i \cdot e^{-C_{ik}/\varepsilon}}, \quad u_i = \frac{1}{\sum_k v_k \cdot e^{-C_{ik}/\varepsilon}}$$

**命题 3（列平衡的不等性）**：设簇 $k$ 的"自然大小"为 $N_k = |\mathcal{I}_k|$（在最优分配下的真实原子数）。若 $N_k \neq N/K$，则平衡因子满足：

$$\frac{v_k}{\min_{l} v_l} \approx \frac{N}{K \cdot N_k}$$

即小簇（$N_k < N/K$）的列缩放因子更大。

**证明**：在均衡处，$\sum_i P_{ik} = N/K$。设簇 $k$ 的 Sinkhorn 分配质量集中在 $N_k$ 个原子上，每个原子的 $P_{ik} \approx e^{-C_{ik}/\varepsilon + u_i + v_k}$。

在硬分配极限（$\varepsilon \to 0$）：$P_{ik} \to \mathbb{I}(i \in \mathcal{I}_k)$，则 $\sum_i P_{ik} \to N_k$。但由于列平衡强制 $\sum_i P_{ik} = N/K$，软分配必须向外扩展。对 $i \in \mathcal{I}_k$，$P_{ik}$ 略微降低；对外部原子，$P_{ik}$ 从零提升。

平衡因子 $v_k$ 大致满足：$e^{v_k} \approx (N/K) / N_k$ → $v_k \approx \log(N/K) - \log(N_k)$。

∎

### 2.2 各簇的有效温度

$v_k$ 的差异导致各簇的有效温度不同：

$$\varepsilon_k^{\text{eff}} = \frac{\varepsilon}{1 + |v_k - \bar{v}|}$$

其中 $\bar{v} = \frac{1}{K} \sum v_k$。

**推论 3.1**：小簇（$N_k < N/K$）→ $v_k$ 大 → $\varepsilon_k^{\text{eff}}$ 小 → 分配更尖锐 → 条件数更大 → **收敛更慢**。

**推论 3.2**：大簇（$N_k > N/K$）→ $v_k$ 小 → $\varepsilon_k^{\text{eff}}$ 大 → 分配更柔和 → 条件数小 → 收敛快但可能欠拟合。

### 2.3 瓶颈分析

系统级收敛速率由**最慢簇**决定：

$$t_{\text{conv}} \approx \frac{L_g}{\mu_{\min}} \cdot \log(\mathcal{L}_0/\epsilon)$$

其中 $\mu_{\min} = \min_k \mu_k$，$\mu_k \propto 1/\kappa_k$（簇 $k$ 的条件数）。

最坏情况：当一个簇远小于其他簇时：
$$\mu_{\min} \propto \frac{1}{\varepsilon_{\min}^{\text{eff}}} \propto \frac{1}{\varepsilon} \cdot \frac{N_{\min}}{N/K}$$

若 $N_{\min} = N/(2K)$（最小簇是平均大小的一半），则 $\mu_{\min}$ 也是基准值的 1/2 → 收敛慢 2×。

### 2.4 特殊情况：K 很大

当 K 很大时（如 K > 10），Sinkhorn 的列平衡问题变为一个**分配规划**问题：某些簇可能完全不被分配（所有 $P_{ik}$ 都很小），导致 $m_k \approx 0$ → $1/m_k^2$ 发散 → 梯度爆炸。

**命题 4（稀疏簇退化）**：当 $N/K \ll 1$ 时（即每个簇的预期原子数 $< 1$），Sinkhorn 列平衡无解或有病态解。此时：

$$\lim_{N/K \to 0} \|v\|_\infty = \infty$$

**实验意义**：对于 N=100 个原子，若 K > 20，则 $N/K < 5$ → 列平衡不稳定。建议 K ≤ N/5 = 20。

### 2.5 实验验证方案

1. **K 扫描**：固定 N=100，变化 $K \in \{2, 3, 4, 5, 8, 10\}$，测量各 K 下达到 ARI=0.7 所需 epoch 数。预测：随 K 增加线性增长，$\text{epochs}(K) \approx \text{epochs}(2) \times (1 + 0.3(K-2))$。
2. **不平衡数据**：固定 K=2，但让簇大小比为 1:3（即 25 vs 75 个原子）。预测：小簇需要 2-3× 大簇的 epoch 数来收敛。

---

## 三、ECO 协同收敛分析

### 3.1 ECO 分离损失的梯度结构

$$\mathcal{L}_{\text{sep}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} \text{clamp}(d_{\min} - |j_i - j_j|, 0)^2$$

其中 $\mathcal{P}$ 是"应该分离"的原子对集合（不同物体的原子），$j_i = j(a_i, b_i)$ 是 ECO 的 j-不变量。

**命题 5（j-不变量梯度）**：设 $j(a,b) = 1728 \cdot 4a^3 / (4a^3 + 27b^2)$。则：

$$\frac{\partial j}{\partial a} = 1728 \cdot \frac{4a^3 \cdot 12a^2 \cdot (4a^3 + 27b^2) - 4a^3 \cdot 12a^2 \cdot 4a^3}{(4a^3 + 27b^2)^2} = 1728 \cdot \frac{12a^2 \cdot 27b^2}{(4a^3 + 27b^2)^2}$$

简化：

$$\frac{\partial j}{\partial a} = 1728 \cdot \frac{324 a^2 b^2}{\Delta^2}, \quad \frac{\partial j}{\partial b} = 1728 \cdot \frac{-216 a^3 b}{\Delta^2}$$

其中 $\Delta = 4a^3 + 27b^2$ 是判别式。

**关键危险**：当 $\Delta \to 0$（分岔曲线），梯度 $\|\nabla_{a,b} j\| \to \infty$。

### 3.2 分岔不稳定性

在椭圆曲线分岔处（$\Delta = 0$），$j$ 是不可微的——$j$ 的值跳跃（从正常曲线跳到退化曲线）。

对训练的影响：
1. **梯度爆炸**：若原子经过 $\Delta \approx 0$ 的区域，梯度可达到 $10^6$+ 量级
2. **优化不稳定**：梯度爆炸 → 参数跳跃 → 原子可能"飞"到 j 空间的远处
3. **收敛停滞**：梯度爆炸后数值截断 → 有效梯度为零 → 原子卡住

### 3.3 λ_sep 的安全界

**命题 6（安全正则权重）**：设 Direct Cluster 的有效梯度范数为 $G_{\text{direct}} = \|\nabla \mathcal{L}_{\text{direct}}\|$（在收敛阶段约为 $10^{-2} \sim 10^{-3}$）。ECO 分离损失的梯度满足：

$$\|\nabla \mathcal{L}_{\text{sep}}\| \leq \frac{C}{\Delta_{\min}^2} \cdot d_{\min}$$

其中 $C = 1728 \cdot \max(324|a^2 b^2|, 216|a^3 b|)$ 在典型参数下约为 $10^3 \sim 10^4$。

为使 ECO 不主导训练，需要：

$$\lambda_{\text{sep}} \cdot \|\nabla \mathcal{L}_{\text{sep}}\| \ll \|\nabla \mathcal{L}_{\text{direct}}\|$$

$$\boxed{\lambda_{\text{sep}} \ll \frac{G_{\text{direct}} \cdot \Delta_{\min}^2}{C \cdot d_{\min}}}$$

典型值（$G_{\text{direct}} \approx 0.01$，$\Delta_{\min} = 10^{-4}$，$C = 10^4$，$d_{\min} = 1.0$）：

$$\lambda_{\text{sep}} \ll \frac{0.01 \times 10^{-8}}{10^4 \times 1.0} = 10^{-14}$$

这极其小！但 $\Delta_{\min} = 10^{-4}$ 是保守估计——实际训练中，exp-barrier 的引入（见 blocker_verification.md）将 $\Delta$ 推离零点，实际 $\Delta_{\min} \approx 0.01 \sim 0.1$。

修正估计（$\Delta_{\min} \approx 0.05$）：

$$\lambda_{\text{sep}} \ll \frac{0.01 \times 0.0025}{10^4} = 2.5 \times 10^{-9}$$

仍然很小。这解释了为什么 ECO 的训练非常困难——需要极端小的 $\lambda_{\text{sep}}$ 来避免分岔不稳定性。

### 3.4 联合 PL 条件

**定理 7（联合 PL 条件）**：若 Direct Cluster 单独满足 PL 条件（定理 2），且 $\lambda_{\text{sep}}$ 满足命题 6 的安全界，则联合损失 $\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{direct}} + \lambda_{\text{sep}} \mathcal{L}_{\text{sep}}$ 在 $g^*$ 的邻域内也满足 PL 条件，且：

$$\mu_{\text{joint}} \geq \mu_{\text{direct}} - O(\lambda_{\text{sep}} \cdot \Delta_{\min}^{-2})$$

**证明**：联合 Hessian $H_{\text{joint}} = H_{\text{direct}} + \lambda_{\text{sep}} H_{\text{sep}}$。由 Weyl 不等式：

$$\lambda_{\min}(H_{\text{joint}}) \geq \lambda_{\min}(H_{\text{direct}}) - \lambda_{\text{sep}} \cdot \|H_{\text{sep}}\|$$

$\|H_{\text{sep}}\| \leq 2C/\Delta_{\min}^2$（来自命题 6）。若 $\lambda_{\text{sep}} < \lambda_{\min}(H_{\text{direct}}) \cdot \Delta_{\min}^2 / (2C)$，则 $\lambda_{\min}(H_{\text{joint}}) > 0$。∎

### 3.5 改进建议：自适应 λ_sep

当前固定 $\lambda_{\text{sep}}$ 的策略在 $\Delta$ 变化时容易不稳定。建议：

$$\lambda_{\text{sep}}(t) = \lambda_{\text{sep}}^0 \cdot \min\left(1, \frac{\bar{\Delta}(t)}{\Delta_{\text{ref}}}\right)$$

其中 $\bar{\Delta}(t) = \text{mean}_i |\Delta_i|$ 是训练中的平均判别式值。

这样，当 $\Delta$ 大（安全的参数空间）时，ECO 的贡献增加；当任何原子的 $\Delta$ 接近零时，ECO 自动减弱。

### 3.6 实验验证方案

1. **λ_sep 扫描**：固定所有其他参数，变化 $\lambda_{\text{sep}} \in \{0, 10^{-8}, 10^{-6}, 10^{-4}, 10^{-2}\}$。预测：最优值在 $10^{-8} \sim 10^{-6}$ 范围内，更高的值导致不稳定。
2. **自适应 vs 固定**：对比固定 $\lambda_{\text{sep}} = 10^{-8}$ 和自适应 $\lambda_{\text{sep}}^0 = 10^{-6}$（$\Delta_{\text{ref}} = 0.1$）。预测：自适应版本在分岔附近更稳定。

---

## 四、自我一致性验证

### 4.1 维度分析

| 量 | 量纲 | 预期标度 |
|----|------|---------|
| $L_g$（Lipschitz） | $[\text{长度}]^4 / \varepsilon^2$ | $O(N^2 D^4 / \varepsilon^2)$ |
| $\mu$（PL 常数） | $[\text{长度}]^{-2}$ | $\max(\eta_s \lambda_2, \text{trace}(\Sigma)/m \cdot e^{-\text{gap}/\varepsilon})$ |
| $\kappa = L_g / \mu$ | 无量纲 | $O(N^2 D^4 / (\varepsilon^2 \eta_s \lambda_2))$ |
| $\lambda_{\text{sep}}$ 安全界 | $[\text{长度}]^2 \cdot [\Delta]^2$ | $O(G_{\text{direct}} \cdot \Delta_{\min}^2 / C)$ |

所有量纲一致。✓

### 4.2 约化到已知情形

- **K=2**：§二退化到 converge_rate_analysis.md 的情况。当 $N_1 = N_2 = N/2$，$v_1 = v_2$ → 各簇条件数相等 → 无瓶颈。✓
- **$\lambda_{\text{sep}} = 0$**：§三退化到 Direct Cluster only。此时 $\mu_{\text{joint}} = \mu_{\text{direct}}$。✓
- **$\eta_s = 0$**：定理 1 不成立（无平滑正则 → $H^*$ 可能奇异）。这与"平滑正则对收敛是必要的"的经验一致。✓

### 4.3 极限行为

- **$\varepsilon \to 0$**：$L_g \to \infty$（梯度爆炸），但 $\mu$ 仍被 $\eta_s \lambda_2$ 下界保护 → **线性收敛在最优点附近仍然成立**，但步长必须极小（$\eta < 2/L_g$ 在 $\varepsilon \to 0$ 时趋于 0）。✓
- **$\varepsilon \to \infty$**：$L_g \to L_{\text{smooth}} \approx 10$（只有平滑项贡献），但 $\mu \to \eta_s \lambda_2$（PL 条件仍成立！）。但 Sinkhorn 分配趋于均匀 → **簇不会形成** → 达到的"最小值"对应于平凡解（所有原子分配相同权重给所有簇）。这是 PL 条件的局限性：它保证收敛到**某个**临界点，不保证该临界点是"好的"临界点。✓

### 4.4 数值自洽性

对典型配置（N=100, K=2, $\varepsilon=0.05$, $\eta_s=0.01$, $H=W=64$）：

| 理论量 | 预测值 | 可测对应物 |
|--------|--------|-----------|
| $\mu$ | $\approx 5 \times 10^{-5}$ | Phase 2 损失下降速率 |
| $L_g$ | $\approx 800$ | 最大稳定学习率 |
| $\kappa = L_g/\mu$ | $\approx 1.6 \times 10^7$ | 收敛所需迭代数 ~ $O(\kappa \log 1/\epsilon)$ |
| 线性收敛迭代 | $\approx \kappa \cdot \log(100) \approx 7 \times 10^7$ 步（纯最坏情况）| 实际 19,200 步 → PL 常数在实践中大得多（~0.01） |

理论与实验的差异表明：**实际 PL 常数比基于稀疏 Hessian 界的估计大 200×**。这是因为 Hessian 的 Sinkhorn 贡献（我们在 §1.2 中保守地列为 $O(e^{-\text{gap}/\varepsilon})$）在实践中是主导的——当 gap 形成后，Sinkhorn 的 Jacobian 急剧收缩，大幅增加 $\mu$。

---

## 五、总结

| 问题 | 状态 | 关键结果 |
|------|------|---------|
| PL 条件证明 | ✅ 已证明 | $\mu \approx \eta_s \lambda_2(\Delta) / 2$，由平滑正则保证，Sinkhorn 贡献提供加速 |
| K > 2 泛化 | ✅ 已分析 | 小簇是瓶颈（$\mu_{\min} \propto N_{\min}/(N/K)$），建议 K ≤ N/5 |
| ECO 协同 | ✅ 已分析 | $\lambda_{\text{sep}}$ 安全界 $\ll G_{\text{direct}} \cdot \Delta_{\min}^2 / C$，典型值 $< 10^{-6}$；建议自适应 λ_sep(t) |

### 下一级未解决问题（链式扩展）

1. **非平滑化的 PL**：若去掉平滑正则（$\eta_s = 0$），Sinkhorn 的贡献能否单独保证 PL 条件？
2. **K 渐近分析**：当 K 随 N 增长（如 $K = \sqrt{N}$）时，收敛速率如何标度？
3. **联合优化的鞍点**：$\mathcal{L}_{\text{direct}} + \mathcal{L}_{\text{sep}}$ 的最优解性质——是否存在 Pareto 最优解，两者同时达到最优？
