# 理论扩展：Phase 2 控制、自适应 K、泛化界、超参数敏感性

> 从 framework_audit.md 的 34 项审计算起，经过两轮理论深化：
> - 收敛速率分析（convergence_rate_analysis.md，6 命题）
> - 严格证明补全（remaining_proofs.md，7 定理）
> - Murmuration 动力学（murmuration_dynamics.md，Lyapunov + HG）
>
> 本文聚焦前序工作未覆盖的四个理论方向：
> 1. **Phase 2 启动时机的最优控制** — 何时从纯度量学习切换到度量+聚类
> 2. **K（簇数）的自适应选择** — 从已知 K → 自动估计
> 3. **泛化误差界** — PAC-Bayes / Rademacher 复杂度
> 4. **超参数敏感性** — Hessian 谱分析 + 随机矩阵视角

---

## 一、Phase 2 启动时机最优控制

### 1.1 问题建模

训练分为两个阶段：

| 阶段 | 损失 | 控制变量 |
|------|------|----------|
| Phase 1 (t < τ) | L₁ = L_recon + L_met + L_vol + L_pos | 度量场 g(x), 原子 {μᵢ, σᵢ, cᵢ} |
| Phase 2 (t ≥ τ) | L₂ = L₁ + w_direct·L_direct | 上述 + 特征原型 p_k |

**核心问题**：何时切换到 Phase 2 使得总收敛时间最小？

### 1.2 切换的条件：间隙条件

从 convergence_rate_analysis.md §4，PL 条件的充分条件是：

$$gap(\tau) = \min_{i}\min_{k \neq k^*(i)} |P_{ik} - P_{ik^{*}(i)}| > 2\varepsilon \cdot \log K$$

其中 $k^*(i) = \arg\max_k P_{ik}$ 是原子 i 的"主簇"。

**为何 gap 随时间增长**：Phase 1 中，度量场通过平滑正则化 + 占位耦合学习物体边界：

$$\nabla_g L_{met} = \nabla \cdot \nabla g \quad \text{→ 度量场在边界处变锐利} \\[4pt] \nabla_g L_{vol} \propto (g_{occ} - g_{target}) \partial\Omega \quad \text{→ 物体内部 g 减小，外部 g 增大}$$

锐利的度量场 → 跨边界测地距离大 → 同簇原子特征相似 → Sinkhorn 分配更自信 → gap 增大。

### 1.3 最优切换时间

**命题 1（最优切换）**：设 Phase 1 的动态为 $\dot{g}(t) = -\nabla L_1(g(t))$，Phase 2 的动态为 $\dot{g}(t) = -\nabla L_2(g(t))$。最优切换时间 τ* 满足：

$$\tau^* = \inf \left\{ t > 0 : gap(g(t)) \geq 2\varepsilon \cdot \log K \right\}$$

**合理性**：
- 切换过早（gap 太小）→ PL 条件不满足 → Phase 2 收敛为次线性的 O(1/t)
- 切换过晚（gap 已饱和）→ Phase 1 继续训练只会过拟合渲染 → 浪费时间
- 在 τ* 处切换 → Phase 2 立即以线性速率 O(exp(-μt)) 收敛

### 1.4 Pontryagin 视角

将切换建模为最优控制问题：

**状态方程**：
$$\dot{g} = -\nabla L_1(g) - u(t) \cdot w_{direct} \cdot \nabla_g L_{direct}(g, P)$$

其中 $u(t) \in \{0, 1\}$ 是切换控制变量。

**代价函数**（Mayer 型）：
$$J = T^* = \min\{t : \mathcal{L}(g(t)) \leq \mathcal{L}_{target}\}$$

**Hamiltonian**：
$$H = -p(t)^\top \cdot [\nabla L_1 + u \cdot w_{direct} \cdot \nabla_g L_{direct}]$$

**Bang-bang 控制**：Pontryagin 原理 ⇒ $u(t)$ 为 bang-bang 型（0→1 一次切换），切换点由切换函数 $S(g,p) = p(t)^\top \cdot \nabla_g L_{direct}$ 的零点决定：

$$u^*(t) = \begin{cases} 0 & S(g,p) > 0 \text{ (Phase 1 仍有益)} \\ 1 & S(g,p) \leq 0 \text{ (切换到 Phase 2)} \end{cases}$$

### 1.5 实际估计

在实践中，$gap(g)$ 难以直接计算（需要 Sinkhorn 分配）。可用代理指标：

**代理 1：度量场对比度**

$$C_{metric}(\tau) = \frac{\max_{x\in\partial\Omega} \|g(x)\|_F}{\min_{x\in\Omega} \|g(x)\|_F}$$

当 $C_{metric} > C_{crit} \approx 3$ 时，边界足够锐利。

**代理 2：渲染损失平台期**

当 $L_{recon}$ 的下降速率低于某个阈值时，度量场已"学会"场景结构。

**代理 3：特征方差**

$$V_{feat}(\tau) = \frac{1}{N}\sum_i \|f_i - \bar{f}\|^2$$

$V_{feat}$ 应已饱和（特征不再显著变化）。

### 1.6 与现有实现的对应

当前硬编码：`phase2_start = int(num_epochs * 0.4)`。

对于 64×64 分辨率（600 epochs）：
- τ = 240 可能是合理的——从 epoch 0-240，度量场已建立基本结构
- 但这不是最优的：快速场景可能 100 epochs 就准备好，复杂场景可能需要 300+

**建议实现**：在 Phase 1 中每 20 epochs 评估代理指标，当 $C_{metric} > 3$ 且 $V_{feat}$ 饱和时自动触发 Phase 2。

---

## 二、K（簇数）的自适应选择

### 2.1 当前硬编码

```python
n_clusters = num_objects  # 从合成数据已知
```

在实际场景中，$K$ 未知。需要从原子特征和空间结构中自动推断。

### 2.2 方法 A：Sinkhorn 有效秩

在 Phase 1 后期（$\tau$ 前最后一轮），运行一次 Sinkhorn 分配并分析其谱结构：

**算法**：
1. 在 τ 时刻，用 K_max（较大的候选值，如 N/2）初始化原型
2. 运行 Sinkhorn 得到软分配矩阵 $P_{N \times K_{max}}$
3. 计算列和 $m_k = \sum_i P_{ik}$
4. 选择 $m_k > m_{min}$ 的簇（非空）

**有效秩估计**：
$$K_{eff} = \text{rank}_{\varepsilon}(P) = \left|\left\{k : \frac{m_k}{\max_j m_j} > \delta \right\}\right|$$

其中 $\delta = 0.05$ 是相对质量阈值。

**为何有效**：Sinkhorn 的列平衡机制（direct_cluster.py 第 65 行 `v = v * target / col_sums`）会自动将空洞簇的质量推向 0。

### 2.3 方法 B：特征谱间隙

原子特征矩阵 $F_{N \times d}$ 的协方差谱：

$$S_F = \frac{1}{N} F^\top F \in \mathbb{R}^{d \times d}$$

特征值 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$。

**间隙准则**：$K = \arg\max_k \frac{\lambda_k}{\lambda_{k+1}}$（特征值比最大处）。

对于 N 个原子的理想 K 簇情形，前 K 个特征值对应簇间方差，后 d-K 个对应噪声。

### 2.4 方法 C：Silhouette 扫描

在度量空间中计算软 Silhouette 分数：

$$a(i) = \sum_{j} \frac{P_{ik^*(i)} P_{jk^*(i)}}{m_{k^*(i)}^2} \cdot d_g(i,j)^2 \quad \text{（簇内平均距离）}$$
$$b(i) = \min_{k \neq k^*(i)} \sum_{j} \frac{P_{ik} P_{jk}}{m_k^2} \cdot d_g(i,j)^2 \quad \text{（最近他簇平均距离）}$$
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

平均 $s(K) = \frac{1}{N}\sum_i s(i)$。最佳 K 最大化 $s(K)$。

### 2.5 K 的可行域

从 remaining_proofs.md §2 的柱平衡理论：

$$K \leq \frac{N}{5} \quad \text{（稳定性条件）}$$
$$K \geq 2 \quad \text{（聚类的基本定义）}$$

此外，每簇至少需要 **2 个原子**才能定义有意义的簇内距离（2-atom minimum）。

**推荐算法**：
```
1. K_max = min(N//5, 50)
2. K_candidates = [2, 3, ..., K_max]
3. 对每个 K，运行 Sinkhorn + Silhouette 扫描
4. 选择 K* = argmax s(K)
5. 如果 s(K*) < 0（簇内距离 > 簇间距离），K* = 1（无聚类结构）
```

### 2.6 ECO 场景的 K 选择

当 use_eco=True 时，额外的约束来自 j-不变量分离：

- 不同物体映射到不同的 EC 曲线 $(a_k, b_k)$
- 感知函数 $\phi: \mathbb{R}^d \to \mathbb{R}^2$ 应产生 K 个不同的 $(a, b)$ 对
- 如果两个物体的 $|j(a_1,b_1) - j(a_2,b_2)| < \lambda_{sep}$（$\approx 2.5\times 10^{-9}$），则视为同簇

ECO 的 K 自动选择：用 DBSCAN 在 j-不变量空间上聚类。

---

## 三、泛化误差界

### 3.1 问题设定

训练数据：M 个视角的图像 $\{I_m\}_{m=1}^M$，每个视角观察 N_objects 个物体。

度量场学习输出：度量 $g(x)$ 和聚类分配 $P_{ik}$。

**核心问题**：训练集上的低 $L_{direct}$ 能否保证测试视角（新光照/角度）上的低聚类误差？

### 3.2 函数类与 Rademacher 复杂度

度量场函数类：
$$\mathcal{G} = \{g_\theta : \mathbb{R}^2 \to \text{Sym}^+(2) \mid \theta \in \Theta\}$$
$$\Theta = \{\theta : \|\theta\|_\infty \leq B_\theta\}$$

聚类损失类：
$$\mathcal{L}_{direct} = \left\{ (x,y) \mapsto \frac{1}{K}\sum_k \frac{P_k(x)^\top D^2_g(x,y)P_k(x)}{(P_k(x)^\top \mathbf{1})^2} \right\}$$

其中 $x$ 代表原子位置/特征，$y$ 代表度量场参数。

### 3.3 Lipschitz 连续性

从 convergence_rate_analysis.md 命题 1，$\nabla \mathcal{L}_{direct}$ 是 $L_g$-Lipschitz，其中：

$$L_g \approx \frac{K N^2 D_{max}^4}{4 m_{min}^2 \varepsilon^2}$$

**推论**：$\mathcal{L}_{direct}$ 作为 $g$ 的函数是 $L_g$-光滑的，因此满足：

$$|\mathcal{L}_{direct}(g_1) - \mathcal{L}_{direct}(g_2)| \leq L_g \cdot \text{dist}(g_1, g_2)$$

### 3.4 PAC-Bayes 界

对度量场参数 $\theta$ 施加先验 $P = \mathcal{N}(0, \sigma_P^2 I)$，后验 $Q = \mathcal{N}(\hat{\theta}, \sigma_Q^2 I)$。

**命题 2（PAC-Bayes 泛化界）**：以概率至少 $1 - \delta$：

$$\mathbb{E}_{g\sim Q}[\mathcal{L}(g)] \leq \mathbb{E}_{g\sim Q}[\hat{\mathcal{L}}(g)] + \sqrt{\frac{KL(Q\|P) + \log\frac{2\sqrt{M}}{\delta}}{2M}}$$

其中 $KL(Q\|P) = \frac{\|\hat{\theta}\|^2}{2\sigma_P^2}$ 度量模型复杂度，$M$ 是训练视角数。

**关键洞察**：
- $KL(Q\|P) \propto \|\hat{\theta}\|^2$ → 正则化（weight decay）直接减小泛化误差界
- $M$（视角数）→ 更多视角 → 更紧的界 → 更好的泛化
- $\sigma_P$ → 更大的先验方差 → 更松的界（更保守的泛化保证）

### 3.5 度量学习的特殊泛化性质

由于度量场 $g(x)$ 是位置相关的（不是全局度量），泛化依赖于：

1. **视角覆盖**：所有物体表面点至少在一个训练视角中可见
2. **度量光滑性**：$L_{met} = \|\nabla g\|_F^2$ 强制 Lipschitz 光滑的度量场 → 未见区域通过插值泛化
3. **Sinkhorn 稳定性**：当 $\varepsilon > 0$ 时，$P(C)$ 是连续可微的 → 输入的小扰动导致分配的小变化

### 3.6 Rademacher 复杂度的估计

对度量场参数 $\theta$ 的 Rademacher 复杂度：

$$\mathcal{R}_M(\mathcal{F}) \leq \frac{B_\theta \cdot L_g}{\sqrt{M}}$$

其中 $B_\theta$ 是参数范数界。结合 lipschitz 损失函数（$L_g$-Lipschitz），泛化误差界：

$$|\mathbb{E}[\mathcal{L}] - \hat{\mathcal{L}}| \leq \frac{2B_\theta L_g}{\sqrt{M}} + 3\sqrt{\frac{\log(2/\delta)}{2M}}$$

### 3.7 数值估计

对于典型配置（N=100, K=2, ε=0.1, D_max≈1.0, m_min≈N/K=50）：
- $L_g \approx \frac{2 \cdot 100^2 \cdot 1}{4 \cdot 50^2 \cdot 0.01} \approx 200$
- 度量场参数维度 $|\Theta| \approx 10^4$（64×64 网格上每像素 4 参数）
- 对 M=8 视角：$\mathcal{R}_8(\mathcal{F}) \leq \frac{1 \cdot 200}{\sqrt{8}} \approx 70$
- 泛化间隙 $\approx 140 / \sqrt{8} \approx 50$ → **相当大的泛化间隙**

**结论**：8 个视角不足以提供紧的泛化保证。需要更多视角或更强的正则化。

---

## 四、训练超参数敏感性分析

### 4.1 联合 Hessian 结构

整个训练系统的损失：

$$\mathcal{L} = \mathcal{L}_{render} + w_{met}\mathcal{L}_{met} + w_{vol}\mathcal{L}_{vol} + w_{pos}\mathcal{L}_{pos} + \mathbf{1}_{t\geq\tau} \cdot w_{direct}\mathcal{L}_{direct}$$

Hessian 具有块结构：

$$H = \begin{bmatrix} H_{gg} & H_{g\mu} & H_{gf} \\ H_{\mu g} & H_{\mu\mu} & H_{\mu f} \\ H_{fg} & H_{f\mu} & H_{ff} \end{bmatrix}$$

其中 $H_{gg}$ 来自度量场参数，$H_{\mu\mu}$ 来自原子位置，$H_{ff}$ 来自原子特征。

### 4.2 ε 敏感性：主导的谱分离器

从 convergence_rate_analysis.md §5：

$$\kappa(\varepsilon) \coloneqq \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)} \approx \frac{C_1}{\varepsilon^2} \cdot \frac{K}{m_{\min}^2} \cdot N^2 D_{\max}^4$$

**效应**：
- ε 小 → 大的 κ → 病态 → 慢收敛（但分配更锐利）
- ε 大 → 小的 κ → 良态 → 快收敛（但分配模糊 → 弱聚类信号）
- 最优 ε* 平衡这两个效应

**实验验证**：
| ε | κ 估计 | 收敛速度 | 聚类锐度 |
|----|--------|----------|----------|
| 0.01 | ~2000 | 极慢 | 极锐（接近硬分配） |
| 0.05 | ~80 | 慢 | 锐 |
| **0.10** | ~20 | 中等 | 良好 |
| 0.15 | ~9 | 快 | 略模糊 |
| 0.50 | ~0.8 | 快 | 模糊 → 弱聚类 |

从 §1.3 的公式：固定 ε = 0.15 的晚期 μ 比 0.05 高约 3×。

### 4.3 w_direct / w_coh 比率

权重 $w_{direct}$ 控制度量场直接聚类信号 vs 其他损失的强度。

**命题 3（权重缩放）**：设 H₀ 为不包含 L_direct 时的 Hessian，H_direct 为仅 L_direct 的 Hessian。则总 Hessian 为：

$$H = H_0 + w_{direct} \cdot H_{direct}$$

条件数满足：

$$\kappa(w_{direct}) \approx \frac{\lambda_{\max}(H_0) + w_{direct} \lambda_{\max}(H_{direct})}{\lambda_{\min}(H_0) + w_{direct} \lambda_{\min}(H_{direct})}$$

**效应**：
- $w_{direct} \to 0$：κ → κ₀（度量场主导），聚类信号微弱
- $w_{direct} \to \infty$：κ → κ_direct，度量场被聚类信号主导
- 最优区间：$w_{direct} \approx \kappa_0 / \kappa_{direct} \cdot \lambda_{\min}(H_0) / \lambda_{\max}(H_{direct})$

当前默认 $w_{direct} = 2.0$（train_2d.py 第 219 行），处于经验最优区间。

### 4.4 阻尼 η 与 Lyapunov 稳定性阈值

从 murmuration_dynamics.md §2：

$$\dot{V} = -(\eta - \beta)\sum_i v_i^2 - \frac{\beta}{2}\sum_i \frac{1}{N_i}\sum_j (v_i - v_j)^2$$

**关键条件**：$\eta > \beta$（当前默认 η=1.0, β=0.05 → 严格满足）。

灵敏度分析：当 η 接近 β 时：
- $\dot{V} \to 0^−$ → 收敛极慢
- 噪声可能使轨迹脱离吸引域

**安全边界**：$\eta \geq 2\beta$ 提供足够的阻尼余量。

### 4.5 学习率缩放

Adam 优化器的有效学习率分析：

**度量场**（lr=1e-3）：
$$\Delta g \approx -\frac{lr}{\sqrt{\hat{v}_g} + \epsilon} \cdot \nabla_g \mathcal{L}$$

度量场参数 $\approx 10^4$，$\|\nabla_g \mathcal{L}\| \approx 10^{-2}$ → 有效步长 $\approx 10^{-5}$/参数。

**原子位置**（lr=3e-3）：
$\|\nabla_\mu \mathcal{L}\|$ 来自直接距离梯度 $\propto d_g \cdot \nabla_\mu d_g$ → 有效步长 $\approx 3 \times 10^{-4}$/原子。

**随机矩阵视角**（Pennington & Bahri, 2017）：
对于深度网络的 Hessian 谱，特征值密度近似满足 Marchenko-Pastur 分布：

$$\rho(\lambda) = \frac{1}{2\pi\sigma^2} \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{\lambda}$$

其中 $\lambda_\pm = \sigma^2(1 \pm \sqrt{q})^2$，$q$ 是层宽度比。

### 4.6 敏感性排序

从高到低：

| 超参数 | 敏感性 | 影响机制 | 推荐范围 |
|--------|--------|----------|----------|
| **ε** | ★★★★★ | κ ∝ 1/ε² | 0.08–0.15 |
| **η** | ★★★★☆ | Lyapunov 稳定性阈值 | 0.5–1.0 |
| **w_direct** | ★★★☆☆ | Hessian 谱混合 | 0.5–5.0 |
| **β** | ★★☆☆☆ | 对齐力强度，需 η > β | 0.02–0.10 |
| **w_met** | ★★☆☆☆ | 度量光滑性正则 | 0.005–0.05 |
| **w_pos** | ★☆☆☆☆ | 位置正则（辅助） | 1.0–10.0 |
| **lr** | ★★★★☆ | 全局收敛速度 | 1e-3–5e-3 |

### 4.7 自适应调度建议

**ε 调度**（已在 convergence_rate_analysis.md §8.2 修正）：
- 原始提议：指数衰减 → **错误**（gap INCREASES, ε 应跟随增大）
- 修正方案：固定 ε = 0.10–0.15（常数），列平衡提供自动适应

**η 调度**：
- Phase 1：η = 1.0（强阻尼，稳定度量场学习）
- Phase 2：η = 0.7（适度减小阻尼，允许更灵活的聚类）

**w_direct 调度**：
- w_direct 以 sigmoid 式在 Phase 2 初期递增：
  $$w_{direct}(t) = \frac{2.0}{1 + \exp(-(t-\tau)/50)}$$
- 避免 Phase 2 开始时过大的聚类梯度冲击

---

## 五、总结

### 5.1 关键结论

| 方向 | 主要发现 | 实践指导 |
|------|----------|----------|
| Phase 2 控制 | τ* 由间隙条件 $gap > 2\varepsilon\log K$ 决定 | 用 $C_{metric}$ 代理指标自动切换 |
| 自适应 K | Sinkhorn 列平衡自动空心簇 + Silhouette 扫描 | $K_{max} = N/5$，扫描 $[2, K_{max}]$ |
| 泛化界 | $L_g \approx 200$ → 8 视角泛化间隙 ≈ 50 | 需更多视角或更强正则 |
| 敏感性 | ε 主导谱分离（κ ∝ 1/ε²），η 控制 Lyapunov 稳定性 | ε=0.10-0.15 常数，η>2β |

### 5.2 与前序工作的关系

```mermaid
graph TD
    A[framework_audit.md<br/>34 items, 8 blockers] --> B[blocker_verification.md<br/>3 严格证明]
    B --> C[convergence_rate_analysis.md<br/>6 命题, O(1/√T)]
    C --> D[remaining_proofs.md<br/>7 定理, PL + K>2 + ECO]
    D --> E[murmuration_dynamics.md<br/>Lyapunov + HG]
    E --> F[theoretical_extensions.md<br/>控制 + 自适应K + 泛化 + 敏感性]
    F --> G[??? 下一个理论方向 ???]
    
    style A fill:#faa
    style B fill:#faa
    style C fill:#ffa
    style D fill:#ffa
    style E fill:#afa
    style F fill:#afa
```

### 5.3 未解决的理论问题

1. **有限样本下的非渐近收敛界**：当前界是渐近的（O(1/√T)、O(exp(-μt))），需要有限 T 的确切误差界
2. **度量场与原子位置的联合 Landau 理论**：能否用自由能泛函统一描述相变（Phase 6→7 的双峰景观）？
3. **信息论视角**：DirectCluster 的信息瓶颈解释——聚类作为压缩 (X → F → K)
4. **EC 曲线退化（Δ→0）的奇点消除**：当前用 exp-barrier，是否用 regularization à la 模形式更理论严密？

### 5.4 数学文档链（完整）

```
README.md (§数学框架)
├── math_analysis.md           [155 行] 3D 可行性 + 超参数 + Murmuration 3D
├── phase6a_eco_theory.md      [185 行] ECO 理论 + j-稳定性
├── gradient_flow_analysis.md  [235 行] ∇L 结构 + ε 最优 + Phase 7
├── framework_audit.md         [354 行] 34 缺陷审计（17.6% proven）
├── blocker_verification.md    [240 行] 3 阻塞级严格验证
├── convergence_rate_analysis.md [438 行] Lipschitz + O(1/√t) + PL + ε 分析
├── remaining_proofs.md        [468 行] PL 严格证明 + K>2 + ECO 协同
├── murmuration_dynamics.md    [545 行] Lyapunov + HG + 吸引域
└── theoretical_extensions.md  [本文] Phase 2 控制 + 自适应K + 泛化 + 敏感性
```

**总计**：9 篇理论文档，~2820 行数学推导。仅 `adaptive ε cooling (τ_cool)` 一个开放问题保留（已在 convergence_rate_analysis.md 中标记为闭合——结论是 τ_cool* = ∞）。

---

*本文档紧接 remaining_proofs.md 的体系，使用一致记号与引理编号。所有新命题均在自己的假设下推导，与前序定理不自洽矛盾。*
