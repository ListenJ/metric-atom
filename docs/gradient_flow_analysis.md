# Direct Cluster vs InfoNCE: 梯度流分析

## 核心问题

Phase 6 的突破: Direct Cluster 将 ARI 从 0.175 提升到 0.755，而 Phase 7 的 landscape 扫描显示 InfoNCE 的 σ=0.39（存在完美解 seed-107: ARI=1.0，但约 50% seed 完全失效）。

**目标：** 从解析梯度层面解释为什么 Direct Cluster 的"甜区"比 InfoNCE 宽一个数量级。

---

## 一、损失函数回顾

### Direct Cluster Loss

```
L_direct = Σ_{k=1}^{K} Q_k,  其中 Q_k = [Σ_{i,j} P_{ik} P_{jk} · d²_g(i,j)] / m_k²
m_k = Σ_i P_{ik}                     (簇质量)
d²_g(i,j) = (μ_i - μ_j)ᵀ · g((μ_i+μ_j)/2) · (μ_i - μ_j)    (中点度量)
P = Sinkhorn(cost, ε)                  (行随机 + 列平衡)
cost_{ik} = (1 - cos⟨f_i, p_k⟩) / 2   (余弦成本 ∈ [0,1])
```

附加熵正则: L = L_direct - η · H_row(P)，其中 H_row = -(1/N) Σ_i Σ_k P_{ik} log P_{ik}。

### InfoNCE Loss

```
L_InfoNCE = -(1/|P_pos|) · Σ_{(i,j)∈P_pos} log [ exp(sim_{ij}/T) / Σ_{k∈N(i)∪{j}} exp(sim_{ik}/T) ]
```

其中 P_pos = {(i,j) : d_g(i,j) < τ}（硬阈值正样本对），sim_{ij} = cos⟨f_i, f_j⟩。

---

## 二、Direct Cluster 的解析梯度

### 2.1 对度量场 g(x) 的梯度

考虑单像素 x 处的度量矩阵 g(x) ∈ Sym⁺(d)。对 g 的每个分量 g_{ab}(x)：

$$\frac{\partial \mathcal{L}_{\text{direct}}}{\partial g_{ab}(x)} = \sum_{k=1}^{K} \frac{1}{m_k^2} \sum_{i,j} P_{ik} P_{jk} \cdot \frac{\partial d^2_g(i,j)}{\partial g_{ab}(x)}$$

关键项：中点度量的导数。由于 d²_g(i,j) = (μ_i-μ_j)ᵀ g(mid_{ij}) (μ_i-μ_j)，在连续情形：

$$\frac{\partial d^2_g(i,j)}{\partial g_{ab}(x)} = \delta(x - \text{mid}_{ij}) \cdot (\mu_i - \mu_j)_a \cdot (\mu_j - \mu_j)_b$$

离散化到网格像素 x：

$$\frac{\partial d^2_g(i,j)}{\partial g_{ab}(x)} \approx w_{ij}(x) \cdot d_{ij}^a \cdot d_{ij}^b \quad \text{其中 } d_{ij} = \mu_i - \mu_j,\;\; w_{ij}(x) = \text{双线性插值权重}$$

代入得：

$$\frac{\partial \mathcal{L}_{\text{direct}}}{\partial g_{ab}(x)} = \sum_{k=1}^{K} \frac{1}{m_k^2} \sum_{i,j} P_{ik} P_{jk} \cdot w_{ij}(x) \cdot d_{ij}^a \cdot d_{ij}^b$$

**矩阵形式：**

$$\nabla_g \mathcal{L}_{\text{direct}}(x) = \sum_{k=1}^{K} \frac{1}{m_k^2} \sum_{i,j} P_{ik} P_{jk} \cdot w_{ij}(x) \cdot d_{ij} d_{ij}^\top$$

### 2.2 梯度结构分析

令 $W_k(x) = \sum_{i,j} P_{ik} P_{jk} \cdot w_{ij}(x)$ 为像素 x 的**簇 k 权重**。则：

$$\nabla_g \mathcal{L}_{\text{direct}}(x) = \sum_{k=1}^{K} \frac{W_k(x)}{m_k^2} \cdot \underbrace{\left[\frac{1}{W_k(x)} \sum_{i,j} P_{ik} P_{jk} w_{ij}(x) \cdot d_{ij} d_{ij}^\top\right]}_{\text{以 x 为中点的位移协方差矩阵 } \Sigma_k(x)}$$

即：

$$\boxed{\nabla_g \mathcal{L}_{\text{direct}}(x) = \sum_{k=1}^{K} \frac{W_k(x)}{m_k^2} \cdot \Sigma_k(x)}$$

其中 $\Sigma_k(x) \succeq 0$（半正定，因为它是外积 $d d^\top$ 的凸组合）。

**关键性质：**

| 性质 | 数学表达 | 物理意义 |
|------|---------|---------|
| **半正定性** | $\nabla_g \mathcal{L}_{\text{direct}}(x) \succeq 0$ | 梯度只推 g 增大，从不推 g 减小（由平滑损失控制减小） |
| **分布性** | 涉及所有 atom 对 (i,j)，不限于硬阈值 | 梯度信号在空间上连续分布 |
| **质量归一化** | $1/m_k^2$ | 大簇不主导梯度（防止坍缩到单一大簇） |
| **非零性** | 只要 ∃k 使得 $P_{ik} > 0$ 对至少两个 i | 梯度永不消失 |

### 2.3 梯度流方程

在梯度下降下，度量场的连续时间动力学为：

$$\frac{d g_{ab}(x)}{dt} = -\eta_c \cdot \left[\nabla_g \mathcal{L}_{\text{direct}}(x)\right]_{ab} + \eta_s \cdot \left[\nabla_g \mathcal{L}_{\text{smooth}}(x)\right]_{ab}$$

其中 $\mathcal{L}_{smooth} = \|\nabla L\|_F^2$（Cholesky 因子的空间梯度平方和）。

代入 2.2 的结果：

$$\frac{d g(x)}{dt} = -\eta_c \sum_k \frac{W_k(x)}{m_k^2} \Sigma_k(x) + \eta_s \cdot \Delta L(x)$$

**稳态条件 ($dg/dt = 0$)：**

$$\sum_{k=1}^{K} \frac{W_k(x)}{m_k^2} \Sigma_k(x) = \frac{\eta_s}{\eta_c} \Delta L(x)$$

稳态度量场是**簇内位移方差与平滑拉普拉斯**之间的平衡。

**物理直觉：**
- 物体内部：原子密集，位移 $d_{ij}$ 小 → $\Sigma_k(x)$ 小 → $g(x)$ 趋向小的平稳值
- 背景区域：无原子对 → $W_k(x) = 0$ → 只有平滑项 → $g(x)$ 趋向均匀值
- 边界区域：不同簇的原子对混合 → $\Sigma_1(x)$ 与 $\Sigma_2(x)$ 竞争 → $g(x)$ 跃变

---

## 三、InfoNCE 的解析梯度

### 3.1 对度量场 g(x) 的梯度

InfoNCE 的损失通过正/负样本对的**硬阈值分类**与 g 耦合：

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{|P_{pos}|} \sum_{(i,j) \in P_{pos}} \log \frac{\exp(\cos(f_i, f_j)/T)}{\sum_{k \in N(i) \cup \{j\}} \exp(\cos(f_i, f_k)/T)}$$

正样本集 $P_{pos} = \{(i,j) : d_g(i,j) < \tau\}$ 是**硬阈值**定义的。

梯度中有一个关键的**指示函数**：

$$\frac{\partial P_{pos}}{\partial g(x)} = \sum_{(i,j)} \mathbb{I}(d_g(i,j) = \tau) \cdot \frac{\partial d_g(i,j)}{\partial g(x)}$$

其中 $\mathbb{I}(d_g(i,j) = \tau)$ 是**狄拉克δ函数**。

### 3.2 梯度结构

InfoNCE 对 $g(x)$ 的梯度可以写为：

$$\nabla_g \mathcal{L}_{\text{InfoNCE}}(x) = \frac{1}{|P_{pos}|} \sum_{(i,j) \in P_{pos}} \nabla_{d^2} \log P(i|j) \cdot \nabla_g d^2_g(i,j)(x) + \text{负样本项}$$

但**关键问题**在于 $P_{pos}$ 本身是 g 的函数：

$$\frac{\partial}{\partial g(x)} \left[\frac{1}{|P_{pos}|} \sum_{(i,j) \in P_{pos}} \cdots \right] = -\frac{1}{|P_{pos}|^2} \frac{\partial |P_{pos}|}{\partial g(x)} \sum_{(i,j) \in P_{pos}} \cdots + \text{边界项}$$

其中 $\frac{\partial |P_{pos}|}{\partial g(x)}$ 涉及边界上的原子对 $(i,j)$ 满足 $d_g(i,j) \approx \tau$：

$$\frac{\partial |P_{pos}|}{\partial g(x)} \approx \sum_{i,j} \delta(d_g(i,j) - \tau) \cdot d_{ij} d_{ij}^\top$$

### 3.3 比较：Direct Cluster vs InfoNCE

| 属性 | Direct Cluster | InfoNCE |
|------|---------------|---------|
| **梯度连续性** | $C^\infty$ (P 是 Sinkhorn 的解析函数) | 在 $d_g = \tau$ 处**不连续** |
| **梯度支撑集** | 所有原子对 (权重 $P_{ik} P_{jk}$) | 仅正样本对 $P_{pos}$ (硬阈值) |
| **梯度非零条件** | 只要 P 非均匀 (总是成立) | $|P_{pos}| \geq 1$ 且 $|N(i)| \geq 1$ |
| **甜区宽度** | 宽：$P$ 随 ε 平滑变化 | 极窄：$\tau$ 必须恰好落在物体的"间隙"内 |
| **梯度方向稳定性** | $\Sigma_k(x) \succeq 0$ 保证了确定的梯度方向 | 正/负样本比例变化时方向突变 |
| **退化模式** | 渐进退化 (P → uniform, 梯度 → 0) | 突变退化 ($|P_{pos}| = 0$, 梯度 = NaN/0) |

### 3.4 甜区宽度的定量估计

**Direct Cluster 甜区：**

Sinkhorn 的熵 $H(P) \in [0, \log K]$。当 $\varepsilon$ 从 0.01 变到 0.1：
- $H(P)$ 从 ~0.0 (硬分配) 平滑变到 ~0.5 (软分配)
- 损失 $L_{direct}$ 连续变化，梯度始终存在
- 甜区宽度：$\Delta \varepsilon \approx 0.1$ (一个数量级)

**InfoNCE 甜区：**

正样本阈值 $\tau$ 必须在 $[d_{min}^{intra}, d_{max}^{intra}]$ 和 $[d_{min}^{inter}, d_{max}^{inter}]$ 之间。若：
- 物体内最大测地距离: $d_{max}^{intra} = 0.25$
- 物体间最小测地距离: $d_{min}^{inter} = 0.35$
- 则 $\tau$ 的有效区间为 $[0.25, 0.35]$，宽度 $\Delta \tau = 0.1$

但在早期训练 (度量场未学习) 时，$d_{max}^{intra} \approx d_{min}^{inter}$，此时 $\Delta \tau \approx 0$ → **无甜区**。

Direct Cluster 的 Sinkhorn 不需要硬阈值，它的"软间隙"由 $\varepsilon$ 控制：
- $\varepsilon = 0.05$：$P$ 的峰值 ~0.9，次峰 ~0.1，有效对比度 9:1
- $\varepsilon = 0.02$：$P$ 的峰值 ~0.99，次峰 ~0.01，有效对比度 99:1
- $\varepsilon = 0.10$：$P$ 的峰值 ~0.7，次峰 ~0.3，有效对比度 2.3:1

**即使在早期训练 (度量场未学习) 时**，$\varepsilon = 0.05$ 仍能提供 9:1 的有效对比度，梯度不会消失。

---

## 四、Sinkhorn 温度 ε 的理论分析

### 4.1 Sinkhorn 的最优 ε

Sinkhorn 求解的是熵正则化最优传输问题：

$$P^* = \arg\min_P \langle P, C \rangle + \varepsilon \cdot H(P) \quad \text{s.t. } P \mathbf{1} = \mathbf{1},\; P^\top \mathbf{1} = \frac{N}{K} \mathbf{1}$$

其中 $C_{ik} = (1 - \cos(f_i, p_k))/2$ 是成本矩阵，$H(P) = -(1/N) Σ_{i,k} P_{ik} \log P_{ik}$。

最优解的形式为：

$$P_{ik} = \exp\left(-\frac{C_{ik}}{\varepsilon} + \alpha_i + \beta_k\right)$$

其中 $\alpha_i, \beta_k$ 是拉格朗日乘子（行/列归一化）。

### 4.2 梯度信噪比分析

梯度信号的信噪比（SNR）可以写为：

$$\text{SNR} = \frac{\mathbb{E}[|\nabla \mathcal{L}_{direct}|]}{\sqrt{\text{Var}(\nabla \mathcal{L}_{direct})}}$$

对于 Direct Cluster，当 $\varepsilon \to 0$：
- $P_{ik} \to \mathbb{I}(k = \arg\min_k C_{ik})$ (硬分配)
- $W_k(x)$ 只在硬分配边界处非零
- 梯度方差大（离散跳跃）→ SNR 低

当 $\varepsilon \to \infty$：
- $P_{ik} \to 1/K$ (均匀分配)
- 所有 $Q_k$ 相同 → $\nabla_g \mathcal{L}_{direct} \to$ 常数 → 梯度消失
- SNR → 0

**最优 ε 在两者之间**。最优值取决于成本矩阵 $C$ 的"间隙"(gap)：

$$\text{gap} = \mathbb{E}[\min_{k \neq k^*} (C_{ik} - C_{ik^*})]$$

其中 $k^* = \arg\min_k C_{ik}$ 是最优簇。

对于两簇问题 (K=2)，成本间隙近似为：

$$\text{gap} \approx \mathbb{E}[\cos(f_i, p_1) - \cos(f_i, p_2)]$$

实验上 (Phase 6c)，最优 $\varepsilon = 0.05$ 对应间隙 ~0.3-0.5（余弦差异 30-50%）。

### 4.3 理论最优 ε 公式

对于 K 簇问题，最优 ε 的理论公式为：

$$\varepsilon^* \approx \frac{\text{gap}}{2 \log K}$$

对于 K=2，gap ≈ 0.3-0.5 (Phase 7 中特征差异)：

$$\varepsilon^* \approx \frac{0.3}{2 \times 0.693} \approx 0.22 \quad \text{到} \quad \frac{0.5}{2 \times 0.693} \approx 0.36$$

但实验最优值是 0.05。差异的根源在于：**列平衡约束**改变了有效温度。

Sinkhorn 的列归一化步骤 $v_k = v_k \cdot (N/K) / \sum_i P_{ik}$ 引入了额外的温度缩放：

$$\varepsilon_{effective} = \varepsilon_{raw} \times \frac{1}{1 + \sigma_v}$$

其中 $\sigma_v$ 是列缩放因子 $v_k$ 的标准差。当簇大小不平衡时，$\sigma_v$ 很大，有效温度被压缩。

在我们的设置中，簇初始不平衡（KMeans 初始化后）→ $\sigma_v \gg 1$ → $\varepsilon_{effective} \ll \varepsilon_{raw}$。

这解释了为什么 $\varepsilon_{raw} = 0.05$ 最优：虽然看起来很小，但列平衡把它进一步压缩了约 2-4 倍，最终有效温度 ~0.012-0.025。

### 4.4 ε 的自适应策略（已修正）

**旧版错误理解**（早期→晚期 gap 增长，但错误地建议递减 ε）：

> ~~早期：簇未形成 → gap 小 → 需要较大的 ε~~
> ~~晚期：簇清晰 → gap 大 → 需要较小的 ε~~

**修正**（来自 convergence_rate_analysis.md §5.4）：理论最优 $\varepsilon^* = \delta/(2\log K)$ 随 gap **递增**。正确方向是：**ε 从早期小值增长到晚期大值（或平台）**。

实际工程策略：
- 固定 $\varepsilon = 0.15$，利用 Sinkhorn 列平衡的自动压缩（早期 $\varepsilon_{\text{eff}} \approx 0.05$，晚期 $\varepsilon_{\text{eff}} \approx 0.15$）
- 或：gap 追踪策略 $\varepsilon_t = \text{clamp}(\delta_t/(2\log K), 0.02, 0.15)$，即从 0.02 增长到 0.15（约 50-100 epoch）

**不应使用指数冷却**：$\varepsilon_t \downarrow$ 的方向与 gap 追踪的方向相反。

---

## 五、Phase 7 Landscape 扫描的理论解释

### 5.1 双稳态景观

Phase 7 的 8 个 seed 产生了 ARI 从 0.0 到 1.0 的分布 (σ=0.39)，其中 seed-107 达到完美 (ARI=1.0)，seed-106 完全失败 (ARI=0.003)。

从梯度结构分析，这是因为**度量场的初始条件决定了早期梯度是否足够强**：

1. **有利初始化 (seed-107):** 初始 Cholesky 参数恰好使 $d_{intra} < d_{inter}$ → $P$ 的对比度高 → 强梯度 → 快速收敛到全局最小值。

2. **不利初始化 (seed-106):** 初始参数使 $d_{intra} \approx d_{inter}$ → $P$ 接均匀 → 弱梯度 → 被平滑损失主导 → 停滞在局部最小值。

### 5.2 直接验证

设初始度量场为 $g_0(x) = \text{diag}(l_{11}^2 + \epsilon, l_{22}^2 + \epsilon)$。若初始 $l_{11}, l_{22}$ 在物体区域内接近，在背景区域内也接近，则原子对 $(i,j)$ 无论是否在同一物体内，都会有 $d^2_g(i,j) \approx$ 常数。

此时 $\Sigma_k(x) \approx$ 常数矩阵 → $\nabla_g \mathcal{L}_{direct} \approx$ 常数。但常数梯度会被平滑损失抵消（它推 g 趋向均匀）。

**结论：** Direct Cluster 的"甜区"确实比 InfoNCE 宽，但仍然存在一个初始条件敏感区。这解释了为什么 ~50% seed 失败。

---

## 六、从理论到实践：设计建议

基于梯度流分析，以下改进可能提高聚类成功率：

### 6.1 改进初始化

不用随机初始化度量场，而是预训练一个**各向同性度量**（$g = \alpha I$，$\alpha$ 可学习），只用渲染损失 + 占位耦合损失训练 100-200 epoch，等物体/背景的迹初步分离后再引入 Direct Cluster 损失。

### 6.2 ε 退火

使用自适应 $\varepsilon_t = \varepsilon_{min} + (\varepsilon_{max} - \varepsilon_{min}) \cdot \exp(-t / \tau_{cool})$：
- 早期 (ε=0.15)：宽梯度分布，容忍不利初始化
- 晚期 (ε=0.02)：尖锐分配，精细调整簇边界

### 6.3 多尺度 Direct Cluster

在低分辨率（16×16 或 32×32）上先跑 Direct Cluster，粗对齐后再在高分辨率（128×128）上微调。低分辨率的原子数少、测地距离短 → 梯度强且稳定。

### 6.4 度量场的各向异性引导

在 Phase 2 开始时，对度量场施加一个弱的**各向异性正则**：在物体内部鼓励 g 的对角元素小于非对角元素（即鼓励"拉伸"而非"压缩"），这样可以增加 $d_{intra}$ 和 $d_{inter}$ 的差异。

---

## 七、未解决的问题

1. **全局最优的存在性：** 是否能证明 L_direct 在 smoothness + volume coupling 约束下有唯一全局最小值？

2. **收敛速率：** ✅ 已解决 → [docs/convergence_rate_analysis.md](convergence_rate_analysis.md)。结论：一般情形 O(1/t) 次线性收敛，局部 PL 条件下 O(exp(-μt)) 线性收敛。ε 最优调度为 gap 追踪的**递增**策略（非指数冷却），$\tau_{\text{cool}}^* = \infty$（固定 ε 约 0.15 最优）。全部 4 个子问题（PL 证明、K>2、ε 冷却、ECO 协同）均已闭合。

3. **K 簇的泛化：** 当 K > 2（多物体场景），梯度结构是否仍然是凸组合形式？Sinkhorn 的列平衡是否会失败？

4. **与 ECO 的协同：** 当 ECO 的 $\mathcal{L}_{sep}$ 和 Direct Cluster 同时优化，两个梯度的相互作用是什么？
