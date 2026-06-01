# Phase 6a: Murmuration-Elliptic Curve 物体表示理论框架

## 1. 核心定义

### 定义 1: 椭圆曲线物体 (Elliptic Curve Object, ECO)

一个物体不是一个区域，而是椭圆曲线上的一个概率流形。

$$O = (E, \mu_E, v)$$

| 符号 | 含义 |
|------|------|
| $E: y^2 = x^3 + ax + b$ | Weierstrass 形式椭圆曲线，$\Delta = -16(4a^3 + 27b^2) \neq 0$ |
| $\mu_E$ | $E$ 上的概率测度（描述物体的"质量分布"） |
| $v: E \to T_p E$ | 切向量场（描述物体的运动方向） |

**关键**：物体的身份由曲线 $(a,b)$ 编码，而非由像素/边界框编码。

### 定义 2: Murmuration 流形动力学

物体上的点不是独立运动的，而是遵循受椭圆曲线约束的 Boids 规则。

设 $\{P_i(t)\}_{i=1}^N \subset E(\mathbb{R})$ 是物体上 $N$ 个采样点，动力学为：

$$\frac{dP_i}{dt} = \underbrace{v_i}_{\text{自身趋势}} + \underbrace{\sum_{j \in \mathcal{N}(i)} w_{ij} \cdot (P_j - P_i)}_{\text{凝聚力 (Cohesion)}} + \underbrace{\sum_{j \in \mathcal{N}(i)} w_{ij} \cdot \log_{P_i}(P_j)}_{\text{对齐 (Alignment)}} + \underbrace{\sum_{j: d(P_i,P_j) < r} w_{ij} \cdot \log_{P_i}(P_j)^{-1}}_{\text{分离 (Separation)}}$$

所有运算在椭圆曲线群上进行：

| 传统 Boids | Murmuration on $E$ |
|------------|---------------------|
| $P_j - P_i$ 是向量差 | $\log_{P_i}(P_j) \in T_{P_i}E$ 是对数映射（切向量） |
| 欧氏距离 $d(P_i, P_j)$ | 椭圆曲线上的测地距离 $d_E(P_i, P_j)$ |
| 加法是向量加法 | 加法是椭圆曲线群运算 $P_i + P_j$ |

### 定义 3: 传感驱动的曲线演化

视觉/多模态信息不改变物体，而是更新定义物体的椭圆曲线。

$$\phi: F_t \to (a_t, b_t)$$

$$(a_t, b_t) = \text{MLP}(\text{concat}[F_t^{\text{visual}}, F_t^{\text{audio}}, (a_{t-1}, b_{t-1})])$$

约束：保持判别式非零 $\Delta_t = -16(4a_t^3 + 27b_t^2) \neq 0$（投影回有效曲线空间）

---

## 2. 稳定性定理

### 定理 1: j-不变量稳定性

当物体外观剧烈变化（非稳态）时，只要动力学连续，物体身份不变。

椭圆曲线的 j-不变量：

$$j(E) = 1728 \cdot \frac{4a^3}{4a^3 + 27b^2}$$

性质：
- $j(E)$ 是曲线的拓扑不变量：同构的曲线有相同的 $j$
- 小扰动 $\delta a, \delta b$ 导致 $\delta j = O(\|\delta\|^2)$（二阶稳定）
- 即使 $(a,b)$ 变化很大，只要 $j$ 不变，物体就是同一个

$$\text{非稳态} \Rightarrow (a_t, b_t) \text{ 变化大} \Rightarrow \text{但 } j(E_t) \approx j(E_0) \Rightarrow \text{身份保持}$$

**对比传统方法**：

| 方法 | 非稳态时的身份保持 |
|------|-------------------|
| 边界框 IoU | 外观变30% → ID切换 |
| 特征余弦相似度 | 光照变一点 → ID切换 |
| **ECO (j-不变量)** | 外观剧变 → j 几乎不变 → ID稳定 |

### 定理 2: Sinkhorn 聚类与 ECO 的兼容性

ECO 版 Direct Loss：

$$\mathcal{L}_{\text{ECO}} = -\sum_i \sum_j P_{ij}^E \log Q_{ij}^E + \lambda \cdot \underbrace{d_H(j(E_i), j(E_j))}_{\text{身份一致性正则化}}$$

其中：
- $P_{ij}^E$ 是在椭圆曲线 $E_i, E_j$ 上计算的 Sinkhorn 分配
- $d_H$ 是匈牙利距离（匹配曲线）
- $\lambda$ 控制身份保持强度

#### Sinkhorn 迭代显式形式（与代码对齐）

`src/losses/direct_cluster.py` 的实际实现如下，给定 $N$ 个原子特征 $\{f_i\}$ 与 $K$ 个原型 $\{p_k\}$：

**成本矩阵**（余弦相似度归一到 $[0,1]$）：

$$C_{ik} = \frac{1 - \cos\langle f_i, p_k \rangle}{2} \in [0, 1]$$

**Sinkhorn 核**（温度参数 $\varepsilon$）：

$$K_{ik} = \exp\left(-\frac{C_{ik}}{\varepsilon}\right)$$

**迭代过程**（共 $T$ 步，先列归一再行归一实现双向平衡）：

$$P^{(0)}_{ik} = K_{ik}, \quad v^{(0)}_k = 1$$

$$\text{for } t = 1, \ldots, T: \quad
\begin{cases}
P^{(t)}_{ik} = \dfrac{K_{ik}\, v^{(t-1)}_k}{\sum_{k'} K_{ik}\, v^{(t-1)}_{k'}} \\[8pt]
v^{(t)}_k = v^{(t-1)}_k \cdot \dfrac{N/K}{\sum_{i'} P^{(t)}_{i'k}}
\end{cases}$$

**收敛性质**：
- $P$ 矩阵**行随机**（每行和 = 1）：每个原子唯一地属于一个簇的软概率
- $P$ 矩阵**列和平衡**为 $N/K$：每个簇的期望大小一致，避免大簇吞并小簇
- $\varepsilon \to 0$ 时退化为 hard assignment；$\varepsilon \to \infty$ 时退化为均匀分布
- 本项目最优 $\varepsilon = 0.05$（Phase 6c 网格搜索结果）

**直接测地聚类损失**（$D_g^2$ 为测地距离平方矩阵，$P_{:,k}$ 为第 $k$ 列）：

$$\mathcal{L}_{\text{direct}} = \sum_{k=1}^{K} \frac{P_{:,k}^\top D_g^2\, P_{:,k}}{(\sum_{i=1}^{N} P_{ik})^2}$$

分母 $(\sum_i P_{ik})^2$ 是该簇的**有效大小平方**，阻止大簇因 $N^2$ 因子主导损失。

**测地距离平方**（中点度量保证对称性）：

$$d_g^2(i, j) = (\mu_i - \mu_j)^\top\, g\!\left(\frac{\mu_i + \mu_j}{2}\right)\, (\mu_i - \mu_j)$$

---

## 1.5 传感函数 φ 的实现架构

`src/losses/eco_cluster.py` 的 $\phi: F_t \to (a_t, b_t)$ 实际是一个 **3 层 MLP**：

$$\phi: \mathbb{R}^{F} \xrightarrow{W_1} \mathbb{R}^{32} \xrightarrow{\text{ReLU}} \xrightarrow{W_2} \mathbb{R}^{32} \xrightarrow{\text{ReLU}} \xrightarrow{W_3} \mathbb{R}^{2} \to (a, b)$$

| 层 | 输入维度 | 输出维度 | 激活 |
|----|---------|---------|------|
| $W_1$ | $F$ (特征维度) | 32 | ReLU |
| $W_2$ | 32 | 32 | ReLU |
| $W_3$ | 32 | 2 | 无（直接输出 $(a,b)$） |

**奇异性保护**（避免掉入 $\Delta = 0$ 的分岔面）：

$$\text{if } |4a^3 + 27b^2| < 10^{-3} \text{ then } (a, b) \leftarrow (a, b) + 0.1 \cdot \text{sign}(a, b)$$

这个 0.1 的扰动在 0.001 阈值边界处把曲线推回非奇异区域，保持 $E(\mathbb{R})$ 拓扑结构稳定。

**Phase 8 附加损失**（鼓励 $(a,b)$ 远离奇点 + j-空间中彼此远离）：

$$\mathcal{L}_{\text{barrier}} = -\text{mean}\left(\log\left(|4a_i^3 + 27b_i^2| + \varepsilon\right)\right)$$

$$\mathcal{L}_{\text{sep}} = \text{mean}\left(\text{clamp}\big(d_{\min} - |j_i - j_k|\big)^2\right), \quad d_{\min} = 10$$

前者将判别式为零的曲面变成排斥墙，后者直接拉开 j-空间中不同曲线的距离。

---

## 3. 完整算法流程

```
输入: 视频帧序列 {I_t}, 多模态特征 {F_t}
输出: 物体轨迹 {O_k = (E_k, μ_k, v_k)}

初始化:
  for each detected object at t=0:
    E_k: y² = x³ + a_k·x + b_k  (从视觉特征回归)
    μ_k: uniform on E_k
    v_k: from optical flow

For t = 1, 2, ...:
  1. 传感更新:
     (a_t^k, b_t^k) = φ(F_t^k, a_{t-1}^k, b_{t-1}^k)
     
  2. Murmuration 演化:
     for each point P_i on E_t^k:
       dP_i/dt = boids_on_E(P_i, {P_j}, E_t^k)
     
  3. Direct Loss 聚类:
     P_t = Sinkhorn(features, E_t^k, ε=0.5)
     
  4. 身份匹配 (用 j-不变量):
     match k_t ↔ k_{t-1} by min |j(E_t^k) - j(E_{t-1}^k')|
     
  5. 非稳态检测:
     if |j(E_t^k) - j(E_0^k)| > τ_j:
       mark as "morphing object" (新物体/融合/分裂)
```

---

## 4. 可验证的预测

| 预测 | 验证方法 | 预期结果 |
|------|---------|---------|
| P1: j-不变量在非稳态下稳定 | 测量 Var(j(E_t)) vs Var(IoU) | Var(j) << Var(IoU) |
| P2: ECO + Direct Loss > Direct Loss alone | 在非稳态视频上对比 ARI | 提升 10-15% |
| P3: 过度变形时 j 突变 = 物体分裂/融合 | 检测 Δj 时产生新 H_0 | 分岔事件可检测 |

---

## 5. 理论贡献定位

| 方法 | 本质 | 局限 |
|------|------|------|
| InfoNCE | "拉近同类，推开异类" | 启发式，无结构 |
| Contrastive | "特征空间聚类" | 隐式，不可解释 |
| **ECO** | **"物体 = 椭圆曲线上的流形"** | 显式数学结构 |

ECO 提供：
- **群结构**（可计算）
- **拓扑不变量**（可证明稳定）
- **动力学**（可预测演化）
- **传感接口**（可融合多模态）

---

## 附录: 数学补充全景图

### 优先级矩阵

| 优先级 | 数学领域 | 为什么必须 | 缺了会怎样 |
|--------|---------|-----------|-----------|
| P0 | 微分几何：流形上的运算 | Boids在E上跑，需要exp/log映射 | 无法实现 |
| P0 | 动力系统：Lyapunov稳定性 | 证明murmuration不会发散 | 审稿人会问 |
| P1 | 最优传输：流形Sinkhorn | Direct Loss要在E上跑 | 只有欧氏Sinkhorn |
| P1 | 分岔理论 | 非稳态的数学描述 | 无法区分"变形"vs"分裂" |
| P2 | 代数几何：模空间 | 理论深度，论文亮点 | 不够强但能跑 |
| P2 | 拓扑学：同调群 | 检测分裂/融合事件 | 可用启发式替代 |
| P3 | 信息几何 | 优雅的梯度推导 | 锦上添花 |

### P0-1: 椭圆曲线群运算

```python
def elliptic_add(P, Q, a, b):
    """椭圆曲线群加法 P + Q"""
    x1, y1 = P; x2, y2 = Q
    if P == Q:
        lam = (3*x1**2 + a) / (2*y1)  # 切线斜率
    else:
        lam = (y2 - y1) / (x2 - x1)   # 割线斜率
    x3 = lam**2 - x1 - x2
    y3 = lam * (x1 - x3) - y1
    return (x3, y3)
```

### P0-2: 紧致性稳定性

$E(\mathbb{R})$ 是紧致流形（拓扑上是环面或两个圆）→ 所有轨道有界 → Boids 不崩溃。

### P1-1: 测地距离（椭圆积分）

$$d_E(P, Q) = \int_{x(P)}^{x(Q)} \frac{dx}{\sqrt{4x^3 - g_2 x - g_3}}$$

### P1-2: 分岔检测

$$\Delta > 0: \text{两个连通分支（物体可能分裂）}$$
$$\Delta < 0: \text{一个连通分支（正常物体）}$$
$$\Delta = 0: \text{奇异点（分岔点！）}$$
