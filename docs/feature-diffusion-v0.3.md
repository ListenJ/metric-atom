# 特征扩散模块设计文档 v0.3

> 项目：MetricAtom
> 日期：2026-05-14
> 状态：已批准 — 可实施
> 审查回应：全部采纳

---

## 1. 背景与动机

当前系统通过 InfoNCE 对比损失实现无监督聚类（ARI=0.175），但瓶颈仍存：

- 特征交互仅发生在损失函数中，信号稀疏，收敛缓慢
- 簇内特征一致性不足，度量场对物体边界的刻画不够锐利
- 3D 扩展面临正样本稀疏、梯度竞争加剧等风险

**核心观察**：度量场已隐式定义原子间的亲和关系，但该信息未被前向计算直接利用。

**目标**：用度量场定义的测地亲和度矩阵，对原子特征进行确定性平滑，使特征平滑成为度量场学习的自然涌现。

---

## 2. 核心设计

### 2.1 总体工作流

```
Phase 1 (epoch 0-249):
  渲染 + 度量场 + 位置正则 → 原子在物体上稳定

Phase 1 末 (epoch 250):
  KMeans 空间聚类 (K=2) → 为每个簇提供特征偏置
  ─── 必要前置，不可移除 ───

Phase 2 (epoch 250-599):
  [扩散]  ← 簇内均质化
  + [InfoNCE]  ← 簇间分离
  → 最终特征
```

**扩散是 KMeans 的补充，不是替代。** KMeans 提供全局特征种子，扩散在其基础上做局部精炼。

### 2.2 亲和矩阵（修正版）

中点度量 + sigmoid 软掩码 + 自适应 $\sigma$：

$$d_{ij}^2 = (\mu_i - \mu_j)^\top g\!\left(\frac{\mu_i + \mu_j}{2}\right) (\mu_i - \mu_j)$$

$$A_{ij} = \exp\!\left(-\frac{d_{ij}^2}{2\,\sigma_i\sigma_j}\right) \cdot \text{sigmoid}\!\left(\frac{\tau_{\max} - d_{ij}}{s}\right)$$

其中：
- **中点度量**：$g((\mu_i + \mu_j)/2)$ 保证 $A_{ij} = A_{ji}$
- **软掩码**：$s = 0.1 \cdot \tau_{\max}$，边界处梯度可正常回传
- **自适应 $\sigma_i$**：$\sigma_i = d_g(\mu_i, \mu_{\text{neighbor}_K})$，$K=5$~$10$

### 2.3 自适应 $\sigma$ 的直觉

密度高的区域：$\sigma$ 小 → 只与最近邻居交互 → 局部精细平滑
稀疏的区域：$\sigma$ 大 → 扩大感受野 → 保持图连通

### 2.4 特征扩散

行归一化 $S = D^{-1}A$，$D_{ii} = \sum_j A_{ij}$：

$$F^{(t+1)} = (1 - \alpha)F^{(t)} + \alpha\, S F^{(t)}$$

- $\alpha = 0.5$（扩散步长）
- $T = 2$（迭代次数，不采用自适应停止以保持确定性）

### 2.5 梯度流

```
L_total ← F' (扩散后特征) ← A ← g (度量场), μ (原子位置)
         ← F (原始特征，独立路径)
```

所有路径可微。软掩码确保边界处梯度不为零。

---

## 3. 与现有组件的集成

| 组件 | 集成方式 | 备注 |
|---|---|---|
| 渲染器 | 无需修改 | 原子颜色由原始特征解码，不依赖扩散后特征 |
| InfoNCE 凝聚损失 | 保留，权重可降至 0.5 | 扩散提供局部平滑，InfoNCE 防止簇间融合 |
| 度量场正则化 | 无冲突 | 扩散质量受益于清晰度量场 |
| KMeans 初始化 | **保留（必要前置）** | Phase 1 末执行，扩散不替代它 |
| 原子管理（播种/剪枝） | 不受影响 | 逻辑不变 |

## 4. 预期收益

| 指标 | 预期变化 | 确定性 | 说明 |
|---|---|---|---|
| 簇内特征方差 | **↓ 30-50%** | 高 | 扩散直接作用于特征平滑 |
| ARI | **↑ 0.02~0.10** | 中 | 间接提升，簇间分离仍靠 InfoNCE + 度量场 |
| 收敛速度 | ↑ 10-20% | 中低 | 需要实验验证 |
| 3D 鲁棒性 | 待验证 | 低 | 可能在稀疏正样本场景有额外价值 |

ARI 突破 0.5 的真正瓶颈是**度量场分离锐利度**（w_vol 与 w_met 的精细平衡）和 **InfoNCE 超参**（$\tau$、温度、权重），非扩散可单独解决。

---

## 5. 风险矩阵（更新版）

| # | 风险 | 概率 | 影响 | 优先级 | 缓解方案 |
|---|---|---|---|---|---|
| R1 | ~~硬截断破坏梯度~~ | ✅ 已修复 | — | — | sigmoid 软掩码 |
| R2 | $\sigma$ 过大 → 全局特征坍缩 | 中 | 高 | P0 | 自适应 $\sigma_i$ + 保留 InfoNCE 安全网 |
| R3 | 自适应 $\sigma$ 的 $K$ 值不合适 | 中 | 中 | P1 | 扫描 $K \in [3, 5, 10, 20]$ |
| R4 | ~~KMeans 被移除后聚类失败~~ | ✅ 已消除 | — | — | 保留 KMeans |
| R5 | 扩散后特征用于渲染 → 质量下降 | 低 | 中 | P2 | 渲染用原始特征，扩散仅用于凝聚损失 |
| R6 | $O(N^2)$ 在 N>1000 时爆炸 | 低(当前) | 高 | P2 | K 近邻近似（未来优化） |

---

## 6. 实施计划

### 步骤 1：实现扩散模块（1 天）

文件：`src/losses/diffusion.py`

```python
def compute_geodesic_affinity(mus, metric_field, K=5, tau_max_factor=3.0, s_factor=0.1):
    """
    对称测地亲和矩阵 + sigmoid 软掩码 + 自适应 sigma。
    返回: (N, N) 亲和矩阵
    """
    N = mus.shape[0]
    mids = (mus.unsqueeze(0) + mus.unsqueeze(1)) / 2  # (N, N, D)
    dx = mus.unsqueeze(0) - mus.unsqueeze(1)           # (N, N, D)
    
    # 中点度量（需批量评估 metric_field）
    g_mid = metric_field(mids.reshape(-1, 2)).reshape(N, N, 2, 2)
    d2 = torch.einsum('ijm,ijmn,ijn->ij', dx, g_mid, dx).clamp(min=0)
    d = torch.sqrt(d2 + 1e-8)
    
    # 自适应 sigma：每个原子取第 K 近邻的测地距离
    d_sorted = d.topk(k=K+1, dim=1, largest=False)[0]  # 包含自身 d=0
    sigma_i = d_sorted[:, K]                            # (N,)
    sigma_prod = sigma_i.unsqueeze(1) * sigma_i.unsqueeze(0)  # (N, N)
    
    # 指数衰减亲和度
    A = torch.exp(-d2 / (2 * sigma_prod + 1e-8))
    
    # sigmoid 软掩码（替代硬截断）
    tau_max = tau_max_factor * sigma_i.mean()
    s = s_factor * tau_max
    soft_mask = torch.sigmoid((tau_max - d) / s)
    
    A = A * soft_mask
    A.fill_diagonal_(0.0)
    return A


def feature_diffusion(F, A, alpha=0.5, T=2):
    """
    可微特征扩散。
    F: (N, d)
    A: (N, N) 亲和矩阵
    返回: (N, d) 扩散后特征
    """
    D = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
    S = A / D
    F_new = F
    for _ in range(T):
        F_new = (1 - alpha) * F_new + alpha * (S @ F_new)
    return F_new
```

### 步骤 2：集成到训练循环（0.5 天）

在 `train_2d.py` 中 Phase 2 的前向渲染前插入：

```python
if epoch >= phase2_start:
    # KMeans 初始化（Phase 1 末，不改动）
    if epoch == phase2_start:
        ...

    # 特征扩散
    mus = torch.stack([a.position for a in atoms])
    A = compute_geodesic_affinity(mus.detach(), metric_field)
    diffused_feats = feature_diffusion(feats, A)
    
    # 用扩散后特征计算 InfoNCE 损失
    loss_coh = contrastive_coherence_loss(atoms, metric_field, ...
                                           diffused_feats=diffused_feats)
```

### 步骤 3：快速验证实验（1 天）

64×64，600 epochs，三组对比：

| 实验 | 配置 | 验证目标 |
|---|---|---|
| A | 基线（无扩散）| ARI ≈ 0.175 |
| B | 基线 + 扩散 | 簇内方差 ↓ 30%+ |
| C | 基线 + 扩散（无 InfoNCE）| 确认 InfoNCE 不可替代 |

### 步骤 4：$K$ 和 $\sigma$ 超参扫描（并行，不阻塞）

扫描 $K \in [3, 5, 10, 20]$，同时推进超参网格搜索。

---

## 7. 当前优先级排序

```
优先级 1 🥇：超参网格搜索      — 主攻 ARI 0.5 的真正瓶颈
优先级 2 🥈：特征扩散实现与验证  — 提升特征质量，辅助项
优先级 3 🥉：3D 调试           — 不阻塞，低优先级推进
```

---

## 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v0.1 | 2026-05-14 | 初始提案 |
| v0.2 | 2026-05-14 | 审查后修订：软掩码、自适应 $\sigma$、$K$ 超参 |
| **v0.3** | **2026-05-14** | **回应审查：保留 KMeans、下调 ARI 预期、对称中点度量、实施计划更新** |
