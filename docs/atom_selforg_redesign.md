# 自组织原子系统：重新设计

> 2026-06-04 | 基于 DirectCluster 路线的 6 个失败实验的教训

---

## 设计原则

从 [postmortem_direct_cluster.md](postmortem_direct_cluster.md) 的 5 条教训推导：

1. **任务要求物体级理解** — 原子必须理解物体才能完成核心任务，而不是把理解作为可选的 bonus
2. **聚类是涌现的** — 没有 Phase 2，没有 KMeans，没有外部强加的聚类 loss。物体分组从原子的交互中自然产生
3. **特征是物体感知的** — 原子的内部状态编码 "我是什么/属于哪个物体"，而不是像素匹配的副产品
4. **系统对扰动鲁棒** — 确定性、可复现、对种子不敏感
5. **度量场是自组织介质** — 不再是重建驱动的几何描述，而是原子间通信和组织的媒介

---

## Part 1: 新任务 — 多视角掩码预测 (Masked Multi-View Prediction)

### 为什么 mask 预测？

```
当前任务: 给定 N 个视角, 用原子渲染所有视角 → L1(渲染, 真实)

问题: 这个任务不要求物体理解。
原子只需要记住 "这个位置该是什么颜色"，不需要知道 "这个颜色属于哪个物体"。

新任务: 给定 N 个视角，其中随机 M 个像素被 mask。
原子必须预测被 mask 像素的颜色。

为什么这强制物体理解:
  - 单个原子看不到全貌，必须参考邻居原子的状态来预测
  - 同一物体上的原子共享视觉属性 (颜色、纹理)
  - 预测需要推理 "这个区域属于哪个物体" → 物体级理解
```

### 任务形式化

```
输入:
  {I_1, I_2, ..., I_V}    — V 个视角的完整图像
  {M_1, M_2, ..., M_V}    — 对应的随机 mask (遮挡~30% 像素)

每 epoch:
  1. 对于可见像素: 标准渲染 → L_recon = L1(渲染, I)
  2. 对于 masked 像素:
     a. 找到该像素的 k 个最近原子 (按度量场测地距离)
     b. 每个近邻原子根据其内部状态 s_i 投票出颜色
     c. 加权聚合 → 预测颜色
     d. L_predict = L1(预测, I_masked)

总损失:
  L = w_recon * L_recon + w_predict * L_predict
    + w_smooth * L_smooth(度量场) + w_state * L_state(状态正则)
```

### 为什么这解决了根因

| 旧问题 | 新方案如何解决 |
|--------|---------------|
| 特征来自重建，不编码物体 | 预测 masked 像素 → 必须参考邻居状态 → 状态编码物体信息 |
| 测地距离 ≠ 物体归属 | 测地距离定义邻居 → 好邻居 = 好预测 → 度量场自适应 |
| KMeans 初始化的种子敏感性 | 没有 KMeans。状态从 0 开始，通过任务驱动演化 |
| 聚类是被动的 | 聚类涌现：同物体原子状态相似 → 度量场收缩 → 自然成群 |

---

## Part 2: 新原子模型

### 状态 s_i 取代特征 f_i

```
旧模型:
  Atom2D(mu, radius, color, _feature)  ← _feature 是重建副产品

新模型:
  Atom2D(mu, radius, color, _state)
    _state: (D,) 向量，编码 "我是谁"
    
    _state 通过两个信号更新:
    1. 预测误差信号: 我的状态能帮助预测吗？→ 梯度更新
    2. 邻居状态信号: 邻居原子是什么状态？→ 信息传播
```

### 状态动力学 (State Dynamics)

```
每 epoch 的状态更新:

s_i^{t+1} = (1-α) * s_i^t + α * aggregate( {s_j | j ∈ N(i)} )

其中:
  N(i) = 测地距离 top-k 近邻
  aggregate = 注意力加权平均:
    w_{i→j} = softmax(cos_sim(s_i, s_j) / τ)  ← 相似状态互相加强
    s_i^{agg} = Σ_j w_{i→j} * s_j

这个更新类似于 Graph Attention Network 的消息传递，
但图结构由度量场定义 — 度量场是参数化的自组织邻接矩阵。
```

### 自组织力

```
度量场 g(x) 的更新不仅来自重建，还来自状态相似度:

L_selforg = -Σ_{i,j} cos_sim(s_i, s_j) * d_g(x_i, x_j)

直观:
  - 状态相似的原子 → 被拉近 (减少测地距离)
  - 状态不同的原子 → 被推远 (增加测地距离)
  - 度量场自然地变得 "物体感知": 
    同物体的原子测地距离小，不同物体的原子测地距离大
```

### 预测机制

```
对于 masked 像素 p:

1. 找到 p 的测地近邻原子: N(p) = {atom_i | d_g(x_i, p) < r}
   (p 在度量场中的位置通过附近原子的插值获得)

2. 每个近邻原子预测颜色:
   pred_i = MLP_state2color(s_i)  ← 从状态解码颜色

3. 加权聚合:
   pred(p) = Σ_i w_i * pred_i
   w_i = softmax(-d_g(x_i, p) / ε)  ← 测地近的原子权重更大

4. L_predict = ||pred(p) - I_true(p)||²
```

---

## Part 3: 涌现聚类 — 如何验证

### 不显式聚类，聚类从状态空间自然涌现

```
训练后:
  1. 对所有原子计算状态 s_i
  2. 状态相似度 → 亲和矩阵 A_ij = cos_sim(s_i, s_j)
  3. 谱聚类 on A → 产生分组
  4. 与真实物体标签对比 → ARI

关键区别:
  旧方案: KMeans(epoch 240) → DirectCluster → ARI
  新方案: 训练(状态伴随任务演化) → 任意时刻的状态相似度 → ARI
  
  旧方案中 ARI 是优化目标。
  新方案中 ARI 是涌现属性 — 如果状态学得好，ARI 自然高。
```

### 期望涌现路径

```
epoch 0-100:   状态随机 → 预测差 → 重建驱动度量场 → 度量场平滑
epoch 100-300: 状态开始分化 → 同物体原子状态趋近 (通过消息传递)
                → 度量场响应自组织力 → 同物体区域测地收缩
epoch 300-500:  稳定涌现 → 状态空间出现清晰的簇
                → 度量场编码物体边界
                → 预测精度提升 (理解到位)
```

---

## Part 4: 实现路线图

### Phase 1: 核心架构 (本 session 后续)

```
改动:
  atom_2d.py:
    _feature → _state (维度 16, nn.Parameter)
    +state_distance(self, other_state) 方法
    +predict_color(self) → 从状态解码颜色

  src/losses/self_organize.py (新文件):
    L_selforg(mus, states, metric_field)
    L_predict(masked_positions, atoms, metric_field)
    L_state_smooth(states, geodesic_affinity)

  train_2d.py:
    +mask 生成 (随机遮挡 30% 像素)
    +预测 loss
    +自组织 loss
    -DirectClusterLoss (移除)
    -Phase 2 特殊处理 (移除)
    -KMeans init (移除)
```

### Phase 2: 验证

```
1. train_2d.py --epochs 600 --seed 42
   → 检查状态演化 (t-SNE of states over epochs)
   → 检查 ARI (谱聚类 on states)

2. 多种子 (42, 123, 99, 77) 鲁棒性测试

3. mask 比例消融: 10%, 30%, 50% → 预测精度 vs 聚类质量

4. 对比基线: ARI ≥ 0.87? 种子敏感性改善?
```

### Phase 3: 扩展 (未来)

```
- 多物体场景 (3+ objects)
- 真实图像 (纹理合成场景)
- 动态物体 (moving objects across views)
- 零样本泛化 (新物体类型)
```

---

## Part 5: 与旧架构的本质区别

| 维度 | 旧架构 (DirectCluster) | 新架构 (Self-Organizing) |
|------|----------------------|------------------------|
| 聚类方式 | 外部强加 (KMeans + DirectCluster loss) | 涌现 (状态空间 + 自组织力) |
| 特征来源 | 重建梯度 (间接) | 预测任务 + 邻居传播 (直接) |
| Phase 结构 | Phase 1 (重建) + Phase 2 (聚类) | 统一 (预测+重建+自组织始终活跃) |
| 度量场角色 | 重建几何 → 测地距离 | 自组织介质 + 邻居定义 |
| 种子依赖 | 高 (KMeans 初始聚类决定命运) | 低 (状态从 0 开始，任务驱动演化) |
| 评估 | ARI 是优化目标 | ARI 是涌现属性 |
| 扩展性 | 2D 合成场景专用 | 可扩展到真实图像、多物体、动态场景 |

---

## 附录: 关键超参初步范围

| 超参 | 建议值 | 理由 |
|------|--------|------|
| state_dim | 16 | 继承旧 feature_dim，便于对比 |
| mask_ratio | 0.3 | 足够强制推理，不过度减少重建信号 |
| w_predict | 1.0 | 预测和重建等权重 |
| w_selforg | 0.5 | 温和的自组织力，不过度扭曲度量场 |
| k_neighbors | 5 | 状态传播的近邻数 |
| state_alpha | 0.3 | 状态更新平滑系数 |
