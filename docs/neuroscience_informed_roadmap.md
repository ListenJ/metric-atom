# 神经科学启发路线图（低偏置实现版）

> 目标：把当前神经科学理论中能直接解释 MetricAtom 失败模式（种子方差 σ≈0.39、测地-聚类悬崖）的可复用思想，提炼成最小、可验证的代码改动。
> 约束：不引入 SAM/CLIP/分割先验；优先做“能跑起来”的实验，再决定是否升级。

## 0. 出发点

MetricAtom 的 `feat/selforg` 分支已有：
- 黎曼度量场 `g(x)`（Cholesky / 矩阵指数）
- 感知原子（位置 μ、半径 r、颜色 c、状态 s）
- 体积渲染重建损失
- 自组织损失：测地邻居传播、自组织力、状态对比、掩码像素预测

当前问题：
- DirectCluster 在 2D 双物体场景 ARI 可达 1.0，但 8 seed 标准差 σ≈0.39，50% seed 失败。
- 失败被诊断为“尖锐悬崖”：初始度量/原子分布一旦偏向某侧，损失景观陡峭下跌。

下面 4 个 Phase 按“先稳收敛、再补目标、最后加结构先验”排序。

---

## Phase 0：稳态可塑性（Homeostatic Plasticity）

### 核心思想
神经元通过活动依赖的内禀兴奋性缩放，把群体活动维持在动态范围内，防止“饿死”或“饱和”。

### 对应 MetricAtom 失败模式
种子方差大 ⇄ 部分原子激活过低或过高，Sinkhorn / 对比学习进入甜区外。

### 最小实现
新增 `src/losses/homeostatic.py`：
- `occupancy_target_loss(atom_weights, target_mean=0.5, target_std=0.25)`：让原子在训练视图上的平均占有率 μ 和方差 σ 贴近目标。
- 或者更简单的 `activity_target_loss(existence_probs, target=0.3)`：对存在概率做 sigmoid 反传，避免大量原子 dead。
- 在 `train_2d.py` Phase 1 早阶段加入，权重 0.01~0.05，作为正则项。

### 验证指标
跑 8 seed，看 ARI 均值和标准差 σ 是否下降。若 σ 从 0.39 → 0.25 即算成功。

---

## Phase 1：下一视角预测（Next-View Predictive Coding）

### 核心思想
Thomson & Gornet 2024：代理通过 next-image 预测，在隐空间自发构造认知地图， latent 距离反映空间距离。

### 对应 MetricAtom
当前只有单帧重建；多视图信息没有被显式用于约束隐态。可把“下一视角颜色”作为自监督目标，强迫原子状态编码几何一致性。

### 最小实现
1. 在 `train_2d.py` 数据加载时，每次采样两个相邻相机位姿 `(I_t, I_{t+1})`。
2. 用当前原子渲染 `I_t` 和 `I_{t+1}` 的颜色（已有渲染器）。
3. 增加 `predictive_next_view_loss`：
   - 把 `I_t` 的渲染颜色/状态作为条件，预测 `I_{t+1}` 的像素颜色。
   - 实现方式：在 `src/losses/self_organize.py` 加一个 `state_transition(states_t, camera_delta)`，用一个小 MLP（或线性层）预测 `states_{t+1}`，再解码成颜色。
   - 损失为预测颜色与 `I_{t+1}` 重建颜色之 L1。
4. 权重初始设为 0.1，不影响主重建流。

### 简化版（更低偏置）
不新增网络，直接把“同一像素在相邻视角的颜色一致性”作为正则：
`L_consistent = |render(I_t, p) - render(I_{t+1}, p')|`，其中 p' 是重投影坐标。但实现较重；建议先做状态-transition 版本。

### 验证指标
- 下一视角预测 PSNR。
- ARI 在 3-4 物体场景是否提升。

---

## Phase 2：网格细胞式度量先验（Grid-Cell Metric Prior）

### 核心思想
网格细胞把 2D/3D 空间编码为六边形晶格的叠加；海马位置细胞解码后得到全局位置。其关键性质是**局部等距**（conformal isometry）：神经流形上的局部距离保持物理距离的比例。

### 对应 MetricAtom
度量场 `g(x)` 目前无结构先验，容易被初始悬崖带偏。可给 `g(x)` 加一个“局部近似欧氏 / 低曲率”正则，等价于阻止度量在物体内部剧烈畸变。

### 最小实现
在 `src/geometry/metric_field.py` 加一个可选正则：
```python
def metric_flatness_loss(metric_field, samples):
    g = metric_field(samples)              # (M, d, d)
    trace = g.diagonal(dim1=-2, dim2=-1).sum(-1)
    det = torch.linalg.det(g)
    # 鼓励 g 接近单位阵的常数倍：低各向异性
    loss = ((trace / d) - det.pow(1/d)).mean()
    return loss
```
物理意义：把度量拉回“接近各向同性缩放”，避免空腔内出现极端各向异性。

### 替代：协变导数惩罚（略重）
若 Flatness 不够，可惩罚 `∂g/∂x` 的 Frobenius 范数，即鼓励度量场光滑。

### 验证指标
- 度量场 trace 分布更集中。
- 跨 seed 的测地距离矩阵更稳定。

---

## Phase 3：Hebbian / STDP 风格自组织（可选升级）

### 核心思想
“一起放电，一起连接” + 赢家通吃。已有 `self_organization_loss` 本质上是状态相似性-测地吸引的软 Hebbian 形式。

### 低偏置升级
把当前软 Sinkhorn/softmax 替换为**硬竞争 + 局部学习**的稀疏版：
- 对每个像素/原子，只让 top-k 赢家更新。
- 在 `state_propagation` 里加 Winner-Take-All：每次传播后，只保留每个状态维度最大的 k 个原子更新。

### 验证指标
训练速度、ARI、显存。

---

## Phase 4：柱状预测编码消息传递（可选远景）

### 核心思想
Bastos 等：皮层微柱实现预测编码，不同层负责前馈预测误差、反馈先验、侧向证据。

### 对应 MetricAtom
原子 = 微柱；状态更新 = 层内消息传递。可把当前单层 GAT 扩展为两层：
- 下层：像素级预测误差
- 上层：对象级先验

### 暂不实现
偏置较高，等 Phase 0-2 验证后再评估。

---

## 推荐执行顺序

1. **Phase 0**（1-2 天）：`homeostatic.py` + 跑 8 seed 基准，验证 σ 下降。
2. **Phase 1**（2-3 天）：在 `train_2d.py` 加 next-view 预测分支，权重小。
3. **Phase 2**（1-2 天）：`metric_flatness_loss`，小权重。
4. Phase 3/4：根据前面实验决定。

---

## 风险与回滚

- **过度约束**：Flatness 正则太强会把度量场压成欧氏，破坏边界信号。权重从 1e-4 开始。
- **目标冲突**：Predictive loss 与重建 loss 可能竞争。用 0.1 以下权重，单独 ablation。
- **计算开销**：next-view 需要每步渲染两个视角，训练时间翻倍。可用相邻帧缓存或低分辨率预测。

---

##  bottom line

> 先让系统“不死”——Phase 0 homeostasis 直接针对 σ=0.39；再让系统“看见时间”——Phase 1 next-view 预测提供多视图一致性自监督；最后才加几何先验和结构。每步都必须是可跑、可测、可回滚的最小改动。
