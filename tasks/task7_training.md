# Task 7: 训练循环与验证

## 描述
组装所有模块，实现分阶段训练循环，包括早停、日志、检查点保存。

## 约束词
- 训练循环必须分阶段：
  - 第一阶段（epoch 0-500）：仅优化E_render + E_met + E_vol
  - 第二阶段（epoch 500+）：加入E_coh
- 早期停止基于渲染损失在验证集上的plateau检测（patience=50）
- 每100步记录：原子分布、度量场迹、渲染对比图
- 使用Adam优化器，学习率1e-3
- 不依赖任何高级训练框架（如Lightning），保持代码透明
- 必须支持检查点保存和恢复

## 输出文件
- src/training/trainer.py: Trainer类
- src/training/optimizer.py: 优化器工厂
- src/training/validator.py: 验证器

## 训练伪代码
```python
for epoch in range(max_epochs):
    # 1. 随机选择一帧
    # 2. 生成光线
    # 3. 体积渲染
    # 4. 计算损失
    loss = loss_render + 0.01*loss_met + 0.1*loss_vol
    if epoch > 500:
        loss += 0.1 * loss_coh
    # 5. 反向传播
    # 6. 记录日志
```

## 测试要求
1. 在单场景上训练10步，确保损失下降
2. 验证检查点保存和加载功能
3. 验证分阶段训练策略（前500步coh损失系数为0）
