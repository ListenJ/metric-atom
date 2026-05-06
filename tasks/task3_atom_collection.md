# Task 3: 原子集合与空间索引

## 描述
实现AtomCollection管理原子列表，提供基于均匀网格的空间查询，加速渲染时"仅激活邻域原子"。

## 约束词
- 空间索引必须使用简单的2D均匀网格（cell size = 2×最大原子半径）
- 不允许使用任何外部最近邻库（如faiss, scipy.spatial）
- 查询接口：给定矩形区域，返回该区域内所有原子的索引
- 必须支持动态插入和删除（预留接口）
- 必须是纯Python/PyTorch实现

## 输出文件
- src/atoms/atom_collection.py: AtomCollection类
- tests/test_atom_collection.py: 单元测试

## 测试要求
1. 随机撒100个原子，查询多个区域，验证召回率100%且不包含远域原子
2. 验证空查询返回空列表
3. 性能测试：1000个原子查询应在10ms内完成
