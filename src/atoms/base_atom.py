from abc import ABC, abstractmethod
import torch.nn as nn


class BaseAtom(nn.Module, ABC):
    """
    感知原子基类。
    
    所有原子必须实现 forward 方法，接收查询点和度量函数，
    返回空间权重、密度和特征贡献。
    """
    
    @abstractmethod
    def forward(self, x, metric_fn):
        """
        Args:
            x: (N, D) 查询点坐标，D为空间维度
            metric_fn: 返回 g(x) 的函数，输出 (N, D, D) 或 (1, D, D)
        
        Returns:
            weight: (N,) 空间支持权重 [0, 1]
            density: (N,) 密度值
            feature_contrib: (N, F) 特征贡献
        """
        pass
    
    @property
    @abstractmethod
    def position(self):
        """原子中心位置"""
        pass
    
    @property
    @abstractmethod
    def radius(self):
        """原子有效半径"""
        pass
