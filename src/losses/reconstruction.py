import torch


def l1_loss(pred, target):
    """L1 重建损失"""
    return (pred - target).abs().mean()


def l2_loss(pred, target):
    """L2 重建损失"""
    return ((pred - target) ** 2).mean()


def reconstruction_loss(pred, target, mode='l1'):
    """
    重建损失。
    
    Args:
        pred: (N, C) 预测值
        target: (N, C) 目标值
        mode: 'l1' 或 'l2'
    
    Returns:
        loss: 标量
    """
    if mode == 'l1':
        return l1_loss(pred, target)
    elif mode == 'l2':
        return l2_loss(pred, target)
    else:
        raise ValueError(f"Unknown mode: {mode}")
