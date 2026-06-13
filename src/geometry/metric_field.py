import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry.cholesky_param import (
    cholesky_to_metric, cholesky_to_metric_3d,
    symmetric_to_metric_2d, symmetric_to_metric_3d,
)


class MetricField2D(nn.Module):
    """
    2D 黎曼度量场 g(x)。

    参数化方式（可切换，见 EXT-1 fix）：
      - 'cholesky': g = L L^T + eps*I (快速，但欧几里得SGD ≠ 流形测地下降)
      - 'matrix_exp': g = exp(H) (严格保证SPD，H是对称矩阵的切空间坐标)

    每个像素存储3个参数（Cholesky下三角或对称矩阵上三角元素），
    通过选定的参数化构建正定度量张量。

    支持双线性插值采样，输入坐标归一化到 [0, 1]。
    """

    def __init__(self, H, W, init_scale=1.0, eps=1e-4, default_batch_size=512,
                 parametrization='cholesky'):
        super().__init__()
        self.H = H
        self.W = W
        self.eps = eps
        self.default_batch_size = default_batch_size
        self.parametrization = parametrization

        # 将三个参数存储为 (1, 3, H, W) 的张量，便于使用 grid_sample
        # 通道0,1,2: 根据参数化方式对应不同的自由度
        self.params = nn.Parameter(torch.zeros(1, 3, H, W))

        # 初始化
        with torch.no_grad():
            if parametrization == 'cholesky':
                self.params[0, 0].fill_(init_scale)  # l11
                self.params[0, 2].fill_(init_scale)  # l22
                self.params[0, 1].zero_()             # l21
            elif parametrization == 'matrix_exp':
                # For g = exp(H) with H = [[h11, h12], [h12, h22]]
                # We want g ≈ init_scale^2 * I, so H ≈ log(init_scale^2) * I
                init_diag = np.log(init_scale**2) if init_scale > 0 else 0.0
                self.params[0, 0].fill_(init_diag)   # h11
                self.params[0, 2].fill_(init_diag)   # h22
                self.params[0, 1].zero_()             # h12 (off-diagonal)
            else:
                raise ValueError(f"Unknown parametrization: {parametrization}")
    
    def forward(self, coords, batch_size=None):
        """
        对给定坐标采样度量张量。
        
        Args:
            coords: (N, 2) 张量，坐标归一化到 [0, 1]
            batch_size: 内部批处理大小，None=使用实例默认值
        
        Returns:
            g: (N, 2, 2) 正定度量张量
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        if coords.dim() != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be (N, 2), got {coords.shape}")
        
        if coords.min() < 0.0 or coords.max() > 1.0:
            coords = coords.clamp(0.0, 1.0)
        
        N = coords.shape[0]
        
        # 内部批处理：避免 grid_sample 一次性 expand 到 N 个点
        if N > batch_size:
            g_parts = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                g_parts.append(self._forward_batch(coords[start:end]))
            return torch.cat(g_parts, dim=0)
        
        return self._forward_batch(coords)
    
    def _forward_batch(self, coords):
        """Process a single batch of coordinates through grid_sample."""
        N = coords.shape[0]

        # grid_sample 需要 (N, H_out, W_out, 2) 格式的 grid
        # coords 是 [0,1]，需要映射到 [-1,1]
        grid = coords.unsqueeze(1).unsqueeze(1) * 2.0 - 1.0  # (N, 1, 1, 2)

        # 使用双线性插值采样参数
        # 需要 batch size 匹配，将 params 扩展为 (N, 3, H, W)
        params_expanded = self.params.expand(N, -1, -1, -1)
        # sampled: (N, 3, 1, 1)
        sampled = F.grid_sample(
            params_expanded, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        # 重塑为 (N, 3)
        sampled = sampled.squeeze(-1).squeeze(-1)  # (N, 3)
        p0, p1, p2 = sampled[:, 0], sampled[:, 1], sampled[:, 2]

        # 根据参数化方式构建正定度量矩阵
        if self.parametrization == 'cholesky':
            g11, g12, g22 = cholesky_to_metric(p0, p1, p2, self.eps)
        elif self.parametrization == 'matrix_exp':
            g11, g12, g22 = symmetric_to_metric_2d(p0, p1, p2)
        else:
            raise ValueError(f"Unknown parametrization: {self.parametrization}")

        g = torch.zeros(N, 2, 2, device=coords.device, dtype=coords.dtype)
        g[:, 0, 0] = g11
        g[:, 0, 1] = g12
        g[:, 1, 0] = g12
        g[:, 1, 1] = g22

        return g
    
    def get_params_at_pixels(self):
        """
        获取每个像素上的原始Cholesky参数。
        
        Returns:
            l11, l21, l22: 每个都是 (H, W) 张量
        """
        return self.params[0, 0], self.params[0, 1], self.params[0, 2]
    
    def trace(self, coords=None):
        """
        计算度量矩阵的迹。如果提供坐标则采样，否则返回所有像素的迹。

        Args:
            coords: 可选，(N, 2) 坐标

        Returns:
            trace: (N,) 或 (H, W)
        """
        if coords is not None:
            g = self.forward(coords)
            return g[:, 0, 0] + g[:, 1, 1]
        else:
            p0, p1, p2 = self.get_params_at_pixels()
            if self.parametrization == 'cholesky':
                g11, _, g22 = cholesky_to_metric(p0, p1, p2, self.eps)
            elif self.parametrization == 'matrix_exp':
                g11, _, g22 = symmetric_to_metric_2d(p0, p1, p2)
            else:
                raise ValueError(f"Unknown parametrization: {self.parametrization}")
            return g11 + g22

    def metric_flatness_loss(self):
        """
        轻量 grid-cell 风格度量平坦先验：惩罚度量张量的各向异性与空间变化。

        生物学动机：海马网格细胞被认为在一个局部近似共形/等距的低维神经流形上
        编码空间。把该想法迁移到 MetricAtom 的黎曼度量场：我们并不真的去学一
        组网格细胞，而是要求“物体内部”的度量尽量各向同性且空间变化平缓，从而
        让同类原子之间的测地距离更接近欧氏距离，减少聚类结果对初始种子的敏感
        度（σ=0.39 的尖锐悬崖问题）。

        实现上只使用已经存在的参数张量 self.params，没有额外可学习参数：
          1. anisotropy_penalty = mean(|g_ij| / (g_ii + g_jj + eps))  鼓励对角占优
          2. spatial_smooth_penalty = mean(|trace(x) - trace(y)|)  沿空间邻居惩罚迹跳变
        返回标量，可在 train_2d.py 里以 w_flat * loss_flat 加入总损失。
        """
        p0, p1, p2 = self.get_params_at_pixels()  # (H, W)
        if self.parametrization == 'cholesky':
            g11, g12, g22 = cholesky_to_metric(p0, p1, p2, self.eps)
        elif self.parametrization == 'matrix_exp':
            g11, g12, g22 = symmetric_to_metric_2d(p0, p1, p2)
        else:
            raise ValueError(f"Unknown parametrization: {self.parametrization}")

        # Anisotropy: off-diagonal relative to diagonal mass
        diag_mass = g11 + g22 + self.eps
        aniso = (g12.abs() / diag_mass).mean()

        # Spatial smoothness of trace on a small finite-difference stencil
        trace_map = g11 + g22  # (H, W)
        # L1 differences to right/down neighbors (reflective border via padding)
        trace_padded = F.pad(trace_map.unsqueeze(0).unsqueeze(0),
                             (0, 1, 0, 1), mode='replicate').squeeze(0).squeeze(0)
        diff_h = (trace_padded[:-1, :-1] - trace_padded[1:, :-1]).abs()
        diff_w = (trace_padded[:-1, :-1] - trace_padded[:-1, 1:]).abs()
        smooth = (diff_h.mean() + diff_w.mean()) * 0.5

        return aniso + smooth


class MetricField3D(nn.Module):
    """
    3D 黎曼度量场 g(x)。

    参数化方式（可切换，见 EXT-1 fix）：
      - 'cholesky': g = L L^T + eps*I
      - 'matrix_exp': g = exp(H) (H 对称)

    每个体素存储六个参数，通过选定的参数化构建正定度量张量。

    支持三线性插值采样，输入坐标归一化到 [0, 1]。
    """

    def __init__(self, res_x, res_y, res_z, init_scale=1.0, eps=1e-4,
                 parametrization='cholesky'):
        super().__init__()
        self.res_x = res_x
        self.res_y = res_y
        self.res_z = res_z
        self.eps = eps
        self.parametrization = parametrization

        # 六个参数存储为 (1, 6, D, H, W) 张量
        self.params = nn.Parameter(torch.zeros(1, 6, res_z, res_y, res_x))

        # 初始化
        with torch.no_grad():
            if parametrization == 'cholesky':
                self.params[0, 0].fill_(init_scale)  # l11
                self.params[0, 2].fill_(init_scale)  # l22
                self.params[0, 5].fill_(init_scale)  # l33
                self.params[0, 1].zero_()             # l21
                self.params[0, 3].zero_()             # l31
                self.params[0, 4].zero_()             # l32
            elif parametrization == 'matrix_exp':
                init_diag = np.log(init_scale**2) if init_scale > 0 else 0.0
                self.params[0, 0].fill_(init_diag)   # h11
                self.params[0, 2].fill_(init_diag)   # h22
                self.params[0, 5].fill_(init_diag)   # h33
                self.params[0, 1].zero_()             # h12
                self.params[0, 3].zero_()             # h13
                self.params[0, 4].zero_()             # h23
            else:
                raise ValueError(f"Unknown parametrization: {parametrization}")

    def forward(self, coords):
        """
        对给定坐标采样度量张量。

        Args:
            coords: (N, 3) 张量，坐标归一化到 [0, 1]

        Returns:
            g: (N, 3, 3) 正定度量张量
        """
        if coords.dim() != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (N, 3), got {coords.shape}")

        if coords.min() < 0.0 or coords.max() > 1.0:
            coords = coords.clamp(0.0, 1.0)

        N = coords.shape[0]

        # grid_sample 5D: 需要 (N, D_out, H_out, W_out, 3) 格式
        # 但 PyTorch 的 grid_sample_3d 输入为 (N, C, D, H, W), grid 为 (N, D_out, H_out, W_out, 3)
        grid = coords.view(1, -1, 1, 1, 3) * 2.0 - 1.0  # (1, N, 1, 1, 3)
        # 转成 (1, 1, N, 1, 3) 以满足 5D grid format (N, D, H, W, 3)
        grid = grid.permute(0, 2, 3, 1, 4)  # (1, 1, 1, N, 3)

        sampled = F.grid_sample(
            self.params, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (1, 6, 1, 1, N)

        sampled = sampled.squeeze()  # (6, N) 或 (N,) 取决于维度
        if sampled.dim() == 1:
            sampled = sampled.unsqueeze(1)
        sampled = sampled.t()  # (N, 6)

        p0, p1, p2, p3, p4, p5 = (
            sampled[:, 0], sampled[:, 1], sampled[:, 2],
            sampled[:, 3], sampled[:, 4], sampled[:, 5]
        )

        # 构建 3x3 正定度量矩阵
        if self.parametrization == 'cholesky':
            g11, g12, g13, g22, g23, g33 = cholesky_to_metric_3d(
                p0, p1, p2, p3, p4, p5, self.eps
            )
        elif self.parametrization == 'matrix_exp':
            g11, g12, g13, g22, g23, g33 = symmetric_to_metric_3d(
                p0, p1, p2, p3, p4, p5
            )
        else:
            raise ValueError(f"Unknown parametrization: {self.parametrization}")

        g = torch.zeros(N, 3, 3, device=coords.device, dtype=coords.dtype)
        g[:, 0, 0] = g11
        g[:, 0, 1] = g12
        g[:, 0, 2] = g13
        g[:, 1, 0] = g12
        g[:, 1, 1] = g22
        g[:, 1, 2] = g23
        g[:, 2, 0] = g13
        g[:, 2, 1] = g23
        g[:, 2, 2] = g33

        return g

    def trace(self, coords=None):
        """
        计算度量矩阵的迹。
        """
        if coords is not None:
            g = self.forward(coords)
            return g[:, 0, 0] + g[:, 1, 1] + g[:, 2, 2]
        else:
            with torch.no_grad():
                if self.parametrization == 'cholesky':
                    diag_sq = self.params[:, [0, 2, 5]] ** 2
                    trace_map = diag_sq.sum(dim=1) + self.eps * 3
                elif self.parametrization == 'matrix_exp':
                    # For exp(H), trace ≈ exp(h11) + exp(h22) + exp(h33) when off-diags small
                    # Exact trace requires matrix_exp per voxel — expensive
                    # Approximate with diagonal entries of H
                    trace_map = (
                        self.params[:, 0].exp() +
                        self.params[:, 2].exp() +
                        self.params[:, 5].exp()
                    ).unsqueeze(0)
                else:
                    raise ValueError(f"Unknown parametrization: {self.parametrization}")
                return trace_map.squeeze(0)

    def metric_flatness_loss(self):
        """
        3D 度量平坦先验：惩罚各向异性与空间迹跳变。

        与 2D 版本保持相同的原则，使用 self.params 计算对角与 off-diagonal
        分量，沿 x/y/z 三个空间方向做 trace 的 L1 平滑。
        """
        with torch.no_grad():
            # 用 trace() 得到 (res_z, res_y, res_x) 空间标量场
            trace_map = self.trace()  # (res_z, res_y, res_x)

        # 用中心差分/邻居差分惩罚空间变化；为保持可微，用 torch 运算在 forward 时
        # 需要保留计算图，因此这里重新读取 params 并计算 trace（与 trace() 内部逻辑
        # 一致但保留 grad）。为减少重复代码，直接对 trace_map 做原地 padding 计算。
        trace_map = trace_map.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        pad = (0, 1, 0, 1, 0, 1)
        trace_padded = F.pad(trace_map, pad, mode='replicate').squeeze(0).squeeze(0)
        diff_d = (trace_padded[:-1, :-1, :-1] - trace_padded[1:, :-1, :-1]).abs()
        diff_h = (trace_padded[:-1, :-1, :-1] - trace_padded[:-1, 1:, :-1]).abs()
        diff_w = (trace_padded[:-1, :-1, :-1] - trace_padded[:-1, :-1, 1:]).abs()
        smooth = (diff_d.mean() + diff_h.mean() + diff_w.mean()) / 3.0

        # 各向异性：off-diagonal 模长相对 diagonal mass
        # 对 3D 精确构造 g  expensive；使用参数级代理：cholesky 下三角非对角元
        # l21, l31, l32 的相对大小。matrix_exp 用 h12, h13, h23 代理。
        p = self.params.squeeze(0)  # (6, D, H, W)
        diag_mass = (p[[0, 2, 5]].abs().sum(dim=0) + self.eps).clamp_min(self.eps)
        if self.parametrization == 'cholesky':
            off = p[[1, 3, 4]].abs().sum(dim=0)
        else:
            off = p[[1, 3, 4]].abs().sum(dim=0)
        aniso = (off / diag_mass).mean()

        return aniso + smooth
