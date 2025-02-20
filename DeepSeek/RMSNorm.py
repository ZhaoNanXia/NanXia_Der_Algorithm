import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # (self.dim, )指定了归一化的形状
        output = F.rms_norm(x, (self.dim, ), self.weight, self.eps)
        return output


class RMSNorm1(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # 计算均方根 (RMS)
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        # 标准化并缩放
        return self.weight * (x / rms)


class RMSNorm2(nn.Module):
    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
        """
        :param normalized_shape: 输入张量的形状
        :param eps(float): 一个极小的常数，防止分母为零
        :param elementwise_affine(bool):若为True,
        :param device: 设备
        :param dtype: 数据类型
        """
        super().__init__()
        self.rms_norm = nn.RMSNorm(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(self, x):
        output = self.rms_norm(x)
        return output


input = torch.randn(1, 2, 2)
print(input)
rms_norm = RMSNorm(2)
rms_norm1 = RMSNorm1(2)
rms_norm2 = RMSNorm2(2)
output = rms_norm(input)
output1 = rms_norm1(input)
output2 = rms_norm2(input)
print(output, output1, output2)
