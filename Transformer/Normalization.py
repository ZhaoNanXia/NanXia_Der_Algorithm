import numpy as np
import torch
import torch.nn as nn

# nn.BatchNorm1d：适用于 2D 或 3D 输入数据（如全连接层输出或序列数据）
BN1 = nn.BatchNorm1d(num_features=4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
# nn.BatchNorm2d：适用于 4D 输入数据（如卷积特征图）
BN2 = nn.BatchNorm2d(num_features=4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
# 适用于 5D 输入数据（如 3D 卷积特征图）
BN3 = nn.BatchNorm3d(num_features=4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)


# # nn.LayerNorm：适合处理任何维度的输入
# LN1 = nn.LayerNorm(normalized_shape=2, eps=1e-5, elementwise_affine=True, device=None, dtype=None)


class LayerNorm:
    """ 层归一化 """
    def __init__(self, normalized_shape, epsilon=1e-5):
        super(LayerNorm).__init__()
        self.epsilon = epsilon  # 初始化一个小常数，防止分母为0
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # 可训练缩放参数：初始化为1
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # 可训练平移参数：初始化为0

    def forward(self, x):
        # dim=-1代表在输入的最后一个维度即特征维度上计算样本均值和方差
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        print(f'mean: {mean}, var: {var}')
        # 对数据进行归一化
        x_normal = self.gamma * ((x - mean) / torch.sqrt(var + self.epsilon)) + self.beta
        return x_normal


# 使用自定义的LayerNorm
LN = LayerNorm(normalized_shape=4)
input_data = torch.randn(2, 1, 3, 4)  # [batch_size, num_channel, len_seq, num_features]
print("Input Data: ", input_data)
print("Custom LayerNorm Output: ", LN.forward(input_data))

# 使用 nn.LayerNorm
# input_data = torch.randn(1, 2, 3)  # [batch_size, len_seq, num_features]
# print("Input Data: ", input_data)
# LN1 = nn.LayerNorm(normalized_shape=3, eps=1e-5, elementwise_affine=True, device=None, dtype=None)
# print("nn.LayerNorm Output: ", LN1(input_data))


class BatchNorm:
    """ 批归一化 """
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9, training=True):
        super(BatchNorm).__init__()

        self.gamma = nn.Parameter(torch.ones(num_features))  # 可训练缩放参数：初始化为1
        self.beta = nn.Parameter(torch.zeros(num_features))  # 可训练平移参数：初始化为0
        self.epsilon = epsilon  # 初始化一个小常数，防止分母为0
        self.momentum = momentum  # 初始化动量参数，用于计算运行时均值和方差
        self.run_mean = None  # 初始化运行时均值
        self.run_var = None  # 初始化运行时方差
        self.training = training

    def forward(self, x):
        # 如果运行时均值和方差未初始化，则使用当前批次数据初始化
        if self.run_mean is None or self.run_var is None:
            self.run_mean = torch.mean(x, dim=0, keepdim=True)
            self.run_var = torch.var(x, dim=0, keepdim=True)
        # 训练模式
        if self.training:
            # dim=0代表在批次维度上计算数据的均值和方差
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True)
            print(f'batch_mean: {batch_mean}, batch_var: {batch_var}')
            # 对当前批次数据归一化
            x_normal = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            # 基于当前批次的均值和方差更新运行时均值和方差
            self.run_mean = self.momentum * self.run_mean + (1 - self.momentum) * batch_mean
            self.run_var = self.momentum * self.run_var + (1 - self.momentum) * batch_var
        # 测试模式
        else:
            # 使用运行时均值和方差归一化
            x_normal = (x - self.run_mean) / np.sqrt(self.run_var + self.epsilon)
        # 返回应用了缩放平移参数后的结果
        return self.gamma * x_normal + self.beta


# 使用自定义的BatchNorm
# BN = BatchNorm(num_features=4)
# input_data = torch.randn(2, 1, 3, 4)  # [batch_size, num_channel, len_seq, num_features]
# print("Input Data: ", input_data)
# print("Custom LayerNorm Output: ", BN.forward(input_data))
