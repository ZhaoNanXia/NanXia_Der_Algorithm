import torch
import torch.nn as nn
import math

from matplotlib import pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 30):
        """
        :param d_model: 输入词嵌入向量(token)的维度
        :param dropout: 随机丢弃
        :param max_len: 可以根据需求调整，适用于不同长度的序列
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)  # 初始化一个dropout层

        """ 计算位置编码并将其存储在pe张量中 """
        pe = torch.zeros(max_len, d_model)  # 创建一个大小为max_len×d_model的全零张量
        # 生成一个从0到max_len的整数序列，并添加一个维度
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算频率因子div_term，用于缩放不同位置的正余弦函数
        # torch.arange()首先生成一个偶数索引，.float()转换为浮点类型
        # math.log()用于计算自然对数，底数为e，因为e^((-i/d_model)*ln(10000))=10000^(-i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # 生成位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用 cos
        draw_positional_encoding(pe)  # 可视化位置编码的效果
        # 在第一个维度添加一个维度batch_size，并调整形状为 (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 将位置编码向量pe注册存储为模型的缓冲区，为模型的一部分而不会被自动梯度更新以及优化器处理
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # 将位置编码与输入嵌入相加，其中可以动态截取与输入序列等长的位置编码，无需额外处理
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        # 初始化一个可学习的位置编码参数
        self.pos_embed = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        # 将可学习的位置编码与输入嵌入相加
        return x + self.pos_embed[:x.size(1), :]


def draw_positional_encoding(absolute_positional_encoding):
    # 展示绝对位置编码的效果
    plt.figure(figsize=(12, 8))
    plt.imshow(absolute_positional_encoding, cmap='viridis')
    plt.colorbar()
    plt.title("Absolute Positional Encoding", fontsize=16)
    plt.xlabel("d_model dimensions", fontsize=16)
    plt.ylabel("Position in Sentence", fontsize=16)
    plt.xticks(fontsize=16)  # 设置x轴刻度字体大小
    plt.yticks(fontsize=16)  # 设置y轴刻度字体大小
    plt.show()


if __name__ == '__main__':
    # 随机生成一个词嵌入张量,形状为[seq_len, batch_size, d_model]
    # seq_len（序列长度）：表示输入序列中的 token 数量。例如，如果输入一个包含 60 个单词的句子，seq_len 就是 60。
    # batch_size（批量大小）：表示同时处理的样本数量。例如，如果一次处理 32 个句子，batch_size 就是 32。
    # d_model（模型维度）：表示每个 token 的嵌入维度。这是模型中每个 token 的特征向量的大小。
    input = torch.randn(30, 1, 64)
    print('input:', input)
    model = PositionalEncoding(64)
    output = model(input)
    print('output:', output)

