import torch
import torch.nn as nn
from SelfAttention import SelfAttention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, input_dim, dropout=0.0):
        """
        :param num_heads: 多头自注意力机制的头数
        :param input_dim: 输入特征维度
        :param dropout: 随机丢弃
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert input_dim % num_heads == 0, 'Input_dim cannot be divisible by the num_heads.'
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.self_attention = SelfAttention(self.head_dim)  # 自注意力层
        self.linear = nn.Linear(input_dim, input_dim)  # 输出的线性层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_feature, feature_dim = x.shape
        assert feature_dim == self.num_heads * self.head_dim, \
            f"Input feature_dim ({feature_dim}) must match num_heads ({self.num_heads}) * head_dim ({self.head_dim})."
        head_inputs = x.view(batch_size, num_feature, self.num_heads, self.head_dim).transpose(1, 2)
        head_outputs = [self.self_attention(head_inputs[:, i, :, :]) for i in range(self.head_dim)]
        concat_output = torch.cat(head_outputs, dim=-1)
        output = self.linear(concat_output)
        return output


if __name__ == '__main__':
    input_data = torch.randn(1, 2, 4)
    print('input_data:', input_data)
    model = MultiHeadSelfAttention(num_heads=2, input_dim=4)
    final_output = model(input_data)
    print(final_output)
