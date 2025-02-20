import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        """
        :param input_dim: 输入特征的维度
        :param dropout: 随机丢弃
        """
        super(SelfAttention, self).__init__()
        self.scale = input_dim ** -0.5  # 缩放
        self.projections = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(3)])  # 线性映射层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 通过线性映射层生成q,k,v
        q, k, v = [proj(x) for proj in self.projections]
        # q与k^T相乘并进行缩放
        attn = torch.matmul(q, k.transpose(1, 2)) * self.scale
        if mask is not None:
            # mask用于处理序列数据时标记哪些位置的元素应该被忽略或者不参与计算
            # 将mask中值为0对应的attn张量中的值替换为无穷大，这些值经softmax后注意力权重为0
            attn = attn.mask_fill(mask == 0, float('inf'))
        attn = attn.softmax(dim=-1)  # 在最后一个维度上进行softmax
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # 注意力分数加权v
        return output


if __name__ == '__main__':
    model = SelfAttention(input_dim=2)  # input_dim必须与输入样本的特征维度（feature_dim）一致
    # 随机生成一个输入样本[batch_size, num_feature, feature_dim]
    input_data = torch.randn(1, 4, 2)
    final_output = model(input_data)
    print('final_output:', final_output)
