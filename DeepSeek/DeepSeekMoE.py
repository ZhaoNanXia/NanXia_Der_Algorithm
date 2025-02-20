import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from kernel import act_quant, weight_dequant, fp8_gemm
from config import ModelArgs


def linear(x, weight, bias=None):
    """
    对输入数据应用一个线性变换：y=xw^T+b
    量化（Quantization）是一种将数据从高精度表示转换为低精度表示的过程。
    在深度学习和计算机科学中，量化通常用于减少模型的存储空间、提高计算效率，同时尽量保持模型的性能。
    """
    if weight.element_size() > 1:
        # weight.element_size()->返回每个元素占用的字节数，如果元素大小大于1说明权重以常见的浮点数格式存储。
        return F.linear(x, weight, bias)
    elif ModelArgs.gemm_impl == "bf16":  # 如果希望使用bfloat16精度进行计算
        weight = weight_dequant(weight, weight.scale)  # 使用权重附带的缩放因子将量化的权重转换回浮点格式
        return F.linear(x, weight, bias)
    else:  # 其他情况
        # 对输入进行量化处理，返回量化后的x及缩放因子
        # block_size，预定义的常量，用于分块量化
        x, scale = act_quant(x, ModelArgs.block_size)
        # 利用fp8_gemm执行专为8位浮点设计的矩阵乘法
        y = fp8_gemm(x, scale, weight, weight.size)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """ 支持量化权重和偏置可选的线性层 """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = ModelArgs.block_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        # 如果权重中每个元素的大小为1个字节（即量化权重），则计算并创建权重缩放因子，用于权重的反量化过程
        if self.weight.element_size() == 1:
            # 计算输入（输出）特征维度在块（block_size）大小下的块数
            # 加上 self.block_size - 1 以确保即使输入（输出）特征不是block_size的整数倍也能正确计算块数
            scale_out_features = (out_features + self.block_size - 1) // self.block_size
            scale_in_features = (in_features + self.block_size - 1) // self.block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            # 注册参数 scale=None
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """ 具有列并行性的线性层，将输出特征分布到多个计算设备上，以实现分布式训练 """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        self.world_size = ModelArgs.world_size
        assert out_features % self.world_size == 0, \
            f"Output features must be divisible by world size (world_size={self.world_size})"
        # 计算每个设备处理的输出特征数量，确保设备间的均匀分布，从而实现列并行
        self.part_out_features = out_features // self.world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor):
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """ 具有行并行性的线性层，将输入特征分布到多个计算设备上，以实现分布式训练 """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        self.world_size = ModelArgs.world_size
        assert in_features % self.world_size == 0, \
            f"Input features must be divisible by world size (world_size={self.world_size})"
        # 计算每个设备处理的输入特征数量，确保设备间的均匀分布，从而实现列并行
        self.part_in_features = in_features // self.world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor):
        y = linear(x, self.weight, self.bias)
        if self.world_size > 1:  # 如果有多个进程，则使用dist.all_reduce将多个进程上的计算结果汇总合并
            dist.all_reduce(y)  # 默认求和
        if self.bias is not None:
            y += self.bias
        return y


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.layer1 = ColumnParallelLinear(dim, inter_dim)
        self.layer2 = ColumnParallelLinear(dim, inter_dim)
        self.layer3 = RowParallelLinear(inter_dim, dim)

    def forward(self, x: torch.Tensor):
        x1 = F.silu(self.layer1(x))
        x2 = self.layer2(x)
        output = self.layer3(x1 * x2)
        return output


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts  # 每个输入激活的专家数
        self.n_groups = args.n_expert_groups  # 路由专家组数量
        self.topk_groups = args.n_limited_groups  # 每个输入选择激活的路由专家组数
        self.score_func = args.score_func  # 打分函数（softmax or sigmoid）
        self.route_scale = args.route_scale  # 缩放因子
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x):
        scores = linear(x, self.weight)
        if self.score_func == 'softmax':
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        origin_scores = scores  # 保存归一化后的原始分数
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)  # 如果无偏置，则取每组的最大值
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)  # 如果有偏置，则取每组top2值之和
            # 对每个样本在组维度上选出topk_groups个组
            group_indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            # 创建一个掩码，只有选中的组位置为True
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, group_indices, True)
            # 将掩码扩展到最后一维，然后与scores相乘再展平为二维向量
            scores = (scores * mask.unsqueeze(-1).flatten(1))
        # 在所有专家中选出得分最高的topk个专家
        expert_indices = torch.topk(scores, self.topk, dim=-1)[1]
        # 从原始得分中提取出选中专家对应的权重
        weights = origin_scores.gather(1, expert_indices)
        if self.score_func == 'sigmoid':
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), expert_indices


class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        """
        :param dim: 输入特征维度
        :param hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.layer1 = nn.Linear(dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, dim)
        self.layer3 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        x1 = F.silu(self.layer1(x))  # x：[batch_size, input_dim]
        x2 = self.layer2(x1)
        x3 = self.layer3(x)
        output = x2 * x3
        return output


class DeepSeekMoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.world_size = args.world_size
        assert args.n_routed_experts % self.world_size == 0,\
            f"Number of experts must be divisible by world size (world_size={self.world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.n_local_experts = args.n_routed_experts // self.world_size  # 每个本地进程负责的专家数量
        self.experts_start_idx = args.rank * self.n_local_experts  # 当前进程所处理的起始专家索引
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts  # 当前进程所处理的终止专家索引
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim)
                                      if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor):
        shape = x.size()
        x = x.view(-1, self.dim)  # 将输入调整为[batch_size, dim]形状
        weights, indices = self.gate(x)  # 通过门控机制计算输入数据应该路由到哪些专家及对应的专家权重
        y = torch.zeros_like(x)  # 创建一个与x形状相同、数据类型相同、且所有元素都为0的张量
        # 计算每个专家被激活的次数
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        # 循环遍历当前进程负责的专家索引范围
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:  # 如果当前专家没有激活，则跳过
                continue
            expert = self.experts[i]  # 获取对应的专家模块
            idx, top = torch.where(indices == i)  # 获取属于当前专家i的输入数据的索引
            y[idx] += expert(x[idx] * weights[idx, top, None])  # 根据 输入数据并乘以对应权重的结果 进行计算
        z = self.shared_experts(x)  # 计算共享专家的输出
        if self.world_size > 1:  # 合并多个进程的结果
            dist.all_reduce(y)
        return (y + z).view(shape)  # 计算最终的输出并恢复原始张量的形状
