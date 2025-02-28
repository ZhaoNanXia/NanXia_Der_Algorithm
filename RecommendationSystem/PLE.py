import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List


@dataclass
class ModelArgs:
    """
    Args:
        input_dim: 原始输入的维度
        expert_hidden_dim: 专家网络的隐藏维度
        expert_dim: 专家网络的输出维度
        num_specific_experts: 任务特定专家的数量
        num_shared_experts: 共享专家的数量
        num_tasks: 任务数量
        num_layers: 多层抽取网络的层数
        tower_hidden_dim: 塔网络的隐藏维度
    """
    input_dim: int = 4
    expert_dim: int = 4
    num_specific_experts: int = 2
    num_shared_experts: int = 2
    num_tasks: int = 2
    num_layers: int = 2
    tower_hidden_dim: int = 2
    expert_hidden_dim: int = 8


class ExpertNet(nn.Module):
    """ 专家网络模块：由两个线性层组成 """
    def __init__(self, input_dim: int, expert_hidden_dim: int, expert_dim: int, dropout: float = 0.):
        super(ExpertNet, self).__init__()
        self.expert_layer = nn.Sequential(
            nn.Linear(input_dim, expert_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden_dim, expert_dim)
        )

    def forward(self, x: torch.Tensor):
        output = self.expert_layer(x)
        return output


class GateNet(nn.Module):
    """
    门控网络模块: 线性层＋Softmax
    PLE中门控网络有两处用途：用于任务特定专家的门控网络和用于共享专家的门控网络
    """
    def __init__(self, input_dim: int, num_experts: int):
        super(GateNet, self).__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        """ 根据输入x通过门控网络计算加权权重 """
        gate_weights = self.gate_layer(x)  # [batch_size, num_experts]
        return gate_weights


class ExtractionNet(nn.Module):
    """
    单层提取网络(Extraction networks),用于PLE中构建多层提取网络(Multi-level extraction networks)
    """
    def __init__(self, input_dim: int, args: ModelArgs):
        super(ExtractionNet, self).__init__()
        self.num_tasks = args.num_tasks

        """ 特定于任务的专家网络组：每个任务都有num_specific_expert个专家，一共有num_tasks个任务"""
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([ExpertNet(input_dim, args.expert_hidden_dim, args.expert_dim)
                           for _ in range(args.num_specific_experts)])
            for _ in range(args.num_tasks)
        ])
        """ 共享专家组：所有任务通用,一共num_shared_experts个专家 """
        self.shared_experts = nn.ModuleList([ExpertNet(input_dim, args.expert_hidden_dim, args.expert_dim)
                                             for _ in range(args.num_shared_experts)])
        """ 每个任务的门控网络：每个任务一个，一共有num_tasks个门控网络 """
        self.task_gates = nn.ModuleList([
            GateNet(input_dim, args.num_specific_experts + args.num_shared_experts)
            for _ in range(args.num_tasks)
        ])
        """ 用于计算共享表示的门控网络，每层一个，共num_layers个 """
        self.shared_gate = GateNet(input_dim, args.num_shared_experts + args.num_specific_experts * args.num_tasks)

    def forward(self, task_reps: List[torch.Tensor], shared_rep: torch.Tensor):
        # 任务表示->task_reps: [num_tasks, batch_size, input_dim]
        # 共享表示->shared_rep: [batch_size, input_dim]
        """ 计算共享专家的输出，只需计算一次 """
        shared_experts_output = torch.stack([expert(shared_rep) for expert in self.shared_experts], dim=0)
        # 输出维度->[num_shared_experts, batch_size, expert_dim]

        """ 计算任务特定专家门控网络的输出：每个任务都要计算一次 """
        task_experts_output = [torch.stack([expert(task_reps[task_idx])
                                            for expert in self.specific_experts[task_idx]], dim=0)
                               for task_idx in range(self.num_tasks)]
        # 输出维度->[num_tasks, num_specific_experts, batch_size, expert_dim]

        """ 更新任务表示 """
        new_task_reps = []
        for task_idx in range(self.num_tasks):
            # 输出维度->[num_specific_experts + num_shared_experts, batch_size, expert_dim]
            experts_output = torch.cat([shared_experts_output, task_experts_output[task_idx]], dim=0)
            # 计算任务门控权重, 输出维度->[batch_size, num_specific_experts + num_shared_experts]
            task_gate_weights = self.task_gates[task_idx](task_reps[task_idx])
            # 加权求和
            new_task_rep = torch.einsum('ebd,be->bd', experts_output, task_gate_weights)
            new_task_reps.append(new_task_rep)

        """ 更新共享表示 """
        # 输出维度->[num_tasks * num_specific_experts + num_shared_experts, batch_size, expert_dim]
        all_experts_output = torch.cat([shared_experts_output] + task_experts_output, dim=0)
        # 共享门控权重, 输出维度->[batch_size, num_tasks * num_specific_experts + num_shared_experts]
        shared_gate_weights = self.shared_gate(shared_rep)
        # 加权求和
        new_shared_rep = torch.einsum('ebd,be->bd', all_experts_output, shared_gate_weights)

        return new_task_reps, new_shared_rep


class TowerNet(nn.Module):
    """ 任务塔网络: 用于每个任务输出最终的结果 """
    def __init__(self, input_dim: int, tower_hidden_dim: int):
        super(TowerNet, self).__init__()
        self.tower_layer = nn.Sequential(
            nn.Linear(input_dim, tower_hidden_dim),
            nn.ReLU(),
            nn.Linear(tower_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        return self.tower_layer(x)


class PLE(nn.Module):
    """ Progressive Layered Extraction (PLE) 主模型 """
    def __init__(self, args: ModelArgs):
        super(PLE, self).__init__()
        self.num_tasks = args.num_tasks
        self.num_layers = args.num_layers

        """ 多层提取网络：一共有num_layers层 """
        current_input_dim = args.input_dim  # 第一层时专家的输入维度为原始输入特征维度
        self.Extraction_layers = nn.ModuleList()
        for _ in range(args.num_layers):
            self.Extraction_layers.append(ExtractionNet(current_input_dim, args))
            current_input_dim = args.expert_dim  # 从第二层开始专家的输入维度为上一层专家的输出维度

        """ 任务塔网络：每个任务一个，一共有num_tasks个 """
        self.task_towers = nn.ModuleList([TowerNet(args.expert_dim, args.tower_hidden_dim)
                                          for _ in range(args.num_tasks)])

    def forward(self, x: torch.Tensor):
        # x: 原始输入,[batch_size, input_dim]
        shared_rep = x  # 用于输入共享专家的共享表示向量
        # 用于输入任务特定专家的任务表示向量，每个任务都有一个，一共num_tasks个
        task_reps = [x.clone() for _ in range(self.num_tasks)]  # [num_tasks, batch_size, input_dim]
        """ 通过多层提取网络计算输出的任务表示和共享表示 """
        for extraction_layer in self.Extraction_layers:
            task_reps, shared_rep = extraction_layer(task_reps, shared_rep)
        """ 仅利用最终的任务表示向量通过任务塔网络计算每个任务的输出结果 """
        task_outputs = []
        for task_idx in range(self.num_tasks):
            # task_reps[task_idx]：[batch_size, expert_dim]
            task_output = self.task_towers[task_idx](task_reps[task_idx])  # [batch_size, 1]
            task_outputs.append(task_output.squeeze(-1))  # [batch_size]
        return task_outputs


if __name__ == '__main__':
    arg = ModelArgs()
    model = PLE(arg)
    # 随机生成一个特征向量，batch_size=1, input_dim=4
    Input = torch.randn(1, arg.input_dim)
    final_outputs = model(Input)
    print(final_outputs)
