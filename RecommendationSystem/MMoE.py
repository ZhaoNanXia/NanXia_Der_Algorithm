import torch
import torch.nn as nn


class Expert(nn.Module):
    """ 单个专家网络 """
    def __init__(self, input_dim, expert_dim, dropout=0.1):
        """
        :param input_dim: 输入特征的维度
        :param expert_dim: 专家网络的输出维度
        """
        super(Expert, self).__init__()
        self.expert_layer = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.expert_layer(x)


class Gate(nn.Module):
    """ 单个门控网络 """
    def __init__(self, input_dim, num_experts, dropout=0.1):
        """
        :param input_dim: 输入特征的维度
        :param num_experts: 专家数量，输出维度要与专家数量匹配，得以使每个专家都有一个权重
        """
        super(Gate, self).__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.gate_layer(x)


class Task(nn.Module):
    """ 任务塔网络 """
    def __init__(self, input_dim, output_dim, dropout=0.1):
        """
        :param input_dim: 输入特征的维度
        :param output_dim: 输出维度
        """
        super(Task, self).__init__()
        self.task_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.task_layer(x)


class MMoE(nn.Module):
    """ Multi-gate Mixture-of-Experts """
    def __init__(self, input_dim, expert_dim, num_experts, num_tasks, num_gates):
        """
        :param input_dim: 输入特征的维度
        :param expert_dim: 专家网络的输出维度
        :param num_experts: 专家数量
        :param num_tasks: 任务数量
        :param num_gates: 门控网络的数量
        """
        super(MMoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, expert_dim) for _ in range(num_experts)])
        self.gates = nn.ModuleList([Gate(input_dim, num_experts) for _ in range(num_gates)])
        self.task_layers = nn.ModuleList([Task(expert_dim, 1) for _ in range(num_tasks)])

    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts]  # 计算所有专家神经网络的输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # 堆叠所有专家神经网络的输出
        final_outputs = []  # 用于存储每个任务的最终输出结果
        '''计算每个任务的输出'''
        for i, gate in enumerate(self.gates):
            gate_weight = gate(x)  # 计算每个门控神经网络的输出
            # 使用门控神经网络的输出加权专家神经网络的输出
            # 使用unsqueeze()增加一个维度的原因是expert_outputs为堆叠所有专家输出之后的结果，需要保持维度一致才能加权计算
            weight_output = expert_outputs * gate_weight.unsqueeze(-1)
            combined_output = torch.sum(weight_output, dim=1)  # 对加权后的专家网络输出求和
            # 计算任务塔网络的输出
            task_output = self.task_layers[i](combined_output)
            # 移除最后一个维度并将结果添加进最终输出列表中
            final_outputs.append(task_output.squeeze(-1))
        return final_outputs


if __name__ == '__main__':
    input_data = torch.randn(1, 4)  # 随机生成一个张量，batch_size=1, feature_dim=4
    print(input_data)
    model = MMoE(input_dim=4, expert_dim=8, num_experts=3, num_tasks=3, num_gates=3)
    output = model(input_data)
    print(f'final_output: {output}')
