import torch
import torch.nn as nn


class DynamicWeightManager:
    """ 动态权重管理 """
    def __init__(self, initial_weights, gammas):
        """
        Args:
            initial_weights (List[float]): 初始权重 [w_0_1, w_0_2, ...]
            gammas (List[float]): 权重更新比例 [γ_1, γ_2, ...]
        """
        self.initial_weights = torch.tensor(initial_weights)
        self.gammas = torch.tensor(gammas)
        self.epoch = 0  # 当前训练轮次

    def get_weights(self):
        """ 获取当前轮次的动态权重 """
        return self.initial_weights * (self.gammas ** self.epoch)

    def step(self):
        """ 每轮训练后调用，更新 epoch """
        self.epoch += 1


class PLELoss(nn.Module):
    def __init__(self, task_types, weight_manager):
        super(PLELoss, self).__init__()
        self.task_types = task_types
        self.weight_manager = weight_manager

        # 定义各任务的损失函数
        self.loss_funcs = nn.ModuleList()
        for task_type in task_types:
            if task_type == "classification":  # 分类任务使用交叉熵损失
                self.loss_funcs.append(nn.BCEWithLogitsLoss(reduction='none'))
            elif task_type == "regression":  # 回归任务使用MSE损失
                self.loss_funcs.append(nn.MSELoss(reduction='none'))
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

    def forward(self, pre_value, truth_value, masks):
        total_loss = 0.0
        # 获取当前动态权重
        weights = self.weight_manager.get_weights().to(truth_value.device)
        for task_idx, (pre_v, loss_fn) in enumerate(zip(pre_value, self.loss_funcs)):
            # 提取任务标签和掩码
            label = truth_value[:, task_idx]  # [batch_size]
            mask = masks[:, task_idx]  # [batch_size]
            valid_samples = mask.sum()  # 有效样本数
            if valid_samples == 0:
                continue  # 跳过无有效样本的任务
            # 计算逐样本损失（未求平均）
            task_loss = loss_fn(pre_value, label)  # [batch_size]
            # 应用掩码并求平均
            masked_loss = (task_loss * mask).sum() / valid_samples
            # 加权损失
            weighted_loss = masked_loss * weights[task_idx]
            total_loss += weighted_loss
        return total_loss
