import lap
import numpy as np
import pandas as pd
import torch
from torch import nn

class CCCLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # 计算均值，保持维度便于后续计算
        mean_true = torch.mean(y_true, dim=1, keepdim=True)
        mean_pred = torch.mean(y_pred, dim=1, keepdim=True)

        # 计算方差
        var_true = torch.var(y_true, dim=1, unbiased=False)
        var_pred = torch.var(y_pred, dim=1, unbiased=False)

        # 计算协方差
        covariance = torch.mean((y_true - mean_true) * (y_pred - mean_pred), dim=1)

        # 计算CCC
        numerator = 2 * covariance
        denominator = var_true + var_pred + torch.squeeze((mean_true - mean_pred) ** 2) + self.eps

        ccc = numerator / denominator

        # 处理数值不稳定
        ccc = torch.clamp(ccc, -1.0, 1.0)

        # 计算平均CCC
        mean_ccc = torch.mean(ccc)

        # 损失 = 1 - CCC
        loss = 1.0 - mean_ccc

        return loss



class SmoothCCCLoss(nn.Module):
    def __init__(self, alpha, eps = 1e-12):
        super().__init__()
        self.eps = eps
        self.alpha = alpha

    def exponential_smoothing(self, data):
        smoothed_data = torch.zeros_like(data)
        smoothed_data[0] = data[0]
        for t in range(1, data.size(0)):
            smoothed_data[t] = self.alpha * data[t] + (1 - self.alpha) * smoothed_data[t - 1]
        return smoothed_data

    def forward(self, y_pred, y_true):
        # 应用指数平滑
        y_true = self.exponential_smoothing(y_true)
        y_pred = self.exponential_smoothing(y_pred)

        # 计算均值，保持维度便于后续计算
        mean_true = torch.mean(y_true, dim=1, keepdim=True)
        mean_pred = torch.mean(y_pred, dim=1, keepdim=True)

        # 计算方差
        var_true = torch.var(y_true, dim=1, unbiased=False)
        var_pred = torch.var(y_pred, dim=1, unbiased=False)

        # 计算协方差
        covariance = torch.mean((y_true - mean_true) * (y_pred - mean_pred), dim=1)

        # 计算CCC
        numerator = 2 * covariance
        denominator = var_true + var_pred + torch.squeeze((mean_true - mean_pred) ** 2) + self.eps

        ccc = numerator / denominator

        # 处理数值不稳定
        ccc = torch.clamp(ccc, -1.0, 1.0)

        # 计算平均CCC
        mean_ccc = torch.mean(ccc)

        # 损失 = 1 - CCC
        loss = 1.0 - mean_ccc

        return loss


class SmoothMSELoss(nn.Module):
    def __init__(self, alpha, eps = 1e-12):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.criterion = nn.MSELoss()

    def exponential_smoothing(self, data):
        smoothed_data = torch.zeros_like(data)
        smoothed_data[0] = data[0]
        for t in range(1, data.size(0)):
            smoothed_data[t] = self.alpha * data[t] + (1 - self.alpha) * smoothed_data[t - 1]
        return smoothed_data

    def forward(self, y_pred, y_true):
        # 应用指数平滑
        y_true = self.exponential_smoothing(y_true)
        y_pred = self.exponential_smoothing(y_pred)

        loss = self.criterion(y_pred, y_true)

        return loss
