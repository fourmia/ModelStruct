'''
Author: LZD
Date: 2021-09-08 15:14:01
LastEditTime: 2021-09-08 15:37:05
LastEditors: Please set LastEditors
Description: create loss func
FilePath: \A模型框架\XNnowcasting\loss\loss.py
'''
import torch
import torch.nn as nn

class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        pass

