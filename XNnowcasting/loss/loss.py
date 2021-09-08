'''
Author: LZD
Date: 2021-09-08 15:14:01
LastEditTime: 2021-09-08 15:15:21
LastEditors: Please set LastEditors
Description: create loss func
FilePath: \A模型框架\XNnowcasting\loss\loss.py
'''


class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = torch.ones_like(input) * balancing_weights[0]  # 阈值初始化为1
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]  # loss中采用pixel，因此需要将雨量阈值转化为对应的像素阈值
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()    # 此步是为了根据目标雨量的大小生成权重网格，weights+差值的目的在于更新较大阈值
        weights = weights * mask.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))

