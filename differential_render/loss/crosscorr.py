import torch
import torch.nn as nn

class CrossCorrLoss(nn.Module):
    def __init__(self):
        super(CrossCorrLoss, self).__init__()

    def forward(self, pred, target):
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        pred_var = pred - pred_mean
        target_var = target - target_mean
        
        nominator = torch.sum(pred_var * target_var) ** 2
        denominator = torch.sum(pred_var * pred_var) + torch.sum(target_var * target_var) + 1e-8
        
        cc = nominator / denominator / torch.numel(pred)
        
        return -cc