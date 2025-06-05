import torch.nn as nn
from pytorch_msssim import ssim

class SSIMLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        if self.reduction == 'mean':
            loss = 1 - ssim(pred, target, data_range=1, size_average=True)
        elif self.reduction == 'sum':
            loss = 1 - ssim(pred, target, data_range=1, size_average=False)
        return loss