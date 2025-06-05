import torch
import torch.nn as nn
class GradientLoss(nn.Module):
    def __init__(self, reduction='mean', penalty='l1'):
        super(GradientLoss, self).__init__()
        self.reduction = reduction
        self.penalty = penalty

    def forward(self, pred):
        dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        if self.reduction == 'mean': d = torch.mean(dx) + torch.mean(dy)
        elif self.reduction == 'sum': d = torch.sum(dx) + torch.sum(dy)
        else: raise Exception('Invalid reduction type.')
        # grad = d / 3.0

        return d