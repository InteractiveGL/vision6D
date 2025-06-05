"""
   Dense pose estimation module no. 1

   Copyright (C) 2020 Siemens AG
   SPDX-License-Identifier: MIT for non-commercial use otherwise see license terms
   Author 2020 Sergey Zakharov
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .unet_parts import up, outconv
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

__all__ = ['ResNet', 'resnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def _freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=32):
        self.inplanes = 64
        self.type = 'uv'
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # head U
        self.up1_u = up(384, 128)
        self.up2_u = up(192, 64)
        self.up3_u = up(128, 64)
        self.up4_u = up(64, 64, add_shortcut=False)

        # head V
        self.up1_v = up(384, 128)
        self.up2_v = up(192, 64)
        self.up3_v = up(128, 64)
        self.up4_v = up(64, 64, add_shortcut=False)

        # head Mask
        self.up1_mask = up(384, 128)
        self.up2_mask = up(192, 64)
        self.up3_mask = up(128, 64)
        self.up4_mask = up(64, 64, add_shortcut=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Output
        self.out_u = outconv(64, 256)
        self.out_v = outconv(64, 256)
        self.out_mask = outconv(64, num_classes)

        _freeze_module(self.conv1)
        _freeze_module(self.bn1)
        _freeze_module(self.layer1)

        # Criterions
        self.class_weights_uv = torch.ones(256)
        self.class_weights_uv[0] = 0.01
        self.class_weights = torch.ones(num_classes)
        self.class_weights[0] = 0.01

        self.criterion_uv = nn.CrossEntropyLoss(weight=self.class_weights_uv)
        self.criterion_mask_id = nn.CrossEntropyLoss(weight=self.class_weights)

        self.optimizer = optim.Adam(list(self.parameters()), weight_decay=0.00004, lr=0.0003) #TODO was lr=0.0003
        self.scheduler = MultiStepLR(self.optimizer, milestones=[500, 1000, 1500, 2000, 2500,
                                                                 3000, 3500, 4000, 4500, 5000, 5500,
                                                                 6000, 6500, 7000, 7500, 8000, 8500], gamma=0.75)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 480 * 640 * 3
        x1 = self.conv1(x)

        # 240 * 320 * 64
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.maxpool(x1)

        # 120 * 160 * 64
        x3 = self.layer1(x2)
        x3 = self.layer2(x3)

        # 60 * 80 * 128
        x4 = self.layer3(x3)

        # head U
        x_u = self.up1_u(x4, x3)
        x_u = self.up2_u(x_u, x2)
        x_u = self.up3_u(x_u, x1)
        x_u = self.up4_u(x_u, x)

        # head V
        x_v = self.up1_v(x4, x3)
        x_v = self.up2_v(x_v, x2)
        x_v = self.up3_v(x_v, x1)
        x_v = self.up4_v(x_v, x)

        # head Mask
        x_mask = self.up1_mask(x4, x3)
        x_mask = self.up2_mask(x_mask, x2)
        x_mask = self.up3_mask(x_mask, x1)
        x_mask = self.up4_mask(x_mask, x)

        u = self.out_u(x_u)
        v = self.out_v(x_v)
        mask = self.out_mask(x_mask)

        self.predicted_u, self.predicted_v, self.predicted_mask_id = u, v, mask

    def optimize(self, rgb, gt_uv, gt_mask_id):

        # Forward pass
        self.train()
        self.forward(rgb)
        self.zero_grad()

        # Define losses + backward pass
        self.loss_u = self.criterion_uv(self.predicted_u, gt_uv[:, 0])
        self.loss_v = self.criterion_uv(self.predicted_v, gt_uv[:, 1])
        self.loss_mask_id = self.criterion_mask_id(self.predicted_mask_id, gt_mask_id)

        self.loss = self.loss_u + self.loss_v + self.loss_mask_id
        self.loss.backward()
        self.optimizer.step()

        return self.loss_u + self.loss_v, self.loss_mask_id

    def read_network_output(self, single=True):
        uv = torch.stack([self.predicted_u.max(1)[1], self.predicted_v.max(1)[1]], dim=-1).squeeze()
        uv_np = uv.detach().cpu().numpy()
        if single:
            mask_id_np = ((uv > 0).sum(-1) == 2).int().detach().cpu().numpy()
        else:
            mask_id_np = self.predicted_mask_id.max(dim=1, keepdim=True)[1].squeeze().detach().cpu().numpy()

        return mask_id_np, uv_np


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

