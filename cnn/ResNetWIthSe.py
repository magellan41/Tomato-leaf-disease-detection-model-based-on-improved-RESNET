import torch
import torch.nn as nn

from cnn.attention_layer import SqueezeExcitation


class ResNetWithSE(nn.Module):
    def __init__(self, base_model, reduction_ratio=16):
        super(ResNetWithSE, self).__init__()
        self.reduction_ratio = reduction_ratio

        # 保留原始 ResNet 的前几层
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # 包装每个 layer，加入 SE 模块
        self.layer1 = self._add_se_to_layer(base_model.layer1, 64)
        self.layer2 = self._add_se_to_layer(base_model.layer2, 128)
        self.layer3 = self._add_se_to_layer(base_model.layer3, 256)
        self.layer4 = self._add_se_to_layer(base_model.layer4, 512)

        # 保留全局池化和全连接层
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc

    def _add_se_to_layer(self, layer, in_channels):
        """为 ResNet 的每个 layer 添加 SE 模块"""
        layers = []
        for block in layer:
            layers.append(block)  # 原始 Block
            layers.append(SqueezeExcitation(in_channels, self.reduction_ratio))  # 添加 SE 模块
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x