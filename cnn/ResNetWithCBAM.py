import torch
import torch.nn as nn
from cnn.attention_layer import CBAM

class ResNetWithCBAM(nn.Module):
    def __init__(self, base_model):
        super(ResNetWithCBAM, self).__init__()
        self.base_model = base_model
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.cbam1(self.base_model.layer1(x))
        x = self.cbam2(self.base_model.layer2(x))
        x = self.cbam3(self.base_model.layer3(x))
        x = self.cbam4(self.base_model.layer4(x))

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x



