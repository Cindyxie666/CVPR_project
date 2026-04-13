"""IResNet (Improved ResNet) backbone used by ArcFace.

Architecture follows the insightface implementation so that pretrained
weights from the official model zoo can be loaded directly:
  https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch

The model expects 112×112 RGB input and produces L2-normalised 512-d
face embeddings.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(
            inplanes, planes, 3, stride=stride, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    """IResNet backbone for ArcFace face recognition.

    Standard configurations::

        IResNet-18:  layers=[2, 2, 2, 2]
        IResNet-50:  layers=[3, 4, 14, 3]
        IResNet-100: layers=[3, 13, 30, 3]
    """

    fc_scale = 7 * 7  # spatial size after all downsampling for 112×112 input

    def __init__(
        self,
        layers: List[int],
        dropout: float = 0.0,
        num_features: int = 512,
    ):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(IBasicBlock, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(IBasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(IBasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(IBasicBlock, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: type,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised 512-d face embeddings."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return F.normalize(x, p=2, dim=1)


# --------------- Factory helpers ---------------

def iresnet18(**kwargs) -> IResNet:
    return IResNet([2, 2, 2, 2], **kwargs)


def iresnet50(**kwargs) -> IResNet:
    return IResNet([3, 4, 14, 3], **kwargs)


def iresnet100(**kwargs) -> IResNet:
    return IResNet([3, 13, 30, 3], **kwargs)
