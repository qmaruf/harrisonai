import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights

import segmentation_models_pytorch as smp


class StdUpsample(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(nout)

    def forward(self, x):
        x = self.bn(F.relu(self.conv(x)))
        return x


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ChannelReduce(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super(ChannelReduce, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        return self.conv(x)


# def convrelu(in_channels, out_channels, kernel, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
#         nn.ReLU(inplace=True),
#     )


class PetNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
    ):
        super(PetNet, self).__init__()
        # self.ups = nn.ModuleList()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = nn.Sequential(*self.base_layers[5])
        self.layer3 = nn.Sequential(*self.base_layers[6])
        self.layer4 = nn.Sequential(*self.base_layers[7])

        self.downs = [
            nn.Sequential(*self.base_layers[:3]),
            nn.Sequential(*self.base_layers[3:5]),
            self.base_layers[5],
            self.base_layers[6],
            self.base_layers[7],
        ]

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.ups = nn.ModuleList(
            [
                ConvRelu(256 + 512, 512, 3, 1),
                ConvRelu(128 + 512, 256, 3, 1),
                ConvRelu(64 + 256, 256, 3, 1),
                ConvRelu(64 + 256, 128, 3, 1),
            ]
        )

        self.conv_original_size0 = ConvRelu(3, 64, 3, 1)
        self.conv_original_size1 = ConvRelu(64, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 1, 1)

        self.r1 = nn.Conv2d(128, 64, 1)
        self.r2 = nn.Conv2d(64, 32, 1)
        self.r3 = nn.Conv2d(32, 1, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.channel_reduce = ChannelReduce()
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, 39)

    def forward(self, input):
        skip_conns = []
        x = input
        for down in self.downs:
            x = down(x)
            skip_conns.append(x)

        clf = self.avgpool(x)
        clf = clf.view(clf.size(0), -1)
        clf = self.fc(clf)
        clf = self.sigmoid(clf)

        # pdb.set_trace()
        skip_conns = skip_conns[:-1][::-1]
        for i, skip_conn in enumerate(skip_conns):
            x = self.upsample(x)
            x = torch.cat([x, skip_conn], dim=1)
            x = self.ups[i](x)

        x = self.upsample(x)

        x = self.channel_reduce(x)

        seg = self.sigmoid(x)
        return seg, clf
