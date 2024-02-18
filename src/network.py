import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ConvRelu(nn.Module):
    """
    Convolutional layer followed by ReLU
    """

    def __init__(self, in_channels, out_channels, kernel, padding):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ChannelReduce(nn.Module):
    """
    Reduce channel size to 1
    """

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


class PetNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
    ):
        """
        UNet model for pet segmentation with resnet18 backbone
        """
        super(PetNet, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_layers = list(self.base_model.children())

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

        self.conv_last = nn.Conv2d(64, 1, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.channel_reduce = ChannelReduce()
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, 39)

        self.clf_block = nn.Sequential(
            nn.Conv2d(512, 512, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 2, 2),
        )

    def forward(self, input):
        skip_conns = []
        x = input
        for down in self.downs:
            x = down(x)
            skip_conns.append(x)

        clf = self.clf_block(x)
        clf = clf.view(clf.size(0), -1)
        clf = self.fc(clf)
        clf = self.sigmoid(clf)

        skip_conns = skip_conns[:-1][::-1]
        for i, skip_conn in enumerate(skip_conns):
            x = self.upsample(x)
            x = torch.cat([x, skip_conn], dim=1)
            x = self.ups[i](x)

        x = self.upsample(x)
        x = self.channel_reduce(x)
        seg = self.sigmoid(x)

        return seg, clf
