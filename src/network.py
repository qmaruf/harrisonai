import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class PetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

        self.seg_head = nn.Sequential(
            nn.Unflatten(1, (16, 32, 1)),
            nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=False),
        )

        self.clf_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 39),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.backbone(x)
        clf = self.clf_head(x)
        seg = self.seg_head(x)
        return clf, seg
