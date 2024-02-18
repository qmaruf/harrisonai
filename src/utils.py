import numpy as np
import torch
import torch.nn as nn
from config import config


class DiceLoss(nn.Module):
    """
    Dice Loss
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: torch.tensor, targets: torch.tensor, smooth: float = 1):
        """
        Forward pass
        args:
            inputs: torch.tensor
            targets: torch.tensor
            smooth: int
        returns:
            loss: torch.tensor
        """
        targets = torch.where(targets > 0.5, 1.0, 0.0)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class Metrics:
    def __init__(self):
        pass

    def get_precision_recall(self, dl_labels, dl_outputs):
        """
        Get mean precision and recall for multi-label classification
        args:
            dl_labels: np.ndarray [all classifier labels]
            dl_outputs: np.ndarray [all classifier predictions]
        returns:
            precision: float
            recall: float
        """
        dl_outputs = np.where(dl_outputs > 0.5, 1, 0)
        precisions, recalls = [], []
        for i in range(config.n_classes):
            tp = np.sum((dl_labels[:, i] == 1) & (dl_outputs[:, i] == 1))
            fp = np.sum((dl_labels[:, i] == 0) & (dl_outputs[:, i] == 1))
            fn = np.sum((dl_labels[:, i] == 1) & (dl_outputs[:, i] == 0))
            precision = tp / (tp + fp) if (tp + fp > 0) else 0
            recall = tp / (tp + fn) if (tp + fn > 0) else 0
            precisions.append(precision)
            recalls.append(recall)

        precision = np.mean(precisions)
        recall = np.mean(recalls)
        return precision, recall

    def get_iou_score(self, seg_lbls_dl: np.ndarray, seg_prds_dl: np.ndarray, smooth=1):
        """
        Get mean IoU score
        args:
            seg_lbls_dl: np.ndarray
            seg_prds_dl: np.ndarray
            smooth: int
        returns:
            iou: float
        """

        seg_prds_dl = np.where(seg_prds_dl > 0.5, 1, 0)
        seg_lbls_dl = np.where(seg_lbls_dl > 0.5, 1, 0)

        intersection = np.logical_and(seg_lbls_dl, seg_prds_dl)
        union = np.logical_or(seg_lbls_dl, seg_prds_dl)
        iou = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
        return iou
