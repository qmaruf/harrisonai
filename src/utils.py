import torch.nn as nn
import torch
from data import DataProcessor
import numpy as np
from config import config


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
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
        pet_id_to_labels = DataProcessor().pet_id_to_label
        metrics_dict = {}
        precisions, recalls = [], []
        for i in range(config.n_classes):
            tp = np.sum((dl_labels[:, i] == 1) & (dl_outputs[:, i] == 1))
            fp = np.sum((dl_labels[:, i] == 0) & (dl_outputs[:, i] == 1))
            fn = np.sum((dl_labels[:, i] == 1) & (dl_outputs[:, i] == 0))
            precision = tp / (tp + fp) if (tp + fp > 0) else 0
            recall = tp / (tp + fn) if (tp + fn > 0) else 0
            precisions.append(precision)
            recalls.append(recall)
            metrics_dict[pet_id_to_labels[i]] = {
                "precision": precision,
                "recall": recall,
            }
        metrics_dict["precision"] = np.mean(precisions)
        metrics_dict["recall"] = np.mean(recalls)
        return metrics_dict
