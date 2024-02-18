import sys

sys.path.append("src")
from unittest.mock import patch

import numpy as np
import pytest
import torch
from utils import DiceLoss, Metrics


def test_dice_loss():
    dice_loss = DiceLoss()
    inputs = torch.tensor([0.7, 0.3, 0.9], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    intersection = (inputs * targets).sum()
    expected_loss = 1 - (2.0 * intersection + 1) / (inputs.sum() + targets.sum() + 1)
    assert torch.isclose(dice_loss(inputs, targets), torch.tensor(expected_loss))


@patch("utils.config.n_classes", new=2)
def test_get_precision_recall():
    metrics = Metrics()
    dl_labels = np.array([[1, 0], [0, 1], [1, 1]])
    dl_outputs = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.8]])
    precision, recall = metrics.get_precision_recall(dl_labels, dl_outputs)

    dl_outputs = np.where(dl_outputs > 0.5, 1, 0)
    expected_precision = 1
    expected_recall = 1

    assert np.isclose(precision, expected_precision)
    assert np.isclose(recall, expected_recall)


def test_get_iou_score():
    metrics = Metrics()
    seg_lbls_dl = np.array([[1, 0, 1], [0, 1, 1]])
    seg_prds_dl = np.array([[0.6, 0.4, 0.9], [0.2, 0.8, 0.7]])
    iou = metrics.get_iou_score(seg_lbls_dl, seg_prds_dl)
    expected_iou = 1
    assert np.isclose(iou, expected_iou)


# Run the tests
if __name__ == "__main__":
    pytest.main()
