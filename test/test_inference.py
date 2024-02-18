import sys

sys.path.append("src")
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from inference import generate_labels, generate_mask, postprocess, preprocess


def test_generate_labels():
    clf_preds = np.array([0.6, 0.7] + [0.1] * 37)
    labels = generate_labels(clf_preds)
    assert labels["cat"] == 1
    assert labels["dog"] == 1


def test_generate_mask():
    side = 256
    seg_preds = np.random.rand(1, side, side)
    mask = generate_mask(seg_preds)
    assert mask.shape == (side, side)


def test_preprocess():
    side = 256
    img = np.random.randint(0, 255, (side, side, 3), dtype=np.uint8)
    preprocessed_img = preprocess(img)
    assert preprocessed_img.shape == torch.Size([1, 3, side, side])


@patch("cv2.imread")
@patch("PIL.Image.Image.save", MagicMock())
def test_postprocess(mock_imread):
    side = 256
    mock_imread.return_value = np.random.randint(
        0, 255, (side, side, 3), dtype=np.uint8
    )
    clf_preds = torch.randn(1, 39)
    seg_preds = torch.randn(1, 1, 224, 224)
    img = torch.randn(1, 3, 224, 224)
    img_path = "path/to/image.jpg"

    img_byte_arr = postprocess(clf_preds, seg_preds, img, img_path)
    assert isinstance(img_byte_arr, bytes)
