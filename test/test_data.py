import sys

sys.path.append("src")
import numpy as np
import torch
from config import config
from data import Transforms


def test_train_transform():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    transformed_image = Transforms.train_transform(image=image)["image"]
    assert transformed_image.shape == torch.Size([3, config.max_side, config.max_side])
    assert isinstance(transformed_image, torch.Tensor)


def test_test_transform():
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    transformed_image = Transforms.test_transform(image=image)["image"]
    assert transformed_image.shape == torch.Size([3, config.max_side, config.max_side])
    assert isinstance(transformed_image, torch.Tensor)


def test_mask_transform():
    mask = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
    transformed_mask = Transforms.mask_transform(image=mask)["image"]
    assert transformed_mask.shape == torch.Size([1, config.max_side, config.max_side])
    assert isinstance(transformed_mask, torch.Tensor)
