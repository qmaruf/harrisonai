import sys

sys.path.append("src")
import io
from glob import glob
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from breeds import CAT_BREEDS, DOG_BREEDS, PET_ID_TO_LABEL
from config import config
from data import Transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

net = torch.load(config.weight_path, map_location="cpu")
net.eval()


def generate_labels(clf_preds: np.ndarray) -> Dict:
    """
    Generate labels from classifier predictions
    args:
        clf_preds: np.ndarray
    returns:
        labels: dict
    """
    clf_preds = np.where(clf_preds > 0.5, 1.0, 0.0)
    labels = {"cat": 0, "dog": 0, "cat_breed": 0, "dog_breed": 0}
    for k, v in PET_ID_TO_LABEL.items():
        if clf_preds[k] == 1:
            if v in ["cat", "dog"]:
                labels[v] = 1
            else:
                if v in CAT_BREEDS:
                    labels["cat_breed"] = v
                elif v in DOG_BREEDS:
                    labels["dog_breed"] = v
    return labels


def generate_mask(seg_preds: np.ndarray) -> np.ndarray:
    """
    Generate mask from segmentation predictions
    args:
        seg_preds: np.ndarray
    returns:
        mask: np.ndarray
    """
    mask = np.where(seg_preds > 0.5, 1.0, 0.0)[0] * 255
    return mask


def preprocess(img: np.ndarray) -> torch.Tensor:
    """
    Preprocess image
    args:
        img: np.ndarray
    returns:
        padded: torch.Tensor
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = Transforms.test_transform(image=img)["image"]
    img = img / 255
    _, h, w = img.shape
    padded = torch.zeros((3, config.max_side, config.max_side))
    padded[:, 0:h, 0:w] = img
    padded = padded.unsqueeze(0)
    padded = padded.to(config.device)
    return padded


def postprocess(
    clf_preds: torch.tensor, seg_preds: torch.tensor, img: np.ndarray, img_path: str
) -> str:
    """
    Draw mask on image
    args:
        clf_preds: torch.tensor
        seg_preds: torch.tensor
        img: np.ndarray
        img_path: str
    returns:
        mask_path: str
    """
    clf_preds = clf_preds[0].data.cpu().numpy()
    seg_preds = seg_preds[0].data.cpu().numpy()
    seg_preds = generate_mask(seg_preds)

    orig_img = cv2.imread(img_path)
    scale = max(orig_img.shape[0], orig_img.shape[1]) / config.max_side
    scaled_h = int(orig_img.shape[0] / scale)
    scaled_w = int(orig_img.shape[1] / scale)
    scaled_img = img[0][:, :scaled_h, :scaled_w]
    scaled_mask = seg_preds[:scaled_h, :scaled_w]

    masked_img = draw_segmentation_masks(
        (scaled_img * 255).to(torch.uint8),
        torch.tensor(scaled_mask, dtype=bool),
        alpha=0.6,
        colors=["red"],
    )
    masked_img = to_pil_image(masked_img)
    mask_path = img_path.replace(".jpg", "_mask.jpg")
    masked_img.save(mask_path)

    img_byte_arr = io.BytesIO()
    masked_img.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def inference_img(img_path: str) -> Tuple[Dict, str]:
    """
    Inference on single image
    args:
        img_path: str
    returns:
        predicted_labels: Dict
        mask_path: str
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    img = preprocess(img)

    with torch.no_grad():
        seg_preds, clf_preds = net(img)

    predicted_labels = generate_labels(clf_preds[0].data.cpu().numpy())
    print(predicted_labels)
    masked_img_byte = postprocess(clf_preds, seg_preds, img, img_path)
    return predicted_labels, masked_img_byte


def inference_imgs():
    img_paths = glob(f"{config.imgs_path}/*/image.jpg")
    for i, img_path in enumerate(img_paths):
        if i > 10:
            break
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        img = preprocess(img)

        with torch.no_grad():
            seg_preds, clf_preds = net(img)

        clf_preds = clf_preds[0].data.cpu().numpy()
        seg_preds = seg_preds[0].data.cpu().numpy()

        clf_preds = generate_labels(clf_preds)
        seg_preds = generate_mask(seg_preds)

        img = img[0].permute(1, 2, 0).cpu()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[1].imshow(seg_preds)
        plt.title(",".join(clf_preds))
        plt.savefig(f"infs/{i}.jpg")
        plt.close()


if __name__ == "__main__":
    inference_imgs()
