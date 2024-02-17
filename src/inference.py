import torch
from data import DataProcessor
from glob import glob
import cv2
from config import config
from data import Transforms
import numpy as np
import torch
import matplotlib.pyplot as plt

pet_id_to_labels = DataProcessor().pet_id_to_label
img_paths = glob(f"{config.imgs_path}/*/image.jpg")
net = torch.load("model.pth").to(config.device)
net.eval()

def generate_labels(clf_preds):
    clf_preds = np.where(clf_preds > 0.5, 1.0, 0.0)
    labels = []
    for k, v in pet_id_to_labels.items():
        if clf_preds[k] == 1:
            labels.append(v)
    return labels

def generate_mask(seg_preds):
    seg_preds = np.where(seg_preds > 0.5, 1.0, 0.0)[0] * 255
    return seg_preds

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = Transforms.test_transform(image=img)["image"]
    img = img / 255
    _, h, w = img.shape
    padded = torch.zeros((3, config.max_side, config.max_side))
    padded[:, 0:h, 0:w] = img
    padded = padded.unsqueeze(0)
    padded = padded.to(config.device)
    return padded

def inference():
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

inference()
