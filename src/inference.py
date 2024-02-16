import torch
from data import DataProcessor, Utils
from glob import glob
import cv2
from config import config
from data import Transforms
import numpy as np
import matplotlib.pyplot as plt


pet_id_to_labels = DataProcessor().pet_id_to_label
# train_dl, test_dl = Utils.get_data_loader(
#     imgs_path=config.imgs_path, info_path=config.info_path
# )

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


def inference():
    for i, img_path in enumerate(img_paths):
        if i > 10:
            break
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_raw = img.copy()
        img = Transforms.test_transform(image=img)["image"]
        img = img / 255
        _, h, w = img.shape
        square = torch.zeros((3, 256, 256))
        square[:, 0:h, 0:w] = img
        square = square.unsqueeze(0)
        square = square.to(config.device)

        with torch.no_grad():
            clf_preds, seg_preds = net(square)
        clf_preds = clf_preds[0].data.cpu().numpy()
        seg_preds = seg_preds[0].data.cpu().numpy()

        clf_preds = generate_labels(clf_preds)
        seg_preds = generate_mask(seg_preds)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_raw)
        axes[1].imshow(seg_preds)
        plt.title(",".join(clf_preds))
        plt.savefig(f"infs/{i}.jpg")
        plt.close()

    # for _, row in test_df.iterrows():
    #     # print (row)
    #     img_id = row["Sample_ID"]
    #     img_path = f"{config.data_path}/{img_id}/image.jpg"
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = test_transform(image=img)["image"]
    #     img = img / 255
    #     _, h, w = img.shape
    #     square = torch.zeros((3, 256, 256))
    #     square[:, 0:h, 0:w] = img
    #     square = square.unsqueeze(0)
    #     square = square.to(config.device)
    #     preds = net(square)[0]
    #     preds = preds.data.cpu().numpy()
    #     preds = np.where(preds > 0.5, 1.0, 0.0)
    #     preds = generate_labels(preds, pet_id_to_labels)
    #     print(row["Breed"], "|", preds)


inference()
