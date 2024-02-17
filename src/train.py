from loguru import logger
import torch.nn as nn
import torch
import cv2
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
from data import DataProcessor
from config import config
from network import PetNet
from utils import DiceLoss, Metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

# wandb.login()
# run = wandb.init(project="harrison.ai")

train_dl, test_dl = DataProcessor().get_data_loader()
net = PetNet().to(config.device)
seg_loss_func = DiceLoss()
clf_loss_func = nn.BCELoss()
optimizer = optim.SGD(
    [
        dict(params=net.parameters(), lr=config.learning_rate),
    ]
)
scheduler = ReduceLROnPlateau(optimizer, "min")


def get_metric(dl):
    metrics = Metrics()
    clf_lbls_dl, clf_preds_dl = [], []
    for data in tqdm(dl, total=len(dl.dataset) // config.batch_size, desc="metric"):
        imgs, clf_lbls, seg_lbls = data
        imgs, clf_lbls, seg_lbls = imgs.cuda(), clf_lbls.cuda(), seg_lbls.cuda()
        with torch.no_grad():
            seg_preds, clf_preds = net(imgs)

        clf_preds = torch.where(clf_preds > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        clf_lbls_dl.append(clf_lbls.data.cpu().numpy())
        clf_preds_dl.append(clf_preds.data.cpu().numpy())

    clf_lbls_dl = np.concatenate(clf_lbls_dl)
    clf_preds_dl = np.concatenate(clf_preds_dl)
    metrics_dict = metrics.get_precision_recall(clf_lbls_dl, clf_preds_dl)
    logger.info(
        f'precision {metrics_dict["precision"]:.4f} recall {metrics_dict["recall"]:.4f}'
    )


def one_epoch(dl, train):
    losses = []
    seg_losses, clf_losses = [], []

    pbar = tqdm(total=len(dl.dataset) // config.batch_size)
    for data in dl:
        pbar.update(1)
        imgs, lbls, segs = data
        imgs, lbls, segs = imgs.cuda(), lbls.cuda(), segs.cuda()

        if train:
            seg_preds, clf_preds = net(imgs)
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                seg_preds, clf_preds = net(imgs)

        seg_loss = seg_loss_func(seg_preds, segs)
        clf_loss = clf_loss_func(clf_preds.float(), lbls.float())
        loss = (seg_loss + clf_loss) / 2.0

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        seg_losses.append(seg_loss.item())
        clf_losses.append(clf_loss.item())

        desc = f"{'train' if train else 'test'} | {np.mean(losses): .5f} | {np.mean(seg_losses): .5f} | {np.mean(clf_losses): .5f}"
        pbar.set_description(desc)

    return np.mean(losses)


def train_pet_net():
    for epoch in range(config.epochs):
        train_loss = one_epoch(train_dl, train=True)
        torch.save(net, "model.pth")
        test_loss = one_epoch(test_dl, train=False)
        get_metric(test_dl)
        scheduler.step(test_loss)

        # wandb.log(
        #     {
        #         "train_loss": train_loss,
        #         "test_loss": test_loss,
        #     }
        # )


if __name__ == "__main__":
    train_pet_net()
