import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from config import config
from data import DataProcessor
from loguru import logger
from network import PetNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import DiceLoss, Metrics

wandb.login()
run = wandb.init(project="harrison.ai")

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


def get_metric(dl: torch.utils.data.DataLoader):
    """
    Get precision, recall and iou metric for dataloader and log to wandb
    args:
        dl: torch.utils.data.DataLoader
    returns:
        None
    """
    metrics = Metrics()
    clf_lbls_dl, clf_preds_dl = [], []
    seg_lbls_dl, seg_prds_dl = [], []

    for data in tqdm(dl, total=len(dl.dataset) // config.batch_size, desc="metric"):
        imgs, clf_lbls, seg_lbls = data
        imgs, clf_lbls, seg_lbls = imgs.cuda(), clf_lbls.cuda(), seg_lbls.cuda()
        with torch.no_grad():
            seg_preds, clf_preds = net(imgs)

        clf_lbls_dl.append(clf_lbls.data.cpu().numpy())
        clf_preds_dl.append(clf_preds.data.cpu().numpy())

        seg_lbls_dl.append(seg_lbls.data.cpu().numpy())
        seg_prds_dl.append(seg_preds.data.cpu().numpy())

    clf_lbls_dl, clf_preds_dl = (
        np.concatenate(clf_lbls_dl),
        np.concatenate(clf_preds_dl),
    )
    seg_lbls_dl, seg_prds_dl = np.concatenate(seg_lbls_dl), np.concatenate(seg_prds_dl)

    precision, recall = metrics.get_precision_recall(clf_lbls_dl, clf_preds_dl)
    iou = metrics.get_iou_score(seg_lbls_dl, seg_prds_dl)
    logger.info(f"precision {precision:.4f} recall {recall:.4f} | iou {iou:.4f}")

    wandb.log({"precision": precision, "recall": recall, "iou": iou})


def one_epoch(dl: torch.utils.data.DataLoader, train: bool) -> float:
    """
    One epoch of training or testing for the dataloader
    args:
        dl: torch.utils.data.DataLoader
        train: bool
    returns:
        loss: float
    """
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
        clf_loss = clf_loss_func(clf_preds.float(), lbls.float()) * 5.0
        loss = (seg_loss + clf_loss) / 2.0

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        seg_losses.append(seg_loss.item())
        clf_losses.append(clf_loss.item())

        desc = f"{'train' if train else 'test'} | {np.mean(losses): .5f} | {np.mean(seg_losses): .5f} | {np.mean(clf_losses): .5f}"
        pbar.set_description(desc)

    if train:
        wandb.log(
            {
                "train_loss": np.mean(losses),
                "train_seg_loss": np.mean(seg_losses),
                "train_clf_loss": np.mean(clf_losses),
            }
        )
    else:
        wandb.log(
            {
                "test_loss": np.mean(losses),
                "test_seg_loss": np.mean(seg_losses),
                "test_clf_loss": np.mean(clf_losses),
            }
        )

    return np.mean(losses)


def train_pet_net() -> None:
    """
    Train pet net and save best model
    """
    max_loss = 1e9
    for epoch in range(config.epochs):
        _ = one_epoch(train_dl, train=True)
        test_loss = one_epoch(test_dl, train=False)
        if test_loss < max_loss:
            max_loss = test_loss
            torch.save(net, config.weight_path)
        get_metric(test_dl)
        scheduler.step(test_loss)


if __name__ == "__main__":
    train_pet_net()
