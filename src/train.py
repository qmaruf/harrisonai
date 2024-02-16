from loguru import logger
import torch.nn as nn
import torch
import cv2
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import wandb
from data import Utils
from config import config
from network import PetNet

# wandb.login()
# run = wandb.init(project="harrison.ai")

train_dl, test_dl = Utils.get_data_loader(
    imgs_path=config.imgs_path, info_path=config.info_path
)

from time import time


class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = outputs.view(-1)
        outputs = torch.where(outputs > 0.5, 1.0, 0.0)

        targets = targets.view(-1)
        targets = torch.where(targets > 0.5, 1.0, 0.0)

        # intersection = (outputs * targets).sum()
        # total = (outputs + targets).sum()
        # union = total - intersection
        # IoU = (intersection + self.smooth) / (union + self.smooth)

        intersection = (outputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            outputs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


net = PetNet().to(config.device)
clf_criterion = nn.BCELoss()
mse_criterion = nn.MSELoss()
seg_criterion = JaccardLoss()
optimizer = optim.SGD(net.parameters(), lr=config.learning_rate)
scheduler = ReduceLROnPlateau(optimizer, "min")


def run_one_epoch(dl, train):
    losses = []
    clf_losses, mse_losses, seg_losses = [], [], []

    pbar = tqdm(total=len(dl.dataset) // config.batch_size)
    for data in dl:
        pbar.update(1)
        inputs, clf_labels, seg_labels = data

        inputs = inputs.to(config.device)
        clf_labels = clf_labels.to(torch.float32).to(config.device)
        seg_labels = seg_labels.to(torch.float32).to(config.device)

        optimizer.zero_grad()

        if train:
            clf_preds, seg_preds = net(inputs)
        else:
            with torch.no_grad():
                clf_preds, seg_preds = net(inputs)

        clf_loss = clf_criterion(clf_preds.float(), clf_labels.float()) * 5.0
        mse_loss = mse_criterion(seg_preds, seg_labels) * 5.0
        seg_loss = seg_criterion(seg_preds, seg_labels)
        loss = (clf_loss + mse_loss + seg_loss) / 3

        losses.append(loss.item())
        clf_losses.append(clf_loss.item())
        mse_losses.append(mse_loss.item())
        seg_losses.append(seg_loss.item())

        desc = f'{"train" if train else "test"} | {np.mean(losses[-100:]): .4f} | clf {np.mean(clf_losses[-100:]): .4f} | seg {np.mean(seg_losses[-100:]): .4f} | mse {np.mean(mse_losses[-100:]): .4f}'
        pbar.set_description(desc)

        if train:
            loss.backward()
            optimizer.step()

    return np.mean(losses)


def run_eval(dl):
    net.eval()
    dl_labels, dl_outputs = [], []
    for data in tqdm(dl, total=len(dl.dataset) // config.batch_size):
        inputs, clf_labels, seg_labels = data
        # import pdb

        # pdb.set_trace()
        inputs = inputs.to(config.device)

        with torch.no_grad():
            clf_preds, seg_preds = net(inputs)

        # import pdb

        # pdb.set_trace()

        seg_pred = seg_preds[0][0]
        seg_pred = torch.where(seg_pred > 0.5, 1.0, 0.0) * 255
        seg_pred = seg_pred.data.cpu().numpy()
        cv2.imwrite(f"debug/{time()}.jpg", seg_pred)

        clf_labels = clf_labels.data.cpu().numpy()
        clf_preds = clf_preds.data.cpu().numpy()
        clf_preds = np.where(clf_preds > 0.5, 1.0, 0.0)

        dl_labels.append(clf_labels)
        dl_outputs.append(clf_preds)

    dl_labels = np.concatenate(dl_labels)
    dl_outputs = np.concatenate(dl_outputs)
    metrics_dict = Utils.get_metrics(dl_labels, dl_outputs)

    return metrics_dict


def train_pet_net():
    for epoch in range(config.epochs):
        net.train()
        train_loss = run_one_epoch(train_dl, train=True)
        net.eval()
        test_loss = run_one_epoch(test_dl, train=False)
        scheduler.step(test_loss)
        metrics_dict = run_eval(train_dl)

        torch.save(net, "model.pth")

        logger.info(
            f"epoch {epoch} precision {metrics_dict['precision']: .4f} recall {metrics_dict['recall']: .4f}"
        )

        # wandb.log(
        #     {
        #         "precision": metrics_dict["mean_precision"],
        #         "recall": metrics_dict["mean_recall"],
        #         "train_loss": train_loss,
        #         "test_loss": test_loss,
        #     }
        # )


if __name__ == "__main__":
    train_pet_net()
