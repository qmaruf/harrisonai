import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision

from torchvision import datasets
from glob import glob
import pandas as pd
import cv2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import ast
from tqdm import tqdm
from loguru import logger
import wandb
from data import Utils, DataProcessor
from config import config
from network import PetNet

wandb.login()
run = wandb.init(project="harrison.ai")

train_dl, test_dl = Utils.get_data_loader(
    imgs_path=config.imgs_path, info_path=config.info_path
)

net = PetNet().to(config.device)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=config.learning_rate)
scheduler = ReduceLROnPlateau(optimizer, "min")


def run_one_epoch(dl, train):
    losses = []
    desc = "train" if train else "test"
    for data in tqdm(dl, total=len(dl.dataset) // config.batch_size, desc=desc):
        inputs, labels = data
        inputs = inputs.to(config.device)
        labels = labels.to(torch.float32).to(config.device)

        optimizer.zero_grad()

        if train:
            outputs = net(inputs)
        else:
            with torch.no_grad():
                outputs = net(inputs)

        loss = criterion(outputs.float(), labels.float())
        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()

    return np.mean(losses)


def run_eval(dl):
    net.eval()
    dl_labels, dl_outputs = [], []
    for data in tqdm(dl, total=len(dl.dataset) // config.batch_size):
        inputs, labels = data
        inputs = inputs.to(config.device)

        with torch.no_grad():
            outputs = net(inputs)

        labels = labels.data.cpu().numpy()
        outputs = outputs.data.cpu().numpy()
        outputs = np.where(outputs > 0.5, 1.0, 0.0)

        dl_labels.append(labels)
        dl_outputs.append(outputs)

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
        metrics_dict = run_eval(test_dl)
        
        wandb.log({
            'precision': metrics_dict['mean_precision'],
            'recall': metrics_dict['mean_recall']
        })
        


if __name__ == "__main__":
    train_pet_net()
