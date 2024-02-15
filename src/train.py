import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
from glob import glob
import pandas as pd
import cv2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
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


wandb.login()
run = wandb.init(project="harrison.ai")


class config:
    data_path = "../dataset/data/"
    info_path = "../dataset/pets_dataset_info.csv"
    device = torch.device("cuda")
    batch_size = 16
    learning_rate = 0.1
    epochs = 32


train_transform = A.Compose(
    [
        A.geometric.resize.LongestMaxSize(max_size=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(3, p=0.5),
        A.ColorJitter(p=0.5),
        A.PixelDropout(0.005, p=0.5),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [A.geometric.resize.LongestMaxSize(max_size=256), ToTensorV2()]
)


class DataProcessor:
    def __init__(self, df_path=None):
        if df_path:
            self.df = pd.read_csv(df_path)
        self.cat_breeds = [
            "Abyssinian",
            "Bengal",
            "Birman",
            "Bombay",
            "British_Shorthair",
            "Egyptian_Mau",
            "Maine_Coon",
            "Persian",
            "Ragdoll",
            "Russian_Blue",
            "Siamese",
            "Sphynx",
        ]

        self.dog_breeds = [
            "american_bulldog",
            "american_pit_bull_terrier",
            "basset_hound",
            "beagle",
            "boxer",
            "chihuahua",
            "english_cocker_spaniel",
            "english_setter",
            "german_shorthaired",
            "great_pyrenees",
            "havanese",
            "japanese_chin",
            "keeshond",
            "leonberger",
            "miniature_pinscher",
            "newfoundland",
            "pomeranian",
            "pug",
            "saint_bernard",
            "samoyed",
            "scottish_terrier",
            "shiba_inu",
            "staffordshire_bull_terrier",
            "wheaten_terrier",
            "yorkshire_terrier",
        ]
        self.pet_label_to_id = {
            k: v
            for v, k in enumerate(["cat", "dog"] + self.cat_breeds + self.dog_breeds)
        }
        self.pet_id_to_label = {v: k for k, v in self.pet_label_to_id.items()}

    def get_pet_and_breed_types(self, breeds):
        labels = [0] * len(self.pet_label_to_id)
        for idx, breed in enumerate(breeds):
            pet_type = 0 if breed in self.cat_breeds else 1
            labels[pet_type] = 1
            breed_type = self.pet_label_to_id[breed]
            labels[breed_type] = 1
        return labels

    def create_multi_label_df(self):
        breeds = self.df["Breed"].apply(lambda x: ast.literal_eval(x))
        multi_label_cols = list(self.pet_label_to_id.keys())
        self.df[multi_label_cols] = breeds.apply(self.get_pet_and_breed_types).apply(
            pd.Series
        )
        df_mlbl = self.df.copy()
        return df_mlbl


pet_id_to_labels = DataProcessor().pet_id_to_label


class HarrisonPetDataset(Dataset):
    def __init__(self, imgs_path, info_df, train=True):
        self.imgs_path = imgs_path
        df = info_df

        sample_ids = info_df["Sample_ID"]
        cat_id = list(df.columns.values).index("cat")

        self.image_paths, self.mask_paths, self.labels = [], [], []
        for sample_id in sample_ids:
            self.image_paths.append(f"{config.data_path}/{sample_id}/image.jpg")
            self.mask_paths.append(f"{config.data_path}/{sample_id}/mask.jpg")
            label = df[df["Sample_ID"] == sample_id].iloc[:, cat_id:].values[0]
            self.labels.append(label)

        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            image = train_transform(image=image)["image"]
        else:
            image = test_transform(image=image)["image"]

        image = image / 255.0
        _, h, w = image.shape
        square = torch.zeros((3, 256, 256))
        square[:, 0:h, 0:w] = image

        return square

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.read_image(image_path)
        label = self.labels[idx].astype(float)
        return image, label


df = DataProcessor(config.info_path).create_multi_label_df()
df = df.sample(frac=1)
n = int(df.shape[0] * 0.8)
train_df = df.iloc[:n, :].reset_index()
test_df = df.iloc[n:, :].reset_index()

train_ds = HarrisonPetDataset(imgs_path=config.data_path, info_df=train_df, train=True)


test_ds = HarrisonPetDataset(imgs_path=config.data_path, info_df=test_df, train=False)

train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)


class HarrisonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 39)
        self.sigmode = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmode(x)
        return x


net = HarrisonNet().to(config.device)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=config.learning_rate)
scheduler = ReduceLROnPlateau(optimizer, "min")


net = HarrisonNet().to(config.device)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=config.learning_rate)


def run_one_epoch(dl, train):
    losses = []
    for data in tqdm(dl, total=len(dl.dataset) // config.batch_size):
        inputs, labels = data
        labels = labels.to(torch.float32)

        inputs = inputs.to(config.device)
        labels = labels.to(config.device)
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

        labels = labels.data.cpu().numpy()
        inputs = inputs.to(config.device)

        with torch.no_grad():
            outputs = net(inputs)

        outputs = outputs.data.cpu().numpy()
        outputs = np.where(outputs > 0.5, 1.0, 0.0)

        dl_labels.append(labels)
        dl_outputs.append(outputs)

    dl_labels = np.concatenate(dl_labels)
    dl_outputs = np.concatenate(dl_outputs)

    metrics_dict = {}
    precisions, recalls = [], []
    for i in range(39):
        tp = np.sum((dl_labels[:, i] == 1) & (dl_outputs[:, i] == 1))
        fp = np.sum((dl_labels[:, i] == 0) & (dl_outputs[:, i] == 1))
        fn = np.sum((dl_labels[:, i] == 1) & (dl_outputs[:, i] == 0))
        precision = tp / (tp + fp) if (tp + fp > 0) else 0
        recall = tp / (tp + fn) if (tp + fn > 0) else 0
        precisions.append(precision)
        recalls.append(recall)
        metrics_dict[pet_id_to_labels[i]] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": f"{precision: .2f}",
            "recall": f"{recall: .2f}",
        }
        logger.info(
            f"{pet_id_to_labels[i]} | tp {tp} | fp {fp} | fn {fn} | precision {precision:.5f} | recall {recall:.5f}"
        )
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    return mean_precision, mean_recall


def generate_labels(preds, id_to_label):
    labels = []
    for k, v in id_to_label.items():
        if preds[k] == 1:
            labels.append(v)
    return labels


for epoch in range(config.epochs):
    net.train()
    train_loss = run_one_epoch(train_dataloader, train=True)
    net.eval()
    test_loss = run_one_epoch(test_dataloader, train=False)
    scheduler.step(test_loss)
    precision, recall = run_eval(test_dataloader)
    print(
        f"epoch {epoch} | train loss {train_loss:.5f} | test loss {test_loss: .5f} | precision {precision:.5f} | recall {recall:.5f}"
    )
    PATH = "model.pt"
    torch.save(net, PATH)
    wandb.log(
        {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "precision": precision,
            "recall": recall,
        }
    )


metrics_dict = run_eval(test_dataloader)


def inference():
    for _, row in test_df.iterrows():
        # print (row)
        img_id = row["Sample_ID"]
        img_path = f"{config.data_path}/{img_id}/image.jpg"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_transform(image=img)["image"]
        img = img / 255
        _, h, w = img.shape
        square = torch.zeros((3, 256, 256))
        square[:, 0:h, 0:w] = img
        square = square.unsqueeze(0)
        square = square.to(config.device)
        preds = net(square)[0]
        preds = preds.data.cpu().numpy()
        preds = np.where(preds > 0.5, 1.0, 0.0)
        preds = generate_labels(preds, pet_id_to_labels)
        print(row["Breed"], "|", preds)
