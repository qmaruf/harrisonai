import numpy as np
from loguru import logger
import torch
import pandas as pd
import ast
from breeds import cat_breeds, dog_breeds
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import DataLoader
from config import config


class PetDataset(Dataset):
    def __init__(self, imgs_path, info_df, train=True):
        sample_ids = info_df["Sample_ID"]
        cat_id = list(info_df.columns.values).index("cat")

        self.image_paths, self.mask_paths, self.labels = [], [], []
        for sample_id in sample_ids:
            self.image_paths.append(f"{imgs_path}/{sample_id}/image.jpg")
            self.mask_paths.append(f"{imgs_path}/{sample_id}/mask.jpg")
            label = (
                info_df[info_df["Sample_ID"] == sample_id].iloc[:, cat_id:].values[0]
            )
            self.labels.append(label)

        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            image = Transforms.train_transform(image=image)["image"]
        else:
            image = Transforms.test_transform(image=image)["image"]

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


class DataProcessor:
    def __init__(self, df_path=None):
        if df_path:
            self.df = pd.read_csv(df_path)

        self.pet_label_to_id = {
            k: v for v, k in enumerate(["cat", "dog"] + cat_breeds + dog_breeds)
        }
        self.pet_id_to_label = {v: k for k, v in self.pet_label_to_id.items()}

    def get_pet_and_breed_types(self, breeds):
        labels = [0] * len(self.pet_label_to_id)
        for idx, breed in enumerate(breeds):
            pet_type = 0 if breed in cat_breeds else 1
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


class Transforms:
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


class Utils:
    @staticmethod
    def get_data_loader(imgs_path, info_path):
        df = DataProcessor(info_path).create_multi_label_df()
        df = df.sample(frac=1)
        n = int(df.shape[0] * 0.8)
        train_df = df.iloc[:n, :].reset_index()
        test_df = df.iloc[n:, :].reset_index()

        train_ds = PetDataset(imgs_path=imgs_path, info_df=train_df, train=True)
        test_ds = PetDataset(imgs_path=imgs_path, info_df=test_df, train=False)

        train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

        return train_dl, test_dl

    @staticmethod
    def get_metrics(dl_labels, dl_outputs):
        pet_id_to_labels = DataProcessor().pet_id_to_label
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
                "precision": precision,
                "recall": recall,
            }
            logger.info(
                f"{pet_id_to_labels[i]} | tp {tp} | fp {fp} | fn {fn} | precision {precision:.5f} | recall {recall:.5f}"
            )
        metrics_dict["mean_precision"] = np.mean(precisions)
        metrics_dict["mean_recall"] = np.mean(recalls)
        return metrics_dict
