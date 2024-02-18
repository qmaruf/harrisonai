from pathlib import Path

import torch


class config:
    imgs_path = "dataset/data/"
    info_path = "dataset/pets_dataset_info.csv"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 16
    learning_rate = 0.1
    epochs = 512
    debug = False
    max_side = 256
    n_classes = 39
    weight_path = "weights/model.pth"
    storage_path = "server/uploaded_files"
    api_endpoint = "http://127.0.0.1:8000/inference"


Path("weights").mkdir(exist_ok=True)
Path("server/uploaded_files").mkdir(exist_ok=True)
