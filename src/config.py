import torch


class config:
    imgs_path = "dataset/data/"
    info_path = "dataset/pets_dataset_info.csv"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 16
    learning_rate = 0.1
    epochs = 512
    debug = True
    max_side = 512
