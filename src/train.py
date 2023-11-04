import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets

from controllers.training import train_model
from core.datasets import DatasetUtils
from core.settings import DATAMODEL_PATH, DATASETS_FOLDER
from core.transforms import get_transorms
from utils import store_to_json_file


def train_model_start():
    data_transforms = get_transorms()
    data_dir = DATASETS_FOLDER
    init_model = DatasetUtils(DATAMODEL_PATH)

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=4, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    # Generate new classes
    json_classes = {index: value for index, value in enumerate(class_names)}
    store_to_json_file(
        json_classes,
        # TODO: create env variable
        "./datasets/classes.json",
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model.load_model()

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Start train model
    model = train_model(
        model,
        device,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=5,
    )

    init_model.backup_model()
    torch.save(model, DATAMODEL_PATH)


if __name__ == "__main__":
    train_model_start()
