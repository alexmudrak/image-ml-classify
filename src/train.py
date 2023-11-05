import os
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets

from controllers.training import train_model
from core.datasets import DatasetUtils
from core.settings import DATAMODEL_PATH, DATASETS_FOLDER
from core.transforms import get_transorms
from utils.file_utils import store_to_json_file


def normalize_dataset():
    # Путь к исходному каталогу с исходными данными
    source_directory = os.path.join(DATASETS_FOLDER, "train")

    # Путь к целевому каталогу, в котором нужно проверить и
    # создать несоответствующие каталоги
    target_directory = os.path.join(DATASETS_FOLDER, "val")

    # Получаем список каталогов из исходного и целевого каталогов
    source_dirs = os.listdir(source_directory)
    target_dirs = os.listdir(target_directory)

    # Проходимся по списку каталогов в целевом каталоге и удаляем лишние
    for target_dir in target_dirs:
        if target_dir not in source_dirs:
            target_dir_path = os.path.join(target_directory, target_dir)
            if os.path.isdir(target_dir_path):
                shutil.rmtree(target_dir_path)

    # Проходимся по списку каталогов в исходном каталоге
    for source_dir in source_dirs:
        # Проверяем, есть ли данный каталог в целевом каталоге
        if source_dir not in target_dirs:
            # Если каталога нет в целевом каталоге, создаем его
            target_dir_path = os.path.join(target_directory, source_dir)
            os.mkdir(target_dir_path)

            # Получаем список файлов в исходном каталоге
            source_dir_path = os.path.join(source_directory, source_dir)
            files_to_move = os.listdir(source_dir_path)

            # Вычисляем количество файлов для перемещения (30%)
            num_files_to_move = int(0.3 * len(files_to_move))

            # Выбираем случайные файлы для перемещения
            files_to_move = random.sample(files_to_move, num_files_to_move)

            # Перемещаем выбранные файлы из исходного каталога в целевой
            for file_to_move in files_to_move:
                source_file_path = os.path.join(source_dir_path, file_to_move)
                target_file_path = os.path.join(target_dir_path, file_to_move)
                shutil.move(source_file_path, target_file_path)

    print("Dataset normalize complete.")


def train_model_start():
    data_transforms = get_transorms()
    data_dir = DATASETS_FOLDER
    init_model = DatasetUtils(DATAMODEL_PATH)

    normalize_dataset()

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
    val_class_names = image_datasets["val"].classes

    if len(class_names) < len(val_class_names):
        raise IndexError("Train classes count less then Validate classes")

    # Generate new classes
    json_classes = {index: value for index, value in enumerate(class_names)}
    store_to_json_file(
        json_classes,
        # TODO: create env variable
        "./datasets/classes.json",
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model.load_model()

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Start train model
    model = train_model(
        model,
        device,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        exp_lr_scheduler,
        num_epochs=5,
    )

    init_model.backup_model()
    torch.save(model, DATAMODEL_PATH)


if __name__ == "__main__":
    train_model_start()
