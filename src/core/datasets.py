import os
import random
import shutil
from datetime import datetime

import torch
from torchvision import models

from core.settings import BACKUPS_FOLDER, DATASETS_FOLDER
from services.yandex_disk import YandexDisk


class CoreDataset:
    @staticmethod
    def normalize_dataset():
        source_directory = os.path.join(DATASETS_FOLDER, "train")
        target_directory = os.path.join(DATASETS_FOLDER, "val")

        source_dirs = os.listdir(source_directory)
        target_dirs = os.listdir(target_directory)

        for target_dir in target_dirs:
            if target_dir not in source_dirs:
                target_dir_path = os.path.join(target_directory, target_dir)
                if os.path.isdir(target_dir_path):
                    shutil.rmtree(target_dir_path)

        for source_dir in source_dirs:
            if source_dir not in target_dirs:
                target_dir_path = os.path.join(target_directory, source_dir)
                os.mkdir(target_dir_path)

                source_dir_path = os.path.join(source_directory, source_dir)
                files_to_move = os.listdir(source_dir_path)

                num_files_to_move = int(0.3 * len(files_to_move))

                files_to_move = random.sample(files_to_move, num_files_to_move)

                for file_to_move in files_to_move:
                    source_file_path = os.path.join(source_dir_path, file_to_move)
                    target_file_path = os.path.join(target_dir_path, file_to_move)
                    shutil.move(source_file_path, target_file_path)
        # TODO: Add logger
        print("Dataset normalize complete.")

    @staticmethod
    def cloud_load():
        # This method should provide condition for
        # choice the cloud system which store dataset.
        # If not set in config, should be skiped.
        client = YandexDisk()
        client.sync_data()


class DatasetUtils:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self) -> models.ResNet:
        try:
            model = torch.load(self.model_path, map_location=self.device)
        except FileNotFoundError:
            model = models.resnet18(weights="IMAGENET1K_V1")
            model.to(self.device)
            torch.save(model, self.model_path)
        return model

    def backup_model(self):
        # TODO: move implementation to file utils
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        model_name = os.path.basename(self.model_path)
        base_filename = os.path.splitext(model_name)[0]
        backup_path = os.path.join(
            BACKUPS_FOLDER, f"{base_filename}_backup_{timestamp}.pth"
        )

        try:
            shutil.copy(self.model_path, backup_path)
            print(f"Model backed up to {backup_path}")
        except FileNotFoundError:
            print("Error: The source model file does not exist.")
        except Exception as e:
            print(f"An error occurred while creating a backup: {str(e)}")
