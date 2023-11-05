import os
import shutil
from datetime import datetime

import torch
from torchvision import models

from core.settings import BACKUPS_FOLDER


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
