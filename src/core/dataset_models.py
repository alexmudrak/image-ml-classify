import torch
from torchvision import models

from utils.file_utils import backup_file


class CoreDatasetModel:
    def __init__(self, dataset_model_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = dataset_model_path
        self.model = None

    def load_model(self) -> models.ResNet:
        try:
            model = torch.load(self.model_path, map_location=self.device)
        except FileNotFoundError:
            model = models.resnet18(weights="IMAGENET1K_V1")
            model.to(self.device)
            torch.save(model, self.model_path)
        self.model = model
        return self.model

    def backup_model(self):
        backup_file(self.model_path)
