import torch
from torchvision import models

from core.logger import app_logger
from utils.file_utils import backup_file

logger = app_logger(__name__)


class CoreDatasetModel:
    def __init__(self, dataset_model_path: str) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = dataset_model_path
        self.model = None

    def load_model(self) -> models.ResNet:
        try:
            logger.info(f"Loading model from file... {self.model_path}")
            model = torch.load(self.model_path, map_location=self.device)
        except FileNotFoundError:
            logger.info(
                "Model file not found, creating a new model... "
                f"{self.model_path}"
            )
            model = models.resnet18(weights="IMAGENET1K_V1")
            model.to(self.device)
            torch.save(model, self.model_path)
        self.model = model
        logger.info(f"Use Device: {self.device}...")
        return self.model

    def backup_model(self):
        backup_file(self.model_path)
