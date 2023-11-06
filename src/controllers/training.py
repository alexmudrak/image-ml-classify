# Based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import multiprocessing
import os
import pickle
import time
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets

from core.dataset_models import CoreDatasetModel
from core.datasets import CoreDataset
from core.logger import app_logger
from core.schemas import TrainingStatus
from core.settings import CLOUD_TYPE, DATAMODEL_PATH, DATASETS_FOLDER
from core.transforms import CoreTranform
from utils.file_utils import store_to_json_file

logger = app_logger(__name__)


class TrainingController:
    def __init__(self, status_db: str) -> None:
        self.status_db = status_db
        self.status = self._load_status()
        self.model_object = None
        self.model = None
        self.model_criterion = None
        self.model_optimizer = None
        self.model_scheduler = None
        self.model_folder_path = DATAMODEL_PATH
        self.dataset_loaders = None
        self.dataset_sizes = None
        self.dataset_class_size = None
        self.dataset_folder_path = DATASETS_FOLDER
        self.transforms = None
        self.cloud_service = CLOUD_TYPE

    def get_status(self) -> TrainingStatus:
        return self.status

    def _set_status(self, status_name: TrainingStatus) -> None:
        self.status = status_name
        self._save_status()

    def _load_status(self):
        # TODO: Move to file utils
        if os.path.exists(self.status_db):
            with open(self.status_db, "rb") as status_file:
                status = pickle.load(status_file)
                return status
        else:
            return TrainingStatus.READY

    def _save_status(self):
        # TODO: Move to file utils
        with open(self.status_db, "wb") as status_file:
            pickle.dump(self.status, status_file)

    def run_train(self, epoch_count: int = 5, hard_run: bool = False) -> None:
        if (
            self.status
            in [
                TrainingStatus.MODEL_TRAINING,
                TrainingStatus.DATASET_SYNCHRONIZATION,
            ]
            and not hard_run
        ):
            return

        self._set_status(TrainingStatus.MODEL_TRAINING)

        training_process = multiprocessing.Process(
            target=self._train, args=(epoch_count,)
        )
        training_process.start()

    def _train(self, epoch_count: int) -> None:
        base_start_time = time.time()

        self._set_status(TrainingStatus.DATASET_SYNCHRONIZATION)
        self._prepeare_transforms()
        self._prepeare_dataset()
        self._prepeare_model()

        dataset_sync_time = time.time() - base_start_time
        logger.info(
            f"Dataset Synchronization Time: {dataset_sync_time} seconds"
        )

        start_time = time.time()

        self._set_status(TrainingStatus.MODEL_TRAINING)
        self._train_process(epoch_count)

        model_training_time = time.time() - start_time
        logger.info(f"Model Training Time: {model_training_time} seconds")

        start_time = time.time()

        if self.model_object:
            self.model_object.backup_model()
            torch.save(self.model, self.model_folder_path)

        self._set_status(TrainingStatus.READY)

        ready_time = time.time() - base_start_time
        logger.info(f"Ready Time: {ready_time} seconds")

    def _prepeare_transforms(self):
        self.transforms = CoreTranform.get_transorms()

    def _prepeare_dataset(self):
        logger.info("Preparing dataset...")
        if not self.transforms:
            raise ValueError("Transforms not set. Please set a valid value.")

        if self.cloud_service:
            # Check and download dataset from CLOUD service
            logger.info(
                "Checking and downloading dataset from the cloud "
                f"service... {self.cloud_service}"
            )
            CoreDataset.cloud_load()

        CoreDataset.normalize_dataset()

        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(self.dataset_folder_path, x), self.transforms[x]
            )
            for x in ["train", "val"]
        }
        self.dataset_loaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=4, shuffle=True, num_workers=4
            )
            for x in ["train", "val"]
        }

        self.dataset_sizes = {
            x: len(image_datasets[x]) for x in ["train", "val"]
        }

        class_names = image_datasets["train"].classes
        val_class_names = image_datasets["val"].classes

        json_classes = {
            index: value for index, value in enumerate(class_names)
        }
        logger.info(
            f"Storing class information to JSON file... {json_classes}"
        )
        store_to_json_file(
            json_classes,
            # TODO: create env variable
            "./datasets/classes.json",
        )
        self.dataset_class_size = len(class_names)

        if self.dataset_class_size < len(val_class_names):
            raise IndexError(
                "The number of training classes is less than the "
                "number of validation classes."
            )
        logger.info("Finished preparing dataset...")

    def _prepeare_model(self):
        if (
            not self.dataset_loaders
            or not self.dataset_sizes
            or not self.dataset_class_size
        ):
            raise ValueError("Dataset information is missing or incomplete.")

        self.model_object = CoreDatasetModel(self.model_folder_path)
        self.model_object.load_model()
        self.model = self.model_object.model

        if not self.model:
            raise ValueError("Model loading failed or is missing.")

        for param in self.model.parameters():
            param.requires_grad = False

        num_input_neurons = self.model.fc.in_features

        logger.debug(
            "Configuring the model's final layers... "
            f"Count: {self.dataset_class_size} "
        )

        self.model.fc = nn.Linear(num_input_neurons, self.dataset_class_size)
        self.model_criterion = nn.CrossEntropyLoss()
        self.model_optimizer = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9
        )
        self.model_scheduler = lr_scheduler.StepLR(
            self.model_optimizer, step_size=7, gamma=0.1
        )

    def _train_process(self, epoch_count: int):
        since = time.time()
        if (
            not self.model
            or not self.dataset_loaders
            or not self.model_optimizer
            or not self.model_object
            or not self.model_criterion
            or not self.model_optimizer
            or not self.model_scheduler
            or not self.dataset_sizes
        ):
            raise ValueError(
                "Some required components are missing or incomplete."
            )

        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(
                tempdir, "best_model_params.pt"
            )

            torch.save(self.model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(epoch_count):
                logger.info(f"Epoch {epoch}/{epoch_count - 1}")

                for phase in ["train", "val"]:
                    if phase == "train":
                        self.model.train()
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0

                    for inputs, labels in self.dataset_loaders[phase]:
                        inputs = inputs.to(self.model_object.device)
                        labels = labels.to(self.model_object.device)

                        self.model_optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == "train"):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = self.model_criterion(outputs, labels)
                            if phase == "train":
                                loss.backward()
                                self.model_optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == "train":
                        self.model_scheduler.step()

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = (
                        running_corrects.double() / self.dataset_sizes[phase]
                    )
                    logger.info(
                        f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
                    )

                    if phase == "val" and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(
                            self.model.state_dict(), best_model_params_path
                        )

            time_elapsed = time.time() - since
            logger.info(
                f"Training complete in {time_elapsed // 60:.0f}m "
                f"{time_elapsed % 60:.0f}s"
            )
            logger.info(f"Best val Acc: {best_acc:.4f}")

            self.model.load_state_dict(torch.load(best_model_params_path))
