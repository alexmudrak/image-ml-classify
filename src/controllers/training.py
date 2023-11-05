# Based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import multiprocessing
import os
import pickle
import time
from tempfile import TemporaryDirectory

import torch

from core.dataset_models import CoreDatasetModel
from core.datasets import CoreDataset
from core.schemas import TrainingStatus
from core.settings import CLOUD_TYPE, DATAMODEL_PATH, DATASETS_FOLDER


class TrainingController:
    def __init__(self, status_db: str) -> None:
        self.status_db = status_db
        self.status = self._load_status()
        self.model = None
        self.dataset = None
        self.transforms = None
        self.dataset_folder_path = DATASETS_FOLDER
        self.dataset_model_folder_path = DATAMODEL_PATH
        self.cloud_service = CLOUD_TYPE

    def get_status(self) -> TrainingStatus:
        return self.status

    def _set_status(self, status_name: TrainingStatus) -> None:
        self.status = status_name
        self._save_status()

    def _load_status(self):
        if os.path.exists(self.status_db):
            with open(self.status_db, "rb") as status_file:
                status = pickle.load(status_file)
                return status
        else:
            return TrainingStatus.READY

    def _save_status(self):
        with open(self.status_db, "wb") as status_file:
            pickle.dump(self.status, status_file)

    def run_train(self, epoch_count: int = 5) -> None:
        if self.status in [
            TrainingStatus.MODEL_TRAINING,
            TrainingStatus.DATASET_SYNCHRONIZATION,
        ]:
            return

        self._set_status(TrainingStatus.MODEL_TRAINING)

        training_process = multiprocessing.Process(
            target=self._train, args=(epoch_count,)
        )
        training_process.start()

    def _train(self, epoch_count: int) -> None:
        self._set_status(TrainingStatus.DATASET_SYNCHRONIZATION)
        self._prepeare_dataset()
        self._prepeare_model()
        self._prepeare_transforms()
        self._set_status(TrainingStatus.MODEL_TRAINING)
        self._train_process(epoch_count)
        self._set_status(TrainingStatus.READY)

    def _prepeare_dataset(self):
        if self.cloud_service:
            # Check and download dataset from CLOUD
            # service
            CoreDataset.cloud_load()
        CoreDataset.normalize_dataset()
        # QUESTION: What sould to set?
        self.dataset = ...

    def _prepeare_model(self):
        self.model_object = CoreDatasetModel(self.dataset_model_folder_path)
        self.model_object.load_model()
        self.model = self.model_object.model

    def _prepeare_transforms(self):
        # TODO: Get and set transforms
        pass

    def _train_process(self, epoch_count: int):
        pass


def train_model(
    model,
    device,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int = 25,
):
    # TODO: need to setup num_epochs by
    #       body value from user request
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # TODO: Add logger
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            # TODO: Add logger
            print()

        time_elapsed = time.time() - since

        # TODO: Add logger
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        # TODO: Add logger
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model
