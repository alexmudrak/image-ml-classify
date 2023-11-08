from unittest.mock import MagicMock, patch

import torch
from torchvision import models, os

from core.dataset_models import CoreDatasetModel
from tests.conftest import TEST_FOLDER

TEST_MODEL_PATH = os.path.join(TEST_FOLDER, "test_model.pth")
core_dataset = CoreDatasetModel(TEST_MODEL_PATH)


def test_load_model_without_existing_file():
    model = core_dataset.load_model()

    assert isinstance(model, models.ResNet)
    assert model.to(device=torch.device("cpu")).training


def test_load_model_with_existing_file():
    model = models.resnet18()
    torch.save(model, TEST_MODEL_PATH)

    loaded_model = core_dataset.load_model()

    assert isinstance(loaded_model, models.ResNet)
    assert loaded_model.to(device=torch.device("cpu")).training


@patch("core.dataset_models.backup_file", return_value=MagicMock())
def test_backup_model(mock_backup):
    mock_backup.return_value = MagicMock()
    core_dataset.backup_model()
