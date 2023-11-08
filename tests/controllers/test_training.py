import os
import shutil
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from controllers.training import TrainingController
from core.datasets import CoreDataset
from core.transforms import CoreTranform
from tests.conftest import TEST_FOLDER


@pytest.fixture
def temp_dir():
    with TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def mock_cloud_load():
    with patch.object(CoreDataset, "cloud_load") as mock:
        yield mock


@pytest.fixture
def mock_normalize_dataset():
    with patch.object(CoreDataset, "normalize_dataset") as mock:
        yield mock


@pytest.fixture
def mock_get_transforms():
    with patch.object(CoreTranform, "get_transorms") as mock:
        yield mock


@pytest.fixture
def mock_train_process():
    with patch.object(TrainingController, "_train_process") as mock:
        yield mock


@pytest.fixture
def mock_torch_save():
    with patch("torch.save") as mock:
        yield mock


@pytest.fixture
def mock_torch_load():
    with patch("torch.load") as mock:
        yield mock


@pytest.fixture
def mock_torch_set_grad_enabled():
    with patch("torch.set_grad_enabled") as mock:
        yield mock


@pytest.fixture
def mock_torch_max():
    with patch("torch.max") as mock:
        yield mock


@pytest.fixture
def mock_torch_sum():
    with patch("torch.sum") as mock:
        yield mock


def test_prepeare_dataset(
    temp_dir,
    mock_cloud_load,
    mock_normalize_dataset,
    mock_get_transforms,
):
    train_folder_path = os.path.join(TEST_FOLDER, "datasets/train")
    for folder in os.listdir(train_folder_path):
        train_class_folder_path = os.path.join(train_folder_path, folder)
        valid_class_folder_path = os.path.join(
            TEST_FOLDER, "datasets/val", folder
        )
        shutil.copytree(train_class_folder_path, valid_class_folder_path)

    with (
        patch(
            "controllers.training.DATASET_FOLDER",
            os.path.join(TEST_FOLDER, "datasets"),
        ),
        patch(
            "controllers.training.DATASET_MODEL_PATH",
            os.path.join(TEST_FOLDER, "datamodels"),
        ),
        patch(
            "controllers.training.DATASET_CLASSES_PATH",
            os.path.join(TEST_FOLDER, "datamodels", "classes.json"),
        ),
    ):
        controller = TrainingController(
            status_db=os.path.join(temp_dir, "status.pkl")
        )

        mock_cloud_load.return_value = None
        mock_normalize_dataset.return_value = None
        mock_get_transforms.return_value = {
            "train": MagicMock(),
            "val": MagicMock(),
        }

        controller._prepeare_dataset()

        mock_cloud_load.assert_called_once()
        mock_normalize_dataset.assert_called_once()
        mock_get_transforms.assert_called_once()


def test_train_process_bad_initialization_value(
    temp_dir,
):
    controller = TrainingController(
        status_db=os.path.join(temp_dir, "status.pkl")
    )

    with pytest.raises(ValueError) as excinfo:
        controller._train_process(epoch_count=5)

    assert "Some required components are missing or incomplete." in str(
        excinfo.value
    )


def test_train_process(
    temp_dir,
    mock_torch_save,
    mock_torch_load,
    mock_torch_sum,
    mock_torch_max,
    mock_torch_set_grad_enabled,
):
    mock_loader = MagicMock()
    mock_loader.inputs = [1, 1, 1, 1]
    mock_loader.labels = [2, 2, 2, 2]

    controller = TrainingController(
        status_db=os.path.join(temp_dir, "status.pkl")
    )
    controller.model = MagicMock()
    controller.model_object = MagicMock()
    controller.dataset_sizes = {"train": 1, "val": 2}

    inputs_mock = MagicMock()
    inputs_mock.to().size.return_value = 123
    controller.dataset_loaders = {
        "train": MagicMock(),
        "val": MagicMock(),
    }
    controller.dataset_loaders["train"].__iter__.return_value = iter(
        [(inputs_mock, inputs_mock)]
    )
    controller.dataset_loaders["val"].__iter__.return_value = iter(
        [(inputs_mock, inputs_mock)]
    )

    controller.model_optimizer = MagicMock()

    loss = MagicMock()
    loss.backward = MagicMock(return_value=123)
    loss.item = MagicMock(return_value=123)
    controller.model_criterion = MagicMock(return_value=loss)

    controller.model_scheduler = MagicMock()

    mock_torch_save.return_value = None
    mock_torch_load.return_value = None
    mock_torch_max.return_value = (0, 0)
    mock_torch_sum.return_value = 0
    mock_torch_set_grad_enabled.return_value = MagicMock()

    controller._train_process(epoch_count=5)

    mock_torch_save.assert_called_once()
    mock_torch_load.assert_called_once()
