import os
import shutil
from unittest.mock import MagicMock, patch

import pytest

from core.datasets import CoreDataset
from tests.conftest import TEST_FOLDER


@pytest.fixture
def tmp_dataset_structure():
    base_path = os.path.join(TEST_FOLDER, "datasets")
    structure = {
        "train_path": os.path.join(base_path, "train"),
        "valid_path": os.path.join(base_path, "val"),
        "base_path": base_path,
    }
    return structure


def test_normilize_dataset(tmp_dataset_structure):
    shutil.rmtree(
        tmp_dataset_structure["valid_path"],
        ignore_errors=True,
    )
    os.makedirs(tmp_dataset_structure["valid_path"], exist_ok=True)

    CoreDataset.normalize_dataset(
        tmp_dataset_structure["base_path"],
        "train",
        "val",
    )

    assert os.path.exists(
        os.path.join(tmp_dataset_structure["valid_path"], "class_1")
    )
    assert os.path.exists(
        os.path.join(tmp_dataset_structure["valid_path"], "class_2")
    )


@patch("core.datasets.YandexDisk", return_value=MagicMock())
def test_cloud_load_with_valid_cloud_type(mock_YandexDisk):
    cloud_type = "yandex"

    CoreDataset.cloud_load(cloud_type)

    mock_YandexDisk.assert_called_once()
    CoreDataset.cloud_client.sync_data.assert_called_once()


def test_cloud_load_with_invalid_cloud_type():
    cloud_type = "invalid_cloud"

    with pytest.raises(NotImplementedError) as exc_info:
        CoreDataset.cloud_load(cloud_type)

    assert "Unknown cloud service name" in str(exc_info.value)


def test_cloud_load_with_no_cloud_type():
    cloud_type = None

    with pytest.raises(ValueError) as exc_info:
        CoreDataset.cloud_load(cloud_type)

    assert "Cloud service name is not set" in str(exc_info.value)
