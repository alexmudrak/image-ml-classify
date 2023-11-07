from unittest.mock import AsyncMock, patch

import pytest

from services.yandex_disk import YandexDisk


@pytest.fixture
def yandex_disk_instance():
    return YandexDisk()


@pytest.fixture
def mock_yadisk():
    return AsyncMock()


def test_sync_data_empty_remote_dataset(yandex_disk_instance, mock_yadisk):
    with patch("services.yandex_disk.YaDisk", return_value=mock_yadisk), patch(
        "os.makedirs"
    ), patch("os.listdir"):
        yandex_disk_instance.sync_data()
    mock_yadisk.listdir.assert_awaited()


def test_sync_data_missed_cloud_values():
    with patch("services.yandex_disk.CLOUD_ID", None):
        with pytest.raises(
            ValueError, match="One or more required parameters are missing"
        ):
            client = YandexDisk()
            client.sync_data()
