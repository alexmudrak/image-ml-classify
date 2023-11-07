import os
import shutil
from unittest.mock import patch

import pytest

TEST_FOLDER = "test_folder"


@pytest.fixture(scope="function", autouse=True)
def cleanup_test_folders():
    yield
    shutil.rmtree(os.path.join(TEST_FOLDER), ignore_errors=True)


@pytest.fixture
def app():
    with (
        patch("core.settings.DATASET_FOLDER", os.path.join(TEST_FOLDER)),
        patch(
            "core.settings.BACKUPS_FOLDER", os.path.join(TEST_FOLDER, "backup")
        ),
        patch(
            "core.settings.DATASET_MODEL_PATH",
            os.path.join(TEST_FOLDER, "test_model"),
        ),
        patch(
            "core.settings.LOCAL_TRAIN_DATASET_PATH",
            os.path.join(TEST_FOLDER, "dataset"),
        ),
        patch(
            "core.settings.LOCAL_VALID_DATASET_PATH",
            os.path.join(TEST_FOLDER, "dataset", "valid"),
        ),
        patch(
            "core.settings.DATASET_MODEL_STATUS_DB_PATH",
            os.path.join(TEST_FOLDER, "dataset", "train"),
        ),
    ):
        from app import create_app

        app = create_app()
        yield app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_home_endpoint(client):
    response = client.get("/")
    assert response.status_code == 404
