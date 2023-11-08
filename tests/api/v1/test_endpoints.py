import os
from io import BytesIO
from unittest.mock import patch

import pytest

from api.v1.endpoints import ClassificationController, TrainingController
from tests.conftest import TEST_FOLDER


@pytest.fixture
def mock_get_classify_image():
    with patch.object(ClassificationController, "get_classify_image") as mock:
        yield mock


@pytest.fixture
def mock_get_status_train():
    with patch.object(TrainingController, "get_status") as mock:
        yield mock


@pytest.fixture
def mock_run_train():
    with patch.object(TrainingController, "run_train") as mock:
        yield mock


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
        patch(
            "validators.request_validators.SECRET_KEY",
            "mockkey",
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


def test_post_classify_image_success(
    client,
    mock_get_classify_image,
):
    mock_get_classify_image.return_value = "mock answer"

    data = dict(
        file=(BytesIO(b"mock file content"), "mock_file.test"),
    )

    response = client.post(
        "/api/v1/image", content_type="multipart/form-data", data=data
    )

    assert response.status_code == 200
    assert response.json["answer"] == mock_get_classify_image.return_value


def test_post_classify_not_send_image(
    client,
):
    response = client.post("/api/v1/image")

    assert response.status_code == 400


def test_get_classify_bad_method(
    client,
):
    response = client.get("/api/v1/image")

    assert response.status_code == 405


def test_get_train_model_status_bad_key(client):
    response = client.get("/api/v1/train")

    assert response.status_code == 401


def test_get_train_model_status(client, mock_get_status_train):
    mock_get_status_train().value = "Mock status"

    response = client.get(
        "/api/v1/train",
        headers={"X-Key": "mockkey"},
    )

    assert response.status_code == 200
    assert response.json["status"] == "Mock status"


def test_post_run_train(client, mock_get_status_train, mock_run_train):
    mock_get_status_train().value = "Mock status"
    mock_run_train.return_value = True

    response = client.post(
        "/api/v1/train",
        headers={"X-Key": "mockkey"},
    )

    assert response.status_code == 202
    assert response.json["status"] == "Mock status"


def test_post_run_train_bad_key(client):
    response = client.post(
        "/api/v1/train",
        headers={"X-Key": "badkey"},
    )

    assert response.status_code == 401
