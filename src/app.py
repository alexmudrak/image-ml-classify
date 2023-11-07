import os

from flask import Flask
from flask_restful import Api

from api.v1.endpoints import ClassificateImageAPI, TrianImageModelAPI
from core.logger import app_logger
from core.settings import (
    BACKUPS_FOLDER,
    DATASET_FOLDER,
    DATASET_MODEL_PATH,
    DATASET_MODEL_STATUS_DB_PATH,
    DEBUG,
    LOCAL_TRAIN_DATASET_PATH,
    LOCAL_VALID_DATASET_PATH,
)


def create_app() -> Flask:
    app = Flask(__name__)
    api = Api(app)
    register_endpoints_v1(api)
    return app


def register_endpoints_v1(api: Api) -> None:
    api_v1 = "/api/v1/"
    api.add_resource(ClassificateImageAPI, api_v1 + "image")
    api.add_resource(TrianImageModelAPI, api_v1 + "train")


# TODO: move to file utils
# Function to check and create directories
def check_and_create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")


logger = app_logger(__name__)

# TODO: move to file utils
directories_to_check = [
    DATASET_FOLDER,
    DATASET_MODEL_PATH,
    LOCAL_TRAIN_DATASET_PATH,
    LOCAL_VALID_DATASET_PATH,
    BACKUPS_FOLDER,
]
check_and_create_directories(directories_to_check)

app = create_app()

# TODO: move to file utils
if os.path.exists(DATASET_MODEL_STATUS_DB_PATH):
    logger.info(f"Removed old status DB file: {DATASET_MODEL_STATUS_DB_PATH}")
    try:
        os.remove(DATASET_MODEL_STATUS_DB_PATH)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6767, debug=DEBUG)
