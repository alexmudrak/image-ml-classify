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
from utils.file_utils import check_and_create_directories, remove_file

directories_to_check = [
    DATASET_FOLDER,
    DATASET_MODEL_PATH,
    LOCAL_TRAIN_DATASET_PATH,
    LOCAL_VALID_DATASET_PATH,
    BACKUPS_FOLDER,
]


def create_app() -> Flask:
    app = Flask(__name__)
    api = Api(app)
    register_endpoints_v1(api)
    check_and_create_directories(directories_to_check)
    remove_file(DATASET_MODEL_STATUS_DB_PATH)
    return app


def register_endpoints_v1(api: Api) -> None:
    api_v1 = "/api/v1/"
    api.add_resource(ClassificateImageAPI, api_v1 + "image")
    api.add_resource(TrianImageModelAPI, api_v1 + "train")


logger = app_logger(__name__)
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6767, debug=DEBUG)
