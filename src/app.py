import os

from flask import Flask
from flask_restful import Api

from api.v1.endpoints import ClassificateImageAPI, TrianImageModelAPI
from core.logger import app_logger
from core.settings import DATASET_MODEL_STATUS_DB_PATH, DEBUG


def create_app() -> Flask:
    app = Flask(__name__)
    api = Api(app)
    register_endpoints_v1(api)
    return app


def register_endpoints_v1(api: Api) -> None:
    api_v1 = "/api/v1/"
    api.add_resource(ClassificateImageAPI, api_v1 + "image")
    api.add_resource(TrianImageModelAPI, api_v1 + "train")


logger = app_logger(__name__)
app = create_app()

if os.path.exists(DATASET_MODEL_STATUS_DB_PATH):
    logger.info(f"Removed old status DB file: {DATASET_MODEL_STATUS_DB_PATH}")
    os.remove(DATASET_MODEL_STATUS_DB_PATH)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6767, debug=DEBUG)
