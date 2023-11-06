from flask import request
from flask_restful import Resource
from werkzeug.exceptions import UnsupportedMediaType

from controllers.classification import ClassificationController
from controllers.training import TrainingController
from core.responses import AppResponses, Response
from core.settings import DB_STATUS_FILE
from validators.request_validators import RequestValidator


class ClassificateImageAPI(Resource):
    def post(self) -> Response:
        file = request.files["file"]
        answer = ClassificationController.get_classify_image(file)
        return AppResponses.return_answer(answer)


class TrianImageModelAPI(Resource):
    def __init__(self) -> None:
        self.training_controller = TrainingController(DB_STATUS_FILE)

    def get(self) -> Response:
        if not RequestValidator.is_valid_x_key(request):
            return AppResponses.error_not_valid_x_key()
        current_status = self.training_controller.get_status()
        return AppResponses.return_status(current_status.value, 200)

    def post(self) -> Response:
        if not RequestValidator.is_valid_x_key(request):
            return AppResponses.error_not_valid_x_key()

        try:
            request_json = request.json or {}
        except UnsupportedMediaType:
            request_json = {}

        epoch = request_json.get("epoch", 5)
        hard_run = request_json.get("hard_run", False)

        self.training_controller.run_train(
            epoch_count=epoch,
            hard_run=hard_run,
        )
        current_status = self.training_controller.get_status()
        return AppResponses.return_status(current_status.value, 202)
