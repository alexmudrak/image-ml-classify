import os

from flask import request
from flask_restful import Resource

from controllers.classification import ClassificationController
from controllers.training import TrainingController
from core.responses import AppResponses
from core.settings import DB_STATUS_FILE
from validators.request_validators import RequestValidator

# TODO: maybe need to refactor
if os.path.exists(DB_STATUS_FILE):
    os.remove(DB_STATUS_FILE)


class ClassificateImageAPI(Resource):
    def post(self):
        if not RequestValidator.is_valid_x_key(request):
            return AppResponses.error_not_valid_x_key()
        file = request.files["file"]
        answer = ClassificationController.get_classify_image(file)
        return AppResponses.return_answer(answer)


class TrianImageModelAPI(Resource):
    def __init__(self):
        self.training_controller = TrainingController(DB_STATUS_FILE)

    def get(self):
        if not RequestValidator.is_valid_x_key(request):
            return AppResponses.error_not_valid_x_key()
        current_status = self.training_controller.get_status()
        return AppResponses.return_status(current_status.value, 200)

    def post(self):
        if not RequestValidator.is_valid_x_key(request):
            return AppResponses.error_not_valid_x_key()
        # TODO: Implement train epoch by request body
        # TODO: Implement hard run from request body
        self.training_controller.run_train(5)
        current_status = self.training_controller.get_status()
        return AppResponses.return_status(current_status.value, 202)
