from flask import request
from flask_restful import Resource

from controllers.classification import ClassificationController
from core.responses import AppResponses
from validators.request_validators import RequestValidator


class ClassificateImageAPI(Resource):
    def post(self):
        if not RequestValidator.is_valid_x_key(request):
            return AppResponses.error_not_valid_x_key()
        file = request.files["file"]
        answer = ClassificationController.classify_image(file)
        return AppResponses.return_answer(answer)


class TrianImageModelAPI(Resource):
    def get(self):
        return "TEST", 200

    def post(self):
        return "RUN TRAIN", 202
