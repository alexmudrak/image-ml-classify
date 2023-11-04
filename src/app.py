from flask import Flask
from flask_restful import Api

from api.v1.endpoints import ClassificateImageAPI, TrianImageModelAPI
from core.settings import DEBUG


def create_app() -> Flask:
    app = Flask(__name__)
    api = Api(app)
    register_endpoints_v1(api)
    return app


def register_endpoints_v1(api: Api) -> None:
    api_v1 = "/api/v1/"
    api.add_resource(ClassificateImageAPI, api_v1 + "image")
    api.add_resource(TrianImageModelAPI, api_v1 + "train")


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=6767, debug=DEBUG)
