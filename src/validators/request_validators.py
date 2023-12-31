from flask import Request

from core.settings import SECRET_KEY


class RequestValidator:
    @staticmethod
    def is_valid_x_key(request: Request) -> bool:
        if (
            "X-Key" in request.headers
            and request.headers["X-Key"] == SECRET_KEY
        ):
            return True
        return False
