from collections import namedtuple

Response = namedtuple("Response", ["data", "status_code"])


class AppResponses:
    @staticmethod
    def error_not_valid_x_key() -> Response:
        return Response({"error": "Forbidden access"}, 401)

    @staticmethod
    def return_answer(answer: str) -> Response:
        return Response({"answer": answer}, 200)

    @staticmethod
    def return_status(status: str, code: int) -> Response:
        return Response({"status": status}, code)
