class AppResponses:
    @staticmethod
    def error_not_valid_x_key() -> tuple[dict, int]:
        return {"error": "Forbidden access"}, 401

    @staticmethod
    def return_answer(answer: str) -> tuple[dict, int]:
        return {"answer": answer}, 200
