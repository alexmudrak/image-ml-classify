import json


def get_from_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def store_to_json_file(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(
            data,
            json_file,
            ensure_ascii=False,
            indent=4,
        )


def get_key_from_dict(dictionary: dict, value: str) -> str:
    for key, val in dictionary.items():
        if val == value:
            return key

    if not dictionary:
        return "0"

    keys = list(dictionary.keys())
    last_key = int(keys[-1])

    last_key_number = int(last_key)
    new_key = str(last_key_number + 1)

    return new_key
