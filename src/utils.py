import json
import os
import shutil


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


def remove_all_folders(base_dir: str) -> None:
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    print(f"Каталог {item_path} удален.")
                except Exception as e:
                    print(f"Не удалось удалить каталог {item_path}: {e}")
    else:
        print(f"Папка {base_dir} не существует.")
