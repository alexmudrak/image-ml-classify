import json
import os
import shutil
from datetime import datetime

from core.settings import BACKUPS_FOLDER


def get_from_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def store_to_json_file(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(
            data,
            json_file,
            ensure_ascii=False,
            indent=4,
        )


def backup_file(file_path: str) -> None:
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.basename(file_path)
    backup_path = (
        f"./{BACKUPS_FOLDER}/{os.path.splitext(file_name)[0]}_backup_{timestamp}.pth"
    )

    try:
        shutil.copy(file_path, backup_path)
        print(f"Model backed up to {backup_path}")
    except FileNotFoundError:
        print("Error: The source model file does not exist.")
    except Exception as e:
        print(f"An error occurred while creating a backup: {str(e)}")


def remove_all_folders(base_dir: str) -> None:
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    print(f"Directory {item_path} removed.")
                except Exception as e:
                    print(f"Failed to remove directory {item_path}: {e}")
    else:
        print(f"Folder {base_dir} does not exist.")
