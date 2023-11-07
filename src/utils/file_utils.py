import json
import os
import pickle
import shutil
from datetime import datetime
from typing import Any

from core.logger import app_logger
from core.settings import BACKUPS_FOLDER

logger = app_logger(__name__)


def get_from_json_file(path: str) -> dict:
    logger.info(f"Trying get json data from {path}")
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


def get_from_pickle(path: str) -> Any | None:
    if os.path.exists(path):
        with open(path, "rb") as file:
            result = pickle.load(file)
            return result


def store_to_pickle(path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(path, file)


def backup_file(file_path: str) -> None:
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.basename(file_path)
    backup_path = os.path.join(
        BACKUPS_FOLDER,
        f"{os.path.splitext(file_name)[0]}_backup_{timestamp}.pth",
    )

    try:
        shutil.copy(file_path, backup_path)
        logger.info(f"Model backed up to {backup_path}")
    except FileNotFoundError:
        logger.error("Error: The source model file does not exist.")
    except Exception as e:
        logger.error(f"An error occurred while creating a backup: {str(e)}")


def remove_all_folders(base_dir: str) -> None:
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    logger.warning(
                        f"Directory '{item_path}' has been "
                        "successfully removed."
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to remove directory {item_path}: {e}"
                    )
    else:
        logger.info(f"The folder '{base_dir}' does not exist.")


def check_and_create_directories(directories: list[str]) -> None:
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")


def remove_file(file_path: str):
    if os.path.exists(file_path):
        logger.info(f"Removed file: {file_path}")
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
