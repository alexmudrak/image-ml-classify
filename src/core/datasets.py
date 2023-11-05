import os
import random
import shutil

from core.settings import CLOUD_TYPE, DATASETS_FOLDER
from services.yandex_disk import YandexDisk


class CoreDataset:
    @staticmethod
    def normalize_dataset():
        source_directory = os.path.join(DATASETS_FOLDER, "train")
        target_directory = os.path.join(DATASETS_FOLDER, "val")

        source_dirs = os.listdir(source_directory)
        target_dirs = os.listdir(target_directory)

        for target_dir in target_dirs:
            if target_dir not in source_dirs:
                target_dir_path = os.path.join(target_directory, target_dir)
                if os.path.isdir(target_dir_path):
                    shutil.rmtree(target_dir_path)

        for source_dir in source_dirs:
            if source_dir not in target_dirs:
                target_dir_path = os.path.join(target_directory, source_dir)
                os.mkdir(target_dir_path)

                source_dir_path = os.path.join(source_directory, source_dir)
                files_to_move = os.listdir(source_dir_path)

                num_files_to_move = int(0.3 * len(files_to_move))

                files_to_move = random.sample(files_to_move, num_files_to_move)

                for file_to_move in files_to_move:
                    source_file_path = os.path.join(source_dir_path, file_to_move)
                    target_file_path = os.path.join(target_dir_path, file_to_move)
                    shutil.move(source_file_path, target_file_path)
        # TODO: Add logger
        print("Dataset normalize complete.")

    @staticmethod
    def cloud_load():
        # This method should provide condition for
        # choice the cloud system which store dataset.
        # If not set in config, should be skiped.
        if not CLOUD_TYPE:
            raise ValueError("Cloud service name not set.")

        match CLOUD_TYPE.lower():
            case "yandex":
                client = YandexDisk()
            case _:
                raise NotImplementedError("Unknown cloud service name")
        client.sync_data()
