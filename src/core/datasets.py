import os
import random
import shutil

from core.logger import app_logger
from core.settings import FILE_COUNTS_TO_VALID
from services.yandex_disk import YandexDisk
from utils.file_utils import remove_all_folders

logger = app_logger(__name__)


class CoreDataset:
    cloud_client = None

    @staticmethod
    def normalize_dataset(
        dataset_folder_path: str,
        train_folder_name: str,
        validate_folder_name: str,
    ) -> None:
        """
        Normalize the dataset by redistributing images from the 'train' folder
        to the 'val' folder.

        This method is responsible for ensuring that both the 'train' and 'val'
        folders have a consistent set of classes and a similar distribution of
        images.

        It achieves this by redistributing a portion (30%) of image from
        'train' to 'val'.
        """
        source_directory = os.path.join(
            dataset_folder_path,
            train_folder_name,
        )
        target_directory = os.path.join(
            dataset_folder_path,
            validate_folder_name,
        )

        remove_all_folders(target_directory)

        source_dirs = os.listdir(source_directory)
        target_dirs = os.listdir(target_directory)

        for target_dir in target_dirs:
            logger.debug(f"Checking target directory: {target_dir}")
            if target_dir not in source_dirs:
                target_dir_path = os.path.join(target_directory, target_dir)
                if os.path.isdir(target_dir_path):
                    shutil.rmtree(target_dir_path)

        for source_dir in source_dirs:
            logger.debug(f"Processing source directory: {source_dir}")
            if source_dir not in target_dirs:
                target_dir_path = os.path.join(target_directory, source_dir)
                os.mkdir(target_dir_path)

                source_dir_path = os.path.join(source_directory, source_dir)
                files_to_move = os.listdir(source_dir_path)
                # TODO: Need to add behavior if in train folder
                #       3 or less files.
                logger.info(
                    f"Redistributing {FILE_COUNTS_TO_VALID * 100}% of files "
                    f"({source_dir}) to the 'val' folder..."
                )
                num_files_to_move = int(
                    FILE_COUNTS_TO_VALID * len(files_to_move)
                )

                if num_files_to_move > 0:
                    files_to_move = random.sample(
                        files_to_move, num_files_to_move
                    )

                    for file_to_move in files_to_move:
                        logger.debug(f"Moving file: {file_to_move}")
                        source_file_path = os.path.join(
                            source_dir_path, file_to_move
                        )
                        target_file_path = os.path.join(
                            target_dir_path, file_to_move
                        )
                        shutil.move(source_file_path, target_file_path)
                else:
                    shutil.rmtree(source_dir_path)
                    shutil.rmtree(target_dir_path)

        logger.info("Dataset normalization complete.")

    @staticmethod
    def cloud_load(cloud_type: str | None) -> None:
        if not cloud_type:
            raise ValueError(
                "Cloud service name is not set. Please set a valid cloud "
                "service name before proceeding."
            )

        match cloud_type.lower():
            case "yandex":
                CoreDataset.cloud_client = YandexDisk()
            case _:
                raise NotImplementedError(
                    f"Unknown cloud service name: {cloud_type}. "
                    "Please provide a valid service name."
                )
        CoreDataset.cloud_client.sync_data()
