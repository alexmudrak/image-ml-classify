# TODO: Implement method, for uploading backup of
#       model.
import asyncio
import os
import shutil
import time

from yadisk_async.yadisk import YaDisk

from core.logger import app_logger
from core.settings import (
    CLOUD_ID,
    CLOUD_SECRET,
    CLOUD_TOKEN,
    CLOUD_TRAIN_DATASET_PATH,
    LOCAL_TRAIN_DATASET_PATH,
)

logger = app_logger(__name__)


class YandexDisk:
    def __init__(self) -> None:
        self.id = CLOUD_ID
        self.token = CLOUD_TOKEN
        self.secret = CLOUD_SECRET
        self.remote_dataset_path = CLOUD_TRAIN_DATASET_PATH
        self.local_dataset_path = LOCAL_TRAIN_DATASET_PATH

    def sync_data(self) -> None:
        if (
            not self.id
            or not self.token
            or not self.secret
            or not self.remote_dataset_path
        ):
            raise ValueError("One or more required parameters are missing")

        client = YaDisk(
            id=self.id,
            secret=self.secret,
            token=self.token,
        )
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self._process_sync(
                client,
                self.remote_dataset_path,
                self.local_dataset_path,
            )
        )

    async def _process_sync(
        self,
        client: YaDisk,
        remote_folder_path: str,
        local_folder_path: str,
        level: int = 0,
    ) -> None:
        """
        Recursively synchronize files and folders between a remote and local
        directory.

        This method compares the contents of the specified remote folder with
        the local folder. It will download files from the remote folder to the
        local folder if they do not exist locally.

        If a remote folder contains subdirectories, it will also be recursively
        synchronized.

        Folders that exist only locally and not on the remote are deleted if
        level is 0 (base level).
        """
        remote_objects = await client.listdir(remote_folder_path)
        local_objects = os.listdir(local_folder_path)

        base_level_folders = []
        tasks = []
        async for remote_object in remote_objects:
            if not remote_object.name:
                continue

            logger.debug(f"Checking {remote_object.path}...")

            remote_file_path = os.path.join(
                remote_folder_path, remote_object.name
            )
            local_file_path = os.path.join(
                local_folder_path, remote_object.name
            )

            if await remote_object.is_dir():
                base_level_folders.append(remote_object.name)
                if not os.path.exists(local_file_path):
                    os.makedirs(local_file_path)

                await self._process_sync(
                    client, remote_file_path, local_file_path, level + 1
                )
            else:
                if remote_object.name not in local_objects:
                    tasks.append(
                        self._download_file(
                            client,
                            remote_file_path,
                            local_file_path,
                        )
                    )
        if level == 0:
            for local_dir in os.listdir(local_folder_path):
                local_dir_path = os.path.join(local_folder_path, local_dir)
                if (
                    os.path.isdir(local_dir_path)
                    and local_dir not in base_level_folders
                ):
                    shutil.rmtree(local_dir_path)

        await asyncio.gather(*tasks)

    async def _download_file(
        self,
        client: YaDisk,
        remote_file_path: str,
        local_file_path: str,
    ) -> None:
        logger.info(f"Downloading {remote_file_path} to {local_file_path}")
        await client.download(remote_file_path, local_file_path)


if __name__ == "__main__":
    client = YandexDisk()
    start_time = time.time()
    logger.info("Checking dataset")

    client.sync_data()

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total execution time: {execution_time} seconds")
