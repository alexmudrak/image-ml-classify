# TODO: Implement Yandeks Disk client
#       Need to check folder with dataset
#       check md5 of folders, if new, try to
#       download and run train model.

# TODO: Implement method, for uploading backup of
#       model.

# Est.
# - New download: Total time: 885.48 seconds
# - Not exist: Total time: 331.64 seconds
# - Full new train: Total time: 1075.97 seconds
# - Retrain: Total time: 605.18 seconds

import asyncio
import os
import shutil
import time

from yadisk_async.yadisk import YaDisk

from core.settings import (CLOUD_ID, CLOUD_SECRET, CLOUD_TOKEN,
                           CLOUD_TRAIN_DATASET_PATH, LOCAL_TRAIN_DATASET_PATH)


class YandexDisk:
    def __init__(self):
        self.id = CLOUD_ID
        self.token = CLOUD_TOKEN
        self.secret = CLOUD_SECRET
        self.remote_dataset_path = CLOUD_TRAIN_DATASET_PATH
        self.local_dataset_path = LOCAL_TRAIN_DATASET_PATH

    def sync_data(self):
        if (
            not self.id
            or not self.token
            or not self.secret
            or not self.remote_dataset_path
        ):
            # TODO: create behavior

            return
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
    ):
        remote_objects = await client.listdir(remote_folder_path)
        local_objects = os.listdir(local_folder_path)

        base_level_folders = []
        tasks = []
        async for remote_object in remote_objects:
            if not remote_object.name:
                continue

            remote_file_path = os.path.join(remote_folder_path, remote_object.name)
            local_file_path = os.path.join(local_folder_path, remote_object.name)

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

    async def _download_file(self, client, remote_file_path, local_file_path):
        print(f"Downloading {remote_file_path} to {local_file_path}")
        # TODO: Add logger
        await client.download(remote_file_path, local_file_path)


if __name__ == "__main__":
    # TODO: move credentials to .env
    client = YandexDisk()
    start_time = time.time()

    print("Check dataset")
    client.sync_data()
    print("Start train model")
    # train_model_start()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total executTotal execution time: {execution_time} seconds")
