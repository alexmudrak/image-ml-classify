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
import time

import yadisk_async

from train import train_model_start

DATASET_REMOTE_FOLDER_PATH = "/dataset"
DATASET_FOLDER_PATH = "./datasets/train/"


async def download_file(client, remote_file_path, local_file_path):
    print(f"Downloading {remote_file_path} to {local_file_path}")
    await client.download(remote_file_path, local_file_path)


async def testing(client, remote_folder_path: str, local_folder_path: str):
    if not await client.check_token():
        raise Exception("Not valid token")

    remote_objects = client.listdir(remote_folder_path)
    local_objects = os.listdir(local_folder_path)

    tasks = []
    async for remote_object in await remote_objects:
        remote_file_path = os.path.join(remote_folder_path, remote_object.name)
        local_file_path = os.path.join(local_folder_path, remote_object.name)

        if await remote_object.is_dir():
            if not os.path.exists(local_file_path):
                os.makedirs(local_file_path)

            await testing(client, remote_file_path, local_file_path)
        else:
            if remote_object.name not in local_objects:
                tasks.append(
                    download_file(
                        client,
                        remote_file_path,
                        local_file_path,
                    )
                )

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    client = yadisk_async.YaDisk(
        id="80fa5e9e7f374650bbee111065c60930",
        secret="074c41c91b9c442f8820037db528b3a2",
        token="y0_AgAAAAA1pERaAArEzQAAAADw87B1aITtDWK1Q2mYDdyQbg9Hth_LFGU",
    )
    start_time = time.time()

    print("Check dataset")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        testing(client, DATASET_REMOTE_FOLDER_PATH, DATASET_FOLDER_PATH)
    )

    print("Start train model")
    train_model_start()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total executTotal execution time: {execution_time} seconds")
