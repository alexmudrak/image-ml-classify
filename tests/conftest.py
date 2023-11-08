import os
import shutil

import pytest

TEST_FOLDER = "test_folder"

MAIN_FOLDERS = [
    "datasets",
    "datamodels",
    "datasets/train",
    "datasets/val",
]
CLASSES_FOLDERS = [
    "class_1",
    "class_2",
]
CLASSES_FILES = [
    "img_1.jpg",
    "img_2.jpg",
    "img_3.jpg",
    "img_4.jpg",
]


@pytest.fixture(scope="function", autouse=True)
def base_folder_structure_session():
    for folder in MAIN_FOLDERS:
        os.makedirs(os.path.join(TEST_FOLDER, folder), exist_ok=True)

    for class_folder in CLASSES_FOLDERS:
        class_folder_path = os.path.join(
            TEST_FOLDER, "datasets/train", class_folder
        )
        os.makedirs(class_folder_path, exist_ok=True)
        for class_name in CLASSES_FILES:
            class_file_path = os.path.join(class_folder_path, class_name)
            with open(class_file_path, "w") as file:
                file.write("Mock image file")
    yield
    shutil.rmtree(os.path.join(TEST_FOLDER), ignore_errors=True)
