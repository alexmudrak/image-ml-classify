import json
import os
import pickle
from unittest.mock import MagicMock, patch

from tests.conftest import TEST_FOLDER
from utils.file_utils import (
    backup_file,
    get_from_json_file,
    get_from_pickle,
    logger,
    remove_all_folders,
    remove_file,
    store_to_json_file,
    store_to_pickle,
)


def test_get_from_json_file(tmpdir):
    file = tmpdir.join("test.json")
    test_dict = {"test": "mock value"}

    file.write(json.dumps(test_dict))
    result = get_from_json_file(file.strpath)

    assert result["test"] == "mock value"


def test_store_to_json_file(tmpdir):
    file = tmpdir.join("test.json")
    test_dict = {"test": "mock value"}

    store_to_json_file(test_dict, file.strpath)

    with open(file.strpath, "r") as file:
        result = json.load(file)

    assert result["test"] == "mock value"


def test_get_from_pickle(tmpdir):
    file = tmpdir.join("test.pkl")
    test_dict = {"test": "mock value"}
    with open(file.strpath, "wb") as file_obj:
        pickle.dump(test_dict, file_obj)

    result = get_from_pickle(file.strpath)

    assert result is not None
    assert result["test"] == "mock value"


def test_load_to_pickle(tmpdir):
    file = tmpdir.join("test.pkl")
    test_dict = {"test": "mock value"}
    store_to_pickle(test_dict, file.strpath)

    with open(file.strpath, "rb") as file_obj:
        result = pickle.load(file_obj)

    assert result["test"] == "mock value"


@patch("utils.file_utils.datetime", return_value=MagicMock())
@patch("utils.file_utils.BACKUPS_FOLDER", "test_folder")
def test_backup_file(mock_datetime, tmpdir):
    fixed_timestamp = "2023-01-01_00-00-00"

    mock_now = mock_datetime.now()
    mock_now.strftime.return_value = fixed_timestamp

    file = tmpdir.join("backup.file")
    file.write("test")

    expect_path = os.path.join(
        "test_folder", f"backup_backup_{fixed_timestamp}.pth"
    )

    backup_file(file.strpath)

    assert os.path.exists(expect_path)


def test_backup_file_not_found(tmpdir):
    file = tmpdir.join("backup.file")
    file.write("test")
    with patch(
        "utils.file_utils.shutil.copy",
        side_effect=FileNotFoundError,
    ):
        with patch.object(logger, "error") as mock_logger_error:
            backup_file(file.strpath)

        expected_error_message = "Error: The source model file does not exist."
        mock_logger_error.assert_called_once_with(expected_error_message)


def test_backup_file_some_error(tmpdir):
    file = tmpdir.join("backup.file")
    file.write("test")
    with patch(
        "utils.file_utils.shutil.copy",
        side_effect=Exception("Mock exception"),
    ):
        with patch.object(logger, "error") as mock_logger_error:
            backup_file(file.strpath)

        expected_error_message = (
            "An error occurred while creating a backup: Mock exception"
        )
        mock_logger_error.assert_called_once_with(expected_error_message)


def test_remove_all_folders():
    base_dir = os.path.join(TEST_FOLDER, "folders_for_remove")
    sub_folders = [
        "test_1",
        "test_2",
        "test_3",
    ]

    os.mkdir(base_dir)
    for sub_folder in sub_folders:
        os.mkdir(os.path.join(base_dir, sub_folder))

    with patch.object(logger, "warning") as mock_logger_warning:
        remove_all_folders(base_dir)

        assert mock_logger_warning.call_count == len(sub_folders)
    assert len(os.listdir(base_dir)) == 0


def test_remove_all_folders_error():
    base_dir = os.path.join(TEST_FOLDER, "folders_for_remove")
    sub_folders = [
        "test_1",
        "test_2",
        "test_3",
    ]

    os.mkdir(base_dir)
    for sub_folder in sub_folders:
        os.mkdir(os.path.join(base_dir, sub_folder))
    with patch(
        "utils.file_utils.shutil.rmtree",
        side_effect=Exception("Mock exception"),
    ):
        with patch.object(logger, "error") as mock_logger_error:
            remove_all_folders(base_dir)

            assert mock_logger_error.call_count == len(sub_folders)
    assert len(os.listdir(base_dir)) == 3


def test_remove_all_folders_not_exist():
    base_dir = os.path.join(TEST_FOLDER, "folders_for_remove")

    with patch.object(logger, "info") as mock_logger_info:
        remove_all_folders(base_dir)

        assert mock_logger_info.call_count == 1


def test_remove_file(tmpdir):
    file = tmpdir.join("backup.file")
    file.write("test")
    expected_error_message = f"Removed file: {file.strpath}"

    with patch.object(logger, "info") as mock_logger_info:
        remove_file(file.strpath)

    mock_logger_info.assert_called_once_with(expected_error_message)


def test_remove_file_not_exist(tmpdir):
    file = tmpdir.join("backup.file")
    file.write("test")
    expected_error_message = f"The file '{file.strpath}' does not exist."

    with patch(
        "utils.file_utils.os.remove",
        side_effect=FileNotFoundError("Mock exception"),
    ):
        with patch.object(logger, "error") as mock_logger_error:
            remove_file(file.strpath)

        mock_logger_error.assert_called_once_with(expected_error_message)
