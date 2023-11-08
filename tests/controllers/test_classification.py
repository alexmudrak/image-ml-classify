from unittest.mock import MagicMock, patch

from werkzeug.datastructures import FileStorage

from controllers.classification import ClassificationController


class MockFileStorage(FileStorage):
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


@patch("controllers.classification.get_from_json_file")
@patch("controllers.classification.CoreDatasetModel")
@patch("controllers.classification.CoreTranform")
@patch("controllers.classification.torch.max")
@patch("controllers.classification.Image")
def test_get_classify_image(
    mock_image,
    mock_torch_max,
    mock_transforms,
    mock_dataset_models,
    mock_get_from_json_file,
    tmpdir,
):
    mock_image.return_value = MagicMock()
    mock_transforms.return_value = MagicMock()
    mock_dataset_models.return_value = MagicMock()
    mock_get_from_json_file.return_value = {"1": "Mock class"}

    mock_preds_item = MagicMock()
    mock_preds_item.item.return_value = 1
    mock_preds = [
        mock_preds_item,
    ]
    mock_torch_max.return_value = (0, mock_preds)

    file = tmpdir.join("mock.file")
    file.write(b"Mock IMAGE")
    image_data = open(file.strpath, "rb").read()

    file_storage = MockFileStorage(image_data)

    result = ClassificationController.get_classify_image(file_storage)

    assert result == "Mock class"
