from io import BytesIO

import numpy as np
import torch
from PIL import Image
from werkzeug.datastructures import FileStorage

from core.dataset_models import CoreDatasetModel
from core.settings import DATASET_CLASSES_PATH, DATASET_MODEL_PATH
from core.transforms import CoreTranform
from utils.file_utils import get_from_json_file


class ClassificationController:
    @staticmethod
    def get_classify_image(file: FileStorage) -> str:
        model_init = CoreDatasetModel(DATASET_MODEL_PATH)
        model = model_init.load_model()
        model.eval()

        image = Image.open(BytesIO(file.read())).convert("RGB")
        image_transform = CoreTranform.get_transorms()["val"]
        image = image_transform(image)

        outputs = model(torch.Tensor(np.array(image)).unsqueeze(0))
        _, preds = torch.max(outputs, 1)

        all_labels = get_from_json_file(DATASET_CLASSES_PATH)
        return all_labels[str(preds[0].item())]
