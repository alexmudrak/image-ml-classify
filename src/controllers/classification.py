from io import BytesIO

import numpy as np
import torch
from PIL import Image
from utils.file_utils import get_from_json_file

# TODO: Create CoreDataset class
from core.datasets import DatasetUtils
# TODO: Create CoreConfig class
from core.settings import DATAMODEL_PATH, DATASETS_FOLDER
# TODO: Create CoreTransform class
from core.transforms import get_transorms


class ClassificationController:
    @staticmethod
    def classify_image(file):
        model_init = DatasetUtils(DATAMODEL_PATH)
        model = model_init.load_model()
        model.eval()

        image = Image.open(BytesIO(file.read())).convert("RGB")
        image_transform = get_transorms()["val"]
        image = image_transform(image)

        outputs = model(torch.Tensor(np.array(image)).unsqueeze(0))
        _, preds = torch.max(outputs, 1)

        # TODO: maybe need to SETUP classes file name
        #       by environment value
        all_labels = get_from_json_file(DATASETS_FOLDER + "classes.json")
        return all_labels[str(preds[0].item())]
