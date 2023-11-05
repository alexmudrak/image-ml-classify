import json
from io import BytesIO

import numpy
import torch
from flask import Flask, request
from flask_restful import Api
from PIL import Image

from core.dataset_models import CoreDatasetModel
from core.settings import DATAMODEL_PATH, DATASETS_FOLDER, DEBUG
from core.transforms import get_transorms
from utils.file_utils import get_from_json_file

app = Flask(__name__)
api = Api(app)


@app.route("/api/image", methods=["POST"])
def upload():
    model_init = CoreDatasetModel(DATAMODEL_PATH)
    model = model_init.load_model()
    model.eval()

    file = request.files["file"]

    image = Image.open(BytesIO(file.read())).convert("RGB")
    image_transform = get_transorms()["val"]
    image = image_transform(image)

    outputs = model(torch.Tensor(numpy.array(image)).unsqueeze(0))
    _, preds = torch.max(outputs, 1)

    all_labels = get_from_json_file(DATASETS_FOLDER + "classes.json")
    return json.dumps({"answer": all_labels[str(preds[0].item())]})


@app.route("/api/image_check", methods=["POST"])
def upload_2():
    model_init = CoreDatasetModel("./datamodels/origin_model_bk.pth")
    model = model_init.load_model()
    model.eval()

    file = request.files["file"]
    file.save("./static/output.png")
    image = Image.open("./static/output.png").convert("RGB")
    image_transform = get_transorms()["val"]
    image = image_transform(image)

    outputs = model(torch.Tensor(numpy.array(image)).unsqueeze(0))
    _, preds = torch.max(outputs, 1)

    all_labels = get_from_json_file(DATASETS_FOLDER + "classes_2.json")
    return json.dumps({"answer": all_labels[str(preds[0].item())]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6767, debug=DEBUG)
