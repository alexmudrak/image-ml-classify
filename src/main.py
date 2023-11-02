import json

import numpy
import torch
from flask import Flask, request
from flask_restful import Api
from PIL import Image

from core.datasets import DatasetUtils
from core.settings import DATAMODEL_PATH, DATASETS_FOLDER
from utils import get_from_json_file

app = Flask(__name__)
api = Api(app)


@app.route("/api/image", methods=["POST"])
def upload():
    model = DatasetUtils.load_existing_model(DATAMODEL_PATH)
    model.eval()

    file = request.files["file"]
    file.save("./static/output.png")
    image = Image.open("./static/output.png").convert("RGB")
    image_transform = DatasetUtils.get_transorms()["val"]
    image = image_transform(image)

    outputs = model(torch.Tensor(numpy.array(image)).unsqueeze(0))
    _, preds = torch.max(outputs, 1)

    all_labels = get_from_json_file(DATASETS_FOLDER + "classes.json")
    return json.dumps({"answer": all_labels[str(preds[0].item())]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6767)
