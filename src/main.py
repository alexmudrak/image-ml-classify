import json

import numpy
import torch
from flask import Flask, request
from flask_restful import Api
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

app = Flask(__name__)
api = Api(app)

device = "cpu"
model = torch.load("./models/base_model.pth", map_location=torch.device(device))
test_transforms = transforms.Compose(
    [
        transforms.Resize(260),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        super().__init__()
        self.transform = transform
        self.data_len = len(img_paths)
        self.img_paths = img_paths
        self.labels = labels

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@app.route("/api/image", methods=["POST"])
def upload():
    file = request.files["file"]
    file.save("./static/output.png")

    model.eval()
    image = Image.open("./static/output.png").convert("RGB")
    print(image)
    image = test_transforms(image)

    outputs = model(torch.Tensor([numpy.array(image)]))
    _, preds = torch.max(outputs, 1)

    # all_labels = ['Пробирочная', 'Ящик 2', 'Надпись из железа',
    #               'Окрашивание', 'Ящик 1', 'Ящик 3', 'Шов']

    all_labels = {
        0: "Лабораторная посуда",
        1: "Ультразвуковой метод контроля",
        2: "Гидроабразивная резка",
        3: "Окрашивание",
        4: "Капиллярный метод контроля",
        5: "Магнитопорошковый метод контроля",
        6: "Сварка стали",
    }

    return json.dumps({"answer": all_labels[int(preds[0].item())]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6767)
