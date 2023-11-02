import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import get_from_json_file, get_key_from_dict, store_to_json_file


class ImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        json_classes_path: str = "./datasets/classes.json",
    ):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_register = get_from_json_file(json_classes_path)

        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    class_id = get_key_from_dict(self.class_register, class_name)
                    if class_id not in self.class_register.keys():
                        self.class_register[class_id] = class_name
                        store_to_json_file(json_classes_path, self.class_register)
                    self.image_paths.append(image_path)
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(int(label))

        return image, label


class DatasetUtils:
    @staticmethod
    def get_transorm():
        return transforms.Compose(
            [
                transforms.Resize(260),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def load_existing_model(model_path: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        return model
