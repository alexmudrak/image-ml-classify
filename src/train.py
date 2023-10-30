import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Получение списка путей к изображениям и меток классов
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(class_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(8)

        return image, label


# Загрузка существующей модели
model_path = "./models/model_final_1.0.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=torch.device(device))

# Замораживаем веса существующей модели (опционально)
for param in model.parameters():
    param.requires_grad = False

# Заменяем последний слой модели на новый слой с новым числом классов
num_classes = 10  # Замените на количество классов в ваших новых данных
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Подготовка данных
transform = transforms.Compose(
    [
        transforms.Resize(260),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Замените на ваш загрузчик данных
train_dataset = ImageDataset(root_dir="./datasets/", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Обучение модели
num_epochs = 10  # Замените на желаемое количество эпох обучения

# all_labels = [
#         "Лабораторная посуда",
#         "Ультразвуковой метод контроля",
#         "Гидроабразивная резка",
#         "Окрашивание",
#         "Капиллярный метод контроля",
#         "Магнитопорошковый метод контроля",
#         "Сварка стали",
#     ]

# Настройте цикл обучения
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        # Прямой проход через модель
        outputs = model(inputs)

        # Рассчитать потери и выполнить обратное распространение
        loss = criterion(outputs, labels)
        loss.backward()

        # Обновление весов
        optimizer.step()

        running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}")

torch.save(model, "./models/new_model.pth")
