from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from core.datasets import DatasetUtils, ImageDataset
from core.settings import DATAMODEL_PATH, DATASETS_FOLDER

# from utils import remove_all_folders

# Замените на ваш загрузчик данных
train_dataset = ImageDataset(
    root_dir=DATASETS_FOLDER,
    transform=DatasetUtils.get_transorms()["train"],
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = DatasetUtils.load_existing_model(DATAMODEL_PATH)
# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Обучение модели
# TODO: need to setup by request
num_epochs = 10  # Замените на желаемое количество эпох обучения
# Настройте цикл обучения
try:
    for epoch in range(num_epochs):
        model.train()
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

except KeyboardInterrupt:
    pass
finally:
    torch.save(model, f"./datamodels/new_model_{datetime.now()}.pth")
    # remove_all_folders(DATASETS_FOLDER)
