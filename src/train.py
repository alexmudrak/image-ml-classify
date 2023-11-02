import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from core.datasets import DatasetUtils, ImageDataset

# Замените на ваш загрузчик данных
train_dataset = ImageDataset(
    root_dir="./datasets/", transform=DatasetUtils.get_transorm()
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = DatasetUtils.load_existing_model("./datamodels/base_model.pth")

for param in model.parameters():
    param.requires_grad = False

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

num_classes = len(train_dataset.class_register.keys())
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Обучение модели
num_epochs = 10  # Замените на желаемое количество эпох обучения

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

torch.save(model, "./datamodels/base_model.pth")
