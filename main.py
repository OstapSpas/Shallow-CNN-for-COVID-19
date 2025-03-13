import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Визначаємо архітектуру моделі
class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()

        # Один згортковий шар (згідно зі статтею)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=2)

        # Пулінг (згідно зі статтею)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Автоматичне визначення розміру для FC-шару
        self.flatten_size = self._get_flatten_size()

        # Повнозв’язні шари (згідно зі статтею)
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 3)

    def _get_flatten_size(self):
        """Визначає розмір тензора після згортки"""
        x = torch.randn(1, 3, 224, 224)  # Тестове зображення
        x = self.pool(F.relu(self.conv1(x)))
        return x.view(-1).shape[0]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

# Ініціалізація моделі
model = ShallowCNN()

# Перевіримо модель
model = ShallowCNN()
print(model)

# Функція втрат
# Визначаємо, чи є GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Отримуємо баланс класів
class_counts = torch.tensor([57, 234, 390], dtype=torch.float32)  # Кількість випадків COVID, NORMAL, PNEUMONIA
weights = class_counts.sum() / class_counts  # Ваги у зворотному порядку
weights = weights / weights.sum()  # Нормалізуємо ваги
weights = weights.to(device)

# Оновлена функція втрат
criterion = nn.CrossEntropyLoss(weight=weights)

# Оптимізатор (SGD)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Перевіримо, що все ініціалізувалося без помилок
print(optimizer)

# Шлях до датасету
dataset_base = "chest_xray"

# Трансформації для зображень (зміна розміру, перетворення в тензор, нормалізація)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Завантаження даних
train_dataset = datasets.ImageFolder(root=f"{dataset_base}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{dataset_base}/val", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_base}/test", transform=transform)


# DataLoader для пакетного завантаження
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Перевіримо, скільки класів (має бути 3: NORMAL, PNEUMONIA, COVID)
print("Класи:", train_dataset.classes)

# Перевіримо розмір вибірки
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")


# Навчальні параметри
num_epochs = 10 # Можна змінити на більше, якщо потрібно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Використовуємо GPU, якщо є
model.to(device)

# Цикл навчання
for epoch in range(num_epochs):
    model.train()  # Переводимо модель в режим навчання
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Переміщуємо на GPU, якщо доступний

        optimizer.zero_grad()  # Обнуляємо градієнти

        outputs = model(images)  # Прогін вперед
        loss = criterion(outputs, labels)  # Обчислення втрат
        loss.backward()  # Зворотне поширення градієнтів
        optimizer.step()  # Оновлення ваг

        running_loss += loss.item()

        # Обчислення точності
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # Перевірка на валідації
    model.eval()  # Переключаємо в режим оцінки
    val_correct, val_total = 0, 0
    val_loss = 0.0

    with torch.no_grad():  # Вимикаємо обрахунок градієнтів
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

print("Навчання завершено!")


# Тестування моделі
model.eval()  # Переключаємо модель в режим оцінки
test_correct, test_total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * test_correct / test_total
print(f"\n🎯 Точність на тестовому наборі: {test_accuracy:.2f}%")

# Класи
class_names = train_dataset.classes  # ['COVID', 'NORMAL', 'PNEUMONIA']

# Вивід Precision, Recall, F1-score
print("\n🔍 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Матриця помилок
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Предбачені класи")
plt.ylabel("Реальні класи")
plt.title("Матриця помилок")
plt.show()

