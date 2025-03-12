import os
import shutil
import random

# Налаштування
source_folder = "imagesWithCovid"  # Звідки беремо зображення
dataset_base = "chest_xray"  # База даних, де є train, val, test
target_folders = {
    "train": os.path.join(dataset_base, "train/COVID"),
    "val": os.path.join(dataset_base, "val/COVID"),
    "test": os.path.join(dataset_base, "test/COVID"),
}

# Отримуємо список усіх файлів
all_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.shuffle(all_files)  # Перемішуємо для випадкового розподілу

# Визначаємо кількість файлів для кожного розділу
train_size = int(0.8 * len(all_files))  # 80%
val_size = int(0.1 * len(all_files))    # 10%
test_size = len(all_files) - train_size - val_size  # 10%

# Розподіляємо файли
splits = {
    "train": all_files[:train_size],
    "val": all_files[train_size:train_size + val_size],
    "test": all_files[train_size + val_size:]
}

# Переміщуємо файли у відповідні папки
for split, files in splits.items():
    os.makedirs(target_folders[split], exist_ok=True)  # Створюємо папки, якщо їх немає
    for file in files:
        src = os.path.join(source_folder, file)
        dst = os.path.join(target_folders[split], file)
        shutil.move(src, dst)  # Переміщуємо файл

print("Розподіл завершено!")
print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Шлях до датасету
dataset_base = "chest_xray"

# Трансформації для зображень (зміна розміру, перетворення в тензор, нормалізація)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Зміна розміру зображень
    transforms.ToTensor(),  # Перетворення в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормалізація
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
