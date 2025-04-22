# Shallow CNN
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torchvision
#
# # Shallow CNN model (from the article)
# class ShallowCNN(nn.Module):
#     def __init__(self):
#         super(ShallowCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=2)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.flatten_size = self._get_flatten_size()
#         self.fc1 = nn.Linear(self.flatten_size, 256)
#         self.fc2 = nn.Linear(256, 3)
#
#     def _get_flatten_size(self):
#         x = torch.randn(1, 3, 224, 224)
#         x = self.pool(F.relu(self.conv1(x)))
#         return x.view(-1).shape[0]
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = torch.flatten(x, start_dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.log_softmax(self.fc2(x), dim=1)
#         return x
#
# # Device config
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Initialize model
# model = ShallowCNN().to(device)
# print(model)
#
# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
#
# # Dataset
# dataset_base = "chest_xray"
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# train_dataset = datasets.ImageFolder(root=f"{dataset_base}/train", transform=transform)
# val_dataset = datasets.ImageFolder(root=f"{dataset_base}/val", transform=transform)
# test_dataset = datasets.ImageFolder(root=f"{dataset_base}/test", transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# print("Classes:", train_dataset.classes)
# print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
#
# # Training
# num_epochs = 50
# train_losses, val_losses, val_accuracies = [], [], []
#
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct, total = 0, 0
#
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     train_accuracy = 100 * correct / total
#     avg_loss = running_loss / len(train_loader)
#     train_losses.append(avg_loss)
#
#     model.eval()
#     val_correct, val_total = 0, 0
#     val_loss = 0.0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             val_total += labels.size(0)
#             val_correct += (predicted == labels).sum().item()
#
#     val_accuracy = 100 * val_correct / val_total
#     avg_val_loss = val_loss / len(val_loader)
#     val_losses.append(avg_val_loss)
#     val_accuracies.append(val_accuracy)
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
#
# print("Training finished!")
#
# # Save model
# torch.save(model.state_dict(), "shallow_cnn.pth")
# print("✅ Model saved as shallow_cnn.pth")
#
# # Testing
# model.eval()
# test_correct, test_total = 0, 0
# all_preds, all_labels = [], []
#
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         test_total += labels.size(0)
#         test_correct += (predicted == labels).sum().item()
#         all_preds.extend(predicted.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#
# test_accuracy = 100 * test_correct / test_total
# print(f"\n🎯 Test Accuracy: {test_accuracy:.2f}%")
#
# # Classification report
# class_names = train_dataset.classes
# print("\n🔍 Classification Report:")
# print(classification_report(all_labels, all_preds, target_names=class_names))
#
# # Confusion matrix
# conf_matrix = confusion_matrix(all_labels, all_preds)
# print("\nConfusion Matrix:")
# print(conf_matrix)
#
# plt.figure(figsize=(6, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig("confusion_matrix.png")
# plt.show()
#
# # Loss plot
# plt.figure()
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Validation Loss")
# plt.title("Loss per Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.tight_layout()
# plt.savefig("loss_plot.png")
# plt.show()
#
# # Accuracy plot
# plt.figure()
# plt.plot(val_accuracies, label="Validation Accuracy")
# plt.title("Validation Accuracy per Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy %")
# plt.legend()
# plt.tight_layout()
# plt.savefig("val_accuracy_plot.png")
# plt.show()
#
# # Sample predictions
# def show_predictions():
#     model.eval()
#     images, labels = next(iter(test_loader))
#     outputs = model(images.to(device))
#     _, preds = torch.max(outputs, 1)
#
#     fig = plt.figure(figsize=(12, 6))
#     for i in range(6):
#         ax = fig.add_subplot(2, 3, i+1)
#         img = images[i].permute(1, 2, 0).numpy()
#         img = (img * 0.229 + 0.485).clip(0, 1)
#         ax.imshow(img)
#         ax.set_title(f"Predicted: {class_names[preds[i]]}\nActual: {class_names[labels[i]]}")
#         ax.axis("off")
#
#     plt.tight_layout()
#     plt.savefig("sample_predictions.png")
#     plt.show()
#
# show_predictions()




















# ImprovedCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import numpy as np
import os


class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.flatten_size = self._get_flatten_size()
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 3)

    def _get_flatten_size(self):
        x = torch.randn(1, 3, 224, 224)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.view(-1).shape[0]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.flatten_size)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# --- Data preparation ---
dataset_base = "chest_xray"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=f"{dataset_base}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root=f"{dataset_base}/val", transform=test_transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_base}/test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Initializing the model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN().to(device)
print(model)

# --- Counting classes and weights ---
class_counts = [0] * len(train_dataset.classes)
for _, label in train_dataset:
    class_counts[label] += 1
class_weights = [sum(class_counts) / c for c in class_counts]
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# --- Losses, optimizer, scheduler ---
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

print(optimizer)
print("Classes:", train_dataset.classes)
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# --- Training ---
num_epochs = 50
patience = 5
best_val_loss = float('inf')
early_stop_counter = 0
train_losses, val_losses, val_accuracies = [], [], []
best_model_path = "best_model.pth"

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    #Validation
    model.eval()
    val_loss = 0.0
    val_correct, val_total = 0, 0
    with torch.no_grad():
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
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

print("The training is over!")
print(f"✅ The best model is saved as {best_model_path}")

# --- Loading the best model ---
model.load_state_dict(torch.load(best_model_path))

# --- Testing ---
model.eval()
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
print(f"\n🎯 Accuracy on the test set: {test_accuracy:.2f}%")

print("\n🔍 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# --- Error matrix ---
conf_matrix = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --- Loss Plot ---
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()

# --- Accuracy Plot ---
plt.figure()
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")
plt.legend()
plt.savefig("val_accuracy_plot.png")
plt.show()

# --- Sample Predictions ---
def show_predictions():
    model.eval()
    images, labels = next(iter(test_loader))
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(12, 6))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1)
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * 0.229 + 0.485).clip(0, 1)
        ax.imshow(img)
        ax.set_title(f"Predicted: {train_dataset.classes[preds[i]]}\nActual: {train_dataset.classes[labels[i]]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.show()

show_predictions()
