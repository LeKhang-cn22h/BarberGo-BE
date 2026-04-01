import os
import cv2
import csv
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ================== CONFIG ==================
DATASET_PATH = "dataset-1"

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_PATH = "acne_best.pth"
CSV_LOG_PATH = "training_metrics.csv"

# ================== PORE REMOVAL ==================
def pore_removal(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    blur = cv2.medianBlur(img, 5)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    result = cv2.addWeighted(img, 0.4, closing, 0.6, 0)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

class PoreRemovalTransform:
    def __call__(self, img):
        return pore_removal(img)

# ================== TRANSFORMS ==================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    PoreRemovalTransform(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    PoreRemovalTransform(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================== LOAD DATA ==================
train_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "validation"), transform=val_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "test"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print("Classes:", train_dataset.classes)
print("Class to index:", train_dataset.class_to_idx)

# ================== MODEL ==================
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ================== INIT CSV ==================
with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

# ================== TRAIN ==================
best_val_acc = 0

for epoch in range(EPOCHS):
    print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

    # ---- TRAIN ----
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        probs = torch.sigmoid(preds)
        predicted = (probs > 0.5).float()

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = correct / total

    # ---- VALIDATE ----
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, labels)

            val_loss += loss.item()

            probs = torch.sigmoid(preds)
            predicted = (probs > 0.5).float()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # ---- WRITE CSV ----
    with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc])

    # ---- SAVE BEST ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("Saved best model!")
# ================== TEST + CONFUSION MATRIX ==================
print("\n Evaluating on TEST set...")

model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        preds = model(imgs)
        probs = torch.sigmoid(preds)
        predicted = (probs > 0.5).float()

        all_preds.extend(predicted.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

# Accuracy
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = (all_preds == all_labels).mean()
print(f"TEST ACCURACY: {test_acc:.4f}")



# ================== CONFUSION MATRIX ==================
print(" Generating confusion matrix...")

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=train_dataset.classes
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Acne Detection")
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()

print("Saved: confusion_matrix.png")

print("\n Training completed!")
print("Best Val Acc:", best_val_acc)
print("Model saved to:", MODEL_SAVE_PATH)
print("CSV log saved to:", CSV_LOG_PATH)
# ================== PLOT TRAINING CHART ==================
print("\nGenerating training chart...")

df = pd.read_csv(CSV_LOG_PATH)

plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
plt.plot(df["epoch"], df["val_acc"], label="Val Acc")

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Metrics")
plt.legend()
plt.grid(True)
plt.savefig("train_chart.png", dpi=200)
plt.close()

print("Saved: train_chart.png")
