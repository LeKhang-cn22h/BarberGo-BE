"""
Acne Classifier Training - SIMPLIFIED METRICS + CHARTS - 50 EPOCHS
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import os
import cv2
import numpy as np
from PIL import Image
import json
import csv
import matplotlib.pyplot as plt

print("="*70)
print("ACNE TRAINING - WITH PORE REMOVAL + SIMPLIFIED EXPORT")
print("="*70)

# ==================== CONFIG ====================
DATASET_DIR = "dataset-1"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "acne_best_pore_removed.pth"

#  Files for export
METRICS_CSV = "training_metrics.csv"
CHART_PATH = "training_chart.png"
SELECTED_EPOCHS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")


# ==================== CUSTOM PREPROCESSING ====================
class PoreRemovalTransform:
    """Custom transform để loại bỏ lỗ chân lông trước khi train"""
    def __init__(self, apply_to_train=True):
        self.apply_to_train = apply_to_train

    def __call__(self, img):
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        smooth = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel, iterations=1)
        final = cv2.GaussianBlur(closed, (3, 3), 0)

        img_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


# ==================== DATA AUGMENTATION ====================
train_transform = transforms.Compose([
    PoreRemovalTransform(apply_to_train=True),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
])

val_transform = transforms.Compose([
    PoreRemovalTransform(apply_to_train=True),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== LOAD DATASET ====================
print("\nLoading dataset...")
train_dataset = datasets.ImageFolder(f"{DATASET_DIR}/train", transform=train_transform)
val_dataset = datasets.ImageFolder(f"{DATASET_DIR}/validation", transform=val_transform)

print(f" Label mapping: {train_dataset.class_to_idx}")
print(f"Train: {len(train_dataset)} images")
print(f"Val: {len(val_dataset)} images")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=True)

# ==================== MODEL ====================
print("\nBuilding model...")
model = models.mobilenet_v2(weights='IMAGENET1K_V1')

print("Freezing early layers...")
for name, param in model.features.named_parameters():
    layer_num = int(name.split('.')[0]) if name.split('.')[0].isdigit() else 0
    if layer_num < 12:
        param.requires_grad = False
    else:
        param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.2),
    nn.Linear(256, 1)
)

model = model.to(DEVICE)
print("Model ready!")

# ==================== LOSS & OPTIMIZER ====================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6  
)


# ==================== SIMPLIFIED METRICS LOGGER ====================
class SimplifiedMetricsLogger:
    """Simplified logger - only 4 metrics + chart"""

    def __init__(self, csv_path, chart_path, selected_epochs):
        self.csv_path = csv_path
        self.chart_path = chart_path
        self.selected_epochs = selected_epochs
        self.all_metrics = []

        # Initialize CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc'])

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Log only 4 core metrics"""
        self.all_metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}"
            ])

    def plot_training_curves(self):
        """Plot training curves"""
        epochs = [m['epoch'] for m in self.all_metrics]
        train_losses = [m['train_loss'] for m in self.all_metrics]
        train_accs = [m['train_acc'] for m in self.all_metrics]
        val_losses = [m['val_loss'] for m in self.all_metrics]
        val_accs = [m['val_acc'] for m in self.all_metrics]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.chart_path, dpi=300, bbox_inches='tight')
        print(f"\n Training chart saved: {self.chart_path}")
        plt.close()

    def print_latex_table(self):
        """Print LaTeX table code"""
        print("\n" + "="*70)
        print("LATEX TABLE CODE")
        print("="*70)

        print("\n\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Kết quả Training và Validation Model CNN}")
        print("\\begin{tabular}{|c|c|c|c|c|}")
        print("\\hline")
        print("\\textbf{Epoch} & \\textbf{Train Loss} & \\textbf{Train Acc} & \\textbf{Val Loss} & \\textbf{Val Acc} \\\\")
        print("\\hline")

        for epoch_num in self.selected_epochs:
            for metric in self.all_metrics:
                if metric['epoch'] == epoch_num:
                    print(f"{epoch_num} & {metric['train_loss']:.4f} & {metric['train_acc']:.4f} & {metric['val_loss']:.4f} & {metric['val_acc']:.4f} \\\\")
                    print("\\hline")
                    break

        print("\\end{tabular}")
        print("\\end{table}")
        print("\n" + "="*70)


# ==================== TRAINING ====================
def train():
    best_val_acc = 0.0
    # Initialize logger
    metrics_logger = SimplifiedMetricsLogger(METRICS_CSV, CHART_PATH, SELECTED_EPOCHS)

    for epoch in range(EPOCHS):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*70}")

        start_time = time.time()

        # ===== TRAIN PHASE =====
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # ===== VALIDATION PHASE =====
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        epoch_time = time.time() - start_time

        # Log metrics
        metrics_logger.log(epoch + 1, train_loss, train_acc, val_loss, val_acc)

        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f" BEST MODEL SAVED! (Val Acc: {val_acc:.4f})")

        #  Scheduler vẫn giữ để điều chỉnh learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE - 50 EPOCHS!")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print("="*70)

    # Export results
    metrics_logger.plot_training_curves()
    metrics_logger.print_latex_table()


if __name__ == "__main__":
    train()