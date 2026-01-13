from torchvision import datasets
import os

DATASET_DIR = "dataset-1"  # ← Thay đường dẫn của bạn

print(" Checking dataset structure...\n")

# Check folders exist
required_folders = [
    f"{DATASET_DIR}/train/Acne",
    f"{DATASET_DIR}/train/Non_Acne",
    f"{DATASET_DIR}/val/Acne",
    f"{DATASET_DIR}/val/Non_Acne",
    f"{DATASET_DIR}/test/Acne",
    f"{DATASET_DIR}/test/Non_Acne",
]

print("Folder structure:")
for folder in required_folders:
    exists = os.path.exists(folder)
    status = "Ok" if exists else "No"
    count = len(os.listdir(folder)) if exists else 0
    print(f"   {status} {folder}: {count} files")

# Load dataset
print("\n Loading with ImageFolder...")
try:
    train_dataset = datasets.ImageFolder(f"{DATASET_DIR}/train")
    val_dataset = datasets.ImageFolder(f"{DATASET_DIR}/val")

    print(f"\n Dataset loaded!")
    print(f"   Train classes: {train_dataset.classes}")
    print(f"   Train class_to_idx: {train_dataset.class_to_idx}")
    print(f"   Val classes: {val_dataset.classes}")

    # Count samples per class
    from collections import Counter

    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)

    print(f"\n Train distribution:")
    for class_idx, count in train_counts.items():
        class_name = train_dataset.classes[class_idx]
        print(f"   {class_name} (index {class_idx}): {count} images")

    print(f"\n Val distribution:")
    for class_idx, count in val_counts.items():
        class_name = val_dataset.classes[class_idx]
        print(f"   {class_name} (index {class_idx}): {count} images")

    #  Check class mapping
    print(f"\n  IMPORTANT:")
    if 'acne' in train_dataset.class_to_idx:
        acne_idx = train_dataset.class_to_idx['acne']
        print(f"   'acne' is mapped to index: {acne_idx}")
        print(f"   In training, label {acne_idx} = acne (HAS acne)")

    # Check a few samples
    print(f"\n Sample check (first 5 files):")
    for i in range(min(5, len(train_dataset))):
        img_path, label = train_dataset.samples[i]
        class_name = train_dataset.classes[label]
        print(f"   {i + 1}. {os.path.basename(img_path)} → {class_name} (label={label})")

except Exception as e:
    print(f"\n ERROR: {e}")