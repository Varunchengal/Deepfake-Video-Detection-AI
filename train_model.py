# train_model.py
# CPU-friendly ResNet18 fine-tuning for deepfake face classification

import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_model.pth")

MAX_EPOCHS = 20
BATCH_SIZE = 16  # Increased from 8 to 16 for BatchNorm stability
EARLY_STOPPING_PATIENCE = 6
NUM_WORKERS = 0

FEATURES_LR = 0.0001    # Learning rate for fine-tuning layer3 and layer4
CLASSIFIER_LR = 0.0005
WEIGHT_DECAY = 0.001

SEED = 42
DEVICE = torch.device("cpu")

# --------------------------------------------------
# 2. REPRODUCIBILITY
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

# --------------------------------------------------
# 3. DATASET CLASS (0 = REAL, 1 = FAKE)
# --------------------------------------------------

class RealFakeImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        folder_names = [
            name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))
        ]
        name_lookup = {name.lower(): name for name in folder_names}
        if "real" not in name_lookup or "fake" not in name_lookup:
            raise FileNotFoundError(
                f"Expected 'real' and 'fake' folders in {directory}. Found: {folder_names}"
            )
        real_name = name_lookup["real"]
        fake_name = name_lookup["fake"]
        return [real_name, fake_name], {real_name: 0, fake_name: 1}

# --------------------------------------------------
# 4. TRANSFORMS (Aligned with Inference)
# --------------------------------------------------

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]),
}

# --------------------------------------------------
# 5. LOAD DATASET
# --------------------------------------------------

image_datasets = {
    split: RealFakeImageFolder(os.path.join(DATA_DIR, split), data_transforms[split])
    for split in ["train", "val"]
}
train_dataset = image_datasets["train"]
val_dataset = image_datasets["val"]

train_counts = [0, 0]
for _, label in train_dataset.samples:
    train_counts[label] += 1

val_counts = [0, 0]
for _, label in val_dataset.samples:
    val_counts[label] += 1

dataloaders = {
    "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
    "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
}
dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

print(f"Train: {dataset_sizes['train']} (Real: {train_counts[0]}, Fake: {train_counts[1]})")
print(f"Val  : {dataset_sizes['val']} (Real: {val_counts[0]}, Fake: {val_counts[1]})")

# --------------------------------------------------
# 6. LOAD RESNET18 & UNFREEZE LAYER3 + LAYER4 + FC
# --------------------------------------------------

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer3 and layer4 to learn subtle deepfake texture features
for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

# Add dropout to prevent overfitting
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, 2)
)
model = model.to(DEVICE)

def set_frozen_layers_eval(m):
    """Keep BatchNorm statistics fixed for frozen layers."""
    m.conv1.eval()
    m.bn1.eval()
    m.layer1.eval()
    m.layer2.eval()

# --------------------------------------------------
# 7. LOSS, OPTIMIZER & SCHEDULER
# --------------------------------------------------

total_train = sum(train_counts)
class_weights = torch.tensor(
    [total_train / (2.0 * train_counts[0]), total_train / (2.0 * train_counts[1])],
    dtype=torch.float32,
    device=DEVICE,
)

# Label smoothing reduces model overconfidence
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

optimizer = optim.AdamW(
    [
        {"params": model.layer3.parameters(), "lr": FEATURES_LR * 0.5},
        {"params": model.layer4.parameters(), "lr": FEATURES_LR},
        {"params": model.fc.parameters(), "lr": CLASSIFIER_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

# --------------------------------------------------
# 8. TRAINING LOOP
# --------------------------------------------------

best_val_accuracy = 0.0
best_val_loss = float("inf")
best_model_weights = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

os.makedirs(MODEL_DIR, exist_ok=True)

for epoch in range(MAX_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
            set_frozen_layers_eval(model)  # Preserve frozen BatchNorm layers
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]
        print(f"{phase.upper():5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}%")

        if phase == "val":
            # Save model if validation accuracy improves or loss decreases at same accuracy
            if (epoch_acc > best_val_accuracy) or (
                epoch_acc == best_val_accuracy and epoch_loss < best_val_loss
            ):
                best_val_accuracy = epoch_acc
                best_val_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                torch.save(best_model_weights, MODEL_PATH)
                print("  --> Best model saved!")
            else:
                epochs_without_improvement += 1

    scheduler.step()

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping triggered.")
        break

print(f"\nBest Val Accuracy: {best_val_accuracy * 100:.2f}%")