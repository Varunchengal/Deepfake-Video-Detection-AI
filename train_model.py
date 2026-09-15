# train_model.py - Real Dataset Training Script using PyTorch & Transfer Learning

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os

print("--- Deepfake Detection Model Training Started ---")

# 1. Data Augmentation & Normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # analyse image in flip mode
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 2. load photos from dataset
data_dir = 'dataset'

try:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"Dataset Loaded Successfully!")
    print(f"Classes Found: {class_names} (0: {class_names[0]}, 1: {class_names[1]})")
    print(f"Training Images: {dataset_sizes['train']} | Validation Images: {dataset_sizes['val']}")

except Exception as e:
    print(f"\n[ERROR] ഡാറ്റാസെറ്റ് ലോഡ് ചെയ്യാൻ കഴിഞ്ഞില്ല! 'dataset/train', 'dataset/val' ഫോൾഡറുകൾ കൃത്യമാണോ എന്ന് നോക്കുക.")
    print(f"Error Details: {e}")
    exit()

# 3. Load Pre-trained ResNet18 (Transfer Learning)
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freez first layer
for param in model.parameters():
    param.requires_grad = False

# last layer changes in real/fake classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 4. Model training (3 Epochs only)
num_epochs = 3
print("\nTraining in progress... (ഇതിന് കുറച്ചു സമയം എടുക്കും, ദയവായി കാത്തിരിക്കുക)")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 15)
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in dataloaders[phase]:
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        print(f"{phase.upper()} -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} ({int(epoch_acc*100)}%)")

# 5. Saving trained model
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "deepfake_model.pth")
torch.save(model.state_dict(), model_path)

print("\n" + "="*50)
print(f"✅ Training Complete! സ്വന്തമായി ട്രെയിൻ ചെയ്ത മോഡൽ ഇവിടെ സേവ് ആയി: {model_path}")
print("="*50)