import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


# 1. Configuration
class Config:
    data_dir = "dataset_nobg"  # Folder dataset after remove background
    batch_size = 32  # Adjust based on your RAM
    num_epochs = 100  # Maximum epochs
    early_stopping_patience = 5  # Stop if no improvement for 5 epochs
    min_delta = 0.001  # Minimum accuracy improvement to qualify as better
    lr = 0.001  # Learning rate
    num_classes = 2  # 0=man, 1=woman
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom padding function
def pad_to_square(img):
    w, h = img.size
    max_dim = max(w, h)
    padding = (
        (max_dim - w) // 2,  # Left
        (max_dim - h) // 2,  # Top
        (max_dim - w + 1) // 2,  # Right
        (max_dim - h + 1) // 2  # Bottom
    )
    return transforms.functional.pad(img, padding, fill=0)

# 2. Data Preparation with Augmentation
def prepare_data():
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),  # Random crop for augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        root=str(Path(Config.data_dir) / "train"),
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=str(Path(Config.data_dir) / "test"),
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, test_loader, train_dataset.classes


# 3. Enhanced Model Definition
def create_model():
    model = models.resnet34(pretrained=True)  # Using larger model

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout for regularization
        nn.Linear(num_ftrs, Config.num_classes)
    )

    # Unfreeze last few layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    return model.to(Config.device)


# 4. Training Function with Early Stopping and LR Scheduling
def train_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1, min_lr=1e-6)

    best_acc = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(Config.num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch + 1}/{Config.num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f} | LR: {current_lr:.6f}")

        # Update scheduler
        scheduler.step(epoch_acc)

        # Early stopping check
        if epoch_acc - best_acc > Config.min_delta:
            best_acc = epoch_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                break

    return history


# 5. Enhanced Visualization
def plot_history(history):
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Learning rate plot
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# 6. Main Execution
if __name__ == "__main__":
    # Initialize
    train_loader, test_loader, class_names = prepare_data()
    print(f"Class mapping: {class_names} -> {list(range(len(class_names)))}")

    model = create_model()
    print(f"Using device: {Config.device}")
    print(f"Model architecture:\n{model}")

    # Check class balance in first batch
    for inputs, labels in train_loader:
        print("\nClass distribution in first batch:", torch.bincount(labels))
        break

    # Train
    print("\nStarting training...")
    history = train_model(model, train_loader, test_loader)

    # Visualize
    plot_history(history)

    print("\nTraining completed! Best model saved as 'best_model.pth'")