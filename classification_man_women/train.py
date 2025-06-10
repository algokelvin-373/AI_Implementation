import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# 1. Configuration
class Config:
    data_dir = "dataset_nobg"  # Folder dataset setelah remove background
    batch_size = 32            # Jumlah gambar per batch (sesuaikan dengan RAM)
    num_epochs = 10            # Jumlah iterasi training (bisa diganti 100)
    lr = 0.001                 # Learning rate (BUKAN MSE error!)
    num_classes = 2            # 2 kelas: man (0) dan woman (1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Data Preparation
def prepare_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        root=str(Path(Config.data_dir) / "train"),
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        root=str(Path(Config.data_dir) / "test"),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes


# 3. Model Definition
def create_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, Config.num_classes)
    return model.to(Config.device)


# 4. Training Function
def train_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

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

        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)

        print(f"Epoch {epoch + 1}/{Config.num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}")

        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "best_model.pth")

    return history


# 5. Visualization
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

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

    # Train
    history = train_model(model, train_loader, test_loader)

    # Visualize
    plot_history(history)

    print("Training completed! Best model saved as 'best_model.pth'")