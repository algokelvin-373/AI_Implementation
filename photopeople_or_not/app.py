# app.py - Full-body vs Non-Human Image Classifier

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import argparse
import pandas as pd
from tabulate import tabulate  # Untuk tampilan tabel rapi

# Notes:
# Model klasifikasi gambar berbasis Deep Learning menggunakan Convolutional Neural Network (CNN)
# dengan arsitektur ResNet-18 dan teknik Trasnfer Learning.
# 🔹 Arsitektur Model
# Pre-trained ResNet-18 diambil dari torchvision.models.
# Layer terakhir (fc) diganti agar sesuai dengan jumlah kelas (dalam hal ini 2: fullbody dan non-human).
# Model dilatih ulang (fine-tuned) pada dataset lokal.
# 🔹 Proses Pelatihan
# Optimizer: Adam (dengan learning rate 0.0001)
# Loss Function: Cross-Entropy Loss (umum untuk klasifikasi)
# Epoch: 10
# Batch Size: 32
# Data Augmentation: Hanya resize dan normalisasi (belum ada augmentasi seperti rotasi, flip, dll — bisa jadi peluang peningkatan)

def train_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.0001

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet norms
    ])

    # Check dataset path
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Dataset tidak ditemukan di '{train_dir}' atau '{val_dir}'. "
                                f"Pastikan struktur foldernya benar.")

    # Load dataset
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Jumlah data latih: {len(train_dataset)}")
    print(f"Jumlah data validasi: {len(val_dataset)}")
    print(f"Kelas: {train_dataset.classes}")  # e.g., ['full body', 'non-human']

    # Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # total kelas
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # List untuk menyimpan log tiap epoch
    log_data = []

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
        val_acc = 100. * correct_val / total_val

        # Simpan hasil epoch ke log
        log_data.append({
            'Epoch': epoch + 1,
            'Train Loss': f"{avg_loss:.4f}",
            'Train Acc (%)': f"{train_acc:.2f}",
            'Val Acc (%)': f"{val_acc:.2f}"
        })

        # Tampilkan tabel tiap epoch (opsional: bisa dihapus jika terlalu banyak)
        # Tapi kita akan tampilkan ringkasan akhir saja, atau update tabel secara dinamis

        # Tampilkan tabel hasil semua epoch di akhir
        print("\n" + "=" * 60)
        print("📊 RINGKASAN PELATIHAN PER EPOCH")
        print("=" * 60)
        df_log = pd.DataFrame(log_data)
        print(tabulate(df_log, headers='keys', tablefmt='grid', showindex=False))

        # Simpan model
        os.makedirs("models", exist_ok=True)
        model_path = "models/fullbody_classifier_22072025000001.pth"
        torch.save(model.state_dict(), model_path)
        print(f"\n✅ Model berhasil disimpan di: {model_path}")

        # Simpan log ke file CSV (opsional, sangat berguna untuk analisis lanjut)
        log_df = pd.DataFrame(log_data)
        log_df.to_csv("training_log.csv", index=False)
        print(f"📊 Log pelatihan disimpan ke 'training_log.csv'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih model deteksi full-body manusia.")
    args = parser.parse_args()

    print("🚀 Memulai pelatihan model...")
    train_model()