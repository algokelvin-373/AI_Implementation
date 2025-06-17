import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn

# 1. Inisialisasi Model
model = models.resnet34(weights=None)  # Harus sesuai dengan training
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)

# 2. Load State Dict
try:
    state_dict = torch.load("checkpoints/best_model4.pth", weights_only=True)
    model.load_state_dict(state_dict)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 4. Transformasi Data
def pad_to_square(img):
    w, h = img.size
    max_dim = max(w, h)
    padding = (
        (max_dim - w) // 2,
        (max_dim - h) // 2,
        (max_dim - w + 1) // 2,
        (max_dim - h + 1) // 2
    )
    return transforms.functional.pad(img, padding, fill=0)

transform = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Dataset & DataLoader
val_dataset = datasets.ImageFolder("dataset_nobg/test", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Total correct: {correct}')
print(f"Akurasi Validasi: {100 * correct / total:.2f}%")