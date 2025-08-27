import argparse
import time

from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
def load_model(model_path):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# Transformation input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function predict
def predict_image(image_path, model):
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"❌ File is not found: {image_path}")
        return None
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat membuka gambar: {e}")
        return None

    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return 'Human' if predicted.item() == 0 else 'Non-Human'

# Main function
def main():
    parser = argparse.ArgumentParser(description="Prediksi apakah gambar berisi manusia full-body atau bukan.")
    parser.add_argument("image_path", type=str, help="Path ke gambar yang akan diprediksi")
    args = parser.parse_args()

    # Path model
    model_path = "models/fullbody_classifier_22072025000001.pth"

    # Load model
    print("🚀 Memuat model...")
    model = load_model(model_path)

    # Predict image
    print(f"\nGambar: {args.image_path}")

    start = time.time() * 1000.0
    result = predict_image(args.image_path, model)
    end = time.time() * 1000.0

    if result is not None:
        print(f"Hasil Prediksi: {result}")

    print(f"Time execution: {end - start} ms")

if __name__ == "__main__":
    main()