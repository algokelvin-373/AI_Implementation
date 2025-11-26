import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time

from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import glob

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


# Function predict single image
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


# Predict multiple images in a folder
def predict_folder(folder_path, model):
    print(f"\n🔍 Memproses folder: {folder_path}")
    # Cari semua file gambar
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_paths:
        print("❌ Tidak ada gambar ditemukan di folder.")
        return

    print(f"✅ Ditemukan {len(image_paths)} gambar:")
    results = {}
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"  → {filename} ... \t\t", end="")
        start = time.time() * 1000.0
        result = predict_image(img_path, model)
        end = time.time() * 1000.0
        if result is not None:
            results[filename] = result
            print(f"{result} ({end - start:.2f} ms)")
        else:
            results[filename] = "Error"
            print("❌ Gagal")

    return results

# Main function
def main():
    parser = argparse.ArgumentParser(description="Prediksi apakah gambar berisi manusia full-body atau bukan.")
    parser.add_argument("--folder", type=str, default="./dataset/predict",
                        help="Path ke folder yang berisi gambar untuk diprediksi (default: ./dataset/predict)")
    args = parser.parse_args()

    # Path model
    model_path = "models/fullbody_classifier_13092025000001.pth"

    # Load model
    print("🚀 Memuat model...")
    model = load_model(model_path)

    # Predict folder
    results = predict_folder(args.folder, model)

    # Print summary
    if results:
        print("\n📊 Ringkasan Hasil Prediksi:")
        human_count = sum(1 for r in results.values() if r == 'Human')
        non_human_count = sum(1 for r in results.values() if r == 'Non-Human')
        error_count = sum(1 for r in results.values() if r == 'Error')
        print(f"  Human: {human_count}")
        print(f"  Non-Human: {non_human_count}")
        print(f"  Error: {error_count}")

if __name__ == "__main__":
    main()