# app.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

# === Configuration ===
MODEL_PATH = '../models/saved_model_method3_20251125_144745.h5'
IMG_SIZE = (150, 150)
CLASS_NAMES = ['Top', 'Bottom', 'Shoes', 'Other']

# Load model (single saat start)
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

class ClothingClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Klasifikasi Pakaian")
        self.root.geometry("600x700")
        self.root.resizable(False, False)

        # Title
        title = tk.Label(root, text="👗 AI Klasifikasi Pakaian", font=("Arial", 18, "bold"))
        title.pack(pady=10)

        # Add Button upload
        self.btn_upload = tk.Button(root, text="Upload Gambar", command=self.upload_image, font=("Arial", 12))
        self.btn_upload.pack(pady=10)

        # Area Image
        self.image_label = tk.Label(root, bg="lightgray", width=400, height=300)
        self.image_label.pack(pady=10)

        # Area Result
        self.result_label = tk.Label(root, text="Belum ada prediksi", font=("Arial", 14), fg="blue")
        self.result_label.pack(pady=10)

        self.confidence_label = tk.Label(root, text="", font=("Arial", 12), fg="green")
        self.confidence_label.pack()

        # Description
        info = tk.Label(root, text="Model: 4 kelas (Top, Bottom, Shoes, Other)\nAkurasi: ~97%", font=("Arial", 10), fg="gray")
        info.pack(side="bottom", pady=10)

    def upload_image(self):
        # Choose file image
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        try:
            # Upload dan show image
            img = Image.open(file_path)
            img_display = img.copy()
            img_display.thumbnail((400, 300))
            photo = ImageTk.PhotoImage(img_display)
            self.image_label.configure(image=photo, bg="white")
            self.image_label.image = photo

            # Prediction
            prediction, confidence = self.predict_image(img)

            # Show Result
            self.result_label.config(text=f"Prediksi: {prediction}")
            self.confidence_label.config(text=f"Akurasi: {confidence:.2f}%")

        except Exception as e:
            messagebox.showerror("Error", f"Gagal memproses gambar:\n{str(e)}")

    def predict_image(self, img):
        # Preprocessing: resize, normalisasi
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        if img_array.shape == (150, 150, 4):  # RGBA → RGB
            img_array = img_array[:, :, :3]
        elif img_array.shape == (150, 150):    # Grayscale → RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        return predicted_class, confidence

# === Run GUI Apps ===
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model tidak ditemukan: {MODEL_PATH}")
        print("Pastikan file model ada di folder 'models/'")
        exit(1)

    root = tk.Tk()
    app = ClothingClassifierApp(root)
    root.mainloop()