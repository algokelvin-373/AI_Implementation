from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime

# Configurations
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 50

# Path ke folder data
train_dir = '../data/processed/train'
val_dir = '../data/processed/val'

# ✅ Data Augmentation - LEBIH LEMBUT (untuk stabilitas)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,  # ← Dikurangi dari 20 → lebih realistis
    width_shift_range=0.1,  # ← Dikurangi dari 0.2
    height_shift_range=0.1,  # ← Dikurangi dari 0.2
    horizontal_flip=True,
    zoom_range=0.1,  # ← Dikurangi dari 0.2 → jangan terlalu ekstrem
    fill_mode='nearest'
)

# ✅ Validation: hanya rescale
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ✅ Build CNN - Lebih Ringan & Stabil
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    # layers.BatchNormalization(),  # ← DIHAPUS UNTUK STABILITAS (opsional)
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.BatchNormalization(),  # ← DIHAPUS UNTUK STABILITAS (opsional)
    layers.MaxPooling2D(2, 2),

    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    # layers.BatchNormalization(),  # ← DIHAPUS UNTUK STABILITAS (opsional)
    layers.MaxPooling2D(2, 2),

    # Classifier Head
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),  # ← Dikurangi dari 0.5 → lebih ringan
    layers.Dense(4, activation='softmax')  # 4 Class: Top, Bottom, Shoes, Other
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks untuk cegah overfitting & stabilisasi
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Training model
print("Ready training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# Save model with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f'../models/saved_model_method3_{timestamp}.h5')
print(f"Model berhasil disimpan sebagai: saved_model_method3_{timestamp}.h5")