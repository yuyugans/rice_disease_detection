
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ==========================================
# 1. KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
SEED = 42

CLASSES = ["blast", "blight", "tungro"]

AUTOTUNE = tf.data.AUTOTUNE

# Reproducibility
tf.keras.utils.set_random_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("GPU tersedia:", tf.config.list_physical_devices("GPU"))


# ==========================================
# 2. MEMUAT DATASET
# ==========================================

print("\nMemuat dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    class_names=CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "validation",
    class_names=CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

print("\nUrutan kelas:", train_ds.class_names)


# ==========================================
# 3. MENGHITUNG CLASS WEIGHT
# ==========================================

class_counts = np.array([
    len(list(
        (DATASET_DIR / "train" / class_name).glob("*.jpg")
    ))
    for class_name in CLASSES
])

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASSES)),
    y=np.concatenate([
        np.full(count, index)
        for index, count in enumerate(class_counts)
    ])
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(class_weights)
}

print("\nDistribusi training:", class_counts)
print("Class weights:", class_weights)


# ==========================================
# 4. OPTIMASI DATA PIPELINE
# ==========================================

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomContrast(0.10),
], name="data_augmentation")


# ==========================================
# 6. MEMBANGUN ARSITEKTUR CNN
# ==========================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="input_image"
)

# Normalisasi piksel 0-255 menjadi 0-1
x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

# Augmentasi hanya aktif saat training
x = data_augmentation(x)

# Convolutional Block 1
x = tf.keras.layers.Conv2D(
    32, (3, 3), padding="same", activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# Convolutional Block 2
x = tf.keras.layers.Conv2D(
    64, (3, 3), padding="same", activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# Convolutional Block 3
x = tf.keras.layers.Conv2D(
    128, (3, 3), padding="same", activation="relu"
)(x)

x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)

# Feature extraction
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Regularization
x = tf.keras.layers.Dropout(0.40)(x)

# Dense layer
x = tf.keras.layers.Dense(
    128, activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.30)(x)

# Output layer
outputs = tf.keras.layers.Dense(
    3, activation="softmax", name="predictions"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="Rice_Disease_CNN"
)


# ==========================================
# 7. COMPILATION MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n=== ARSITEKTUR MODEL CNN ===")
model.summary()


# ==========================================
# 8. CALLBACKS
# ==========================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=0.00001,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "best_rice_cnn.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]


# ==========================================
# 9. TRAINING MODEL
# ==========================================

print("\n=== MEMULAI TRAINING CNN ===")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ==========================================
# 10. MENYIMPAN MODEL TERAKHIR
# ==========================================

model.save(MODEL_DIR / "rice_disease_cnn.keras")

print("\nModel berhasil disimpan!")


# ==========================================
# 11. MENYIMPAN RIWAYAT TRAINING
# ==========================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    MODEL_DIR / "training_history.json",
    "w"
) as file:

    json.dump(history_data, file, indent=4)


# ==========================================
# 12. VISUALISASI AKURASI DAN LOSS
# ==========================================

plt.figure(figsize=(12, 5))

# Grafik akurasi
plt.subplot(1, 2, 1)

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Grafik loss
plt.subplot(1, 2, 2)

plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "training_graph.png",
    dpi=300
)

plt.show()

print("\n=== TRAINING SELESAI ===")
