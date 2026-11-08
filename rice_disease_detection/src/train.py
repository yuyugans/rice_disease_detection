
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 80
SEED = 42

CLASSES = ["blast", "blight", "tungro"]
AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(SEED)

print("TensorFlow:", tf.__version__)
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

print("Urutan kelas:", train_ds.class_names)

# ==========================================
# 3. CLASS WEIGHT
# ==========================================

class_counts = np.array([
    len(list(
        (DATASET_DIR / "train" / name).glob("*.jpg")
    ))
    for name in CLASSES
])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASSES)),
    y=np.concatenate([
        np.full(count, index)
        for index, count in enumerate(class_counts)
    ])
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(class_weights_array)
}

print("\nJumlah gambar:", class_counts)
print("Class weights:", class_weights)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================================
# 4. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
], name="data_augmentation")

# ==========================================
# 5. ARSITEKTUR CNN
# ==========================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="input_image"
)

x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
x = data_augmentation(x)

# Convolutional block 1
x = tf.keras.layers.Conv2D(
    16, 3, padding="same", use_bias=False
)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)

# Convolutional block 2
x = tf.keras.layers.Conv2D(
    32, 3, padding="same", use_bias=False
)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)

# Convolutional block 3
x = tf.keras.layers.Conv2D(
    64, 3, padding="same", use_bias=False
)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)

# Feature extraction
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Regularization
x = tf.keras.layers.Dropout(0.40)(x)

# Dense layer dengan L2 regularization
x = tf.keras.layers.Dense(
    64,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001)
)(x)

x = tf.keras.layers.Dropout(0.40)(x)

outputs = tf.keras.layers.Dense(
    3,
    activation="softmax",
    name="predictions"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="Rice_Disease_CNN"
)

# ==========================================
# 6. COMPILATION
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n=== ARSITEKTUR MODEL CNN ===")
model.summary()

# ==========================================
# 7. CALLBACKS
# ==========================================

best_model_path = MODEL_DIR / "best_rice_cnn.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=0.000001,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ==========================================
# 8. TRAINING
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
# 9. SIMPAN MODEL TERBAIK
# ==========================================

best_model = tf.keras.models.load_model(best_model_path)

final_model_path = MODEL_DIR / "rice_disease_cnn.keras"
best_model.save(final_model_path)

print("\nModel terbaik berhasil disimpan:", final_model_path)

# ==========================================
# 10. SIMPAN RIWAYAT TRAINING
# ==========================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    MODEL_DIR / "training_history.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(history_data, file, indent=4)

# ==========================================
# 11. GRAFIK TRAINING
# ==========================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

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
print("Epoch dijalankan:", len(history.history["accuracy"]))
print(
    "Validation accuracy terbaik:",
    max(history.history["val_accuracy"])
)
print(
    "Validation loss terbaik:",
    min(history.history["val_loss"])
)
