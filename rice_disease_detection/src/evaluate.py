
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset" / "processed"
MODEL_PATH = BASE_DIR / "models" / "rice_disease_cnn.keras"
MODEL_DIR = BASE_DIR / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
CLASSES = ["blast", "blight", "tungro"]

# ==========================================
# 2. MEMERIKSA MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}\n"
        "Jalankan src/train.py terlebih dahulu."
    )

# ==========================================
# 3. MEMUAT DATA TEST
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    class_names=CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("Urutan kelas:", test_ds.class_names)

# ==========================================
# 4. MEMUAT MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel berhasil dimuat.")
print("\n=== EVALUASI MODEL ===")

# ==========================================
# 5. MELAKUKAN PREDIKSI
# ==========================================

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_ds
])

predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)
print("\n=== JUMLAH PREDIKSI SETIAP KELAS ===")

for index, class_name in enumerate(CLASSES):
    jumlah = int(np.sum(y_pred == index))
    print(f"{class_name}: {jumlah} gambar")

# ==========================================
# 6. MENGHITUNG AKURASI
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

print(f"\nAkurasi Model: {accuracy * 100:.2f}%")

# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASSES,
    zero_division=0
)

print("\n=== CLASSIFICATION REPORT ===")
print(report)

report_path = MODEL_DIR / "classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

fig, ax = plt.subplots(figsize=(8, 6))

disp.plot(
    cmap="Blues",
    values_format="d",
    ax=ax,
    colorbar=False
)

plt.title("Confusion Matrix - Rice Disease CNN")
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "confusion_matrix.png",
    dpi=300
)

plt.show()

# ==========================================
# 9. MENYIMPAN HASIL EVALUASI
# ==========================================

results = {
    "accuracy": float(accuracy),
    "accuracy_percentage": float(accuracy * 100),
    "total_test_images": int(len(y_true)),
    "class_names": CLASSES,
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )
}

with open(
    MODEL_DIR / "evaluation_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

print("\n=== EVALUASI SELESAI ===")
print("Hasil evaluasi tersimpan di folder models.")
