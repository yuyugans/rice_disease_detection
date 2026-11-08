
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Lokasi dataset
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset" / "raw"

CLASSES = ["blast", "blight", "tungro"]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total_images = 0
class_counts = {}

print("=== EKSPLORASI DATASET PENYAKIT PADI ===\n")

# Menampilkan jumlah gambar setiap kelas
for class_name in CLASSES:

    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        print(f"Folder tidak ditemukan: {class_name}")
        continue

    images = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    ]

    valid_images = []
    invalid_images = []

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is not None:
            valid_images.append(image_path)
        else:
            invalid_images.append(image_path)

    class_counts[class_name] = len(valid_images)
    total_images += len(valid_images)

    print(f"Kelas       : {class_name}")
    print(f"Jumlah valid: {len(valid_images)}")
    print(f"Jumlah rusak: {len(invalid_images)}")
    print("-" * 35)

print(f"\nTotal gambar valid: {total_images}")

# Visualisasi distribusi dataset
plt.figure(figsize=(8, 5))

plt.bar(
    class_counts.keys(),
    class_counts.values()
)

plt.title("Distribusi Dataset Penyakit Daun Padi")
plt.xlabel("Kelas Penyakit")
plt.ylabel("Jumlah Gambar")

plt.tight_layout()
plt.show()

# Menampilkan sampel gambar
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, class_name in zip(axes, CLASSES):

    class_dir = DATASET_DIR / class_name

    images = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    ]

    if not images:
        ax.set_title(f"{class_name} - kosong")
        ax.axis("off")
        continue

    image = cv2.imread(str(images[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image)
    ax.set_title(class_name.capitalize())
    ax.axis("off")

plt.tight_layout()
plt.show()
