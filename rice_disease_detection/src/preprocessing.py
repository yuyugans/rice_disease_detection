
import cv2
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# KONFIGURASI
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "dataset" / "raw"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

CLASSES = ["blast", "blight", "tungro"]

IMAGE_SIZE = (224, 224)

RANDOM_STATE = 42

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==========================================
# MEMBACA DATASET
# ==========================================

image_paths = []
labels = []

print("Membaca dataset...\n")

for class_name in CLASSES:

    class_dir = RAW_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {class_dir}"
        )

    files = [
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    ]

    valid_count = 0

    for file_path in files:

        image = cv2.imread(str(file_path))

        if image is None:
            print(f"Gambar rusak dilewati: {file_path.name}")
            continue

        image_paths.append(str(file_path))
        labels.append(class_name)

        valid_count += 1

    print(f"{class_name}: {valid_count} gambar valid")


print("\nTotal gambar valid:", len(image_paths))


# ==========================================
# PEMBAGIAN DATASET
# 70% TRAIN, 15% VALIDATION, 15% TEST
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)


# ==========================================
# CEK FOLDER OUTPUT
# ==========================================

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        "Folder processed tidak kosong. "
        "Pindahkan atau hapus hasil preprocessing "
        "sebelumnya sebelum menjalankan ulang."
    )


# ==========================================
# FUNGSI PREPROCESSING
# ==========================================

def process_and_save(paths, labels, split_name):

    print(f"\nMemproses {split_name}...")

    for index, (path, label) in enumerate(
        zip(paths, labels)
    ):

        image = cv2.imread(path)

        if image is None:
            print(f"Gagal membaca: {path}")
            continue

        # Resize gambar
        image = cv2.resize(
            image,
            IMAGE_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # Konversi BGR ke RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Simpan ke folder kelas
        save_dir = OUTPUT_DIR / split_name / label
        save_dir.mkdir(parents=True, exist_ok=True)

        # Nama file dibuat unik
        filename = f"{index:05d}.jpg"

        save_path = save_dir / filename

        # OpenCV menyimpan dalam format BGR
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(str(save_path), image)


# ==========================================
# MENYIMPAN DATASET
# ==========================================

process_and_save(X_train, y_train, "train")
process_and_save(X_val, y_val, "validation")
process_and_save(X_test, y_test, "test")


# ==========================================
# MENYIMPAN INFORMASI KELAS
# ==========================================

class_info = {
    "classes": CLASSES,
    "image_size": list(IMAGE_SIZE),
    "normalization": "pixel / 255.0",
    "random_state": RANDOM_STATE
}

with open(
    OUTPUT_DIR / "class_info.json",
    "w"
) as file:

    json.dump(class_info, file, indent=4)


# ==========================================
# HASIL AKHIR
# ==========================================

print("\n=== PREPROCESSING SELESAI ===")

for split_name in ["train", "validation", "test"]:

    print(f"\n{split_name.upper()}")

    for class_name in CLASSES:

        folder = OUTPUT_DIR / split_name / class_name

        count = len(list(folder.glob("*.jpg")))

        print(f"{class_name}: {count}")

print("\nLokasi hasil:", OUTPUT_DIR)
