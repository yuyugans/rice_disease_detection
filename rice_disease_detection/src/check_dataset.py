from pathlib import Path
from PIL import Image, ImageDraw
import random

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset" / "processed"

CLASSES = ["blast", "blight", "tungro"]
SPLITS = ["train", "validation", "test"]

random.seed(42)

for split in SPLITS:
    print(f"\n=== {split.upper()} ===")

    for class_name in CLASSES:
        folder = DATASET_DIR / split / class_name
        files = list(folder.glob("*.jpg"))

        print(f"{class_name}: {len(files)} gambar")

        if files:
            samples = random.sample(files, min(3, len(files)))

            for file in samples:
                try:
                    with Image.open(file) as image:
                        image.verify()
                except Exception:
                    print("Gambar rusak:", file)