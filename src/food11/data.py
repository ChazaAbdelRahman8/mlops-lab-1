from pathlib import Path
from collections import defaultdict
import shutil

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "food11_raw"

PROCESSED_DIR = DATA_DIR / "food11_processed"
MINI_DIR = DATA_DIR / "food11_processed_mini"


CATEGORIES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}

SPLITS = ["training", "evaluation", "validation"]

IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_category(filename):
    try:
        category_id = int(filename.split("_")[0])
        return CATEGORIES.get(category_id)
    except (ValueError, IndexError):
        return None


def resize_and_save(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        )
        image.save(destination)


def prepare_directories():
    for directory in [PROCESSED_DIR, MINI_DIR]:
        if directory.exists():
            shutil.rmtree(directory)


def process_dataset():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at:\n{RAW_DIR}"
        )

    prepare_directories()

    mini_counts = defaultdict(int)

    total_processed = 0
    total_mini = 0

    for split in SPLITS:
        source_split = RAW_DIR / split

        if not source_split.exists():
            print(f"Warning: {source_split} does not exist.")
            continue

        print(f"Processing {split}...")

        files = sorted(
            file
            for file in source_split.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        for source_file in files:
            category = get_category(source_file.name)

            if category is None:
                continue

            processed_destination = (
                PROCESSED_DIR
                / split
                / category
                / source_file.name
            )

            resize_and_save(
                source_file,
                processed_destination,
            )

            total_processed += 1

            key = (split, category)

            if mini_counts[key] < MINI_LIMIT:
                mini_destination = (
                    MINI_DIR
                    / split
                    / category
                    / source_file.name
                )

                resize_and_save(
                    source_file,
                    mini_destination,
                )

                mini_counts[key] += 1
                total_mini += 1

    print("\nData preparation completed.")
    print(f"Processed images: {total_processed}")
    print(f"Mini images: {total_mini}")
    print("Image size: 128x128")


if __name__ == "__main__":
    process_dataset()