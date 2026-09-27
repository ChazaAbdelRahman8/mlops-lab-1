from pathlib import Path
import shutil

from PIL import Image


# ============================================================
# Paths
# ============================================================

# Current file:
# mlops-lab-1/src/food11/data.py
#
# parents[2] -> mlops-lab-1/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "food11_raw"
PROCESSED_DIR = DATA_DIR / "food11_processed"
MINI_DIR = DATA_DIR / "food11_processed_mini"


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (128, 128)

# Maximum number of images per class per split
# in the mini dataset
MINI_LIMIT = 100

SPLITS = [
    "training",
    "evaluation",
    "validation",
]

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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# Category detection
# ============================================================

def get_category_from_filename(filename: str):
    """
    Food-11 filenames start with the category number.

    Examples:
        0_268.jpg  -> Bread
        1_15.jpg   -> Dairy product
        5_90.jpg   -> Meat
        10_30.jpg  -> Vegetable-Fruit
    """

    try:
        class_id = int(filename.split("_")[0])

        if class_id in CATEGORIES:
            return CATEGORIES[class_id]

    except (ValueError, IndexError):
        pass

    return None


# ============================================================
# Image processing
# ============================================================

def resize_and_save(source_path: Path, destination_path: Path):
    """
    Resize one image to 128x128 and save it.
    """

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Image.open(source_path) as image:
        image = image.convert("RGB")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        )

        image.save(destination_path)


# ============================================================
# Output folder preparation
# ============================================================

def clean_output_folders():
    """
    Remove old generated processed datasets.
    """

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    if MINI_DIR.exists():
        shutil.rmtree(MINI_DIR)


def create_output_folders():
    """
    Create all split/category folders.

    Example:
        food11_processed/
            training/
                Bread/
                Dairy product/
                ...
    """

    for output_root in [
        PROCESSED_DIR,
        MINI_DIR,
    ]:
        for split in SPLITS:
            for category in CATEGORIES.values():
                folder = output_root / split / category

                folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )


# ============================================================
# Process one split
# ============================================================

def process_split(split: str):
    """
    Process training, evaluation, or validation.
    """

    source_folder = RAW_DIR / split

    print()
    print("=" * 60)
    print(f"Processing {split}")
    print("=" * 60)
    print(f"Source: {source_folder}")

    if not source_folder.exists():
        print("ERROR: Split folder does not exist.")
        return 0, 0

    # Find images directly or recursively
    image_files = sorted(
        file
        for file in source_folder.rglob("*")
        if (
            file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    print(f"Images found: {len(image_files)}")

    if len(image_files) == 0:
        print("WARNING: No images found in this split.")
        return 0, 0

    full_counts = {
        category: 0
        for category in CATEGORIES.values()
    }

    mini_counts = {
        category: 0
        for category in CATEGORIES.values()
    }

    full_total = 0
    mini_total = 0

    for image_path in image_files:

        category = get_category_from_filename(
            image_path.name
        )

        if category is None:
            print(
                f"Skipping file because category "
                f"could not be detected: {image_path.name}"
            )
            continue

        # ====================================================
        # Full processed dataset
        # ====================================================

        full_destination = (
            PROCESSED_DIR
            / split
            / category
            / image_path.name
        )

        resize_and_save(
            image_path,
            full_destination,
        )

        full_counts[category] += 1
        full_total += 1

        # ====================================================
        # Mini processed dataset
        # Maximum 100 images per category per split
        # ====================================================

        if mini_counts[category] < MINI_LIMIT:

            mini_destination = (
                MINI_DIR
                / split
                / category
                / image_path.name
            )

            resize_and_save(
                image_path,
                mini_destination,
            )

            mini_counts[category] += 1
            mini_total += 1

    # ========================================================
    # Print counts
    # ========================================================

    print()
    print(f"Category counts for {split}:")
    print("-" * 60)

    for category in CATEGORIES.values():

        print(
            f"{category:20} "
            f"Full: {full_counts[category]:4} | "
            f"Mini: {mini_counts[category]:4}"
        )

    return full_total, mini_total


# ============================================================
# Main dataset processing
# ============================================================

def process_dataset():

    print()
    print("=" * 60)
    print("Food-11 Data Preparation")
    print("=" * 60)

    print()
    print("Raw dataset:")
    print(RAW_DIR)

    # --------------------------------------------------------
    # Check raw dataset
    # --------------------------------------------------------

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"\nRaw dataset not found.\n\n"
            f"Expected path:\n"
            f"{RAW_DIR}\n"
        )

    # --------------------------------------------------------
    # Clean previous outputs
    # --------------------------------------------------------

    print()
    print("Cleaning old processed datasets...")

    clean_output_folders()

    # --------------------------------------------------------
    # Create category folders
    # --------------------------------------------------------

    print("Creating output folders...")

    create_output_folders()

    total_full = 0
    total_mini = 0

    # --------------------------------------------------------
    # Process each split
    # --------------------------------------------------------

    for split in SPLITS:

        full_count, mini_count = process_split(split)

        total_full += full_count
        total_mini += mini_count

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 60)
    print("Data preparation completed")
    print("=" * 60)

    print(f"Full processed images: {total_full}")
    print(f"Mini processed images: {total_mini}")
    print(
        f"Image dimensions: "
        f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
    )

    print()
    print("Processed dataset:")
    print(PROCESSED_DIR)

    print()
    print("Mini dataset:")
    print(MINI_DIR)

    if total_full == 0:
        print()
        print(
            "ERROR: No images were processed."
        )

        print(
            "Check that food11_raw contains "
            "training, evaluation, and validation folders "
            "with image files inside them."
        )

    else:
        print()
        print("Done successfully.")


# ============================================================
# Run script
# ============================================================

if __name__ == "__main__":
    process_dataset()