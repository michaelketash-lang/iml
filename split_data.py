import os
import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "validation"

VAL_SPLIT = 0.2
SEED = 42


def main():
    # same split every time we run the script
    random.seed(SEED)

    if not TRAIN_DIR.exists():
        print(f"[FAIL] Could not find {TRAIN_DIR}")
        return

    # create validation
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    classes = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes. Starting physical split...")

    total_moved = 0

    for class_dir in classes:
        class_name = class_dir.name
        val_class_dir = VAL_DIR / class_name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        # taking all the pictures
        images = [f for f in class_dir.iterdir() if
                  f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg'}]

        # if no pictures so we probably already splitted in the past
        if len(images) == 0:
            continue

        # mixing pictures
        random.shuffle(images)

        # calculate 20%
        num_val = int(len(images) * VAL_SPLIT)
        val_images = images[:num_val]

        # moving pictures and not copy
        for img_path in val_images:
            shutil.move(str(img_path), str(val_class_dir / img_path.name))

        total_moved += num_val
        print(f"moved {num_val} images for class {class_name}")

    if total_moved > 0:
        print(
            f"\n[SUCCESS] Moved {total_moved} images to the validation folder.")
    else:
        print(
            "\n[INFO] No images were moved. The split might have already been done.")


if __name__ == "__main__":
    main()