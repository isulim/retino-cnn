"""
Remove unused CSV files and one directory with unused images.
Create directories for each class of diabetic retinopathy and move files to their respective category.
"""

import os
import shutil

from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from pandas import read_csv


DR_CLASSES = {
    0: "healthy",
    1: "DR",
    2: "DR",
    3: "DR",
    4: "DR"
}


def remove_unused_files(path: Path):
    """Remove unused CSV files and one directory with unused images."""

    train_labels19 = Path(path, 'labels', "trainLabels19.csv")
    test_images19 = Path(path, 'labels', "testImages19.csv")
    train_labels15 = Path(path, 'labels', "trainLabels15.csv")
    test_labels15 = Path(path, 'labels', "testLabels15.csv")

    for file in tqdm([train_labels19, train_labels15, test_labels15, test_images19]):
        if file.exists():
            os.remove(file)
            print(f"Removed {file}")

    imgs = Path(path, 'resized_test19')
    if imgs.exists():
        shutil.rmtree(imgs)
        print(f"Removed {imgs} directory")


def rename_files(path: Path):
    """Rename files in the directory to match the format of the labels CSV file."""

    images = Path(path, 'resized_traintest15_train19')
    if images.exists():
        shutil.move(images, Path(path, 'images'))
        print(f"Renamed `raw/resized_traintest15_train19` to `raw/images`")

    labels = Path(path, 'labels', 'traintestLabels15_trainLabels19.csv')
    if labels.exists():
        shutil.move(labels, Path(path, 'labels.csv'))
        shutil.rmtree(Path(path, "labels"))
        print(f"Renamed `raw/labels/traintestLabels15_trainLabels19` to raw/labels.csv")


def create_dr_classes(path: Path):
    """Create directories for each class of diabetic retinopathy."""

    for severity in DR_CLASSES.values():
        os.makedirs(Path(path, severity), exist_ok=True)


def move_images_to_categories(path: Path):
    """Move images to their respective category directories."""

    from tqdm import tqdm
    images = Path(path, "images")

    labels = read_csv(Path(path, "labels.csv"))
    print("Moving images to their respective category directories...")
    for image in tqdm(images.glob("*")):
        label = labels[labels["image"] == image.stem]["level"].values[0]
        shutil.move(image, Path(path, DR_CLASSES[label], image.name))

    shutil.rmtree(images)


if __name__ == "__main__":

    load_dotenv()
    kaggle_dir: str = os.getenv("KAGGLE_FILES_DIR", "")
    raw_path = Path(kaggle_dir, "raw")

    remove_unused_files(raw_path)
    rename_files(raw_path)
    create_dr_classes(raw_path)
    move_images_to_categories(raw_path)
