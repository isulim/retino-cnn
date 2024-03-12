"""
Remove unused CSV files and one directory with unused images.
"""

import os
import shutil

from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


def remove_unused_files(path: Path):
    """Remove unused CSV files and one directory with unused images."""

    train_labels19 = Path(path, "trainLabels19.csv")
    train_labels15 = Path(path, "trainLabels15.csv")
    test_labels15 = Path(path, "testLabels15.csv")

    for file in tqdm([train_labels19, train_labels15, test_labels15]):
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
    shutil.move(images, Path(path, 'images'))
    print(f"Renamed `raw/resized_traintest15_train19` to `raw/images`")
    labels = Path(path, 'labels', 'traintestLabels15_trainLabels19.csv')
    shutil.move(labels, Path(path, 'labels.csv'))
    shutil.rmtree(Path(path, "labels"))
    print(f"Renamed `raw/labels/traintestLabels15_trainLabels19` to raw/labels.csv")


if __name__ == "__main__":

    load_dotenv()
    kaggle_dir: str = os.getenv("KAGGLE_FILES_DIR", "")
    raw_path = Path(kaggle_dir, "raw")

    remove_unused_files(raw_path)
    rename_files(raw_path)

