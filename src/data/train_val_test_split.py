"""Split folders with images into train and validation sets using `split-folders`."""

import os

from pathlib import Path

from dotenv import load_dotenv
from splitfolders import ratio  # type: ignore


if __name__ == "__main__":

    load_dotenv()
    seed = os.getenv("SEED")
    kaggle_dir: str = os.getenv("KAGGLE_FILES_DIR", "")

    raw_path = Path(kaggle_dir, "raw")
    processed_path = Path(kaggle_dir, "processed")

    print("Splitting folders into train, validation, test sets (70%/20%/10%).")
    ratio(raw_path, output=processed_path, seed=seed, ratio=(0.8, 0.1, 0.1))
    print(f"Done - files stored in {str(processed_path)}")
