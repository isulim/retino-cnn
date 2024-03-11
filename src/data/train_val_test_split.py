"""Split folders with images into train and validation sets using `split-folders`."""

import os
from pathlib import Path

from splitfolders import ratio  # type: ignore
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()
    seed = os.getenv("SEED")
    kaggle_dir = os.getenv("KAGGLE_FILES_DIR", "")
    source_path = Path(kaggle_dir, "raw")
    output_path = Path(kaggle_dir, "processed")
    os.makedirs(output_path, exist_ok=True)

    ratio(source_path, output=output_path, seed=seed, ratio=(0.7, 0.2, 0.1))
    print("Splitting folders into train, validation, test sets.")
