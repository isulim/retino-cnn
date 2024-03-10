"""Split folders with images into train and validation sets using `split-folders`."""

import os
from splitfolders import ratio
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()
    seed = os.getenv("SEED")
    source_path = os.path.join(os.getenv("KAGGLE_FILES_DIR"), "raw")
    output_path = os.path.join(os.getenv("KAGGLE_FILES_DIR"), "processed")
    os.makedirs(output_path, exist_ok=True)

    ratio(source_path, output=output_path, seed=seed, ratio=(0.7, 0.2, 0.1))
    print("Splitting folders into train, validation, test sets.")
