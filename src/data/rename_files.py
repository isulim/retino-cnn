"""
Rename filenames to lowercase and replace spaces with underscores.
Move the directories one up.
"""

import os
import shutil

from pathlib import Path

from dotenv import load_dotenv


def rename_files(path: Path):
    for root, dirs, files in path.walk():
        for file in files:
            filename = file.replace(" ", "_").replace("(", "").replace(")", "").replace("__", "_").lower()
            if file != filename:
                os.rename(f"{path}/{file}", f"{path}/{filename}")
                print(f"Renamed {file} to {filename}")

        for dir in dirs:
            rename_files(Path(f"{path}/{dir}"))
            dirname = dir.replace(" ", "_").replace("(", "").replace(")", "").lower()
            if dir != dirname:
                os.rename(f"{path}/{dir}", f"{path}/{dirname}")
                print(f"Renamed {dir} to {dirname}")


if __name__ == "__main__":
    load_dotenv()
    kaggle_dir: str = os.getenv("KAGGLE_FILES_DIR")
    raw_path = Path(kaggle_dir, "raw")
    rename_files(raw_path)

    # Move the directories one up
    shutil.move(Path(raw_path, "brain_tumor_data_set", "brain_tumor_data_set", "brain_tumor"), Path(raw_path, "tumor"))
    shutil.move(Path(raw_path, "brain_tumor_data_set", "brain_tumor_data_set", "healthy"), Path(raw_path, "healthy"))
    shutil.rmtree(Path(raw_path, "brain_tumor_data_set"))
