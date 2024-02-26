import os

from dotenv import load_dotenv


def rename_files(path: str):
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = file.replace(" ", "_").replace("(", "").replace(")", "").replace("__", "_").lower()
            if file != filename:
                os.rename(f"{path}/{file}", f"{path}/{filename}")
                print(f"Renamed {file} to {filename}")

        for dir in dirs:
            rename_files(f"{path}/{dir}")
            dirname = dir.replace(" ", "_").replace("(", "").replace(")", "").lower()
            if dir != dirname:
                os.rename(f"{path}/{dir}", f"{path}/{dirname}")
                print(f"Renamed {dir} to {dirname}")


if __name__ == "__main__":
    load_dotenv()
    path = os.getenv("KAGGLE_FILES_DIR")
    rename_files(path)
