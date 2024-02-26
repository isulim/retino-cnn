import os

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
    path = os.getenv("KAGGLE_FILES_DIR")
    os.makedirs(path, exist_ok=True)

    dataset = os.getenv("KAGGLE_DATASET")

    from kaggle import api
    api.authenticate()
    api.dataset_download_files(dataset=dataset, path=path, quiet=False, unzip=True)
