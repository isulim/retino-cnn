"""Download dataset from Kaggle using Kaggle API."""


import os
import zipfile

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def download_dataset(ds: str, output_path: str | Path):
    api.authenticate()
    api.dataset_download_files(dataset=ds, path=output_path, quiet=False, unzip=False)


def unzip_dataset(api, ds: str, output_path: str | Path):
    try:
        outfile: str = api.split_dataset_string(ds)[1]
        with zipfile.ZipFile(f"{output_path}/{outfile}.zip") as z:
            z.extractall(output_path)
        print('Unzipped all files.')

        os.remove(f"{output_path}/{outfile}.zip")
        print('Deleted zip file.')

    except zipfile.BadZipFile as e:
        raise ValueError(
            'Bad zip file, please report on '
            'www.github.com/kaggle/kaggle-api', e)
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    kaggle_dir: str = os.getenv("KAGGLE_FILES_DIR", "")
    raw_path: Path = Path(kaggle_dir, "raw")
    os.makedirs(raw_path, exist_ok=True)

    dataset_name: str = os.getenv("KAGGLE_DATASET", "")

    from kaggle import api  # type: ignore

    unzip = input("Do you want to unzip the files after download? (Y/n): ")
    download_dataset(dataset_name, raw_path)

    if not unzip or unzip.lower() == "y":
        unzip_dataset(api, dataset_name, raw_path)
    else:
        print("Files downloaded successfully")
