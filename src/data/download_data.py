import os
import zipfile

from dotenv import load_dotenv
load_dotenv()


def download_dataset(dataset):
    api.authenticate()
    api.dataset_download_files(dataset=dataset, path=path, quiet=False, unzip=False)


def unzip_dataset(api, dataset, path):
    try:
        outfile = api.split_dataset_string(dataset)[1]
        with zipfile.ZipFile(f"{path}/{outfile}.zip") as z:
            z.extractall(path)
        print('Unzipped all files.')

        os.remove(f"{path}/{outfile}.zip")
        print('Deleted zip file.')

    except zipfile.BadZipFile as e:
        raise ValueError(
            'Bad zip file, please report on '
            'www.github.com/kaggle/kaggle-api', e)
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    path = os.getenv("KAGGLE_FILES_DIR")
    os.makedirs(path, exist_ok=True)

    dataset = os.getenv("KAGGLE_DATASET")
    from kaggle import api

    download_dataset(dataset)

    unzip = input("Do you want to unzip the files? (Y/n): ")
    if unzip.lower() == "y":
        unzip_dataset(api, dataset, path)
    else:
        print("Files downloaded successfully")
