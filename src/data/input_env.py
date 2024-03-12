"""Input environment variables."""


def ask_kaggle_config():
    """Ask for Kaggle config directory."""
    kaggle_dir = input("Enter Kaggle config directory [.kaggle]: ")

    if not kaggle_dir:
        kaggle_dir = ".kaggle"

    return kaggle_dir


def ask_files_dir():
    """Ask for Kaggle files directory."""
    files_dir = input("Enter directory for data files [data]: ")

    if not files_dir:
        files_dir = "data"

    return files_dir


def ask_kaggle_dataset():
    """Ask for Kaggle dataset name."""
    dataset = input("Enter Kaggle dataset name: ")
    if not dataset:
        raise ValueError("Please enter a valid dataset name.")
    return dataset


def confirm(files_dir: str, dataset: str, kaggle_dir: str):
    """Confirm input."""
    confirm = input(f"""Is provided information correct?
    DATA_FILES_DIR: {files_dir}
    KAGGLE_CONFIG_DIR: {kaggle_dir}
    KAGGLE_DATASET: {dataset}
    (Y/n): """)
    if not confirm or confirm.lower() == "y":
        with open(".env", "a") as f:
            f.write(f"KAGGLE_FILES_DIR={files_dir}\n")
            f.write(f"KAGGLE_CONFIG_DIR={kaggle_dir}\n")
            f.write(f"KAGGLE_DATASET={dataset}\n")
    else:
        print("Please re-enter the information.")


if __name__ == '__main__':
    kaggle_dir = ask_kaggle_config()
    files_dir = ask_files_dir()
    dataset = ask_kaggle_dataset()
    confirm(files_dir, dataset, kaggle_dir)
