# brain-cnn
An app using CNN to determine weather cancer is/not present in given brain CT scan.

## Environment
Dev environment is managed using `poetry`: [https://python-poetry.org](https://python-poetry.org).
```bash
poetry install
```

## Dataset

### About dataset
`Brain Tumor Dataset` from Kaggle: [https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)  
- 4602 images, including 2513 tumor and 2087 healthy scans


### Download dataset
To download the dataset, first create a personal Kaggle API token under  
`Settings -> Account -> API`  [https://www.kaggle.com/settings](https://www.kaggle.com/settings)  
and store it in `${KAGGLE_CONFIG_DIR}/kaggle.json` directory (or `~/.kaggle/kaggle.json` which is default for Kaggle API).

Next:
```bash
make download-dataset # Download dataset from Kaggle (with optional unzipping)
make rename-files # Rename files to remove whitespaces and parentheses
```
