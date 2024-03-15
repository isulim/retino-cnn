# retino-cnn
An app using CNN to determine severity od diabetic retinopathy.

## Environment
Dev environment is managed using `poetry`: [https://python-poetry.org](https://python-poetry.org).
```bash
poetry install --with dev
```

## Dataset

### About dataset
`Resized 2015 & 2019 Diabetic Retinopathy Detection` from Kaggle: [https://www.kaggle.com/datasets/c7934597/resized-2015-2019-diabetic-retinopathy-detection](https://www.kaggle.com/datasets/c7934597/resized-2015-2019-diabetic-retinopathy-detection)  
Dataset contains 92 404 images of human retina including:
- 67 156 healthy
- 6 583 mild cases
- 14 160 moderate cases
- 2 288 severe cases
- 2 217 proliferative cases  

In total about 72.5% of dataset are healthy retinas and 27.5% are not-healthy with varying severity.   
I simplify this to binary classification problem: healthy vs not-healthy.

### Download dataset
To download the dataset, first create a personal Kaggle API token under  
`Settings -> Account -> API`  [https://www.kaggle.com/settings](https://www.kaggle.com/settings)  
and store it in `${KAGGLE_CONFIG_DIR}/kaggle.json` directory (or `~/.kaggle/kaggle.json` which is default for Kaggle API).

Next run make command:
```bash
make all
```
to input environment variables, download the dataset and split it into train, validation and test sets.   
Alternatively, you can run each step separately:
```bash
make input-env # Create input environment file
make download-dataset # Download dataset from Kaggle (with optional unzipping)
make rename-files # Rename files to remove whitespaces and parentheses
make split-data # Split data into train, validation and test sets
```
or if you have `.env` file already created:
```bash
make download-rename-data # Download, rename, split
```
