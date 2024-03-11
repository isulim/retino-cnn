all: download-data rename-files split-data


split-data:
	poetry run python src/data/train_val_test_split.py

rename-files:
	poetry run python src/data/rename_files.py

download-data:
	poetry run python src/data/download_data.py
