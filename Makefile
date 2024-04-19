all: input-env download-data rename-files split-data

process-data: download-data rename-files split-data

download-rename-data: download-data rename-files

split-data:
	poetry run python scripts/data/train_val_test_split.py

rename-files:
	poetry run python scripts/data/rename_files.py

download-data:
	poetry run python scripts/data/download_data.py

input-env:
	poetry run python scripts/data/input_env.py
