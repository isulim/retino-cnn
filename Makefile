all: configure-envs download-data rename-files split-data

download-and-process: download-data rename-files split-data

process-data: rename-files split-data

download-rename-data: download-data rename-files

split-data:
	poetry run python scripts/data/train_val_test_split.py

clean-data:
	poetry run python scripts/data/rename_files.py

download-data:
	poetry run python scripts/data/download_data.py

configure-envs:
	poetry run python scripts/data/configure_envs.py
