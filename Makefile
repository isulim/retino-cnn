all: configure-envs download-data clean-data split-data

download-and-process: download-data clean-data split-data

process-data: clean-data split-data

download-clean-data: download-data clean-data

split-data:
	poetry run python scripts/data/train_val_test_split.py

clean-data:
	poetry run python scripts/data/rename_files.py

download-data:
	poetry run python scripts/data/download_data.py

configure-envs:
	poetry run python scripts/data/configure_envs.py
