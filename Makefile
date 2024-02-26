all: download-data rename-files

rename-files:
	poetry run python src/data/rename_files.py

download-data:
	poetry run python src/data/download_data.py
