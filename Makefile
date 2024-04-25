split-data:
	poetry run python scripts/data/train_val_test_split.py

clean-data:
	poetry run python scripts/data/rename_files.py

download-data:
	poetry run python scripts/data/download_data.py

configure-envs:
	poetry run python scripts/data/configure_envs.py

model-pickle:
	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1EnChn919Nj9_xN31mR2Qq6fTbiyzcF1d' -O resnet34-model.pkl

model-onnx:
	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_Vg2-ozg7I2pv2-BG1yCWopkE3D7HX4n' -O resnet34-model.onnx

all: configure-envs download-data clean-data split-data

download-and-process: download-data clean-data split-data

process-data: clean-data split-data

download-clean-data: download-data clean-data
