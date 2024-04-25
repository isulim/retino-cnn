"""Export model in ONNX format."""

from .resnet import ResNetClassifier



model = ResNetClassifier.load_from_checkpoint("../experiments/models/resnet50/resnet50-model-epoch=12-val_loss=0.457-val_acc=0.804.ckpt")
