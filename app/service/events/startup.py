"""Startup events."""

from pathlib import Path

from litestar import Litestar


from app.models.retino_cnn import RetinoCNNModel


def instantiate_model(app: Litestar):
    """Instantiate the RetinoCNNModel and attach it to the app state."""

    app.state.model = RetinoCNNModel(Path("models/resnet34-model.onnx"))
