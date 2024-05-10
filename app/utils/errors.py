from pathlib import Path
from litestar.exceptions import LitestarException, ValidationException


class ModelNotFoundError(LitestarException):
    def __init__(self, path: Path):
        super().__init__(f"Model not found in location: {path}")


class InvalidOnnxModelError(LitestarException):
    def __init__(self, path: Path):
        super().__init__(f"Invalid ONNX model file: {path}")


class ImageFetchError(ValidationException):
    def __init__(self, url: str):
        super().__init__(f"Failed to fetch image from URL: {url}")
