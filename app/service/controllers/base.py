from tempfile import TemporaryFile
from typing import Annotated, override

import requests
from PIL import Image
from litestar import Controller, Request
from litestar.enums import RequestEncodingType
from litestar.params import Body
from pydantic import BaseModel

from app.models.retino_cnn import RetinoCNNModel
from app.service.schemas.requests import ImageFileRequest, ImageURLRequest


class BaseController(Controller):
    """Base controller for probability prediction by image upload or URL"""

    def _read_image_bytes(self, data: BaseModel):
        """Read image from request data."""

        raise NotImplementedError

    def _get_app_model(self, request: Request) -> RetinoCNNModel:
        """Get the model stored in app state."""

        return request.app.state.model

    def healthy_probability(self, request: Request, data: BaseModel) -> float:
        """Predict the probability of `healthy` for the image."""

        model = self._get_app_model(request)
        img_content = self._read_image_bytes(data)

        with TemporaryFile() as tmp:
            tmp.write(img_content)
            img = Image.open(tmp)
            return model.healthy_probability(img)

    def retino_probability(self, request: Request, data: BaseModel) -> float:
        """Predict the probability of `retinopathy` for the image."""

        model = self._get_app_model(request)
        img_content = self._read_image_bytes(data)

        with TemporaryFile() as tmp:
            tmp.write(img_content)
            img = Image.open(tmp)
            return model.retino_probability(img)

    async def classify(self, request: Request, data: BaseModel) -> str:
        """Classify the image as `healthy` or `retinopathy` based on the healthy threshold."""

        model = self._get_app_model(request)
        img_content = await self._read_image_bytes(data)
        threshold = data.healthy_threshold

        with TemporaryFile() as tmp:
            tmp.write(img_content)
            img = Image.open(tmp)
            return model.classify(img, threshold)


class BaseURLController(BaseController):
    """Base controller for probability prediction by image URL"""

    @override
    def _read_image_bytes(self, data: ImageURLRequest) -> bytes:
        """Read image bytes from URL."""
        image_response = requests.get(data.url, stream=True)

        if not image_response.ok:
            raise ValueError(f"Failed to fetch image from URL: {data.url}")

        return image_response.content


class BaseImageController(BaseController):
    """Base controller for probability prediction by image upload"""

    @override
    async def _read_image_bytes(self, data: Annotated[
            ImageFileRequest,
            Body(media_type=RequestEncodingType.MULTI_PART)
    ]) -> bytes:
        """Read image bytes from request data."""

        image_content = await data.image.read()
        return image_content
