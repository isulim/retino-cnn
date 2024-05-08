from typing import Annotated

from litestar import Request, post
from litestar.enums import RequestEncodingType
from litestar.params import Body

from app.service.controllers.base import BaseImageController, BaseURLController
from app.service.schemas.requests import ImageFileRequest, ImageURLRequest
from app.service.schemas.responses import ClassificationResponse


class ImageClassifyController(BaseImageController):
    """Controller for probability prediction by image upload"""

    path = "/classify"

    @post("")
    async def classify_image(self, request: Request, data: Annotated[ImageFileRequest, Body(media_type=RequestEncodingType.MULTI_PART)]) -> ClassificationResponse:
        """Classify the image as `healthy` or `retinopathy` based on the healthy threshold."""

        classification = await self.classify(request, data)

        return ClassificationResponse(classification=classification)


class UrlClassifyController(BaseURLController):
    """Controller for probability prediction by image URL"""

    path = "/classify"

    @post("")
    async def classify_url(self, request: Request, data: ImageURLRequest) -> ClassificationResponse:
        """Classify the image as `healthy` or `retinopathy` based on the healthy threshold."""

        classification = await self.classify(request, data)

        return ClassificationResponse(classification=classification)
