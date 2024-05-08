from typing import Annotated

from litestar import Request, post
from litestar.enums import RequestEncodingType
from litestar.params import Body

from app.service.controllers.base import BaseURLController, BaseImageController
from app.service.schemas.requests import ImageFileRequest, ImageURLRequest
from app.service.schemas.responses import ProbabilityResponse


class ImagePredictController(BaseImageController):
    """Controller for probability prediction by image upload"""

    path = "/predict"

    @post("/healthy")
    async def healthy(self, request: Request, data: Annotated[ImageFileRequest, Body(media_type=RequestEncodingType.MULTI_PART)]) -> ProbabilityResponse:
        """Predict the probability of `healthy` for the image."""

        return ProbabilityResponse(probability=self.healthy_probability(request, data))

    @post("/retinopathy")
    async def retino(self, request: Request, data: Annotated[ImageFileRequest, Body(media_type=RequestEncodingType.MULTI_PART)]) -> ProbabilityResponse:
        """Predict the probability of `retinopathy` for the image."""

        return ProbabilityResponse(probability=self.retino_probability(request, data))


class UrlPredictController(BaseURLController):
    """Controller for probability prediction by image URL"""

    path = "/predict"

    @post("/healthy")
    async def healthy(self, request: Request, data: ImageURLRequest) -> ProbabilityResponse:
        """Predict the probability of `healthy` for the image."""

        return ProbabilityResponse(probability=self.healthy_probability(request, data))

    @post("/retinopathy")
    async def retino(self, request: Request, data: ImageURLRequest) -> ProbabilityResponse:
        """Predict the probability of `retinopathy` for the image."""

        return ProbabilityResponse(probability=self.retino_probability(request, data))
