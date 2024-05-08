from typing import Optional

from pydantic import BaseModel, Field
from pydantic.networks import HttpUrl
from litestar.datastructures import UploadFile


class ImageURLRequest(BaseModel):
    class Config:
        extra = "forbid"

    url: HttpUrl = Field(
        default=...,
        title="Image URL",
        description="URL of image to analyze",
    )
    healthy_threshold: Optional[float] = Field(
        default=0.5,
        title="Healthy threshold",
        description="Probability threshold for healthy classification",
        ge=0.0,
        le=1.0,
    )


class ImageFileRequest(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    image: UploadFile = Field(
        default=...,
        title="Image upload",
        description="Uploaded file to analyze"
    )
    healthy_threshold: Optional[float] = Field(
        default=0.5,
        title="Healthy threshold",
        description="Probability threshold for healthy classification",
        ge=0.0,
        le=1.0,
    )