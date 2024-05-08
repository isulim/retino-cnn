from pydantic import BaseModel, Field

from app.models.outputs import Category


class ProbabilityResponse(BaseModel):
    class Config:
        extra = "forbid"

    probability: float = Field(
        default=...,
        title="Probability",
        description="Probability of class in the image.",
        ge=0.0,
        le=1.0,
    )


class ClassificationResponse(BaseModel):
    class Config:
        extra = "forbid"

    classification: Category = Field(
        default=...,
        title="DR classification",
        description="Classification of Diabetic Retinopathy in the image (healthy/retinopathy).",
    )
