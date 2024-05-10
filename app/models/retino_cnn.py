from pathlib import Path

import onnx
from onnxruntime import InferenceSession

from PIL import Image
from torchvision import transforms

from app.models.outputs import Category
from app.utils.errors import InvalidOnnxModelError, ModelNotFoundError
from app.service.events import logger


class RetinoCNNModel:
    """Implementation of Diabetic Retinopathy CNN Model used to evaluate images."""

    def __init__(self, model_path: Path) -> None:
        """Initialization of diabetic retinopathy model."""

        self.model = self._load_model(model_path)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.float()),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _predict_proba(self, image: Image) -> float:
        """
        Predicts the probability of diabetic retinopathy for the image.

        Args:
            image: Image to classify.

        Returns:
            Probability of diabetic retinopathy.
        """

        transformed_image = self._transform_image(image)
        output = self.model.run(None, {"input.1": transformed_image.numpy()})
        return float(output[0])

    def healthy_probability(self, image: Image) -> float:
        """
        Predicts the probability of healthy for the image.

        Args:
            image: Image to classify.

        Returns:
            Probability of healthy.
        """

        return 1 - self._predict_proba(image)

    def retino_probability(self, image: Image) -> float:
        """
        Predicts the probability of retinopathy for the image.

        Args:
            image: Image to classify.

        Returns:
            Probability of not healthy.
        """

        return self._predict_proba(image)

    def classify(self, image: Image, threshold: float = 0.5) -> Category:
        """
        Classify the image into healthy/not healthy based on probability and threshold..

        Args:
            image: Image to classify.
            threshold: Threshold for classification.

        Returns:
            Classification of image case into healthy/not healthy.
        """

        proba = self.healthy_probability(image)
        return Category.HEALTHY if proba >= threshold else Category.RETINOPATHY

    @staticmethod
    def _load_model(path: Path) -> InferenceSession:
        """
        Loads the RetinoCNN model.

        Args:
            path: Path to the model in ONNX format.

        Returns:
            RetinoCNN model instance.
        """
        logger.info(
            event="CNN Model Load",
            detail="Loading RetinoCNN model from path",
            model_path=str(path)
        )
        try:
            with path.open(mode="rb") as file:
                name = file.name
                model = onnx.load(file)
                onnx.checker.check_model(model)
        except onnx.onnx_cpp2py_export.checker.ValidationError as exc:
            logger.error(
                event="Invalid ONNX model",
                detail=str(exc),
                exception_type=type(exc),
                path=str(path),
            )
            raise InvalidOnnxModelError from None
        except FileNotFoundError:
            logger.error(
                event="File not found",
                detail="RetinoCNN model not found in location",
                path=str(path),
            )
            raise ModelNotFoundError from None

        logger.info(
            event="Model Loaded",
            detail="RetinoCNN model loaded",
            model_name=name
        )
        return InferenceSession(str(path))

    def _transform_image(self, image: Image) -> Image:
        """
        Transforms converts image to RGB, applies torchvision transforms, unsqueezes dimension by 1.

        Args:
            image: Image to transform.

        Returns:
            Transformed image.
        """
        img = image.convert('RGB')
        img = self.transforms(img)
        return img.unsqueeze(0)
