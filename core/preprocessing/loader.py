import numpy
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
import logging
from core.preprocessing.validator import ImageValidator

logger = logging.getLogger(__name__)


class ImageLoader:
    def __init__(self):
        self.validator = ImageValidator()

    def load(self, file_path: Union[str, Path], target_size: Tuple[int, int] = (224, 224), normalize: bool = True, grayscale: bool = False) -> numpy.ndarray:
        """
        Loads an image and prepares it for the model..
        Args:
            file_path: Path to image
            target_size: (width, height) — size the model expects
            normalize: Whether to normalize (convert to [0, 1])
            grayscale: Convert to grayscale or not
        Returns:
            NumPy array of shape (1, height, width, channels)
        """
        self.validator.validate(file_path)

        img = Image.open(file_path)

        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        img = img.resize(target_size)

        img_array = numpy.array(img, dtype=numpy.float32)

        if grayscale:
            img_array = numpy.expand_dims(img_array, axis=-1)

        if normalize:
            img_array /= 255.0

        img_array = numpy.expand_dims(img_array, axis=0)
        return img_array

    @staticmethod
    def info(file_path: Union[str, Path]) -> dict:
        """
        Returns information about the image.
        Args:
            file_path: Path to image
        Returns:
            dict: Dictionary of the form { "path": str(file_path),
                                "format": img.format,
                                "mode": img.mode,
                                "width": img.width,
                                "height": img.height }
        """
        ImageValidator.validate(file_path)

        with Image.open(file_path) as img:
            info = {
                "path": str(file_path),
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height
            }

        logger.info(f"Image Information: {info}")

        return info
