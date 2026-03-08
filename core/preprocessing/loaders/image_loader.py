import numpy
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import logging
from core.preprocessing.validators.image_validator import ImageValidator
from core.preprocessing.loaders.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class ImageLoader(BaseLoader):
    def __init__(self, validator: Optional[ImageValidator] = None):
        validator = validator or ImageValidator()
        super().__init__(validator)

    def info(self, file_path: Union[str, Path]) -> dict:
        """
        Get information about an image

        Returns: dictionary with image metadata (path, format, mode, width, height)
        """

        self._validate(file_path)

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

    def load(self,
             file_path: Union[str, Path],
             target_size: Tuple[int, int] = (224, 224),
             normalize: bool = True,
             grayscale: bool = False) -> numpy.ndarray:
        """
        Loads an image and prepares it for the model.
        Args:
            file_path: path to image
            target_size: (width, height) — size the model expects
            normalize: whether to normalize (convert to [0, 1])
            grayscale: convert to grayscale or not

        Returns:
            NumPy array of shape (1, height, width, channels)

        Raises:
            FileNotFoundError
            ValueError
            FileTooLargeError
        """
        self._validate(file_path)

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
