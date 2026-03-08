from PIL import Image
from pathlib import Path
from typing import Union, Optional
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

    def load(self, file_path: Union[str, Path]) -> Image.Image:
        """
        Loads an image and prepares it for the model.
        Args:
            file_path: path to image

        Returns:
            PIL.Image.Image: raw image object

        Raises:
            FileNotFoundError
            ValueError
            FileTooLargeError
        """

        self._validate(file_path)

        return Image.open(file_path)
