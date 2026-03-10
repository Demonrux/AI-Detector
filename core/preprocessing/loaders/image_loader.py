from PIL import Image
from pathlib import Path
from typing import Union, Optional
from core.preprocessing import ImageValidator
from core.preprocessing import BaseLoader
import logging

logger = logging.getLogger(__name__)


class ImageLoader(BaseLoader):
    def __init__(self, validator: Optional[ImageValidator] = None):
        validator = validator or ImageValidator()
        super().__init__(validator)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def info(self, image: Image.Image) -> dict:
        """
        Get information about an image

        Returns: dictionary with image metadata (path, format, mode, width, height)
        """

        info = {
            "format": image.format,
            "mode": image.mode,
            "width": image.width,
            "height": image.height
        }

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
