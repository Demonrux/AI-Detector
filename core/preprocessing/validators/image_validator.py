from pathlib import Path
from typing import Union, Set
from core.exceptions.errors import FileTooLargeError
from core.preprocessing.validators.base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class ImageValidator(BaseValidator):
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    MAX_SIZE_MB = 5
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

    def get_allowed_extensions(self) -> Set[str]:
        return ImageValidator.ALLOWED_EXTENSIONS

    def get_max_size_bytes(self) -> int:
        """Return maximum allowed file size in bytes"""
        return self.MAX_SIZE_MB * 1024 * 1024

    def validate(self, file_path: Union[str, Path]) -> bool:
        """
        Checks whether the file can be processed as an image.
        Args:
            file_path: path to the image file

        Returns:
            True if the file is correct

        Raises:
            FileNotFoundError
            ValueError
            FileTooLargeError
        """
        path = Path(file_path)
        logger.info(f"Validate file {path}")

        self._check_exists(path)
        self._check_extension(path)
        self._check_size(path)

        return True
