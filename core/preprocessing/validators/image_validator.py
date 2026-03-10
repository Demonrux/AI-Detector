from pathlib import Path
from typing import Union, Set
from PIL import Image
from core.exceptions.errors import CorruptedImageError
from core.preprocessing.validators.base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class ImageValidator(BaseValidator):
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    MAX_SIZE_MB = 5
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def get_allowed_extensions(self) -> Set[str]:
        return ImageValidator.ALLOWED_EXTENSIONS

    def get_max_size_bytes(self) -> int:
        """
        Return maximum allowed file size in bytes
        """

        return self.MAX_SIZE_BYTES

    @staticmethod
    def _check_integrity(path: Path) -> None:
        """
        Checks if a file is corrupted.
        Uses PIL.verify() for a quick structure check.
        """

        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                _ = img.format, img.size
        except Exception as error:
            raise CorruptedImageError(f"Image file is corrupted: {path} — {error}")

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

        self._check_exists(path)
        self._check_extension(path)
        self._check_size(path)
        self._check_integrity(path)

        return True
