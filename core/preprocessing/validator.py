from pathlib import Path
from typing import Union
from core.exceptions.errors import UnsupportedFormatError, FileTooLargeError


class ImageValidator:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    MAX_SIZE_MB = 5
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

    @staticmethod
    def validate(file_path: Union[str, Path]) -> bool:
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

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() not in ImageValidator.ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported format'{path.suffix}'."
                f"."f"Allowed: {', '.join(ImageValidator.ALLOWED_EXTENSIONS)}")

        file_size = path.stat().st_size
        if file_size > ImageValidator.MAX_SIZE_BYTES:
            raise FileTooLargeError(
                f"File size exceeded: {file_size} byte. "
                f"Max: {ImageValidator.MAX_SIZE_MB} MB"
            )

        # TODO: что файл действительно изображение (не битый)
        return True
