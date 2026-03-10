from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Set
from core.exceptions.errors import UnsupportedFormatError, FileTooLargeError


class BaseValidator(ABC):
    """
    Abstract base class for all validators
    """

    @abstractmethod
    def validate(self, file_path: Union[str, Path]) -> bool:
        """
        Check the file for correctness.

        Args:
            file_path: file path

        Returns:
            True if the file is correct

        Raises:
            FileNotFoundError: file does not exist
            UnsupportedFormatError: unsupported format
            A proper validator should throw specific exceptions.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def get_allowed_extensions(self) -> Set[str]:
        """
        Return the set of allowed extensions
        """
        pass

    @abstractmethod
    def get_max_size_bytes(self) -> int:
        """
        Return maximum allowed file size in bytes
        """
        pass

    @staticmethod
    def _check_exists(file_path: Path) -> None:
        """Checking file existence"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def _check_size(self, file_path: Path) -> None:
        """
        Checking the acceptable file size
        """

        file_size = file_path.stat().st_size
        max_bytes = self.get_max_size_bytes()

        if file_size > max_bytes:
            raise FileTooLargeError(f"File size exceeded: {file_size} byte. " f"Max: {max_bytes // (1024*1024)} MB")

    def _check_extension(self, file_path: Path) -> None:
        """
        Extension check
        """

        ext = file_path.suffix.lower()
        if ext not in self.get_allowed_extensions():
            raise UnsupportedFormatError(f"Format '{ext}' not supported. Allowed: {', '.join(self.get_allowed_extensions())}")
