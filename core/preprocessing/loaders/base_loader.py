# core/loaders/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Any, Optional
import logging
from core.preprocessing.validators.base_validator import BaseValidator

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base class for all loaders"""

    def __init__(self, validator: Optional[BaseValidator] = None):
        """
        Args:
            validator: validator for this file type
        """
        self.validator = validator

    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> Any:
        """
        Load data from file.

        Args:
            file_path: file path

        Returns:
            File in a format suitable for processing
        """
        pass

    @abstractmethod
    def info(self, file_path: Union[str, Path]) -> dict:
        """
        Get information about a file

        Returns:
            Dictionary with file metadata
        """
        pass

    def _validate(self, file_path: Union[str, Path]) -> None:
        """
        File validation, if a validator exists.
        Called in descendants where needed.
        """
        if self.validator:
            self.validator.validate(file_path)
