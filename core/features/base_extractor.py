from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import numpy as np


class BaseExtractor(ABC):
    """
    Abstract base class for all feature extractors.

    Feature extractors take a file (image, text, video, etc.) and extract
    numerical features that can be used by machine learning models.
    """

    @abstractmethod
    def extract(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Extract features from a file.

        Args:
            file_path: Path to the file

        Returns:
            NumPy array of extracted features (1D or 2D)

        Raises:
            FileNotFoundError: if file doesn't exist
            ValueError: if file can't be processed
        """
        pass

    def __call__(self, file_path: Union[str, Path]) -> np.ndarray:
        """Make extractor callable for convenience."""
        return self.extract(file_path)

    def extract_batch(self, file_paths: list) -> np.ndarray:
        """
        Extract features from multiple files.

        Args:
            file_paths: List of paths to files

        Returns:
            2D NumPy array of shape (n_files, n_features)
        """
        features = []
        for path in file_paths:
            features.append(self.extract(path))
        return np.array(features)

