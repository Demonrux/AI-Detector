from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from PIL import Image
import numpy as np


class BaseExtractor(ABC):
    """
    Abstract base class for all feature extractors.

    Feature extractors take a file (image, text, video, etc.) and extract
    numerical features that can be used by machine learning models.
    """

    @abstractmethod
    def extract(self, **kwargs) -> np.ndarray:
        """
        Extract features from preloaded data.

        Args:
            **kwargs: Keyword arguments containing the preloaded data.
                Common patterns:
                    - For images: extract(image=pillow_image)
                    - For text: extract(text=string_content)
                    - For audio: extract(audio=audio_array)
                    - For video: extract(frames=video_frames)

        Returns:
            NumPy array of extracted features

        Raises:
            ValueError: if required keyword arguments are missing or invalid
        """
        pass

    def __call__(self, **kwargs) -> np.ndarray:
        """Make extractor callable for convenience."""
        return self.extract(**kwargs)

    def extract_batch(self, data_list: list, **kwargs) -> np.ndarray:
        """
        Extract features from multiple inputs.

        Args:
            data_list: List of preloaded data objects
            **kwargs: Additional keyword arguments passed to extract()

        Returns:
            2D NumPy array of shape (n_samples, n_features)
        """
        features = []
        for data in data_list:

            feat = self.extract(data=data, **kwargs)
            features.append(feat)
        return np.array(features)

    def get_feature_dim(self) -> int:
        """
        Return the dimensionality of the feature vector.
        Should be overridden by subclasses if they want to report feature dimension.
        """
        raise NotImplementedError("Subclasses should implement this method")
