import numpy as np
from pathlib import Path
from typing import Union, List
from core.features.base_extractor import BaseExtractor


class CompositeExtractor(BaseExtractor):
    """
    Combines multiple extractors into one.

    Example:
        composite = CompositeExtractor([
            EXIFExtractor(),
            CLIPExtractor()
        ])
        features = composite.extract("photo.jpg")  # all features concatenated
    """

    def __init__(self, extractors: List[BaseExtractor]):
        """
        Args:
            extractors: List of feature extractors to combine
        """

        self.extractors = extractors
        self._feature_names = self._combine_feature_names()
        self._feature_dims = self._combine_feature_dims()

    def _combine_feature_names(self) -> list:
        all_names = []
        for ex in self.extractors:
            if hasattr(ex, 'get_feature_names'):
                names = ex.get_feature_names()
            else:
                names = [f'{ex.__class__.__name__}_{i}'
                         for i in range(ex.get_feature_dim())]
            all_names.extend(names)
        return all_names

    def _combine_feature_dims(self) -> int:
        """
        Total feature dimension.
        """

        return sum(ex.get_feature_dim() for ex in self.extractors)

    def extract(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Extract features from all extractors and concatenate.

        Returns:
            Combined numpy array of all features
        """

        all_features = []
        for extractor in self.extractors:
            features = extractor.extract(file_path)
            all_features.append(features)
        return np.concatenate(all_features)

    def get_feature_names(self) -> list:
        return self._feature_names

    def get_feature_dim(self) -> int:
        return self._feature_dims

    def get_exif(self, file_path: Union[str, Path]) -> dict:
        """
        Get EXIF from first extractor that supports it.
        """

        for ex in self.extractors:
            if hasattr(ex, 'get_exif'):
                return ex.get_exif(file_path)
        return {}
