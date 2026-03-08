from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, Optional
import logging
import numpy
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    An abstract base class for all detectors (images, text, video, etc.).
    Defines a common interface for data loading, training, and prediction."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Args:
            model_path: Path to the model file (if None, the default path will be used)
        """

        self.model_path = Path(model_path) if model_path else self._get_default_model_path()
        self.model = None
        self.load_model()
        logger.info(f"Detector {self.__class__.__name__} initialized")

    @abstractmethod
    def _get_default_model_path(self) -> Path:
        """
        Returns the path to the default model for this detector.
        Must be implemented in a child class.
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Loads a model from self.model_path.
        Must be implemented in a child class, taking into account the specific framework.
        """
        pass

    @abstractmethod
    def load_dataset(self, base_dir: Path) -> tuple:
        """
        Loads data from the structure:
            base_dir/
                train/
                    ai/
                    nature/
                val/
                    ai/
                    nature/

        Returns:
            tuple: (X_train, y_train, X_val, y_val)
        """
        pass

    @abstractmethod
    def train(self, dataset_path: Path) -> Any:
        """
        Trains a model on the dataset.

        Returns:
            Trained model
        """

        pass

    @abstractmethod
    def predict(self, input_data: Union[str, Path]) -> Dict[str, Any]:
        """
        Runs the model on the input data.

        Args:
            input_data: Path to file or prepared data

        Returns:
            Dictionary with results (class, confidence)
        """

        pass

    @staticmethod
    def _extract_with_progress(extractor, files, desc) -> numpy.ndarray:
        """
        Extracts features from a list of files with a progress bar.
        """
        logger.info(f"Start extraction: {desc} ({len(files)} files)")
        features = []
        for f in tqdm(files, desc=desc):
            features.append(extractor.extract(f))

        result = numpy.array(features)
        logger.info(f"Completed: {desc}, feature shape: {result.shape}")
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_path})"
