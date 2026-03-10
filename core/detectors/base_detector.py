from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, Optional
import logging
from core.features import BaseExtractor
from core.preprocessing import BaseLoader


class BaseDetector(ABC):
    """
    An abstract base class for all detectors (images, text, video, etc.).
    Defines a common interface for data loading, training, and prediction.
    """

    def __init__(self, extractor: Optional[BaseExtractor] = None, loader: Optional[BaseLoader] = None):
        """
        Args:
            extractor: Feature extractor (if None, uses default from factory)
            loader: Data loader (if None, uses default from factory)
        """

        self.extractor = extractor or self._create_default_extractor()
        self.loader = loader or self._create_default_loader()

        self._model = None
        self._model_path = None

        logging.info(f"{self.__class__.__name__} initialized with " 
                     f"extractor={self.extractor}, "
                     f"loader={self.loader}")

    def __repr__(self) -> str:
        model_info = f"model={self._model_path}" if self._model_path else "no model loaded"
        return f"{self.__class__.__name__}({model_info})"

    @abstractmethod
    def _create_default_extractor(self) -> BaseExtractor:
        """Create default extractor for this detector type"""
        pass

    @abstractmethod
    def _create_default_loader(self) -> BaseLoader:
        """Create default loader for this detector type"""
        pass

    @abstractmethod
    def _get_default_model_path(self) -> Path:
        """
        Returns the path to the default model for this detector.
        Must be implemented in a child class.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: Union[str, Path], force: bool = False):
        """
        Loads a model from file.

        Args:
            model_path: Path to model file
            force: If True, reload even if same model is already loaded.

        Returns:
            Loaded model

        Raises:
            FileNotFoundError: if model file doesn't exist
            ValueError: if model loading fails
        """
        pass

    @abstractmethod
    def _load_dataset(self, base_dir: Path) -> tuple:
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

        Args:
            dataset_path: Path to dataset directory

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
