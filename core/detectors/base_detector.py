from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
from typing import Union, Dict, Any, Optional
import logging
import numpy
from tqdm import tqdm


class BaseDetector(ABC):
    """
    An abstract base class for all detectors (images, text, video, etc.).
    Defines a common interface for data loading, training, and prediction.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Args:
            model_path: Path to the model file (if None, the default path will be used)

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model can't be loaded
        """
        self._model = None
        self._model_path = None

        try:
            if model_path is None:
                model_path = self._get_default_model_path()
                logging.warning(f"No model path provided, using default: {model_path}")

            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model_path = Path(model_path)
            logging.info(f"Model path set to: {model_path}")

            self._model_path = Path(model_path)
            self.load_model(self._model_path)

            if self._model is None:
                raise ValueError(f"Failed to load model from {self._model_path}")

            logging.info(f"Detector initialized with model: {self._model_path}")

        except (FileNotFoundError, ValueError) as e:
            logging.error(f"Failed to initialize detector: {e}")
            raise

        except Exception as e:
            logging.error(f"Unexpected error during initialization: {e}")
            raise RuntimeError(f"Detector initialization failed: {e}") from e

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
        Loads a model from self.model_path.

        Args:
            model_path: Path to model file
            force: If True, reload even if same model is already loaded.
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
    def _extract_with_progress(extractor, files, desc, loader=None, **kwargs) -> numpy.ndarray:
        """
        Extracts features from a list of files with a progress bar.

        Args:
            extractor: Feature extractor
            files: List of file paths
            desc: Description for progress bar
            loader: Loader that converts file paths to data objects
            **kwargs: Additional keyword arguments to pass to extractor

        Returns:
            numpy.ndarray: Array of features
        """

        logging.info(f"Start extraction: {desc} ({len(files)} files)")
        features = []
        skipped = 0
        feature_dim = None

        for file in tqdm(files, desc=desc):
            try:
                data = loader.load(file)

                if isinstance(data, Image.Image):
                    feat = extractor.extract(image=data, **kwargs)
                elif isinstance(data, str):
                    feat = extractor.extract(text=data, **kwargs)
                # Add other data types as needed

                else:
                    feat = extractor.extract(data=data, **kwargs)

                if feature_dim is None:
                    feature_dim = len(feat)

                if len(feat) != feature_dim:
                    raise ValueError(f"Feature dimension mismatch: expected {feature_dim}, got {len(feat)}")

                features.append(feat)

            except Exception as error:
                logging.warning(f"Skipping file {file}: {error}")
                skipped += 1
                if feature_dim is None:
                    continue
                features.append(numpy.zeros(feature_dim))

        if not features:
            raise RuntimeError(f"No features could be extracted for {desc}")

        result = numpy.array(features)
        logging.info(f"Completed: {desc}, feature shape: {result.shape}, skipped: {skipped}")
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self._model_path})"
