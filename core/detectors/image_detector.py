from typing import Union, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier
from core.features import BaseExtractor
from core.preprocessing import BaseLoader
from core.detectors import BaseDetector
import joblib
import logging
import numpy

Dataset = namedtuple('Dataset', ['X_train', 'y_train', 'X_val', 'y_val'])


class ImageDetector(BaseDetector):
    """
    Class for working with image detection
    """

    _DEFAULT_DATASET_PATH = Path("C:\\Users\\DMITRY\\Desktop\\Img_datasets\\imagenet_midjourney")

    def __init__(self, extractor: Optional[BaseExtractor] = None, loader: Optional[BaseLoader] = None):
        """
        Args:
            extractor: Feature extractor (if None, uses default)
            loader: Data loader (if None, uses default)
        """

        super().__init__(extractor, loader)
        self._model = None
        self._model_path = None

    def _create_default_extractor(self) -> BaseExtractor:
        """
        Creates a composite of EXIF and CLIP by default.
        """

        from core.features.composite import CompositeExtractor
        from core.features.exif_extractor import EXIFExtractor
        from core.features.clip_extractor import CLIPExtractor

        return CompositeExtractor([EXIFExtractor(), CLIPExtractor()])

    def _create_default_loader(self) -> BaseLoader:
        """
        Creates a loader with the default validator.
        """
        from core.preprocessing.loaders.image_loader import ImageLoader
        from core.preprocessing.validators.image_validator import ImageValidator

        return ImageLoader(validator=ImageValidator())

    def _get_default_model_path(self) -> Path:
        """Path to the default model."""

        return Path(__file__).parent.parent / "models" / "midjourney.pkl"

    def load_model(self, model_path: Union[str, Path], force: bool = False) -> None:
        """
        Loads a model from file and stores it internally.

        Args:
            model_path: Path to model file
            force: If True, reload even if same model is already loaded

        Raises:
            FileNotFoundError: if model file doesn't exist
            ValueError: if model loading fails
        """
        model_path = Path(model_path)

        if (self._model is not None) and (self._model_path == model_path) and not force:
            logging.info(f"Model {self._model_path} already loaded, skipping")
            return

        if not model_path.exists():
            error_msg = f"Model file not found: {model_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logging.info(f"Loading model from {model_path}")
            self._model = joblib.load(model_path)
            self._model_path = model_path
            logging.info(f"Model loaded successfully: {self._model_path}")

        except Exception as error:
            self._model = None
            self._model_path = None
            error_msg = f"Failed to load model from {model_path}: {error}"
            logging.error(error_msg)
            raise ValueError(error_msg) from error

    def _extract_features_from_files(self, files, desc, **kwargs) -> numpy.ndarray:
        """
        Extracts features from images

        Args:
            files: List of image file paths
            desc: Description for the progress bar
            **kwargs: Additional arguments

        Returns:
            np.ndarray: Array of image features
        """

        logging.info(f"Start extraction: {desc} ({len(files)} files)")
        features = []
        skipped = 0
        feature_dim = None

        for file in tqdm(files, desc=desc, delay=0.5):
            try:
                image = self.loader.load(file)

                feat = self.extractor.extract(image=image, **kwargs)

                if feature_dim is None:
                    feature_dim = len(feat)
                    logging.info(f"Feature dimension detected: {feature_dim}")

                if len(feat) != feature_dim:
                    raise ValueError(f"Dimension mismatch: expected {feature_dim}, got {len(feat)}")

                features.append(feat)

            except Exception as error:
                logging.warning(f"Skipping file {file}: {error}")
                skipped += 1
                if feature_dim is None:
                    continue

        if not features:
            raise RuntimeError(f"No features could be extracted for {desc}")

        result = numpy.array(features)
        logging.info(f"Completed: {desc}, shape: {result.shape}, skipped: {skipped}")
        return result

    def _load_dataset(self, base_dir: Path) -> Dataset:
        """
        Loads a dataset of images.

        Args:
            base_dir: path to the folder with train/val

        Returns:
            (X_train, y_train, X_val, y_val)
        """

        base_dir = Path(base_dir)

        logging.info("=" * 60)
        logging.info(f"LOAD IMAGE DATASET - {base_dir}")
        logging.info("=" * 60)

        train_ai = list((base_dir / "train" / "ai").glob("*.*"))
        train_nature = list((base_dir / "train" / "nature").glob("*.*"))
        val_ai = list((base_dir / "val" / "ai").glob("*.*"))
        val_nature = list((base_dir / "val" / "nature").glob("*.*"))

        logging.info(f"Train AI: {len(train_ai)} files")
        logging.info(f"Train Nature: {len(train_nature)} files")
        logging.info(f"Val AI: {len(val_ai)} files")
        logging.info(f"Val Nature: {len(val_nature)} files")

        x_train_ai = self._extract_features_from_files(train_ai, "Extract train_ai")
        x_train_nature = self._extract_features_from_files(train_nature, "Extract train_nature")
        x_val_ai = self._extract_features_from_files(val_ai, "Extract val_ai")
        x_val_nature = self._extract_features_from_files(val_nature, "Extract val_nature")

        x_train = numpy.vstack([x_train_ai, x_train_nature])
        y_train = [0] * len(train_ai) + [1] * len(train_nature)

        x_val = numpy.vstack([x_val_ai, x_val_nature])
        y_val = [0] * len(val_ai) + [1] * len(val_nature)

        logging.info("=" * 60)
        logging.info("DATA READY")
        logging.info("=" * 60)
        logging.info(f"X_train shape: {x_train.shape}")
        logging.info(f"X_val shape: {x_val.shape}")
        logging.info(f"y_train: {len(y_train)} samples ({sum(y_train)} nature, {len(y_train) - sum(y_train)} ai)")
        logging.info(f"y_val: {len(y_val)} samples ({sum(y_val)} nature, {len(y_val) - sum(y_val)} ai)")

        return Dataset(
            X_train=x_train,
            y_train=numpy.array(y_train),
            X_val=x_val,
            y_val=numpy.array(y_val)
        )

    def train(self,
              dataset_path: Optional[Union[str, Path]] = None,
              save_path: Optional[Union[str, Path]] = None,
              n_estimators: int = 200,
              max_depth: int = 20,
              random_state: int = 42,
              n_jobs: int = -1,
              **kwargs) -> None:
        """
        Trains RandomForest on the dataset.

        Args:
            dataset_path: path to the folder with train/val
            save_path: custom path for saving (if None, generates timestamped name)
            n_estimators: number of trees in the forest
            max_depth: maximum depth of the tree
            random_state: random seed for reproducibility
            n_jobs: number of jobs to run in parallel (-1 uses all processors)
            **kwargs: additional parameters for RandomForestClassifier

        Returns:
            Trained RandomForestClassifier
        """

        if dataset_path is None or dataset_path == self._DEFAULT_DATASET_PATH:
            dataset_path = self._DEFAULT_DATASET_PATH
            logging.info(f"No dataset path provided, using default: {self._DEFAULT_DATASET_PATH}")

        data = self._load_dataset(dataset_path)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )

        logging.info("Model training...")

        model.fit(data.X_train, data.y_train)
        self._model = model

        accuracy = model.score(data.X_val, data.y_val)

        logging.info(f"Accuracy: {accuracy:.4f}")

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            acc_str = f"{accuracy:.4f}".replace(".", "_")

            if self._model_path:
                save_dir = self._model_path.parent
            else:
                save_dir = Path("models")  # или self._get_default_model_path().parent

            save_path = save_dir / f"model_{timestamp}_acc{acc_str}.pkl"

        self.save_model(Path(save_path))

    def predict(self, input_data: Union[str, Path]) -> Dict[str, Any]:
        """
        Predicts for a single image.

        Args:
            input_data: Path to image

        Returns:
            Dictionary with results
        """
        if self._model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")

        image_path = Path(input_data)

        with tqdm(total=1, desc="Loading image", bar_format='{desc}: {elapsed}') as pbar:
            try:
                image = self.loader.load(image_path)
                img_info = self.loader.info(image)
                pbar.update(1)

            except Exception as error:
                logging.error(f"Image validation error: {error}")
                raise ValueError(f"Image validation error: {error}")

        with tqdm(total=1, desc="Extracting features", bar_format='{desc}: {elapsed}') as pbar:
            features = self.extractor.extract(image=image)
            features = features.reshape(1, -1)
            pbar.update(1)

        with tqdm(total=1, desc="Predicting", bar_format='{desc}: {elapsed}') as pbar:
            prediction = self._model.predict(features)[0]
            proba = self._model.predict_proba(features)[0]
            pbar.update(1)

        result = {
            'class': 'AI-generated' if prediction == 0 else 'Real photo',
            'class_id': int(prediction),
            'confidence_ai': float(proba[0]),
            'confidence_real': float(proba[1]),
            'image_info': img_info
        }

        return result

    def save_model(self, path: Optional[Union[str, Path]] = None):
        """
        Saves the trained model.

        Args:
            path: Path to save the model. If None, uses current model_path.

        Raises:
            ValueError: if no model is loaded or no save path provided
        """

        if self._model is None:
            raise ValueError("No model to save. Train or load a model first.")

        if path is None:
            if self._model_path is None:
                raise ValueError("No save path provided and no model_path set")
            save_path = self._model_path
        else:
            save_path = Path(path)

        joblib.dump(self._model, save_path)
        self._model_path = save_path
        logging.info(f"Model saved to: {save_path}")
