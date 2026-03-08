from core.detectors.base_detector import BaseDetector
from typing import Union, Dict, Any, Optional
from pathlib import Path
import numpy
from core.preprocessing.loaders.image_loader import ImageLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from core.features.composite import CompositeExtractor
import joblib
import logging


class ImageDetector(BaseDetector):

    _DEFAULT_DATASET_PATH = Path("C:\\Users\\DMITRY\\Desktop\\Img_datasets\\imagenet_midjourney")

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        :param model_path: path to the model file .rxl
        """
        self.loader = ImageLoader()
        self.extractor = self._create_default_extractor()
        super().__init__(model_path)

    def _get_default_model_path(self) -> Path:
        """Path to the default model."""

        return Path(__file__).parent.parent / "models" / "midjourney.pkl"

    @staticmethod
    def _create_default_extractor() -> CompositeExtractor:
        """Creates a default extractor (EXIF + CLIP)."""

        from core.features.composite import CompositeExtractor
        from core.features.exif_extractor import EXIFExtractor
        from core.features.clip_extractor import CLIPExtractor

        return CompositeExtractor([EXIFExtractor(), CLIPExtractor()])

    def load_model(self, model_path: Union[str, Path], force: bool = False):
        """
        Loads a RandomForest model from file.

        Args:
            model_path: Path to model file. If None, uses current self.model_path
            force: If True, reload even if same model is already loaded

        Returns:
            Loaded model
        """

        if (self._model is not None) and (self.model_path == model_path) and not force:
            logging.warning(f"Model {self.model_path} already loaded, skipping")
            return self._model

        if not model_path.exists():
            logging.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            self._model = joblib.load(model_path)
            self.model_path = model_path
            logging.info(f"Model loaded: {self.model_path}")
            return self._model
        except Exception as error:
            logging.error(f"Model loading error: {error}")
            self._model = None
            raise

    def _load_dataset(self, base_dir: Path) -> tuple:
        """
        Loads a dataset of images.

        Args:
            base_dir: path to the folder with train/val

        Returns:
            (X_train, y_train, X_val, y_val)
        """

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
        x_train_ai = self._extract_with_progress(self.extractor, train_ai, "Extract train_ai")
        x_train_nature = self._extract_with_progress(self.extractor, train_nature, "Extract train_nature")
        x_val_ai = self._extract_with_progress(self.extractor, val_ai, "Extract val_ai")
        x_val_nature = self._extract_with_progress(self.extractor, val_nature, "Extract val_nature")

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

        return x_train, y_train, x_val, y_val

    def train(self,
              dataset_path: Optional[Union[str, Path]] = None,
              n_estimators: int = 200,
              max_depth: int = 20,
              random_state: int = 42,
              n_jobs: int = -1,
              **kwargs) -> RandomForestClassifier:
        """
        Trains RandomForest on the dataset.

        Args:
            dataset_path: path to the folder with train/val
            n_estimators: number of trees in the forest
            max_depth: maximum depth of the tree
            random_state: random seed for reproducibility
            n_jobs: number of jobs to run in parallel (-1 uses all processors)
            **kwargs: additional parameters for RandomForestClassifier

        Returns:
            Trained RandomForestClassifier
        """
        if dataset_path is None:
            dataset_path = self._DEFAULT_DATASET_PATH
            logging.info(f"No dataset path provided, using default: {self._DEFAULT_DATASET_PATH}")

        x_train, y_train, x_val, y_val = self._load_dataset(dataset_path)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )

        logging.info("Model training...")

        model.fit(x_train, y_train)
        self._model = model

        y_prediction = model.predict(x_val)
        accuracy = accuracy_score(y_val, y_prediction)
        logging.info(f"Accuracy:{accuracy:.4f}")

        print("\nReport by class:")
        print(classification_report(y_val, y_prediction, target_names=['AI', 'Nature']))

        cm = confusion_matrix(y_val, y_prediction)
        print("Error Matrix:")
        print(cm)

        print(f"Class model: {model.classes_}")

        return model

    def predict(self, input_data: Union[str, Path], verbose=False) -> Dict[str, Any]:
        """
       Predicts for a single image.

        Args:
            input_data: Path to image
            verbose: Verbose output

        Returns:
            Dictionary with results
        """

        image_path = Path(input_data)

        try:
            img_info = self.loader.info(image_path)
        except Exception as error:
            logging.error(f"Image validation error: {error}")
            raise ValueError(f"Image validation error: {error}")

        features = self.extractor.extract(image_path)
        features = features.reshape(1, -1)

        prediction = self._model.predict(features)[0]
        proba = self._model.predict_proba(features)[0]

        result = {
            'class': 'AI-generated' if prediction == 0 else 'Real photo',
            'class_id': int(prediction),
            'confidence_ai': float(proba[0]),
            'confidence_real': float(proba[1]),
            'image_info': img_info
        }

        if verbose:
            print(f"Image Analysis: {image_path}")

            print("\n" + "=" * 50)
            print("RESULT:")
            print("=" * 50)
            print(f"file: {image_path.name}")
            print(f"Size: {img_info.get('width')}x{img_info.get('height')}")
            print(f"Format: {img_info.get('format')}")
            print(f"AI: {proba[0] * 100:.2f}%")
            print(f"Real: {proba[1] * 100:.2f}%")
            print(f"Verdict: {result['class']}")
            print("=" * 50)

        return result

    def save_model(self, path: Optional[Union[str, Path]] = None):
        """
        Saves the trained model.
        """
        save_path = Path(path) if path else self.model_path
        joblib.dump(self._model, save_path)
        logging.info(f"Model save: {save_path}")
