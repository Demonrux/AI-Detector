
import numpy
import logging
import joblib
import argparse
import sys
from utils.logger import setup_logger
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
from core.preprocessing.loaders.image_loader import ImageLoader
from core.features.composite import CompositeExtractor
from core.features.exif_extractor import EXIFExtractor
from core.features.clip_extractor import CLIPExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


setup_logger()


def extract_with_progress(extractor, files, desc):
    logging.info(f"Start extraction: {desc} ({len(files)} files)")
    features = []
    for f in tqdm(files, desc=desc):
        features.append(extractor.extract(f))

    logging.info(f"Completed: {desc}, feature shape: {numpy.array(features).shape}")
    return numpy.array(features)


def load_dataset(base_dir: Path, extractor):
    """
    Загружает данные из структуры:
    base_dir/
        train/
            ai/
            nature/
        val/
            ai/
            nature/
    """
    logging.info("=" * 60)
    logging.info("LOAD DATASET")
    logging.info("=" * 60)

    train_ai = list((base_dir / "train" / "ai").glob("*.*"))
    train_nature = list((base_dir / "train" / "nature").glob("*.*"))
    val_ai = list((base_dir / "val" / "ai").glob("*.*"))
    val_nature = list((base_dir / "val" / "nature").glob("*.*"))

    logging.info(f"Train AI: {len(train_ai)} files")
    logging.info(f"Train Nature: {len(train_nature)} files")
    logging.info(f"Val AI: {len(val_ai)} files")
    logging.info(f"Val Nature: {len(val_nature)} files")
    logging.info(f"All files: {len(train_ai) + len(train_nature) + len(val_ai) + len(val_nature)}")

    print(f"Train AI: {len(train_ai)}, Train Nature: {len(train_nature)}")
    print(f"Val AI: {len(val_ai)}, Val Nature: {len(val_nature)}")

    X_train_ai = extract_with_progress(extractor, train_ai, "Extract train_ai")
    X_train_nature = extract_with_progress(extractor, train_nature, "Extract train_nature")
    X_val_ai = extract_with_progress(extractor, val_ai, "Extract val_ai")
    X_val_nature = extract_with_progress(extractor, val_nature, "Extract val_nature")

    X_train = numpy.vstack([X_train_ai, X_train_nature])
    y_train = [0] * len(train_ai) + [1] * len(train_nature)

    X_val = numpy.vstack([X_val_ai, X_val_nature])
    y_val = [0] * len(val_ai) + [1] * len(val_nature)

    logging.info("=" * 60)
    logging.info("DATA READY")
    logging.info("=" * 60)
    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"X_val shape: {X_val.shape}")
    logging.info(f"y_train: {len(y_train)} marks ({sum(y_train)} nature, {len(y_train) - sum(y_train)} ai)")
    logging.info(f"y_val: {len(y_val)} marks ({sum(y_val)} nature, {len(y_val) - sum(y_val)} ai)")

    return X_train, y_train, X_val, y_val


def predict_image(model_path: str, image_path: str, extractor=None, verbose=False):
    """
    Predict if an image is AI-generated or real.

    Args:
        model_path: path to saved model (.pkl file)
        image_path: path to image file
        extractor: optional pre-configured extractor (creates new if None)
        verbose: flag for verbose output

    Returns:
        dict with prediction results
    """

    model = joblib.load(model_path)
    print(f"Model loaded: {model_path}")

    loader = ImageLoader()

    try:
        img_info = loader.info(image_path)

        if extractor is None:
            from core.features.composite import CompositeExtractor
            from core.features.exif_extractor import EXIFExtractor
            from core.features.clip_extractor import CLIPExtractor
            extractor = CompositeExtractor([EXIFExtractor(), CLIPExtractor()])

        if verbose:
            print(f"Анализ изображения: {image_path}")

        features = extractor.extract(image_path)
        features = features.reshape(1, -1)

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        class_names = ['AI-generated', 'Real photo']
        result = {
            'class': class_names[pred],
            'class_id': int(pred),
            'confidence_ai': float(proba[0]),
            'confidence_real': float(proba[1]),
            'image_path': str(image_path),
            'image_info': img_info
        }

        if verbose:
            print("\n" + "=" * 50)
            print("РЕЗУЛЬТАТ:")
            print("=" * 50)
            print(f"Файл: {Path(image_path).name}")
            print(f"Размер: {img_info.get('width')}x{img_info.get('height')}")
            print(f"Формат: {img_info.get('format')}")
            print(f"AI: {proba[0] * 100:.2f}%")
            print(f"Real: {proba[1] * 100:.2f}%")
            print(f"Вердикт: {result['class']}")
            print("=" * 50)

        return result

    except Exception as e:
        raise e


def train_model(dataset_path: Path, extractor: CompositeExtractor) -> RandomForestClassifier:
    X_train, y_train, X_val, y_val = load_dataset(dataset_path, extractor)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nAccuracy in validation: {accuracy:.4f}")

    print("\nReport by class:")
    print(classification_report(y_val, y_pred, target_names=['AI', 'Nature']))

    cm = confusion_matrix(y_val, y_pred)
    print("Error Matrix:")
    print(cm)

    print(f"Классы модели: {model.classes_}")
    print(f"0 - это {'AI' if model.classes_[0] == 0 else 'Nature'}")

    return model


def main():
    #extractor = CompositeExtractor([EXIFExtractor(), CLIPExtractor()])
    #dataset_path = Path("C:\\Users\\DMITRY\\Desktop\\Img_datasets\\imagenet_midjourney")

    #train_model(dataset_path, extractor)

    parser = argparse.ArgumentParser(description='AI Image Detector')
    parser.add_argument('--image', '-i', type=str, help='Path to image file')
    parser.add_argument('--model', '-m', type=str,
                        default='core/models/model_20260308_001936_acc0.9990.pkl',
                        help='Path to model file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show features')

    args = parser.parse_args()

    if not args.image:
        print("Укажите путь к изображению: python main.py --image photo.jpg")
        return

    extractor = CompositeExtractor([EXIFExtractor(), CLIPExtractor()])

    result = predict_image(args.model, args.image, extractor, verbose=args.verbose)

    if not args.verbose:
        print(f"{result['class_id']},{result['confidence_ai']:.4f},{result['confidence_real']:.4f}")


if __name__ == "__main__":
    main()
