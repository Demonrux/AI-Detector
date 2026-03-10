import argparse
import json
import sys
from pathlib import Path
from PIL import Image
from PIL.TiffImagePlugin import IFDRational
import numpy
from datetime import datetime

project_root = str(Path(__file__).parent.parent)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.detectors.image_detector import ImageDetector
from core.features.exif_extractor import EXIFExtractor
from core.features.composite import CompositeExtractor
from  core.features.clip_extractor import CLIPExtractor
from utils.logger import setup_logger, disable_logs


def default_serializer(obj):
    """
    Кастомный сериализатор для JSON, обрабатывающий специальные типы.
    Используется в json.dumps(default=default_serializer)

    Args:
        obj: объект для сериализации

    Returns:
        JSON-совместимый тип (str, int, float, list, dict)

    Raises:
        TypeError: если тип не поддерживается
    """

    if isinstance(obj, IFDRational):
        return float(obj)

    elif isinstance(obj, bytes):
        try:
            return obj.decode('utf-8', errors='ignore').strip('\x00')
        except:
            return str(obj)

    elif isinstance(obj, datetime):
        return obj.isoformat()

    elif isinstance(obj, numpy.integer):
        return int(obj)
    elif isinstance(obj, numpy.floating):
        return float(obj)
    elif isinstance(obj, numpy.ndarray):
        return obj.tolist()
    elif isinstance(obj, numpy.bool_):
        return bool(obj)

    elif isinstance(obj, Path):
        return str(obj)

    elif isinstance(obj, (set, tuple)):
        return list(obj)

    elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return obj.to_dict()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    disable_logs()
    parser = argparse.ArgumentParser(description='AI Detector')
    parser.add_argument('--image', '-i', required=True, help='Path to image file')
    parser.add_argument('--model', '-m', help='Path to model file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--json', '-j', action='store_true', help='Output only JSON (for Qt integration)')

    args = parser.parse_args()

    try:
        detector = ImageDetector()
        detector.load_model(args.model)
        result = detector.predict(args.image)

        exif_extractor = EXIFExtractor()

        with Image.open(args.image) as img:
            exif_data = exif_extractor.get_exif_dict(img)

        enhanced_image_info = {
            'path': str(Path(args.image).absolute()),
            'format': result['image_info'].get('format', ''),
            'mode': result['image_info'].get('mode', ''),
            'width': result['image_info'].get('width', 0),
            'height': result['image_info'].get('height', 0),
            'exif': exif_data
        }

        if args.json:
            clean_result = {
                'class': result['class'],
                'class_id': result.get('class_id', 0),
                'confidence_ai': result['confidence_ai'],
                'confidence_real': result['confidence_real'],
                'feature_dimension': detector.extractor.get_feature_dim(),
                'image_info': enhanced_image_info
            }

            print("---RESULT_JSON---")
            print(json.dumps(clean_result, default=default_serializer, ensure_ascii=False))
            print("---END_JSON---")

            sys.stdout.flush()

    except Exception as e:
        error_result = {
            "error": str(e),
            "class": "Error",
            "confidence_ai": 0.0,
            "confidence_real": 0.0
        }

        print("---RESULT_JSON---")
        print(json.dumps(error_result, default=default_serializer, ensure_ascii=False))
        print("---END_JSON---")

        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    setup_logger()
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()

    # detector = ImageDetector()
    # detector.load_model(r"C:\Users\DMITRY\PycharmProjects\AI_detector_core\core\models\midjourney.pkl")
    # print(detector.predict(r"C:\Users\DMITRY\PycharmProjects\AI_detector_core\evaluation\img\ai\midjourney_001.png"))

