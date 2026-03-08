import argparse
import json
import sys
from pathlib import Path
from core.detectors.image_detector import ImageDetector
from utils.logger import setup_logger

project_root = str(Path(__file__).parent.parent)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

setup_logger()


def main():
    parser = argparse.ArgumentParser(description='AI Detector')
    parser.add_argument('--image', '-i', required=True, help='Path to image file')
    parser.add_argument('--model', '-m', help='Path to model file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    try:
        if args.model:
            detector = ImageDetector(model_path=args.model)
        else:
            detector = ImageDetector()

        result = detector.predict(args.image, verbose=args.verbose)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "class": "Error",
            "confidence_ai": 0.0,
            "confidence_real": 0.0
        }, ensure_ascii=False))
        sys.exit(1)


def model() -> None:
    image_path = "C:\\Users\\DMITRY\\PycharmProjects\\AI_detector_core\\evaluation\\img\\ai\\84_midjourney_46.png"

    detector = ImageDetector()
    predict = detector.predict(image_path, verbose=True)

    print(predict["class"])


if __name__ == "__main__":
    # main()
    model()
