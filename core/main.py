import sys
from utils.logger import setup_logger
from pathlib import Path
from core.detectors.image_detector import ImageDetector

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
setup_logger()


def main():
    detector = ImageDetector()
    detector.train()

    result = detector.predict(
        "C:\\Users\\DMITRY\\PycharmProjects\\AI_detector_core\\evaluation\\img\\ai\\84_midjourney_46.png",
        verbose=True)

    print(result)


if __name__ == "__main__":
    main()
