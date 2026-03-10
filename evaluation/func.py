from pathlib import Path
from core.features.exif_extractor import EXIFExtractor
from PIL import Image


def extract_exif_dir():
    extractor = EXIFExtractor()

    nature_folder = Path(r"C:\Users\DMITRY\PycharmProjects\AI_detector_core\evaluation\img\nature")

    for image_path in nature_folder.glob("*"):
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"\n--- {image_path.name} ---")

            with Image.open(image_path) as img:
                exif = extractor.get_exif_dict(img)
                print(exif)