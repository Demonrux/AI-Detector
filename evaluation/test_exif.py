from PIL import Image
from core.features.exif_extractor import EXIFExtractor
import json


def test():
    files = [
        r"C:\Users\DMITRY\Desktop\Img_datasets\n01687978_10119.JPEG",
        r"C:\Users\DMITRY\PycharmProjects\AI_detector_core\evaluation\img\nature\n01689811_976.JPEG"
    ]

    extractor = EXIFExtractor()

    for file_path in files:
        print(f"\n{'=' * 60}")
        print(f"File: {file_path}")
        print(f"{'=' * 60}")

        try:
            with Image.open(file_path) as img:
                exif_data = extractor.get_exif_dict(img)
                print("EXIF Data:")
                print(json.dumps(exif_data, indent=2, ensure_ascii=False))

                if not exif_data:
                    print("No EXIF data found!")

                if hasattr(img, '_getexif'):
                    raw_exif = img._getexif()
                    print(f"Raw EXIF exists: {raw_exif is not None}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "_main__":
    test()
