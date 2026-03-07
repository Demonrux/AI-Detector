from utils.logger import setup_logger
from pathlib import Path
from core.preprocessing.loader import ImageLoader

setup_logger()


def main():
    test_image = Path("C:\\Users\DMITRY\\PycharmProjects\\AI_detector_core\\evaluation\\img\\img1.png")

    loader = ImageLoader()

    print("\n--- Информация о файле ---")
    info = loader.info(test_image)

    for key, value in info.items():
        print(f"{key}: {value}")

    img_array = loader.load(file_path=test_image, target_size=(224, 224), normalize=True, grayscale=False)

    print(img_array)

    print(f"\n--- Результат ---")
    print(f"Форма массива: {img_array.shape}")
    print(f"Тип данных: {img_array.dtype}")
    print(f"Мин. значение: {img_array.min():.3f}")
    print(f"Макс. значение: {img_array.max():.3f}")
    print(f"Среднее: {img_array.mean():.3f}")


if __name__ == "__main__":
    main()
