import numpy
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
import logging
from core.preprocessing.validator import Validator

logger = logging.getLogger(__name__)


class Loader:
    def __init__(self):
        self.validator = Validator()

    def load_image(self, file_path: Union[str, Path],  target_size: Tuple[int, int] = (224, 224),
                   normalize: bool = True,
                   grayscale: bool = False) \
            -> numpy.ndarray:
        """
        Загружает изображение и подготавливает его для модели.
    
        Args:
            file_path: Путь к изображению 
            target_size: (width, height) — размер, который ожидает модель
            normalize: Проводить ли нормализацию (приводить к [0, 1])
            grayscale: Преобразовывать ли в оттенки серого
    
        Returns:
            numpy array формы (1, height, width, channels)
        """
        self.validator.validate_image(file_path)

        img = Image.open(file_path)

        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        img = img.resize(target_size)

        img_array = numpy.array(img, dtype=numpy.float32)

        if grayscale:
            img_array = numpy.expand_dims(img_array, axis=-1)

        if normalize:
            img_array /= 255.0

        img_array = numpy.expand_dims(img_array, axis=0)
        return img_array

    @staticmethod
    def get_image_info(file_path: Union[str, Path]) -> dict:
        """
        Возвращает информацию об изображении без загрузки в память.
        Args:
            file_path: Путь к изображению
        Returns:
            dict: Словарь вида - "path": str(file_path),
                                "format": img.format,
                                "mode": img.mode,
                                "width": img.width,
                                "height": img.height}
        """
        Validator.validate_image(file_path)

        with Image.open(file_path) as img:
            info = {
                "path": str(file_path),
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height
            }

        logger.info(f"Информация о изображении: {info}")

        return info