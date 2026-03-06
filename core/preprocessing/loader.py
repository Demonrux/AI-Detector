import numpy
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import logging
from ..preprocessing.validator import validate_image

logger = logging.getLogger(__name__)


def load_image(
        file_path: Union[str, Path], target_size:
        Tuple[int, int] = (224, 224),
        normalize: bool = True,
        grayscale: bool = False) \
        -> numpy.ndarray:
    """
    Загружает изображение и подготавливает его для модели.

    Args:
        file_path: Путь к изображению (предполагается, что файл уже проверен валидатором)
        target_size: (width, height) — размер, который ожидает модель
        normalize: Делить ли пиксели на 255 (приводить к [0, 1])
        grayscale: Преобразовывать ли в оттенки серого

    Returns:
        numpy array формы (1, height, width, channels)

    Note:
        Функция НЕ проверяет существование файла и формат.
        Предполагается, что валидация уже пройдена до вызова.
    """
    validate_image(file_path)

    img = Image.open(file_path)
    img = img.resize(target_size)
    img_array = numpy.array(img, dtype=numpy.float32)

    if grayscale:
        img = img.convert('L')
        img_array = numpy.expand_dims(img_array, axis=-1)

    else:
        img = img.convert('RGB')

    if normalize:
        img_array /= 255.0

    img_array = numpy.expand_dims(img_array, axis=0)

    return img_array


def get_image_info(file_path: Union[str, Path]) -> dict:
    """
    Возвращает информацию об изображении без загрузки в память.
    """
    validate_image(file_path)

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