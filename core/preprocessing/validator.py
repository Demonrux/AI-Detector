from pathlib import Path
from typing import Union
from core.exceptions.errors import UnsupportedFormatError, FileTooLargeError


class Validator:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    MAX_SIZE_MB = 5
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

    @staticmethod
    def validate_image(file_path: Union[str, Path]) -> bool:
        """
        Проверяет, можно ли работать с файлом как с изображением.

        Args:
            file_path: путь к файлу изображения

        Returns:
            True если файл подходит

        Raises:
            FileNotFoundError: файл не существует
            ValueError: неподдерживаемый формат
            FileTooLargeError: превышен макс размер файла
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        if path.suffix.lower() not in Validator.ALLOWED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Неподдерживаемый формат '{path.suffix}'. "
                f"Разрешены: {', '.join(Validator.ALLOWED_EXTENSIONS)}"
            )

        file_size = path.stat().st_size
        if file_size > Validator.MAX_SIZE_BYTES:
            raise FileTooLargeError(
                f"Файл слишком большой: {file_size} байт. "
                f"Максимум: {Validator.MAX_SIZE_MB} МБ"
            )

        # TODO: можно добавить проверку, что файл действительно изображение (не битый)
        return True
