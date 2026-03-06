import unittest
import tempfile
from core.preprocessing.validator import Validator
from core.exceptions.errors import UnsupportedFormatError
import os
from pathlib import Path


class TestValidator(unittest.TestCase):
    """Тесты для валидатора изображений."""

    def setUp(self):
        self.validator = Validator()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.rmdir(self.test_dir)

    def test_valid_file(self):
        """Проверяет валидацию корректного файла."""
        test_file = Path(self.test_dir) / "test.jpg"
        test_file.touch()

        try:
            self.validator.validate_image(test_file)
        except Exception as e:
            self.fail(f"validate_image выбросил исключение {e}")

    def test_validate_image_unsupported_format(self):
        """Проверяет ошибку при неподдерживаемом формате."""
        test_file = Path(self.test_dir) / "test.gif"
        test_file.touch()

        with self.assertRaises(UnsupportedFormatError):
            self.validator.validate_image(test_file)

    def test_file_not_found(self):
        """Проверяет ошибку при отсутствии файла."""
        with self.assertRaises(FileNotFoundError):
            self.validator.validate_image("nonexistent.jpg")


if __name__ == "__main__":
    unittest.main()