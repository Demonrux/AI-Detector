from pathlib import Path
import os
import numpy
from PIL import Image
import tempfile
import unittest
from core.preprocessing.loader import Loader
import logging

logger = logging.getLogger(__name__)


class TestLoader(unittest.TestCase):
    """Тесты для загрузчика изображений."""

    loader: Loader
    test_dir: Path
    test_image: Path
    broken_file: Path

    def setUp(self):
        self.loader = Loader()
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_image = self.test_dir / "test.jpg"
        self.broken_file = self.test_dir / "broken.jpg"

        img = Image.new('RGB', (100, 100), color='red')
        img.save(self.test_image)

        with open(self.broken_file, 'w') as f:
            f.write("this is not an image")

    def tearDown(self):
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()

    def test_returns_numpy_array(self):
        """Проверяет, что load_image возвращает numpy array."""
        result = self.loader.load_image(self.test_image)
        self.assertIsInstance(result, numpy.ndarray)

    def test_correct_shape(self):
        """Проверяет форму выходного массива (1, height, width, channels)."""
        result = self.loader.load_image(self.test_image, target_size=(224, 224))
        self.assertEqual(result.shape, (1, 224, 224, 3))

    def test_grayscale(self):
        """Проверяет конвертацию в grayscale."""
        result = self.loader.load_image(self.test_image, grayscale=True)
        self.assertEqual(result.shape[-1], 1)

    def test_normalization(self):
        """Проверяет нормализацию значений."""
        result_norm = self.loader.load_image(self.test_image, normalize=True)
        result_no_norm = self.loader.load_image(self.test_image, normalize=False)

        self.assertTrue((result_norm >= 0).all() and (result_norm <= 1).all())
        self.assertTrue((result_no_norm > 1).any())

    def test_file_not_found(self):
        """Проверяет ошибку при отсутствии файла."""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_image("non_existent_file.jpg")

    def test_get_image_info(self):
        """Проверяет получение информации об изображении."""
        info = self.loader.get_image_info(self.test_image)

        self.assertEqual(info["format"], "JPEG")
        self.assertEqual(info["width"], 100)
        self.assertEqual(info["height"], 100)
        self.assertEqual(info["mode"], "RGB")

    def test_broken_image(self):
        """Проверяет поведение с битым файлом."""
        with self.assertRaises(Exception):
            self.loader.load_image(self.broken_file)


if __name__ == "__main__":
    unittest.main()