from pathlib import Path
import numpy
from PIL import Image
import tempfile
import unittest
from core.preprocessing.loaders.image_loader import ImageLoader
from evaluation.runner import TestRunner
import logging

logger = logging.getLogger(__name__)


class TestLoader(unittest.TestCase):
    """Tests for the image loader."""

    loader: ImageLoader
    test_dir: Path
    test_image: Path
    broken_file: Path

    def setUp(self):
        self.loader = ImageLoader()
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
        """Checks that load_image returns a numpy array."""

        result = self.loader.load(self.test_image)
        self.assertIsInstance(result, numpy.ndarray)

    def test_correct_shape(self):
        """Checks the shape of the output array (1, height, width, channels)."""

        result = self.loader.load(self.test_image, target_size=(224, 224))
        self.assertEqual(result.shape, (1, 224, 224, 3))

    def test_grayscale(self):
        """Checks conversion to grayscale."""

        result = self.loader.load(self.test_image, grayscale=True)
        self.assertEqual(result.shape[-1], 1)

    def test_normalization(self):
        """Checks the normalization of values."""

        result_norm = self.loader.load(self.test_image, normalize=True)
        result_no_norm = self.loader.load(self.test_image, normalize=False)

        self.assertTrue((result_norm >= 0).all() and (result_norm <= 1).all())
        self.assertTrue((result_no_norm > 1).any())

    def test_file_not_found(self):
        """Checks for an error if a file does not exist."""

        with self.assertRaises(FileNotFoundError):
            self.loader.load("non_existent_file.jpg")

    def test_get_image_info(self):
        """Checks for receiving image information."""

        info = self.loader.info(self.test_image)

        self.assertEqual(info["format"], "JPEG")
        self.assertEqual(info["width"], 100)
        self.assertEqual(info["height"], 100)
        self.assertEqual(info["mode"], "RGB")

    def test_broken_image(self):
        """Checks behavior with a broken file."""

        with self.assertRaises(Exception):
            self.loader.load(self.broken_file)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner(verbosity=2))
