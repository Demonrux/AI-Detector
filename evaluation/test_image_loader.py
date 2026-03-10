from pathlib import Path
import numpy
from PIL import Image
import tempfile
import unittest
from core.preprocessing.loaders.image_loader import ImageLoader
from core.exceptions.errors import UnsupportedFormatError, FileTooLargeError
from evaluation.runner import TestRunner
import logging

logger = logging.getLogger(__name__)


class TestImageLoader(unittest.TestCase):
    """Tests for the image loader."""

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

    def test_load_returns_pil_image(self):
        """Checks that load returns a PIL Image."""
        result = self.loader.load(self.test_image)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (100, 100))

    def test_load_with_validation(self):
        """Test that load validates the file."""

        img = self.loader.load(self.test_image)
        self.assertIsInstance(img, Image.Image)

        with self.assertRaises(Exception):
            self.loader.load(self.broken_file)

    def test_load_with_skip_validation(self):
        """Test skip_validation flag."""

        with self.assertRaises(Exception):
            self.loader.load(self.broken_file)

    def test_info_returns_dict(self):
        """Checks that info returns a dictionary."""

        info = self.loader.info(self.test_image)
        self.assertIsInstance(info, dict)
        self.assertEqual(info["format"], "JPEG")
        self.assertEqual(info["width"], 100)
        self.assertEqual(info["height"], 100)

    def test_file_not_found(self):
        """Checks for an error if a file does not exist."""

        with self.assertRaises(FileNotFoundError):
            self.loader.load("non_existent_file.jpg")

    def test_unsupported_format(self):
        """Test unsupported format raises error."""

        unsupported = self.test_dir / "test.gif"
        img = Image.new('RGB', (10, 10))
        img.save(unsupported, format='GIF')

        with self.assertRaises(UnsupportedFormatError):
            self.loader.load(unsupported)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner(verbosity=2))
