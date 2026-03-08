import unittest
import tempfile
from core.preprocessing.validators.image_validator import ImageValidator
from core.exceptions.errors import UnsupportedFormatError, FileTooLargeError
from evaluation.runner import TestRunner
import shutil
from pathlib import Path
from PIL import Image


class TestImageValidator(unittest.TestCase):
    """Tests for the image validator."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.validator = ImageValidator()

        self.valid_image = self.test_dir / "valid.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(self.valid_image)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_valid_file(self):
        """Checks the validity of a correct file."""

        try:
            self.validator.validate(self.valid_image)
        except Exception as error:
            self.fail(f"Valid file raised exception: {error}")

    def test_unsupported_format(self):
        """Checks for an error when the format is not supported."""

        test_file = self.test_dir / "test.gif"
        test_file.touch()

        with self.assertRaises(UnsupportedFormatError):
            self.validator.validate(test_file)

    def test_file_not_found(self):
        """Checks for an error if a file does not exist."""

        with self.assertRaises(FileNotFoundError):
            self.validator.validate("nonexistent.jpg")

    def test_file_too_large(self):
        """File is larger than the limit -> FileTooLargeError."""

        large_file = self.test_dir / "large.jpg"

        with open(large_file, 'wb') as file:
            file.write(b'0' * (self.validator.get_max_size_bytes() + 1))

        with self.assertRaises(FileTooLargeError):
            self.validator.validate(large_file)

    def test_corrupted_image(self):
        """Test that validator catches corrupted images."""

        corrupted = self.test_dir / "corrupted.jpg"
        with open(corrupted, 'wb') as file:
            file.write(b'\xFF\xD8\xFF\x00' + b'corrupted data' * 100)

        with self.assertRaises(Exception):
            self.validator.validate(corrupted)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner(verbosity=2))