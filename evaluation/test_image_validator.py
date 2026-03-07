import unittest
import tempfile
from core.preprocessing.validator import ImageValidator
from core.exceptions.errors import UnsupportedFormatError, FileTooLargeError
import shutil
from pathlib import Path


class TestImageValidator(unittest.TestCase):
    """Tests for the image validator."""
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.validator = ImageValidator()

    def tearDown(self):
        shutil.rmtree(self.test_dir)  # 👈 вот так

    def test_valid_file(self):
        """Checks the validity of a correct file"""
        test_file = Path(self.test_dir) / "test.jpg"
        test_file.touch()

        try:
            self.validator.validate(test_file)
        except Exception as e:
            self.fail(f"Error: {e}")

    def test_valid_file_with_different_extension(self):
        """Checking different allowed forma"""
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            test_file = Path(self.test_dir) / f"test{ext}"
            test_file.touch()

            try:
                self.validator.validate(test_file)
            except Exception as e:
                self.fail(f"Format {ext} is not allowed:: {e}")

    def test_validate_image_unsupported_format(self):
        """Checks for an error when the format is not supported."""
        test_file = Path(self.test_dir) / "test.gif"
        test_file.touch()

        with self.assertRaises(UnsupportedFormatError):
            self.validator.validate(test_file)

    def test_unsupported_format_uppercase(self):
        """Uppercase format should also be caught"""
        test_file = Path(self.test_dir) / "test.GIF"
        test_file.touch()

        with self.assertRaises(UnsupportedFormatError):
            self.validator.validate(test_file)

    def test_file_without_extension(self):
        """File without extension -> unsupported format"""
        test_file = Path(self.test_dir) / "test."
        test_file.touch()

        with self.assertRaises(UnsupportedFormatError):
            self.validator.validate(test_file)

    def test_file_not_found(self):
        """Checks for an error if a file does not exist."""
        with self.assertRaises(FileNotFoundError):
            self.validator.validate("nonexistent.jpg")

    def test_file_too_large(self):
        """File is larger than the limit -> FileTooLargeErr"""
        test_file = Path(self.test_dir) / "large.jpg"

        with open(test_file, 'wb') as f:
            f.write(b'0' * (ImageValidator.MAX_SIZE_BYTES + 1))

        with self.assertRaises(FileTooLargeError):
            self.validator.validate(test_file)

    def test_file_exactly_max_size(self):
        """Boundary case of size"""
        test_file = Path(self.test_dir) / "exact.jpg"

        with open(test_file, 'wb') as f:
            f.write(b'0' * ImageValidator.MAX_SIZE_BYTES)

        try:
            self.validator.validate(test_file)
        except FileTooLargeError:
            self.fail()


if __name__ == "__main__":
    unittest.main()