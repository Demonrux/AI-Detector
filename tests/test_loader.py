from pathlib import Path
import pytest
from PIL import Image
import tempfile
from core.preprocessing.validator import validate_image
from core.preprocessing.loader import load_image, get_image_info
from core.exceptions.errors import UnsupportedFormatError, FileTooLargeError
import logging
logger = logging.getLogger(__name__)


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def create_dummy_image(size_bytes: int, suffix: str = ".jpg", mode: str = "RGB") -> Path:
    """Создаёт временный файл-заглушку заданного размера"""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(b'0' * size_bytes)
        return Path(f.name)


def create_real_dummy_image(suffix: str = ".png", size=(100, 100), mode="RGB") -> Path:
    """Создаёт реальное изображение через PIL"""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        img = Image.new(mode, size, color='red')
        img.save(f, format=suffix.replace('.', '').upper())
        return Path(f.name)


# ========== ТЕСТЫ ВАЛИДАЦИИ ==========

def test_validate_image_not_found():
    logger.info("ТЕСТ: проверка несуществующего файла")
    with pytest.raises(FileNotFoundError):
        validate_image("C:/not.png")


def test_validate_image_wrong_format():
    """Неподдерживаемый формат"""
    bad_file = create_dummy_image(100, suffix=".txt")
    try:
        with pytest.raises(UnsupportedFormatError):
            validate_image(bad_file)
    finally:
        bad_file.unlink()


@pytest.mark.parametrize("size_mb,should_pass", [(4, True), (5, True),  (6, False), (10, False)])
def test_validate_image_size(monkeypatch, size_mb, should_pass):
    """Проверка ограничения по размеру (5 МБ)"""
    import core.preprocessing.validator as validator
    monkeypatch.setattr(validator, 'MAX_SIZE_MB', 5)
    monkeypatch.setattr(validator, 'MAX_SIZE_BYTES', 5 * 1024 * 1024)

    test_file = create_dummy_image(size_mb * 1024 * 1024)

    try:
        if should_pass:
            result = validate_image(test_file)
            assert result is True
        else:
            with pytest.raises(FileTooLargeError):
                validate_image(test_file)
    finally:
        test_file.unlink()


def test_validate_image_allowed_formats(monkeypatch):
    """Проверка всех разрешённых форматов"""
    allowed = ['.jpg', '.jpeg', '.png', '.bmp']

    for ext in allowed:
        test_file = create_dummy_image(100, suffix=ext)
        try:
            result = validate_image(test_file)
            assert result is True
        finally:
            test_file.unlink()


# ========== ТЕСТЫ ЗАГРУЗКИ ==========
def test_get_image_info_returns_dict():
    """Проверка получения информации об изображении"""
    test_file = create_real_dummy_image()
    try:
        info = get_image_info(test_file)

        assert isinstance(info, dict)
        assert "path" in info
        assert "format" in info
        assert "mode" in info
        assert "width" in info
        assert "height" in info
        assert info["width"] > 0
        assert info["height"] > 0
    finally:
        test_file.unlink()


@pytest.mark.parametrize("target_size", [(224, 224), (128, 128), (256, 256)])
def test_load_image_shape_rgb(target_size):
    """Проверка формы массива для RGB"""
    test_file = create_real_dummy_image()
    try:
        img = load_image(file_path=test_file, target_size=target_size, normalize=True, grayscale=False)

        assert img.shape == (1, target_size[1], target_size[0], 3)
        assert img.dtype == "float32"
        assert img.min() >= 0.0
        assert img.max() <= 1.0
    finally:
        test_file.unlink()


@pytest.mark.parametrize("target_size", [(224, 224), (128, 128), (100, 100)])
def test_load_image_shape_grayscale(target_size):
    """Проверка формы массива для grayscale"""
    test_file = create_real_dummy_image(mode="L")  # grayscale
    try:
        img = load_image(file_path=test_file, target_size=target_size, normalize=False, grayscale=True)

        assert img.shape == (1, target_size[1], target_size[0], 1)
        assert img.dtype == "float32"
        assert img.min() >= 0
        assert img.max() <= 255
    finally:
        test_file.unlink()


def test_load_image_without_normalize():
    """Проверка без нормализации (значения 0-255)"""
    test_file = create_real_dummy_image()
    try:
        img = load_image(file_path=test_file, normalize=False, grayscale=False)

        assert img.max() > 1.0
        assert img.dtype == "float32"
    finally:
        test_file.unlink()