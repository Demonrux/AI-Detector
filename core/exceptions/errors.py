class ValidationError(Exception):
    """Базовое исключение для ошибок валидации."""
    pass


class UnsupportedFormatError(ValidationError):
    """Неподдерживаемый формат файла."""
    pass


class FileTooLargeError(ValidationError):
    """Файл слишком большой."""
    pass


class CorruptedImageError(Exception):
    """Выбрасывается, когда файл изображения повреждён."""
    pass


class ModelLoadError(Exception):
    """Ошибка загрузки модели."""
    pass


class PredictionError(Exception):
    """Ошибка при предсказании."""
    pass