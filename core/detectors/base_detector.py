from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseDetector(ABC):

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Args:
            model_path: Путь к файлу модели (если None - будет использован путь по умолчанию)
        """

        self.model_path = Path(model_path) if model_path else self._get_default_model_path()
        self.model = None
        self._load_model()
        logger.info(f"Детектор {self.__class__.__name__} инициализирован")

    @abstractmethod
    def _get_default_model_path(self) -> Path:
        """
        Возвращает путь к модели по умолчанию для данного детектора.
        Должен быть реализован в дочернем классе.
        """

        pass

    @abstractmethod
    def _load_model(self):
        """
        Загружает модель из self.model_path.
        Должен быть реализован в дочернем классе с учетом конкретного фреймворка.
        """

        pass

    @abstractmethod
    def preprocess(self, input_data: Union[str, Path, Any]) -> Any:
        """
        Подготавливает входные данные для передачи в модель.

        Args:
            input_data: Путь к файлу или сами данные

        Returns:
            Данные в формате, который ожидает модель
        """

        pass

    @abstractmethod
    def predict(self, processed_data: Any) -> Dict[str, Any]:

        """
        Запускает модель на подготовленных данных.

        Args:
            processed_data: Данные после preprocessing

        Returns:
            Словарь с результатами (минимум: probability)
        """
        pass


    def validate(self, input_data: Union[str, Path]) -> bool:
        """
        Проверяет, можно ли обработать эти данные.
        Базовая реализация - проверяет существование файла.
        Можно переопределить в дочернем классе.

        Args:
            input_data: Путь к файлу

        Returns:
            True если данные валидны, иначе False
        """

    def detect(self, input_data: Union[str, Path]) -> Dict[str, Any]:
        """
        Главный метод: принимает файл, возвращает результат.
        Содержит общую логику для всех детекторов.

        Args:
            input_data: Путь к файлу для анализа

        Returns:
            Словарь с результатами анализа
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_path})"
