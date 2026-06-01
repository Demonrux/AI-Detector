# AI Image Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Детектор изображений, сгенерированных искусственными нейросетями (Midjourney, Stable Diffusion, DALL-E и др.), основанный на анализе частотных артефактов с помощью дискретного вейвлет-преобразования (DWT) и классификатора Random Forest.

## Возможности

- **Вейвлет-анализ** – извлечение 9 признаков (дисперсия, энергия, энтропия) из субдиапазонов LH, HL, HH.
- **Многоканальность** – поддержка RGB, YCbCr и градаций серого.
- **Композитные признаки** – объединение вейвлетов с EXIF-метаданными и CLIP-эмбеддингами.
- **Обучение на нескольких генераторах** – Midjourney, ADM, BigGAN, GLIDE, VQDM, Wukong.
- **Ансамбль моделей** – комбинирование детекторов для повышения обобщающей способности.
- **Кластеризация и EDA** – встроенные инструменты для анализа структуры признакового пространства.

## Структура проекта
```text
ai-image-detector/
├── core/
│   ├── detectors/
│   │   ├── base_detector.py
│   │   ├── image_detector.py
│   │   └── ensemble_detector.py
│   ├── features/
│   │   ├── base_extractor.py
│   │   ├── wavelet_extractor.py
│   │   ├── exif_extractor.py
│   │   ├── clip_extractor.py
│   │   └── composite.py
│   ├── preprocessing/
│   │   ├── loaders/
│   │   │   ├── base_loader.py
│   │   │   └── image_loader.py
│   │   └── validators/
│   │       ├── base_validator.py
│   │       └── image_validator.py
│   ├── exceptions/
│   │   └── errors.py
│   └── main.py
├── evaluation/
│   └── clustering.py
├── models/                 # Сохранённые модели
├── logs/                   # Логи
├── utils/
│   └── logger.py
├── requirements.txt
└── README.md
```

### Основные компоненты

| Компонент | Описание |
|-----------|----------|
| `WaveletExtractor` | DWT (bior1.3, 1 уровень), извлечение var/energy/entropy для LH, HL, HH |
| `EXIFExtractor` | Метаданные изображения (камера, софт, GPS, дата) |
| `CLIPExtractor` | Векторное представление через CLIP (ViT-B/32) |
| `CompositeExtractor` | Объединение признаков из нескольких экстракторов |
| `ImageDetector` | Основной класс детекции (обучение, предсказание, загрузка/сохранение) |
| `EnsembleDetector` | Ансамбль моделей для разных генераторов |
| `ImageLoader` / `ImageValidator` | Загрузка и валидация изображений |

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/ai-image-detector.git
cd ai-image-detector
```
### 2. Создание виртуального окружения
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```
#### Основные зависимости:
- numpy, scipy – численные вычисления
- scikit-learn – Random Forest, метрики
- pywavelets – вейвлет-преобразование
- Pillow – обработка изображений
- torch, clip – CLIP-эмбеддинги
- matplotlib, seaborn – визуализация
- tqdm – прогресс-бары
- joblib – сохранение/загрузка моделей

## Использование
### Командная строка
```bash
python -m core.main --image /path/to/image.jpg --model models/midjourney.pkl
```
### Параметры:
- --image, -i – путь к изображению (обязательный)
- --model, -m – путь к модели (опционально)
- --json, -j – вывод в JSON

### Программное использование
```python
from core.detectors import ImageDetector
from core.features import WaveletExtractor, CompositeExtractor
from core.preprocessing import ImageLoader

# Создание детектора с одним вейвлет-экстрактором
extractor = CompositeExtractor([
    WaveletExtractor(wavelet='bior1.3', levels=1, color_space='gray')
])

detector = ImageDetector(extractor=extractor, loader=ImageLoader())
detector.load_model("models/midjourney.pkl")

# Предсказание
result = detector.predict("path/to/image.jpg")
print(f"Class: {result['class']}")
print(f"Confidence AI: {result['confidence_ai']:.4f}")
print(f"Confidence Real: {result['confidence_real']:.4f}")

# Обучение модели

detector = ImageDetector(extractor=extractor, loader=ImageLoader())
detector.train(
    dataset_path="/path/to/dataset",
    n_estimators=200,
    max_depth=20
)
```
### Ожидаемая структура датасета:

```text
dataset/
├── train/
│   ├── ai/        # AI-изображения
│   └── nature/    # Реальные фото
└── val/
    ├── ai/
    └── nature/
```

