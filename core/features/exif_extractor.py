import numpy as np
from datetime import datetime
from PIL import Image, ExifTags
from core.features.base_extractor import BaseExtractor


class EXIFExtractor(BaseExtractor):
    """Extract features from image EXIF metadata."""

    CAMERA_TAGS = {
        'Make', 'Model', 'ExposureTime', 'FNumber',
        'ISOSpeedRatings', 'FocalLength', 'WhiteBalance',
        'Flash', 'MeteringMode', 'ExposureProgram'
    }

    SOFTWARE_TAGS = {'Software', 'HostComputer'}

    DESCRIPTION_TAGS = {
        'ImageDescription', 'Copyright', 'Artist',
        'XPComment', 'XPSubject', 'XPTitle'
    }

    DATETIME_TAGS = {
        'DateTime', 'DateTimeOriginal', 'DateTimeDigitized',
        'ModifyDate', 'CreateDate'
    }

    GPS_TAGS = {'GPSInfo', 'GPSLatitude', 'GPSLongitude'}

    def __init__(self):
        self.feature_list = [
            'has_exif',
            'has_make',
            'has_model',
            'has_date',
            'has_gps',
            'has_software',
            'has_copyright',
            'has_description',
            'total_exif_fields',
            'software_mentions_ai',
            'software_mentions_editor',
            'valid_date',
            'image_width',
            'image_height',
            'camera_fields_count',
            'software_fields_count',
            'description_fields_count',
            'datetime_fields_count',
            'gps_fields_count'
        ]

    @staticmethod
    def get_exif_dict(image: Image.Image) -> dict:
        """Internal method to get EXIF as dict."""
        exif = image.getexif()
        return {
            ExifTags.TAGS.get(tag_id, str(tag_id)): value
            for tag_id, value in exif.items()
        }

    def extract(self, image: Image.Image = None, **kwargs) -> np.ndarray:
        """
        Extract features from EXIF metadata.

        Args:
            image: PIL Image object (preloaded)
            **kwargs: Additional arguments (for compatibility)

        Returns:
            numpy array of extracted features
        """

        if image is None:
            raise ValueError("EXIFExtractor requires 'image' keyword argument")

        named = self.get_exif_dict(image)
        features = []

        features.append(1 if named else 0)
        features.append(1 if 'Make' in named else 0)
        features.append(1 if 'Model' in named else 0)
        features.append(1 if 'DateTime' in named else 0)
        features.append(1 if any(gps in named for gps in self.GPS_TAGS) else 0)
        features.append(1 if 'Software' in named else 0)
        features.append(1 if 'Copyright' in named else 0)
        features.append(1 if 'ImageDescription' in named else 0)

        features.append(len(named))

        software = str(named.get('Software', '')).lower()
        features.append(1 if any(ai in software for ai in ['midjourney', 'dall-e', 'stable', 'diffusion', 'ai'])
                        else 0)
        features.append(1 if any(ed in software for ed in ['photoshop', 'lightroom', 'elements', 'gimp', 'editor'])
                        else 0)

        date_str = named.get('DateTime', '')
        try:
            datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
            features.append(1)
        except ValueError:
            features.append(0)

        features.append(image.width)
        features.append(image.height)

        features.append(self._count_category_matches(named, self.CAMERA_TAGS))
        features.append(self._count_category_matches(named, self.SOFTWARE_TAGS))
        features.append(self._count_category_matches(named, self.DESCRIPTION_TAGS))
        features.append(self._count_category_matches(named, self.DATETIME_TAGS))
        features.append(self._count_category_matches(named, self.GPS_TAGS))

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _count_category_matches(exif_dict: dict, category_tags: set) -> int:
        """Count how many tags from a category are present in EXIF."""
        return sum(1 for tag in category_tags if tag in exif_dict)

    def get_feature_names(self) -> list:
        return self.feature_list

    def get_feature_dim(self) -> int:
        return len(self.feature_list)
