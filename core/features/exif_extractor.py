from pathlib import Path
from typing import Union
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
    def _get_exif_dict(file_path: Union[str, Path]) -> dict:
        """Internal method to get EXIF as dict."""
        img = Image.open(file_path)
        exif = img.getexif()

        return {
            ExifTags.TAGS.get(tag_id, str(tag_id)): value
            for tag_id, value in exif.items()
        }

    def get_exif(self, file_path: Union[str, Path]) -> dict:
        """
        Extract EXIF metadata as a readable dictionary.
        Useful for debugging and inspection.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with EXIF tags as keys and their values
        """
        result = self._get_exif_dict(file_path)
        img = Image.open(file_path)
        result['ImageWidth'] = img.width
        result['ImageHeight'] = img.height
        return result

    @staticmethod
    def _count_category_matches(exif_dict: dict, category_tags: set) -> int:
        """Count how many tags from a category are present in EXIF."""
        return sum(1 for tag in category_tags if tag in exif_dict)

    def extract(self, file_path: Union[str, Path]) -> np.ndarray:
        """Extract features from EXIF metadata."""
        named = self._get_exif_dict(file_path)
        img = Image.open(file_path)

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
        features.append(1 if any(ai in software for ai in ['midjourney', 'dall-e', 'stable', 'diffusion', 'ai']) else 0)
        features.append(1 if any(ed in software for ed in ['photoshop', 'lightroom', 'elements', 'gimp', 'editor']) else 0)

        date_str = named.get('DateTime', '')
        try:
            datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
            features.append(1)
        except:
            features.append(0)

        features.append(img.width)
        features.append(img.height)

        features.append(self._count_category_matches(named, self.CAMERA_TAGS))
        features.append(self._count_category_matches(named, self.SOFTWARE_TAGS))
        features.append(self._count_category_matches(named, self.DESCRIPTION_TAGS))
        features.append(self._count_category_matches(named, self.DATETIME_TAGS))
        features.append(self._count_category_matches(named, self.GPS_TAGS))

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> list:
        return self.feature_list

    def get_feature_dim(self) -> int:
        """Return number of features."""
        return len(self.feature_list)
