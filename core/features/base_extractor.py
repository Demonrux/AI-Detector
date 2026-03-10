from abc import ABC, abstractmethod
import numpy


class BaseExtractor(ABC):
    """
    Abstract base class for all feature extractors.

    Feature extractors take a file (image, text, video, etc.) and extract
    numerical features that can be used by machine learning models.
    """

    @abstractmethod
    def extract(self, **kwargs) -> numpy .ndarray:
        """
        Extract features from preloaded data.

        Args:
            **kwargs: Keyword arguments containing the preloaded data.
                Common patterns:
                    - For images: extract(image=pillow_image)
                    - For text: extract(text=string_content)
                    - For audio: extract(audio=audio_array)
                    - For video: extract(frames=video_frames)

        Returns:
            NumPy array of extracted features

        Raises:
            ValueError: if required keyword arguments are missing or invalid
        """
        pass

    def __call__(self, **kwargs) -> numpy .ndarray:
        """
        Make extractor callable for convenience.
        """

        return self.extract(**kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def get_feature_dim(self) -> int:
        """
        Return the dimensionality of the feature vector.

        Returns:
            int: dimensionality of the feature
        """

        raise NotImplementedError("Subclasses should implement this method")
