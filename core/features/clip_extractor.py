import torch
import clip
from PIL import Image
import numpy as np
from typing import Optional
from .base_extractor import BaseExtractor
from core.preprocessing.loaders.image_loader import ImageLoader


class CLIPExtractor(BaseExtractor):
    """Extract visual features using OpenAI's CLIP model."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP extractor.

        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14, etc.)
            device: 'cuda', 'cpu', or None for auto-detection
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

        self.feature_dim = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768}.get(model_name, 512)

        self.feature_list = [f'clip_{i}' for i in range(self.feature_dim)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def extract(self, image: Image.Image = None, **kwargs) -> np.ndarray:
        """
        Extract CLIP features from image.

        Args:
            image: PIL Image object (preloaded)
            **kwargs: Additional arguments (for compatibility)

        Returns:
            numpy array of shape (feature_dim, features)
        """

        if image is None:
            raise ValueError("CLIPExtractor requires 'image' keyword argument")

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image_tensor)

        return features.cpu().numpy().flatten().astype(np.float32)

    def get_feature_dim(self) -> int:
        return self.feature_dim
