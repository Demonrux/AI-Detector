import torch
import clip
from pathlib import Path
from typing import Union
import numpy as np
from typing import Optional
from .base_extractor import BaseExtractor
from core.preprocessing.loaders.image_loader import ImageLoader


class CLIPExtractor(BaseExtractor):
    """Extract visual features using OpenAI's CLIP model."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = None, loader: Optional[ImageLoader] = None):
        """
        Initialize CLIP extractor.

        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14, etc.)
            device: 'cuda', 'cpu', or None for auto-detection
            loader: data loader for this type (if not passed, a new one is created)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.loader = loader or ImageLoader()
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)

        self.feature_dim = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768}.get(model_name, 512)

        self.feature_list = [f'clip_{i}' for i in range(self.feature_dim)]

    def extract(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Extract CLIP features from image.

        Returns:
            numpy array of shape (feature_dim, features)
        """

        img = self.loader.load(file_path)

        image = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image)

        return features.cpu().numpy().flatten().astype(np.float32)

    def get_feature_dim(self) -> int:
        return 512
