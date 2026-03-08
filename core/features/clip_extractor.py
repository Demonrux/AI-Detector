import torch
import clip
from PIL import Image
from pathlib import Path
from typing import Union
import numpy as np
from .base_extractor import BaseExtractor


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

    def extract(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Extract CLIP features from image.

        Returns:
            numpy array of shape (feature_dim, features)
        """
        image = self.preprocess(Image.open(file_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image)

        features_np = features.cpu().numpy().flatten()

        return features_np.astype(np.float32)

    def get_feature_names(self) -> list:
        return self.feature_list

    def get_feature_dim(self) -> int:
        return self.feature_dim
