from __future__ import annotations

from typing import Dict

import numpy as np


class DensityFeatureExtractor:
    def __init__(self, prefix: str = "density"):
        self._prefix = prefix

    @property
    def name(self) -> str:
        return self._prefix

    def extract_batch(self, images: np.ndarray) -> Dict[str, object]:
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a numpy array")
        if images.ndim != 3:
            raise ValueError("images must have shape (n_images, height, width)")

        x_density = np.sum(images, axis=1)
        y_density = np.sum(images, axis=2)

        return {
            "x_density": x_density,
            "y_density": y_density,
        }
