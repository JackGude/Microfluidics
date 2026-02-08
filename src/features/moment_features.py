from __future__ import annotations

from typing import Dict

import numpy as np


class CenterOfMassExtractor:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    @property
    def name(self) -> str:
        return "center_of_mass"

    def extract_batch(self, images: np.ndarray) -> Dict[str, object]:
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a numpy array")
        if images.ndim != 3:
            raise ValueError("images must have shape (n_images, height, width)")

        n, h, w = images.shape

        y_coords = np.arange(h, dtype=float)
        x_coords = np.arange(w, dtype=float)

        mass = np.sum(images, axis=(1, 2))
        denom = np.maximum(mass, self.eps)

        x_mass = np.sum(images * x_coords[None, None, :], axis=(1, 2))
        y_mass = np.sum(images * y_coords[None, :, None], axis=(1, 2))

        x_mean = x_mass / denom
        y_mean = y_mass / denom

        x_var = np.sum(images * (x_coords[None, None, :] - x_mean[:, None, None]) ** 2, axis=(1, 2)) / denom
        y_var = np.sum(images * (y_coords[None, :, None] - y_mean[:, None, None]) ** 2, axis=(1, 2)) / denom

        x_std = np.sqrt(np.maximum(x_var, 0.0))
        y_std = np.sqrt(np.maximum(y_var, 0.0))

        features = np.stack([mass, x_mean, y_mean, x_std, y_std], axis=1)
        return {"com_stats": features}
