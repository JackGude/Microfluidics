from __future__ import annotations

from typing import Dict

import numpy as np


class RadialProfileExtractor:
    def __init__(self, n_bins: int = 12, eps: float = 1e-12):
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        self.n_bins = n_bins
        self.eps = eps

    @property
    def name(self) -> str:
        return "radial_profile"

    def extract_batch(self, images: np.ndarray) -> Dict[str, object]:
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a numpy array")
        if images.ndim != 3:
            raise ValueError("images must have shape (n_images, height, width)")

        n, h, w = images.shape
        if h != w:
            raise ValueError("RadialProfileExtractor expects square images")

        yy, xx = np.indices((h, w))
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        r_max = float(np.max(rr))
        r_norm = rr / max(r_max, self.eps)
        bin_idx = np.minimum((r_norm * self.n_bins).astype(int), self.n_bins - 1)

        profile = np.zeros((n, self.n_bins), dtype=float)
        counts = np.zeros(self.n_bins, dtype=float)

        for b in range(self.n_bins):
            mask = bin_idx == b
            counts[b] = float(np.sum(mask))
            if counts[b] <= 0:
                continue
            profile[:, b] = np.sum(images[:, mask], axis=1)

        return {"radial_sum": profile, "radial_count": counts.reshape(1, -1).repeat(n, axis=0)}
