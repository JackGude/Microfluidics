from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _conv2d_batch(images: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if images.ndim != 3:
        raise ValueError("images must have shape (n, h, w)")
    if kernel.shape != (3, 3):
        raise ValueError("kernel must be 3x3")

    n, h, w = images.shape
    padded = np.pad(images, ((0, 0), (1, 1), (1, 1)), mode="constant")

    out = np.zeros((n, h, w), dtype=float)
    for dy in range(3):
        for dx in range(3):
            k = float(kernel[dy, dx])
            if k == 0.0:
                continue
            out += k * padded[:, dy : dy + h, dx : dx + w]

    return out


class SobelProjectionExtractor:
    @property
    def name(self) -> str:
        return "sobel_proj"

    def extract_batch(self, images: np.ndarray) -> Dict[str, object]:
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a numpy array")
        if images.ndim != 3:
            raise ValueError("images must have shape (n_images, height, width)")

        gx_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
        gy_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

        gx = _conv2d_batch(images, gx_kernel)
        gy = _conv2d_batch(images, gy_kernel)

        mag = np.sqrt(gx * gx + gy * gy)

        xproj = np.sum(mag, axis=1)
        yproj = np.sum(mag, axis=2)

        return {
            "sobel_mag_xproj": xproj,
            "sobel_mag_yproj": yproj,
        }


class LaplacianProjectionExtractor:
    def __init__(self, use_abs: bool = True):
        self.use_abs = use_abs

    @property
    def name(self) -> str:
        return "laplacian_proj"

    def extract_batch(self, images: np.ndarray) -> Dict[str, object]:
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a numpy array")
        if images.ndim != 3:
            raise ValueError("images must have shape (n_images, height, width)")

        lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
        lap = _conv2d_batch(images, lap_kernel)

        resp = np.abs(lap) if self.use_abs else lap

        xproj = np.sum(resp, axis=1)
        yproj = np.sum(resp, axis=2)

        return {
            "lap_abs_xproj": xproj,
            "lap_abs_yproj": yproj,
        }
