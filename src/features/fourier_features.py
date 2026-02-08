from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class Fourier1DConfig:
    k_max: int = 5
    include_k0: bool = False


class Fourier1DDensityExtractor:
    def __init__(self, config: Fourier1DConfig = Fourier1DConfig()):
        self.config = config

    @property
    def name(self) -> str:
        return "fft1d_density"

    def extract_batch(self, images: np.ndarray) -> Dict[str, object]:
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a numpy array")
        if images.ndim != 3:
            raise ValueError("images must have shape (n_images, height, width)")

        n, h, w = images.shape
        x_density = np.sum(images, axis=1)  # (n, w)
        y_density = np.sum(images, axis=2)  # (n, h)

        fx = self._fourier_1d_features(x_density)
        fy = self._fourier_1d_features(y_density)

        return {
            "fft1d_x": fx,
            "fft1d_y": fy,
        }

    def _fourier_1d_features(self, signals: np.ndarray) -> np.ndarray:
        if signals.ndim != 2:
            raise ValueError("signals must have shape (n, N)")

        n, N = signals.shape

        if self.config.include_k0:
            ks = np.arange(0, self.config.k_max + 1)
        else:
            ks = np.arange(1, self.config.k_max + 1)

        t = np.arange(N, dtype=float)
        angles = 2.0 * np.pi * np.outer(ks, t) / float(N)  # (K, N)
        cos_basis = np.cos(angles)
        sin_basis = np.sin(angles)

        # (n, N) @ (N, K) -> (n, K)
        cos_coeff = signals @ cos_basis.T
        sin_coeff = signals @ sin_basis.T

        # Concatenate so each row is [cos(k1..kK), sin(k1..kK)]
        return np.concatenate([cos_coeff, sin_coeff], axis=1)


@dataclass(frozen=True)
class Fourier2DConfig:
    kx_max: int = 3
    ky_max: int = 3
    include_k0: bool = True


class Fourier2DLowFreqExtractor:
    def __init__(self, config: Fourier2DConfig = Fourier2DConfig()):
        self.config = config

    @property
    def name(self) -> str:
        return "fft2d_lowfreq"

    def extract_batch(self, images: np.ndarray) -> Dict[str, object]:
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a numpy array")
        if images.ndim != 3:
            raise ValueError("images must have shape (n_images, height, width)")

        n, h, w = images.shape
        if h != w:
            raise ValueError("Fourier2DLowFreqExtractor expects square images")

        kxs, kys = self._k_ranges()

        cosx, sinx = self._trig_basis(kxs, w)
        cosy, siny = self._trig_basis(kys, h)

        # First project along x for each row y.
        # Xc/Xs shape: (n, h, Kx)
        Xc = np.tensordot(images, cosx.T, axes=([2], [0]))
        Xs = np.tensordot(images, sinx.T, axes=([2], [0]))

        # Now combine along y using trig identity:
        # cos(a+b) = cos a cos b - sin a sin b
        # sin(a+b) = sin a cos b + cos a sin b
        # Real(F) = sum I * cos(a+b) = A - B
        # Imag(F) = - sum I * sin(a+b) = -(C + D)
        # where:
        # A = sum I * cosx * cosy
        # B = sum I * sinx * siny
        # C = sum I * sinx * cosy
        # D = sum I * cosx * siny
        A = np.einsum("nhk,yh->nyk", Xc, cosy)
        B = np.einsum("nhk,yh->nyk", Xs, siny)
        C = np.einsum("nhk,yh->nyk", Xs, cosy)
        D = np.einsum("nhk,yh->nyk", Xc, siny)

        real = A - B
        imag = -(C + D)

        # Flatten (Ky, Kx) grid into feature vector.
        real_flat = real.reshape(n, real.shape[1] * real.shape[2])
        imag_flat = imag.reshape(n, imag.shape[1] * imag.shape[2])

        return {
            "fft2_real": real_flat,
            "fft2_imag": imag_flat,
        }

    def _k_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.config.include_k0:
            kxs = np.arange(0, self.config.kx_max + 1)
            kys = np.arange(0, self.config.ky_max + 1)
        else:
            kxs = np.arange(1, self.config.kx_max + 1)
            kys = np.arange(1, self.config.ky_max + 1)
        return kxs.astype(float), kys.astype(float)

    def _trig_basis(self, ks: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(N, dtype=float)
        angles = 2.0 * np.pi * np.outer(ks, t) / float(N)  # (K, N)
        return np.cos(angles), np.sin(angles)
