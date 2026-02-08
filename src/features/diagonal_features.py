from __future__ import annotations

from typing import Dict, List

import numpy as np


class DiagonalProjectionExtractor:
    def __init__(self, prefix: str = "diag"):
        self._prefix = prefix

    @property
    def name(self) -> str:
        return self._prefix

    def extract_batch(self, images: np.ndarray) -> Dict[str, object]:
        if not isinstance(images, np.ndarray):
            raise TypeError("images must be a numpy array")
        if images.ndim != 3:
            raise ValueError("images must have shape (n_images, height, width)")

        n, h, w = images.shape
        if h != w:
            raise ValueError("DiagonalProjectionExtractor expects square images")

        out_len = 2 * h - 1
        tlbr = np.zeros((n, out_len), dtype=float)
        trbl = np.zeros((n, out_len), dtype=float)

        flat = images.reshape(n, h * w)

        tlbr_groups: List[List[int]] = [[] for _ in range(out_len)]
        trbl_groups: List[List[int]] = [[] for _ in range(out_len)]
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                tlbr_groups[(i - j) + (h - 1)].append(idx)
                trbl_groups[(i + j)].append(idx)

        for k in range(out_len):
            tlbr[:, k] = np.sum(flat[:, tlbr_groups[k]], axis=1)
            trbl[:, k] = np.sum(flat[:, trbl_groups[k]], axis=1)

        return {
            "diag_tlbr": tlbr,
            "diag_trbl": trbl,
        }
