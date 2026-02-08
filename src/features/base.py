from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np


@dataclass(frozen=True)
class FeatureOutput:
    name: str
    values: object


class FeatureExtractor(Protocol):
    @property
    def name(self) -> str:
        ...

    def extract_batch(self, images: np.ndarray) -> Dict[str, object]:
        ...
