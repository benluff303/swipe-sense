# utils.py
# Small utilities used across modules

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def l2_normalize(x: np.ndarray, axis=-1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def minmax01(x: np.ndarray, low_pct: float = 5, high_pct: float = 95) -> np.ndarray:
    lo, hi = np.percentile(x, low_pct), np.percentile(x, high_pct)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)

@dataclass
class ImageMeta:
    path: str
    width: int
    height: int
    edge_energy: float
    ahash: int
