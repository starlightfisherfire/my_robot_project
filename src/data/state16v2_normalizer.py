"""State16v2 normalizer — geometric relationships, no velocity."""

import json
import numpy as np
from pathlib import Path


class State16v2Normalizer:
    """Normalizer for state16v2 format.

    Continuous indices: [0,1,4,5,6,7,9,10,11,12,13,14]
    Valid flag index: 15
    """

    VALID_FLAG_INDEX = 15
    CONTINUOUS_INDICES = [0, 1, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None
        self.fitted_ = False

    def fit(self, states: np.ndarray) -> "State16v2Normalizer":
        """Fit normalizer on training data.
        
        Args:
            states: [N, H, 6, 16] or [N, 6, 16] array
        """
        states = np.asarray(states, dtype=np.float32)
        flat = states.reshape(-1, states.shape[-1])
        valid_mask = flat[:, self.VALID_FLAG_INDEX] > 0.5
        valid_tokens = flat[valid_mask]

        self.mean_ = valid_tokens[:, self.CONTINUOUS_INDICES].mean(axis=0).astype(np.float32)
        std = valid_tokens[:, self.CONTINUOUS_INDICES].std(axis=0).astype(np.float32)
        self.std_ = np.where(std < self.eps, 1.0, std)
        self.fitted_ = True
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        """Normalize continuous fields for valid tokens."""
        if not self.fitted_:
            raise RuntimeError("Must fit before transform.")
        states = np.asarray(states, dtype=np.float32)
        out = states.copy()
        flat = out.reshape(-1, out.shape[-1])
        valid_mask = flat[:, self.VALID_FLAG_INDEX] > 0.5
        if valid_mask.any():
            vt = flat[valid_mask].copy()
            vt[:, self.CONTINUOUS_INDICES] = (
                vt[:, self.CONTINUOUS_INDICES] - self.mean_
            ) / self.std_
            flat[valid_mask] = vt
        flat[~valid_mask] = 0.0
        return out

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "mean": self.mean_.tolist(),
                "std": self.std_.tolist(),
                "continuous_indices": self.CONTINUOUS_INDICES,
                "valid_flag_index": self.VALID_FLAG_INDEX,
                "version": "state16v2",
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "State16v2Normalizer":
        with open(path) as f:
            data = json.load(f)
        norm = cls()
        norm.mean_ = np.array(data["mean"], dtype=np.float32)
        norm.std_ = np.array(data["std"], dtype=np.float32)
        norm.fitted_ = True
        return norm
