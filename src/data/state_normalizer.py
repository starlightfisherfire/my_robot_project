from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


class StateNormalizer:
    """
    Normalizer for Paper 1 structured object-state tensors.

    Expected input shape:
        [..., D_raw] where D_raw = 16

    Token feature schema:
        0  x
        1  y
        2  sin(theta)
        3  cos(theta)
        4  vx
        5  vy
        6  omega
        7  size_x
        8  size_y
        9  shape_T
        10 shape_L
        11 shape_other
        12 mass
        13 friction
        14 contact_flag
        15 valid_flag

    v0.1 rule:
        Normalize continuous scalar fields only.
        Do not normalize sin/cos, one-hot shape fields, contact_flag, valid_flag.
        Invalid / padding tokens must remain exactly zero after transform.
    """

    def __init__(
        self,
        continuous_indices: Optional[Iterable[int]] = None,
        valid_flag_index: int = 15,
        eps: float = 1e-6,
    ):
        if continuous_indices is None:
            continuous_indices = [0, 1, 4, 5, 6, 7, 8, 12, 13]

        self.continuous_indices = list(continuous_indices)
        self.valid_flag_index = valid_flag_index
        self.eps = eps

        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.fitted_: bool = False

    def fit(self, states: np.ndarray) -> "StateNormalizer":
        """
        Fit mean/std on valid tokens only.

        states shape:
            [B, H, N, D] or [num_samples, D]
        """
        states = np.asarray(states, dtype=np.float32)

        required_last_dim = max(self.continuous_indices + [self.valid_flag_index]) + 1
        if states.shape[-1] < required_last_dim:
            raise ValueError(
                f"Last dim too small for schema: got {states.shape[-1]}, "
                f"required at least {required_last_dim}"
            )

        flat = states.reshape(-1, states.shape[-1])
        valid_mask = flat[:, self.valid_flag_index] > 0.5

        if valid_mask.sum() == 0:
            raise ValueError("No valid tokens found when fitting StateNormalizer.")

        valid_flat = flat[valid_mask]
        cont = valid_flat[:, self.continuous_indices]

        mean = cont.mean(axis=0)
        std = cont.std(axis=0)
        std = np.where(std < self.eps, 1.0, std)

        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        self.fitted_ = True

        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        """
        Normalize continuous fields for valid tokens.

        Invalid / padding tokens are forced to exactly zero.
        """
        if not self.fitted_:
            raise RuntimeError("StateNormalizer must be fitted before transform().")

        states = np.asarray(states, dtype=np.float32)
        out = states.copy()

        flat = out.reshape(-1, out.shape[-1])
        valid_mask = flat[:, self.valid_flag_index] > 0.5

        if valid_mask.any():
            valid_tokens = flat[valid_mask].copy()
            valid_tokens[:, self.continuous_indices] = (
                valid_tokens[:, self.continuous_indices] - self.mean_
            ) / self.std_
            flat[valid_mask] = valid_tokens

        # Critical for padded obstacle tokens:
        # invalid tokens must remain exactly zero.
        flat[~valid_mask] = 0.0

        return out

    def inverse_transform(self, normalized_states: np.ndarray) -> np.ndarray:
        """
        Recover original continuous fields for valid tokens.

        Invalid / padding tokens are kept exactly zero.
        """
        if not self.fitted_:
            raise RuntimeError("StateNormalizer must be fitted before inverse_transform().")

        normalized_states = np.asarray(normalized_states, dtype=np.float32)
        out = normalized_states.copy()

        flat = out.reshape(-1, out.shape[-1])
        valid_mask = flat[:, self.valid_flag_index] > 0.5

        if valid_mask.any():
            valid_tokens = flat[valid_mask].copy()
            valid_tokens[:, self.continuous_indices] = (
                valid_tokens[:, self.continuous_indices] * self.std_
            ) + self.mean_
            flat[valid_mask] = valid_tokens

        # Keep invalid / padding tokens exactly zero.
        flat[~valid_mask] = 0.0

        return out

    def save(self, path: str | Path) -> None:
        if not self.fitted_:
            raise RuntimeError("Cannot save an unfitted StateNormalizer.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "continuous_indices": self.continuous_indices,
            "valid_flag_index": self.valid_flag_index,
            "eps": self.eps,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "StateNormalizer":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        obj = cls(
            continuous_indices=payload["continuous_indices"],
            valid_flag_index=payload["valid_flag_index"],
            eps=payload["eps"],
        )

        obj.mean_ = np.asarray(payload["mean"], dtype=np.float32)
        obj.std_ = np.asarray(payload["std"], dtype=np.float32)
        obj.fitted_ = True

        return obj