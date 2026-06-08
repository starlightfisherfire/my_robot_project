from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


CostFunction = Callable[[np.ndarray], float]


@dataclass
class CEMResult:
    action_sequence: np.ndarray
    best_cost: float
    mean: np.ndarray
    std: np.ndarray
    cost_history: list[float]


class CEMMPC:
    """
    Minimal CEM optimizer for action-sequence planning.

    This class does not know about MuJoCo or robot states.
    It only solves:

        minimize cost_fn(action_sequence)

    where:
        action_sequence: [horizon, action_dim]
    """

    def __init__(
        self,
        horizon: int = 15,
        action_dim: int = 2,
        num_samples: int = 1024,
        num_elites: int = 64,
        num_iterations: int = 5,
        action_low: float | list[float] | np.ndarray = -1.0,
        action_high: float | list[float] | np.ndarray = 1.0,
        init_std: float = 0.7,
        smoothing: float = 0.2,
        seed: int | None = None,
    ):
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        if num_elites <= 0 or num_elites > num_samples:
            raise ValueError(
                f"num_elites must be in [1, num_samples], got "
                f"num_elites={num_elites}, num_samples={num_samples}"
            )

        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {num_iterations}")

        if not (0.0 <= smoothing <= 1.0):
            raise ValueError(f"smoothing must be in [0,1], got {smoothing}")

        self.horizon = horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.smoothing = smoothing

        self.rng = np.random.default_rng(seed)

        self.action_low = self._to_action_array(action_low)
        self.action_high = self._to_action_array(action_high)

        if np.any(self.action_low >= self.action_high):
            raise ValueError(
                f"action_low must be < action_high, got "
                f"{self.action_low}, {self.action_high}"
            )

        self.init_std = float(init_std)

        if not np.isfinite(self.init_std) or self.init_std <= 0:
            raise ValueError(
                f"init_std must be positive and finite, got {self.init_std}"
            )

    def _to_action_array(self, value: float | list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)

        if arr.ndim == 0:
            arr = np.full((self.action_dim,), float(arr), dtype=np.float64)

        if arr.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action bound shape {(self.action_dim,)}, got {arr.shape}"
            )

        if not np.isfinite(arr).all():
            raise ValueError(f"Action bound must be finite, got {arr}")

        return arr

    def optimize(
        self,
        cost_fn: CostFunction,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> CEMResult:
        """
        Run CEM optimization.

        Args:
            cost_fn:
                Function mapping action_sequence [H, A] to scalar cost.

            init_mean:
                Optional [H, A] initial action mean.

            init_std:
                Optional [H, A] initial std.
        """
        if init_mean is None:
            mean = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        else:
            mean = np.asarray(init_mean, dtype=np.float64).copy()

        if mean.shape != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected init_mean shape {(self.horizon, self.action_dim)}, "
                f"got {mean.shape}"
            )

        if not np.isfinite(mean).all():
            raise ValueError("init_mean contains non-finite values.")

        if init_std is None:
            std = np.full(
                (self.horizon, self.action_dim),
                self.init_std,
                dtype=np.float64,
            )
        else:
            std = np.asarray(init_std, dtype=np.float64).copy()

        if std.shape != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected init_std shape {(self.horizon, self.action_dim)}, "
                f"got {std.shape}"
            )

        if not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("init_std must contain positive finite values.")

        best_sequence = mean.copy()
        best_cost = float("inf")
        cost_history: list[float] = []

        low = self.action_low.reshape(1, 1, self.action_dim)
        high = self.action_high.reshape(1, 1, self.action_dim)

        for _ in range(self.num_iterations):
            samples = self.rng.normal(
                loc=mean[None, :, :],
                scale=std[None, :, :],
                size=(self.num_samples, self.horizon, self.action_dim),
            )

            samples = np.clip(samples, low, high)

            if hasattr(cost_fn, 'evaluate_batch'):
                costs = cost_fn.evaluate_batch(samples)
            else:
                costs = np.asarray([cost_fn(seq) for seq in samples], dtype=np.float64)

            if not np.isfinite(costs).all():
                raise ValueError("CEM received non-finite costs.")

            elite_idx = np.argsort(costs)[: self.num_elites]
            elites = samples[elite_idx]

            elite_mean = elites.mean(axis=0)
            elite_std = elites.std(axis=0) + 1e-6

            if not np.isfinite(elite_mean).all():
                raise ValueError("CEM elite_mean contains non-finite values.")

            if not np.isfinite(elite_std).all() or np.any(elite_std <= 0):
                raise ValueError("CEM elite_std must contain positive finite values.")

            mean = self.smoothing * mean + (1.0 - self.smoothing) * elite_mean
            std = self.smoothing * std + (1.0 - self.smoothing) * elite_std

            if not np.isfinite(mean).all():
                raise ValueError("CEM mean became non-finite.")

            if not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("CEM std became non-positive or non-finite.")

            iter_best_idx = int(np.argmin(costs))
            iter_best_cost = float(costs[iter_best_idx])

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_sequence = samples[iter_best_idx].copy()

            cost_history.append(best_cost)

        return CEMResult(
            action_sequence=best_sequence,
            best_cost=best_cost,
            mean=mean,
            std=std,
            cost_history=cost_history,
        )

    def plan(
        self,
        cost_fn: CostFunction,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> tuple[np.ndarray, CEMResult]:
        """
        Return first action and full CEM result.
        """
        result = self.optimize(
            cost_fn=cost_fn,
            init_mean=init_mean,
            init_std=init_std,
        )

        first_action = result.action_sequence[0].copy()

        return first_action, result