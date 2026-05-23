from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


CostFunction = Callable[[np.ndarray], float]


@dataclass
class MPPIResult:
    action_sequence: np.ndarray
    best_cost: float
    mean: np.ndarray
    std: np.ndarray
    cost_history: list[float]
    diagnostics: dict | None = None  # effective_sample_size, weight_entropy, collapse_rate, etc.


class MPPI:
    """
    Model Predictive Path Integral controller.

    Unlike CEM which keeps only elite samples, MPPI uses exponential
    weighting over ALL samples. This naturally handles multimodal cost
    landscapes (e.g. go-left vs go-right around an obstacle) because
    samples on both sides of an obstacle contribute proportionally
    to their cost, rather than being discarded.

    Interface is compatible with CEMMPC for drop-in replacement.
    """

    def __init__(
        self,
        horizon: int = 15,
        action_dim: int = 2,
        num_samples: int = 1024,
        num_iterations: int = 5,
        action_low: float | list[float] | np.ndarray = -1.0,
        action_high: float | list[float] | np.ndarray = 1.0,
        init_std: float = 0.7,
        temperature: float = 0.1,
        smoothing: float = 0.2,
        seed: int | None = None,
    ):
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive, got {num_iterations}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not (0.0 <= smoothing <= 1.0):
            raise ValueError(f"smoothing must be in [0,1], got {smoothing}")

        self.horizon = horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.smoothing = smoothing
        self.init_std = float(init_std)

        self.rng = np.random.default_rng(seed)

        self.action_low = self._to_action_array(action_low)
        self.action_high = self._to_action_array(action_high)

        if np.any(self.action_low >= self.action_high):
            raise ValueError(
                f"action_low must be < action_high, got "
                f"{self.action_low}, {self.action_high}"
            )

    def _to_action_array(self, value: float | list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full((self.action_dim,), float(arr), dtype=np.float64)
        if arr.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action bound shape {(self.action_dim,)}, got {arr.shape}"
            )
        return arr

    def optimize(
        self,
        cost_fn: CostFunction,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> MPPIResult:
        """
        Run MPPI optimization.

        Uses exponential weighting over all samples instead of elite selection.
        This preserves information from all trajectories and handles
        multimodal cost landscapes better than CEM.
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

        if init_std is None:
            std = np.full(
                (self.horizon, self.action_dim), self.init_std, dtype=np.float64
            )
        else:
            std = np.asarray(init_std, dtype=np.float64).copy()

        if std.shape != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected init_std shape {(self.horizon, self.action_dim)}, "
                f"got {std.shape}"
            )

        best_sequence = mean.copy()
        best_cost = float("inf")
        cost_history: list[float] = []

        # Diagnostics accumulators
        diag_ess: list[float] = []
        diag_entropy: list[float] = []

        low = self.action_low.reshape(1, 1, self.action_dim)
        high = self.action_high.reshape(1, 1, self.action_dim)

        for _ in range(self.num_iterations):
            noise = self.rng.normal(
                scale=std[None, :, :],
                size=(self.num_samples, self.horizon, self.action_dim),
            )
            samples = mean[None, :, :] + noise
            samples = np.clip(samples, low, high)

            costs = np.asarray([cost_fn(seq) for seq in samples], dtype=np.float64)

            if not np.isfinite(costs).all():
                raise ValueError("MPPI received non-finite costs.")

            # Exponential weighting: lower cost → higher weight
            shifted = costs - np.min(costs)
            weights = np.exp(-shifted / self.temperature)
            weights_sum = weights.sum()
            if weights_sum < 1e-10:
                weights = np.ones_like(weights) / self.num_samples
            else:
                weights = weights / weights_sum

            # Diagnostics: effective sample size & weight entropy
            ess = 1.0 / float(np.sum(weights ** 2))
            ent = -float(np.sum(weights * np.log(weights + 1e-12)))
            diag_ess.append(ess)
            diag_entropy.append(ent)

            # Weighted mean update
            weighted_mean = np.einsum("n,nhd->hd", weights, samples)
            diff = samples - weighted_mean[None, :, :]
            weighted_std = np.sqrt(
                np.einsum("n,nhd->hd", weights, diff ** 2) + 1e-6
            )

            mean = self.smoothing * mean + (1.0 - self.smoothing) * weighted_mean
            std = self.smoothing * std + (1.0 - self.smoothing) * weighted_std

            iter_best_idx = int(np.argmin(costs))
            iter_best_cost = float(costs[iter_best_idx])

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_sequence = samples[iter_best_idx].copy()

            cost_history.append(best_cost)

        return MPPIResult(
            action_sequence=best_sequence,
            best_cost=best_cost,
            mean=mean,
            std=std,
            cost_history=cost_history,
            diagnostics={
                "effective_sample_size_mean": float(np.mean(diag_ess)),
                "effective_sample_size_min": float(np.min(diag_ess)),
                "weight_entropy_mean": float(np.mean(diag_entropy)),
                "weight_entropy_min": float(np.min(diag_entropy)),
                "collapse_rate": float(
                    sum(1 for e in diag_ess if e < 0.05 * self.num_samples) / len(diag_ess)
                ),
            },
        )

    def plan(
        self,
        cost_fn: CostFunction,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> tuple[np.ndarray, MPPIResult]:
        result = self.optimize(cost_fn=cost_fn, init_mean=init_mean, init_std=init_std)
        first_action = result.action_sequence[0].copy()
        return first_action, result
