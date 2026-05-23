from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .cem_mpc import CEMMPC, CEMResult


CostFunction = Callable[[np.ndarray], float]


@dataclass
class MultimodalCEMResult:
    action_sequence: np.ndarray
    best_cost: float
    mean: np.ndarray
    std: np.ndarray
    cost_history: list[float]
    mode_costs: list[float]
    best_mode: int


class MultimodalCEMMPC:
    """
    Multi-modal CEM: runs parallel CEM instances with different initial
    means (e.g. left-detour, right-detour, straight) and picks the best.

    Solves CEM's fundamental single-Gaussian limitation when obstacles
    create bimodal optimal solutions.

    Interface is compatible with CEMMPC for drop-in replacement.
    """

    def __init__(
        self,
        horizon: int = 15,
        action_dim: int = 2,
        num_samples: int = 512,
        num_elites: int = 64,
        num_iterations: int = 5,
        action_low: float | list[float] | np.ndarray = -1.0,
        action_high: float | list[float] | np.ndarray = 1.0,
        init_std: float = 0.7,
        smoothing: float = 0.2,
        lateral_offset: float = 0.5,
        seed: int | None = None,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.lateral_offset = lateral_offset
        self.seed = seed

        self._cem_kwargs = dict(
            horizon=horizon,
            action_dim=action_dim,
            num_samples=num_samples,
            num_elites=num_elites,
            num_iterations=num_iterations,
            action_low=action_low,
            action_high=action_high,
            init_std=init_std,
            smoothing=smoothing,
        )

        self.action_low = np.atleast_1d(
            np.asarray(action_low, dtype=np.float64)
        )
        self.action_high = np.atleast_1d(
            np.asarray(action_high, dtype=np.float64)
        )
        if self.action_low.shape == (1,):
            self.action_low = np.full((action_dim,), self.action_low[0])
        if self.action_high.shape == (1,):
            self.action_high = np.full((action_dim,), self.action_high[0])

    def _make_lateral_means(self) -> list[np.ndarray]:
        """Generate initial means: straight, left-detour, right-detour."""
        straight = np.zeros((self.horizon, self.action_dim), dtype=np.float64)

        left = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        mid = self.horizon // 2
        left[:mid, 1] = self.lateral_offset
        left[mid:, 1] = -self.lateral_offset * 0.5

        right = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        right[:mid, 1] = -self.lateral_offset
        right[mid:, 1] = self.lateral_offset * 0.5

        return [straight, left, right]

    def optimize(
        self,
        cost_fn: CostFunction,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> MultimodalCEMResult:
        """
        Run CEM from multiple initial means and return the best result.

        If init_mean is provided, it is used as one mode alongside the
        lateral detour modes. Otherwise, straight/left/right are used.
        """
        modes = self._make_lateral_means()
        if init_mean is not None:
            modes[0] = np.asarray(init_mean, dtype=np.float64).copy()

        results: list[CEMResult] = []
        for i, mode_mean in enumerate(modes):
            seed_i = None if self.seed is None else self.seed + i * 1000
            cem = CEMMPC(**self._cem_kwargs, seed=seed_i)
            result = cem.optimize(cost_fn=cost_fn, init_mean=mode_mean, init_std=init_std)
            results.append(result)

        mode_costs = [r.best_cost for r in results]
        best_mode = int(np.argmin(mode_costs))
        best_result = results[best_mode]

        return MultimodalCEMResult(
            action_sequence=best_result.action_sequence,
            best_cost=best_result.best_cost,
            mean=best_result.mean,
            std=best_result.std,
            cost_history=best_result.cost_history,
            mode_costs=mode_costs,
            best_mode=best_mode,
        )

    def plan(
        self,
        cost_fn: CostFunction,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> tuple[np.ndarray, MultimodalCEMResult]:
        result = self.optimize(cost_fn=cost_fn, init_mean=init_mean, init_std=init_std)
        first_action = result.action_sequence[0].copy()
        return first_action, result
