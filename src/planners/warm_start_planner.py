"""
Warm-Start MPC Planners (CEM & MPPI).

Key idea:
  1. Plan a long action sequence [H, A].
  2. Execute `execute_steps` actions.
  3. At next replan, shift the previous best sequence forward by `execute_steps`,
     and use it as the initial mean with *reduced* std for the "already planned"
     tail portion, and normal std for the new horizon extension.

This gives:
  - Smoother action continuity (no sudden direction changes at replan boundary)
  - Faster convergence (warm-start is already near-optimal)
  - Less wasted exploration in the tail region

Usage:
    from src.planners.warm_start_planner import WarmStartCEM, WarmStartMPPI

    planner = WarmStartCEM(execute_steps=10, warm_std_factor=0.3, **cem_kwargs)
    # or
    planner = WarmStartMPPI(execute_steps=10, warm_std_factor=0.3, **mppi_kwargs)

    # In the MPC loop:
    action, result = planner.plan(cost_fn)
    # execute action...
    planner.update_after_execute(result)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .cem_mpc import CEMMPC, CEMResult
from .mppi import MPPI, MPPIResult


CostFunction = Callable[[np.ndarray], float]


def _shift_and_build_std(
    prev_sequence: np.ndarray,
    execute_steps: int,
    horizon: int,
    action_dim: int,
    base_std: float | np.ndarray,
    warm_std_factor: float,
    reoptimize_std_factor: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift previous action sequence and build aligned std.

    Returns:
        init_mean: [H, A] shifted mean
        init_std:  [H, A] std with low values for warm-start tail, normal for new portion
    """
    prev_H, prev_A = prev_sequence.shape
    assert prev_A == action_dim

    # Number of steps to carry forward from previous plan
    carry = min(prev_H - execute_steps, horizon)

    # Build shifted mean
    init_mean = np.zeros((horizon, action_dim), dtype=np.float64)
    if carry > 0:
        init_mean[:carry] = prev_sequence[execute_steps:execute_steps + carry]

    # Clamp to action bounds
    init_mean = np.clip(init_mean, action_low, action_high)

    # Build aligned std
    if isinstance(base_std, np.ndarray) and base_std.ndim == 2:
        base_std_arr = base_std[:1, :].repeat(horizon, axis=0)  # broadcast
    else:
        base_std_arr = np.full((horizon, action_dim), float(base_std), dtype=np.float64)

    init_std = base_std_arr.copy()

    # Warm-start portion: reduced std (we trust the previous plan)
    if carry > 0:
        init_std[:carry] = base_std_arr[:carry] * warm_std_factor

    # New horizon extension: use reoptimize_std_factor (can be same or different)
    if carry < horizon:
        init_std[carry:] = base_std_arr[carry:] * reoptimize_std_factor

    # Ensure positive
    init_std = np.maximum(init_std, 1e-4)

    return init_mean, init_std


@dataclass
class WarmStartCEMResult:
    """Wraps CEMResult with warm-start metadata."""
    inner_result: CEMResult
    warm_start_steps: int  # how many steps came from previous plan

    @property
    def action_sequence(self):
        return self.inner_result.action_sequence

    @property
    def best_cost(self):
        return self.inner_result.best_cost

    @property
    def mean(self):
        return self.inner_result.mean

    @property
    def std(self):
        return self.inner_result.std

    @property
    def cost_history(self):
        return self.inner_result.cost_history


class WarmStartCEM:
    """
    CEM with warm-start from previous action sequence.

    After each replan, call update_after_execute() to store the tail
    of the current plan for the next warm-start.
    """

    def __init__(
        self,
        execute_steps: int = 10,
        warm_std_factor: float = 0.3,
        reoptimize_std_factor: float = 1.0,
        **cem_kwargs,
    ):
        """
        Args:
            execute_steps: Number of actions executed per replan cycle.
            warm_std_factor: Multiplier for std in the warm-start tail region.
                Lower = more trust in previous plan. E.g. 0.3 means std is 30% of base.
            reoptimize_std_factor: Multiplier for std in the new horizon extension.
                1.0 = normal exploration. Can set > 1.0 for more exploration at the frontier.
            **cem_kwargs: Passed to CEMMPC (horizon, num_samples, etc.)
        """
        self.execute_steps = execute_steps
        self.warm_std_factor = warm_std_factor
        self.reoptimize_std_factor = reoptimize_std_factor
        self._cem = CEMMPC(**cem_kwargs)
        self._prev_sequence: np.ndarray | None = None

    def optimize(
        self,
        cost_fn: CostFunction,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> WarmStartCEMResult:
        """
        Run CEM with warm-start if previous sequence is available.

        If init_mean/init_std are explicitly provided, they override warm-start.
        """
        # Determine if we should use warm-start
        use_warm = (
            self._prev_sequence is not None
            and init_mean is None  # no explicit override
        )

        if use_warm:
            ws_mean, ws_std = _shift_and_build_std(
                prev_sequence=self._prev_sequence,
                execute_steps=self.execute_steps,
                horizon=self._cem.horizon,
                action_dim=self._cem.action_dim,
                base_std=self._cem.init_std,
                warm_std_factor=self.warm_std_factor,
                reoptimize_std_factor=self.reoptimize_std_factor,
                action_low=self._cem.action_low,
                action_high=self._cem.action_high,
            )
            result = self._cem.optimize(cost_fn, init_mean=ws_mean, init_std=ws_std)
            warm_steps = min(
                self._prev_sequence.shape[0] - self.execute_steps,
                self._cem.horizon,
            )
        else:
            result = self._cem.optimize(cost_fn, init_mean=init_mean, init_std=init_std)
            warm_steps = 0

        return WarmStartCEMResult(
            inner_result=result,
            warm_start_steps=warm_steps,
        )

    def plan(
        self,
        cost_fn: CostFunction,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> tuple[np.ndarray, WarmStartCEMResult]:
        result = self.optimize(cost_fn, init_mean=init_mean, init_std=init_std)
        first_action = result.action_sequence[0].copy()
        return first_action, result

    def update_after_execute(self, result: WarmStartCEMResult):
        """Store the full action sequence for next warm-start."""
        self._prev_sequence = result.action_sequence.copy()

    def reset(self):
        """Clear warm-start state (e.g. after environment reset)."""
        self._prev_sequence = None


@dataclass
class WarmStartMPPIResult:
    """Wraps MPPIResult with warm-start metadata."""
    inner_result: MPPIResult
    warm_start_steps: int

    @property
    def action_sequence(self):
        return self.inner_result.action_sequence

    @property
    def best_cost(self):
        return self.inner_result.best_cost

    @property
    def mean(self):
        return self.inner_result.mean

    @property
    def std(self):
        return self.inner_result.std

    @property
    def cost_history(self):
        return self.inner_result.cost_history

    @property
    def diagnostics(self):
        return self.inner_result.diagnostics


class WarmStartMPPI:
    """
    MPPI with warm-start from previous action sequence.

    Same idea as WarmStartCEM but for MPPI.
    """

    def __init__(
        self,
        execute_steps: int = 10,
        warm_std_factor: float = 0.3,
        reoptimize_std_factor: float = 1.0,
        **mppi_kwargs,
    ):
        """
        Args:
            execute_steps: Number of actions executed per replan cycle.
            warm_std_factor: Multiplier for std in the warm-start tail region.
            reoptimize_std_factor: Multiplier for std in the new horizon extension.
            **mppi_kwargs: Passed to MPPI (horizon, num_samples, etc.)
        """
        self.execute_steps = execute_steps
        self.warm_std_factor = warm_std_factor
        self.reoptimize_std_factor = reoptimize_std_factor
        self._mppi = MPPI(**mppi_kwargs)
        self._prev_sequence: np.ndarray | None = None

    def optimize(
        self,
        cost_fn: CostFunction,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> WarmStartMPPIResult:
        use_warm = (
            self._prev_sequence is not None
            and init_mean is None
        )

        if use_warm:
            ws_mean, ws_std = _shift_and_build_std(
                prev_sequence=self._prev_sequence,
                execute_steps=self.execute_steps,
                horizon=self._mppi.horizon,
                action_dim=self._mppi.action_dim,
                base_std=self._mppi.init_std,
                warm_std_factor=self.warm_std_factor,
                reoptimize_std_factor=self.reoptimize_std_factor,
                action_low=self._mppi.action_low,
                action_high=self._mppi.action_high,
            )
            result = self._mppi.optimize(cost_fn, init_mean=ws_mean, init_std=ws_std)
            warm_steps = min(
                self._prev_sequence.shape[0] - self.execute_steps,
                self._mppi.horizon,
            )
        else:
            result = self._mppi.optimize(cost_fn, init_mean=init_mean, init_std=init_std)
            warm_steps = 0

        return WarmStartMPPIResult(
            inner_result=result,
            warm_start_steps=warm_steps,
        )

    def plan(
        self,
        cost_fn: CostFunction,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> tuple[np.ndarray, WarmStartMPPIResult]:
        result = self.optimize(cost_fn, init_mean=init_mean, init_std=init_std)
        first_action = result.action_sequence[0].copy()
        return first_action, result

    def update_after_execute(self, result: WarmStartMPPIResult):
        """Store the full action sequence for next warm-start."""
        self._prev_sequence = result.action_sequence.copy()

    def reset(self):
        """Clear warm-start state."""
        self._prev_sequence = None
