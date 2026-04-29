"""
Smoke test for CEM-MPC toy optimizer.

This script does not use MuJoCo.

It verifies that CEM can optimize a simple action-sequence objective:

    target action sequence = constant vector

Expected behavior:
    final best cost should be much smaller than the zero-action cost.
"""

from __future__ import annotations

import numpy as np

from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import action_smoothness_cost


def main() -> None:
    horizon = 12
    action_dim = 2

    target_action = np.array([0.45, -0.25], dtype=np.float64)
    target_sequence = np.tile(target_action[None, :], (horizon, 1))

    def toy_cost(action_sequence: np.ndarray) -> float:
        tracking_cost = float(np.mean((action_sequence - target_sequence) ** 2))
        smooth_cost = action_smoothness_cost(action_sequence)
        return tracking_cost + 0.1 * smooth_cost

    zero_sequence = np.zeros((horizon, action_dim), dtype=np.float64)
    zero_cost = toy_cost(zero_sequence)

    planner = CEMMPC(
        horizon=horizon,
        action_dim=action_dim,
        num_samples=512,
        num_elites=64,
        num_iterations=6,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
        init_std=0.8,
        smoothing=0.2,
        seed=42,
    )

    first_action, result = planner.plan(toy_cost)

    final_cost = result.best_cost
    mean_action = result.action_sequence.mean(axis=0)

    assert np.isfinite(final_cost), final_cost
    assert final_cost < zero_cost * 0.5, (
        f"CEM did not improve enough: zero_cost={zero_cost}, final_cost={final_cost}"
    )

    assert result.action_sequence.shape == (horizon, action_dim)
    assert first_action.shape == (action_dim,)
    assert np.isfinite(result.action_sequence).all()
    assert np.isfinite(first_action).all()
    assert np.isfinite(result.mean).all()
    assert np.isfinite(result.std).all()
    assert len(result.cost_history) == planner.num_iterations

    print("target_action:", target_action)
    print("zero_cost:", zero_cost)
    print("final_cost:", final_cost)
    print("first_action:", first_action)
    print("mean_action:", mean_action)
    print("cost_history:", result.cost_history)
    print("cem mpc toy debug ok")


if __name__ == "__main__":
    main()