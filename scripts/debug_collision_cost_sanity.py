#!/usr/bin/env python3
"""
Pure numpy sanity check for cumulative collision cost.

No MuJoCo.  No env.  Just numpy + cost_functions.

Run:
  PYTHONPATH=. python scripts/debug_collision_cost_sanity.py
"""

from __future__ import annotations

import numpy as np

from src.planners.cost_functions import (
    CostWeights,
    collision_cost,
    rollout_cost,
)


def make_fake_rollout(
    collision_pattern: np.ndarray,
) -> dict:
    """Build minimal fake inputs for rollout_cost.

    Returns dict with keys matching rollout_cost signature.
    """
    T = len(collision_pattern)
    # Object stays at origin, goal at (0.1, 0, 0)
    object_poses = np.zeros((T, 3), dtype=np.float64)
    object_poses[:, 0] = 0.0  # x
    object_poses[:, 1] = 0.0  # y
    ee_positions = np.zeros((T, 2), dtype=np.float64)
    action_sequence = np.zeros((T - 1, 2), dtype=np.float64)
    goal_pose = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    contact_flags = np.zeros(T, dtype=np.float64)

    return {
        "predicted_object_poses": object_poses,
        "ee_positions": ee_positions,
        "action_sequence": action_sequence,
        "goal_pose": goal_pose,
        "contact_flags": contact_flags,
        "collision_flags": collision_pattern.astype(np.float64),
    }


def main() -> None:
    weights = CostWeights(
        w_pos=50.0,
        w_theta=2.0,
        w_reach=5.0,
        w_no_contact=2.0,
        w_push_alignment=1.0,
        w_collision=20.0,
        w_collision_step=1.0,
        w_action=0.05,
        w_smooth=0.1,
        w_subgoal=0.0,
    )

    horizon = 80

    # Scenario 1: no collision
    flags_none = np.zeros(horizon, dtype=np.float64)
    # Scenario 2: one collision at step 10
    flags_one = np.zeros(horizon, dtype=np.float64)
    flags_one[10] = 1.0
    # Scenario 3: repeated collision steps 10-49 (40 steps)
    flags_repeated = np.zeros(horizon, dtype=np.float64)
    flags_repeated[10:50] = 1.0
    # Scenario 4: stuck on obstacle (60 steps)
    flags_stuck = np.zeros(horizon, dtype=np.float64)
    flags_stuck[10:70] = 1.0

    scenarios = [
        ("no collision", flags_none),
        ("one collision (step 10)", flags_one),
        ("repeated collision (steps 10-49)", flags_repeated),
        ("stuck on obstacle (steps 10-69)", flags_stuck),
    ]

    print("=" * 70)
    print("Collision Cost Sanity Check")
    print(f"  w_collision = {weights.w_collision}")
    print(f"  w_collision_step = {weights.w_collision_step}")
    print("=" * 70)
    print()

    results = []
    for name, flags in scenarios:
        coll_any, coll_count, coll_rate = collision_cost(flags)
        rollout_inputs = make_fake_rollout(flags)
        total_cost = rollout_cost(
            predicted_object_poses=rollout_inputs["predicted_object_poses"],
            ee_positions=rollout_inputs["ee_positions"],
            action_sequence=rollout_inputs["action_sequence"],
            goal_pose=rollout_inputs["goal_pose"],
            weights=weights,
            contact_flags=rollout_inputs["contact_flags"],
            collision_flags=rollout_inputs["collision_flags"],
        )
        collision_part = weights.w_collision * coll_any + weights.w_collision_step * coll_count
        results.append((name, coll_any, coll_count, coll_rate, collision_part, total_cost))

        print(f"  {name}")
        print(f"    collision_any   = {coll_any:.0f}")
        print(f"    collision_count = {coll_count:.0f}")
        print(f"    collision_rate  = {coll_rate:.3f}")
        print(f"    collision cost  = {collision_part:.1f}  (max={weights.w_collision*coll_any:.1f} + step={weights.w_collision_step*coll_count:.1f})")
        print(f"    total cost      = {total_cost:.2f}")
        print()

    # Assertions
    print("Monotonicity checks:")
    costs = [r[5] for r in results]
    for i in range(len(costs) - 1):
        ok = costs[i] < costs[i + 1]
        label = f"  cost({results[i][0]}) < cost({results[i+1][0]})"
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: {costs[i]:.2f} < {costs[i+1]:.2f}")
        assert ok, f"Monotonicity violated: {costs[i]:.2f} >= {costs[i+1]:.2f}"

    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
