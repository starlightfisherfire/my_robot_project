"""
Smoke test for toy oracle rollout.

This script verifies:

ToyPushEnv
→ clone state
→ rollout action sequence
→ restore state
→ rollout_cost
→ CEM-MPC planning through oracle rollout

This is not MuJoCo yet.
It only tests the Oracle-MPC interface on a lightweight deterministic toy env.
"""

from __future__ import annotations

import numpy as np

from src.envs.toy_push_env import ToyPushEnv
from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import CostWeights
from src.planners.oracle_rollout import (
    oracle_rollout_cost,
    rollout_action_sequence,
)


def state_distance_to_goal(env: ToyPushEnv) -> float:
    state = env.clone_state()
    return float(np.linalg.norm(state.object_pose[:2] - state.goal_pose[:2]))


def main() -> None:
    env = ToyPushEnv()
    env.reset(
        object_pose=[0.20, 0.18, 0.0],
        goal_pose=[0.42, 0.18, 0.0],
        ee_pos=[0.10, 0.18],
    )

    horizon = 18
    action_dim = 2

    initial_state = env.clone_state()
    initial_dist = state_distance_to_goal(env)

    zero_actions = np.zeros((horizon, action_dim), dtype=np.float64)

    zero_rollout = rollout_action_sequence(
        env=env,
        action_sequence=zero_actions,
        restore_state=True,
    )

    # Check rollout shapes.
    assert zero_rollout.predicted_object_poses.shape == (horizon + 1, 3)
    assert zero_rollout.ee_positions.shape == (horizon + 1, 2)
    assert zero_rollout.contact_flags.shape == (horizon + 1,)
    assert zero_rollout.collision_flags.shape == (horizon + 1,)

    # Check restore_state worked.
    restored_state = env.clone_state()
    assert np.allclose(restored_state.object_pose, initial_state.object_pose)
    assert np.allclose(restored_state.goal_pose, initial_state.goal_pose)
    assert np.allclose(restored_state.ee_pos, initial_state.ee_pos)
    assert restored_state.step_count == initial_state.step_count

    weights = CostWeights(
        w_pos=10.0,
        w_theta=1.0,
        w_reach=2.0,
        w_no_contact=1.0,
        w_push_alignment=0.5,
        w_collision=20.0,
        w_action=0.02,
        w_smooth=0.05,
        w_subgoal=0.0,
    )

    zero_cost = oracle_rollout_cost(
        env=env,
        action_sequence=zero_actions,
        weights=weights,
        restore_state=True,
    )

    planner = CEMMPC(
        horizon=horizon,
        action_dim=action_dim,
        num_samples=1536,
        num_elites=128,
        num_iterations=7,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
        init_std=0.8,
        smoothing=0.2,
        seed=42,
    )

    def cost_fn(action_sequence: np.ndarray) -> float:
        return oracle_rollout_cost(
            env=env,
            action_sequence=action_sequence,
            weights=weights,
            restore_state=True,
        )

    first_action, result = planner.plan(cost_fn)

    planned_rollout = rollout_action_sequence(
        env=env,
        action_sequence=result.action_sequence,
        restore_state=True,
    )

    final_object_pose = planned_rollout.predicted_object_poses[-1]
    goal_pose = env.clone_state().goal_pose
    final_dist = float(np.linalg.norm(final_object_pose[:2] - goal_pose[:2]))

    planned_cost = oracle_rollout_cost(
        env=env,
        action_sequence=result.action_sequence,
        weights=weights,
        restore_state=True,
    )

    assert np.isfinite(zero_cost), zero_cost
    assert np.isfinite(planned_cost), planned_cost
    assert planned_cost < zero_cost, (
        f"Expected CEM planned cost < zero cost, got "
        f"planned_cost={planned_cost}, zero_cost={zero_cost}"
    )
    assert final_dist < initial_dist, (
        f"Expected final distance < initial distance, got "
        f"final_dist={final_dist}, initial_dist={initial_dist}"
    )

    # Check env was restored again after planning rollouts.
    final_env_state = env.clone_state()
    assert np.allclose(final_env_state.object_pose, initial_state.object_pose)
    assert np.allclose(final_env_state.goal_pose, initial_state.goal_pose)
    assert np.allclose(final_env_state.ee_pos, initial_state.ee_pos)

    print("initial_dist:", initial_dist)
    print("final_dist:", final_dist)
    print("zero_cost:", zero_cost)
    print("planned_cost:", planned_cost)
    print("first_action:", first_action)
    print("best_cost:", result.best_cost)
    print("cost_history:", result.cost_history)
    print("max_contact:", float(np.max(planned_rollout.contact_flags)))
    print("max_collision:", float(np.max(planned_rollout.collision_flags)))
    print("oracle rollout debug ok")


if __name__ == "__main__":
    main()