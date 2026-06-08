from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.envs.mujoco_push_env import MujocoPushEnv, MujocoPushState
from src.planners.cost_functions import CostWeights, rollout_cost, make_staged_cost_weights


@dataclass
class MujocoOracleRolloutResult:
    """
    Rollout result for one MuJoCo candidate action sequence.

    predicted_object_poses:
        [H + 1, 3], includes current / initial object pose at index 0.

    ee_positions:
        [H + 1, 2], includes current / initial EE position at index 0.

    contact_flags:
        [H + 1], contact indicator for each recorded state.

    collision_flags:
        [H + 1], collision indicator for each recorded state.
    """

    predicted_object_poses: np.ndarray
    ee_positions: np.ndarray
    contact_flags: np.ndarray
    collision_flags: np.ndarray


def rollout_action_sequence_mujoco(
    env: MujocoPushEnv,
    action_sequence: np.ndarray,
    restore_state: bool = True,
) -> MujocoOracleRolloutResult:
    """
    Roll out one action sequence using MuJoCo true dynamics.

    Args:
        env:
            MujocoPushEnv with clone_state / restore_state / step.

        action_sequence:
            [H, 2] normalized planar actions.

        restore_state:
            If True, restore env to its original state after rollout.

            This must be True when used inside CEM candidate evaluation,
            because every candidate sequence must start from the same state.
    """
    action_sequence = np.asarray(action_sequence, dtype=np.float64)

    if action_sequence.ndim != 2:
        raise ValueError(
            f"Expected action_sequence [H, A], got {action_sequence.shape}"
        )

    if action_sequence.shape[-1] != 2:
        raise ValueError(f"Expected action_dim=2, got {action_sequence.shape[-1]}")

    if not np.isfinite(action_sequence).all():
        raise ValueError("action_sequence contains non-finite values.")

    start_state: MujocoPushState = env.clone_state()

    object_poses = [env.get_object_pose().copy()]
    ee_positions = [env.get_ee_pos().copy()]
    contact_flags = [env.get_contact_flag()]
    collision_flags = [env.get_collision_flag()]

    for action in action_sequence:
        env.step(action)

        object_poses.append(env.get_object_pose().copy())
        ee_positions.append(env.get_ee_pos().copy())
        contact_flags.append(env.get_contact_flag())
        collision_flags.append(env.get_collision_flag())

    result = MujocoOracleRolloutResult(
        predicted_object_poses=np.asarray(object_poses, dtype=np.float64),
        ee_positions=np.asarray(ee_positions, dtype=np.float64),
        contact_flags=np.asarray(contact_flags, dtype=np.float64),
        collision_flags=np.asarray(collision_flags, dtype=np.float64),
    )

    if restore_state:
        env.restore_state(start_state)

    return result


def mujoco_oracle_rollout_cost(
    env: MujocoPushEnv,
    action_sequence: np.ndarray,
    goal_pose: np.ndarray | None = None,
    weights: CostWeights | None = None,
    restore_state: bool = True,
    obstacle_positions: np.ndarray | None = None,
    obstacle_radii: np.ndarray | None = None,
    cost_mode: str = "current",
) -> float:
    """
    Compute rollout cost using MuJoCo true dynamics.

    Important:
        When used inside a CEM cost_fn, restore_state must be True.

        CEM evaluates many candidate action sequences from the same current
        environment state. If restore_state=False inside CEM, candidate costs
        are no longer comparable.
    """
    if goal_pose is None:
        goal_pose = env.get_goal_pose()

    # Auto-select staged weights when cost_mode is staged and weights not provided
    if weights is None and cost_mode == "staged_contact_obstacle_goal":
        weights = make_staged_cost_weights()

    result = rollout_action_sequence_mujoco(
        env=env,
        action_sequence=action_sequence,
        restore_state=restore_state,
    )

    return rollout_cost(
        predicted_object_poses=result.predicted_object_poses,
        ee_positions=result.ee_positions,
        action_sequence=action_sequence,
        goal_pose=np.asarray(goal_pose, dtype=np.float64),
        weights=weights,
        contact_flags=result.contact_flags,
        collision_flags=result.collision_flags,
        obstacle_positions=obstacle_positions,
        obstacle_radii=obstacle_radii,
        cost_mode=cost_mode,
    )