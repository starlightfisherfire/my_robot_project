from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.envs.toy_push_env import ToyPushEnv, ToyPushState
from src.planners.cost_functions import CostWeights, rollout_cost


@dataclass
class OracleRolloutResult:
    """
    Rollout result for one candidate action sequence.

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


def rollout_action_sequence(
    env: ToyPushEnv,
    action_sequence: np.ndarray,
    restore_state: bool = True,
) -> OracleRolloutResult:
    """
    Roll out one action sequence using the environment's true dynamics.

    For the current stage, env is ToyPushEnv.
    Later, the same interface can be adapted to MuJoCo.

    Args:
        env:
            Environment with clone_state / restore_state / step.

        action_sequence:
            [H, action_dim]

        restore_state:
            If True, restore env to its original state after rollout.

            This should be True when the rollout is used for planning,
            because CEM evaluates many candidate sequences from the same
            starting state.
    """
    action_sequence = np.asarray(action_sequence, dtype=np.float64)

    if action_sequence.ndim != 2:
        raise ValueError(f"Expected action_sequence [H,A], got {action_sequence.shape}")

    if action_sequence.shape[-1] != 2:
        raise ValueError(f"Expected action_dim=2, got {action_sequence.shape[-1]}")

    if not np.isfinite(action_sequence).all():
        raise ValueError("action_sequence contains non-finite values.")

    start_state: ToyPushState = env.clone_state()

    object_poses = [start_state.object_pose.copy()]
    ee_positions = [start_state.ee_pos.copy()]
    contact_flags = [float(start_state.last_contact)]
    collision_flags = [float(start_state.last_collision)]

    for action in action_sequence:
        state = env.step(action)

        object_poses.append(state.object_pose.copy())
        ee_positions.append(state.ee_pos.copy())
        contact_flags.append(float(state.last_contact))
        collision_flags.append(float(state.last_collision))

    result = OracleRolloutResult(
        predicted_object_poses=np.asarray(object_poses, dtype=np.float64),
        ee_positions=np.asarray(ee_positions, dtype=np.float64),
        contact_flags=np.asarray(contact_flags, dtype=np.float64),
        collision_flags=np.asarray(collision_flags, dtype=np.float64),
    )

    if restore_state:
        env.restore_state(start_state)

    return result


def oracle_rollout_cost(
    env: ToyPushEnv,
    action_sequence: np.ndarray,
    goal_pose: np.ndarray | None = None,
    weights: CostWeights | None = None,
    restore_state: bool = True,
) -> float:
    """
    Compute rollout cost using environment true dynamics.

    This is the toy-env version of Oracle-MPC cost evaluation.

    Important:
        When used inside a CEM cost_fn, restore_state must be True.

        CEM evaluates many candidate action sequences from the same current
        environment state. If restore_state=False inside CEM, every cost
        evaluation mutates the environment, making candidate costs invalid
        and non-comparable.

        Use restore_state=False only when deliberately executing or simulating
        a sequence as an environment transition outside the CEM candidate
        evaluation loop.
    """
    start_state = env.clone_state()

    if goal_pose is None:
        goal_pose = start_state.goal_pose

    result = rollout_action_sequence(
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
    )