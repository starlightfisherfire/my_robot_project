"""
Smoke test for MuJoCo oracle rollout.

This script verifies:

MujocoPushEnv
→ clone_state
→ rollout action sequence with MuJoCo true dynamics
→ restore_state
→ compute rollout cost

This is not full Oracle-MPC yet.
It only checks that the MuJoCo true-dynamics rollout interface works.
"""

from __future__ import annotations

import numpy as np

from src.envs.mujoco_push_env import MujocoPushEnv
from src.planners.cost_functions import CostWeights
from src.planners.mujoco_oracle_rollout import (
    mujoco_oracle_rollout_cost,
    rollout_action_sequence_mujoco,
)


def dist_to_goal(env: MujocoPushEnv) -> float:
    object_pose = env.get_object_pose()
    goal_pose = env.get_goal_pose()
    return float(np.linalg.norm(object_pose[:2] - goal_pose[:2]))


def main() -> None:
    env = MujocoPushEnv(
        control_dt=0.1,
        max_speed_mps=0.05,
    )

    # Use a close goal for rollout-interface debugging.
    # This is not the final push-to-pose benchmark setting.
    env.reset(
        object_pose=[0.20, 0.18, 0.0],
        goal_pose=[0.23, 0.18, 0.0],
        ee_pos=[0.10, 0.18],
    )

    horizon = 80
    action_dim = 2

    initial_state = env.clone_state()
    initial_dist = dist_to_goal(env)

    zero_actions = np.zeros((horizon, action_dim), dtype=np.float64)

    # A simple hand-coded push-right sequence.
    # It may overshoot the close goal, so the test should not require
    # final_dist < initial_dist. Instead, it checks contact, object motion,
    # and whether the rollout gets closer to the goal at some point.
    right_actions = np.zeros((horizon, action_dim), dtype=np.float64)
    right_actions[:, 0] = 1.0

    zero_rollout = rollout_action_sequence_mujoco(
        env=env,
        action_sequence=zero_actions,
        restore_state=True,
    )

    right_rollout = rollout_action_sequence_mujoco(
        env=env,
        action_sequence=right_actions,
        restore_state=True,
    )

    assert zero_rollout.predicted_object_poses.shape == (horizon + 1, 3)
    assert zero_rollout.ee_positions.shape == (horizon + 1, 2)
    assert zero_rollout.contact_flags.shape == (horizon + 1,)
    assert zero_rollout.collision_flags.shape == (horizon + 1,)

    assert right_rollout.predicted_object_poses.shape == (horizon + 1, 3)
    assert right_rollout.ee_positions.shape == (horizon + 1, 2)
    assert right_rollout.contact_flags.shape == (horizon + 1,)
    assert right_rollout.collision_flags.shape == (horizon + 1,)

    assert np.isfinite(right_rollout.predicted_object_poses).all()
    assert np.isfinite(right_rollout.ee_positions).all()
    assert np.isfinite(right_rollout.contact_flags).all()
    assert np.isfinite(right_rollout.collision_flags).all()

    # Check restore after rollout.
    restored_state = env.clone_state()
    assert np.allclose(restored_state.qpos, initial_state.qpos)
    assert np.allclose(restored_state.qvel, initial_state.qvel)
    assert np.allclose(restored_state.ctrl, initial_state.ctrl)
    assert np.allclose(restored_state.goal_pose, initial_state.goal_pose)
    assert restored_state.step_count == initial_state.step_count

    weights = CostWeights(
        w_pos=50.0,
        w_theta=1.0,
        w_reach=1.0,
        w_no_contact=1.0,
        w_push_alignment=0.2,
        w_collision=20.0,
        w_action=0.01,
        w_smooth=0.01,
        w_subgoal=0.0,
    )

    zero_cost = mujoco_oracle_rollout_cost(
        env=env,
        action_sequence=zero_actions,
        weights=weights,
        restore_state=True,
    )

    right_cost = mujoco_oracle_rollout_cost(
        env=env,
        action_sequence=right_actions,
        weights=weights,
        restore_state=True,
    )

    final_object_pose = right_rollout.predicted_object_poses[-1]
    goal_pose = env.get_goal_pose()

    distances_to_goal = np.linalg.norm(
        right_rollout.predicted_object_poses[:, :2] - goal_pose[:2],
        axis=1,
    )
    min_dist = float(np.min(distances_to_goal))
    final_dist = float(distances_to_goal[-1])

    object_x_start = float(right_rollout.predicted_object_poses[0, 0])
    object_x_final = float(right_rollout.predicted_object_poses[-1, 0])

    max_contact = float(np.max(right_rollout.contact_flags))
    max_collision = float(np.max(right_rollout.collision_flags))

    assert np.isfinite(zero_cost), zero_cost
    assert np.isfinite(right_cost), right_cost

    assert max_contact in {0.0, 1.0}
    assert max_collision in {0.0, 1.0}

    # The hand-coded right push should at least make contact.
    assert max_contact > 0.5, "Expected right push rollout to make contact."

    # The object should move to the right under a right-push sequence.
    assert object_x_final > object_x_start, (
        f"Expected object x to increase. "
        f"object_x_start={object_x_start}, object_x_final={object_x_final}"
    )

    # It is okay if the final pose overshoots the close goal.
    # For interface sanity, only require that the rollout gets closer at some point.
    assert min_dist < initial_dist, (
        f"Expected rollout to get closer to the goal at some point. "
        f"initial_dist={initial_dist}, min_dist={min_dist}, final_dist={final_dist}"
    )

    # Check env restored after cost calls.
    final_env_state = env.clone_state()
    assert np.allclose(final_env_state.qpos, initial_state.qpos)
    assert np.allclose(final_env_state.qvel, initial_state.qvel)
    assert np.allclose(final_env_state.ctrl, initial_state.ctrl)
    assert np.allclose(final_env_state.goal_pose, initial_state.goal_pose)
    assert final_env_state.step_count == initial_state.step_count

    print("initial_dist:", initial_dist)
    print("min_dist:", min_dist)
    print("final_dist:", final_dist)
    print("zero_cost:", zero_cost)
    print("right_cost:", right_cost)
    print("max_contact:", max_contact)
    print("max_collision:", max_collision)
    print("object_x_start:", object_x_start)
    print("object_x_final:", object_x_final)
    print("object_pose_start:", right_rollout.predicted_object_poses[0])
    print("object_pose_final:", final_object_pose)
    print("ee_pos_start:", right_rollout.ee_positions[0])
    print("ee_pos_final:", right_rollout.ee_positions[-1])
    print("mujoco oracle rollout debug ok")


if __name__ == "__main__":
    main()