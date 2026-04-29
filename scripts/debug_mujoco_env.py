"""
Smoke test for minimal MuJoCo push environment.

This script verifies:

MujocoPushEnv
→ reset
→ step
→ clone_state / restore_state
→ reset_from_template
→ object pose / EE pose / contact flag extraction

This is not Oracle-MPC yet.
It only checks that the first MuJoCo environment scaffold is alive.
"""

from __future__ import annotations

import numpy as np

from src.envs.mujoco_push_env import MujocoPushEnv
from src.interventions.reset_template_loader import (
    get_templates_by_split,
    load_reset_templates,
)


def main() -> None:
    env = MujocoPushEnv()

    state0 = env.reset(
        object_pose=[0.20, 0.18, 0.0],
        goal_pose=[0.42, 0.18, 0.0],
        ee_pos=[0.10, 0.18],
    )

    object_pose0 = env.get_object_pose()
    ee_pos0 = env.get_ee_pos()
    goal_pose0 = env.get_goal_pose()

    assert object_pose0.shape == (3,)
    assert ee_pos0.shape == (2,)
    assert goal_pose0.shape == (3,)
    assert np.isfinite(object_pose0).all()
    assert np.isfinite(ee_pos0).all()
    assert np.isfinite(goal_pose0).all()

    # Step with zero action.
    env.step([0.0, 0.0])
    object_pose_zero = env.get_object_pose()
    ee_pos_zero = env.get_ee_pos()

    assert np.isfinite(object_pose_zero).all()
    assert np.isfinite(ee_pos_zero).all()

    # Restore should return to the original state.
    env.restore_state(state0)
    object_pose_restored = env.get_object_pose()
    ee_pos_restored = env.get_ee_pos()

    assert np.allclose(object_pose_restored[:2], object_pose0[:2], atol=1e-6)
    assert np.allclose(ee_pos_restored, ee_pos0, atol=1e-6)

    # Move right for a short sequence and check finite states.
    for _ in range(30):
        env.step([1.0, 0.0])

    object_pose_after = env.get_object_pose()
    ee_pos_after = env.get_ee_pos()
    contact_flag = env.get_contact_flag()
    collision_flag = env.get_collision_flag()

    assert np.isfinite(object_pose_after).all()
    assert np.isfinite(ee_pos_after).all()
    assert contact_flag in {0.0, 1.0}
    assert collision_flag in {0.0, 1.0}

    # Reset from generated template.
    templates = load_reset_templates("data/sim/metadata/reset_templates_v0.json")
    train_templates = get_templates_by_split(templates, "train_sim_id")
    env.reset_from_template(train_templates[0])

    object_pose_template = env.get_object_pose()
    ee_pos_template = env.get_ee_pos()
    goal_pose_template = env.get_goal_pose()

    assert object_pose_template.shape == (3,)
    assert ee_pos_template.shape == (2,)
    assert goal_pose_template.shape == (3,)
    assert np.isfinite(object_pose_template).all()
    assert np.isfinite(ee_pos_template).all()
    assert np.isfinite(goal_pose_template).all()

    print("object_pose0:", object_pose0)
    print("ee_pos0:", ee_pos0)
    print("goal_pose0:", goal_pose0)
    print("object_pose_after:", object_pose_after)
    print("ee_pos_after:", ee_pos_after)
    print("contact_flag:", contact_flag)
    print("collision_flag:", collision_flag)
    print("object_pose_template:", object_pose_template)
    print("ee_pos_template:", ee_pos_template)
    print("goal_pose_template:", goal_pose_template)
    print("mujoco env debug ok")


if __name__ == "__main__":
    main()