#!/usr/bin/env python3
"""Probe single-step object response in MuJoCo dynamics.

Measures: pusher displacement, object displacement, object speed,
contact/collision flags in one 0.1s env.step().

Usage:
  PYTHONPATH=. python scripts/probe_single_step_object_response.py
"""

import json
import sys
import numpy as np
from pathlib import Path
from src.envs.mujoco_push_env import MujocoPushEnv

# ── Config ──
TEMPLATES_PATH = "data/sim/metadata/reset_templates_obstacle_difficulty_v0.json"
SPLIT = "test_sim_layout_ood_blocking_hard"
TEMPLATE_INDEX = 0
MAX_SPEED_MPS = 1.0
OBJECT_MASSES = [0.050, 0.500]  # kg: 50g and 500g
DT = 0.1  # control_dt


def get_template():
    with open(TEMPLATES_PATH) as f:
        templates = json.load(f)
    filtered = [t for t in templates if t["split"] == SPLIT]
    if TEMPLATE_INDEX >= len(filtered):
        raise ValueError(f"template-index {TEMPLATE_INDEX} out of range")
    return filtered[TEMPLATE_INDEX]


def approach_object(env: MujocoPushEnv, max_steps: int = 200) -> bool:
    """Move pusher toward object center until contact or max_steps."""
    for _ in range(max_steps):
        obj_pos = env.get_object_pose()
        ee_pos = env.get_ee_pos()
        direction = obj_pos[:2] - ee_pos[:2]
        dist = float(np.linalg.norm(direction))

        if dist < 1e-4:
            break

        action = direction / dist  # unit vector toward object
        env.step(action)

        if env.get_contact_flag() > 0:
            return True
    return env.get_contact_flag() > 0


def probe_step(env: MujocoPushEnv, object_mass_kg: float):
    """Execute one push step and measure displacements."""
    # Get initial state
    ee_before = env.get_ee_pos().copy()
    obj_before = env.get_object_pose().copy()
    contact_before = bool(env.get_contact_flag() > 0)

    # Compute direction from ee to goal (or toward object center)
    goal = env.get_goal_pose()
    obj_center = env.get_object_pose()
    direction = goal[:2] - ee_before[:2]
    dist_to_goal = float(np.linalg.norm(direction))
    if dist_to_goal > 1e-4:
        action = direction / dist_to_goal
    else:
        action = np.array([1.0, 0.0])

    # Execute one step
    env.step(action)

    # Get final state
    ee_after = env.get_ee_pos().copy()
    obj_after = env.get_object_pose().copy()
    contact_after = bool(env.get_contact_flag() > 0)
    collision_after = bool(env.get_collision_flag() > 0)

    # Compute displacements
    pusher_displacement = float(np.linalg.norm(ee_after[:2] - ee_before[:2]))
    object_displacement = float(np.linalg.norm(obj_after[:2] - obj_before[:2]))
    object_avg_speed = object_displacement / DT
    pusher_avg_speed = pusher_displacement / DT

    return {
        "object_mass_kg": object_mass_kg,
        "initial_ee_pos": ee_before.tolist(),
        "initial_object_pose": obj_before.tolist(),
        "object_pose_before": obj_before.tolist(),
        "object_pose_after": obj_after.tolist(),
        "object_displacement_m": round(object_displacement, 6),
        "object_avg_speed_mps": round(object_avg_speed, 4),
        "pusher_displacement_m": round(pusher_displacement, 6),
        "pusher_avg_speed_mps": round(pusher_avg_speed, 4),
        "contact_before": contact_before,
        "contact_after": contact_after,
        "collision_after": collision_after,
    }


def main():
    template = get_template()
    print(f"Template: {template['reset_template_id']}")
    print(f"Split: {SPLIT}")
    print()
    print(f"{'='*80}")
    print(f"Single-Step Object Response Probe")
    print(f"max_speed_mps={MAX_SPEED_MPS}, control_dt={DT}s")
    print(f"{'='*80}")
    print()

    for mass in OBJECT_MASSES:
        print(f"\n{'─'*80}")
        print(f"  Object Mass: {mass*1000:.0f}g ({mass} kg)")
        print(f"{'─'*80}")

        # Create env with specified object mass
        env = MujocoPushEnv(
            control_dt=DT,
            max_speed_mps=MAX_SPEED_MPS,
            pusher_mass=0.05,
        )
        env.reset_from_template(template)

        # Set object mass by modifying the MuJoCo model directly
        # Override the geom mass in the model
        obj_body_id = env.object_body_id
        env.model.body_mass[obj_body_id] = mass
        # Recompute derived quantities
        import mujoco
        mujoco.mj_forward(env.model, env.data)

        # Get initial positions
        ee_init = env.get_ee_pos()
        obj_init = env.get_object_pose()
        goal_init = env.get_goal_pose()

        dist_ee_to_obj = float(np.linalg.norm(obj_init[:2] - ee_init[:2]))
        print(f"initial_ee_pos: ({ee_init[0]:.4f}, {ee_init[1]:.4f})")
        print(f"initial_object_pose: x={obj_init[0]:.4f} y={obj_init[1]:.4f} theta={obj_init[2]:.2f}°")
        print(f"goal_pose: ({goal_init[0]:.4f}, {goal_init[1]:.4f})")
        print(f"distance_ee_to_object_center: {dist_ee_to_obj:.4f}m ({dist_ee_to_obj*100:.1f}cm)")

        # Approach object until contact
        print(f"\n  Approaching object...")
        contacted = approach_object(env, max_steps=200)
        print(f"  Contact achieved: {contacted}")

        if not contacted:
            print("  ⚠️  Could not reach object, skipping push step")
            continue

        # Measure pre-contact state
        ee_pre = env.get_ee_pos()
        obj_pre = env.get_object_pose()
        contact_gap = float(np.linalg.norm(obj_pre[:2] - ee_pre[:2]))
        print(f"\n  estimated_contact_gap: {contact_gap:.4f}m ({contact_gap*100:.1f}cm)")

        # Execute one push step with action toward goal
        print(f"\n  Executing one push step (toward goal, {DT}s)...")
        result = probe_step(env, mass)

        # Print results
        print(f"\n  {'─'*40}")
        print(f"  Results for {mass*1000:.0f}g object:")
        print(f"  {'─'*40}")
        print(f"  object_pose_before:   x={obj_pre[0]:.4f} y={obj_pre[1]:.4f} theta={obj_pre[2]:.2f}°")
        print(f"  object_pose_after:    x={result['object_pose_after'][0]:.4f} y={result['object_pose_after'][1]:.4f} theta={result['object_pose_after'][2]:.2f}°")
        print(f"  object_displacement_m: {result['object_displacement_m']:.6f}m ({result['object_displacement_m']*1000:.2f}mm)")
        print(f"  object_avg_speed_mps:  {result['object_avg_speed_mps']:.4f} m/s ({result['object_avg_speed_mps']*100:.1f} cm/s)")
        print(f"  pusher_displacement_m: {result['pusher_displacement_m']:.6f}m ({result['pusher_displacement_m']*1000:.2f}mm)")
        print(f"  pusher_avg_speed_mps:  {result['pusher_avg_speed_mps']:.4f} m/s ({result['pusher_avg_speed_mps']*100:.1f} cm/s)")
        print(f"  contact_before:  {result['contact_before']}")
        print(f"  contact_after:   {result['contact_after']}")
        print(f"  collision_after: {result['collision_after']}")

        # Physics interpretation
        if result['object_displacement_m'] > 0:
            vel_ratio = result['object_avg_speed_mps'] / MAX_SPEED_MPS
            print(f"\n  Physics interpretation:")
            print(f"    Object speed / Pusher target speed = {vel_ratio:.3f}")
            print(f"    Effective mass ratio = {mass/(0.05+mass):.3f}")
            print(f"    Expected velocity (perfectly inelastic) = {MAX_SPEED_MPS * 0.05/(0.05+mass):.3f} m/s")

        del env

    print(f"\n{'='*80}")
    print(f"DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
