#!/usr/bin/env python3
"""
Diagnose why MuJoCo Oracle MPC rollout doesn't move the object.

This script:
1. Loads one reset template
2. Runs CEM-MPC planning
3. Prints detailed rollout information to diagnose the issue
"""

import numpy as np
from pathlib import Path

from src.interventions.reset_template_loader import load_reset_templates
from src.envs.mujoco_push_env import MujocoPushEnv
from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import CostWeights
from src.planners.mujoco_oracle_rollout import (
    mujoco_oracle_rollout_cost,
    rollout_action_sequence_mujoco,
)


def main():
    print("=" * 70)
    print("MuJoCo Oracle MPC Rollout Diagnosis")
    print("=" * 70)
    print()

    # Load one template
    template_path = Path("data/sim/metadata/reset_templates_v0.json")
    templates = load_reset_templates(template_path)
    template = templates[0]

    print(f"Template ID: {template['reset_template_id']}")
    print(f"Split: {template['split']}")
    print(f"Layout family: {template['layout_family']}")
    print(f"Shape family: {template['shape_family']}")
    print()

    # Create environment
    env = MujocoPushEnv(
        control_dt=0.1,
        max_speed_mps=0.05,
    )
    env.reset_from_template(template)

    initial_object_pose = env.get_object_pose()
    initial_ee_pos = env.get_ee_pos()
    initial_goal_pose = env.get_goal_pose()

    print("Initial state:")
    print(f"  Object pose: {initial_object_pose}")
    print(f"  EE position: {initial_ee_pos}")
    print(f"  Goal pose: {initial_goal_pose}")
    print(f"  Object-goal distance: {np.linalg.norm(initial_object_pose[:2] - initial_goal_pose[:2]):.4f} m")
    print(f"  EE-object distance: {np.linalg.norm(initial_ee_pos - initial_object_pose[:2]):.4f} m")
    print()

    # Setup CEM-MPC
    horizon = 80
    weights = CostWeights(
        w_pos=50.0,
        w_theta=2.0,
        w_reach=5.0,
        w_no_contact=2.0,
        w_push_alignment=1.0,
        w_collision=20.0,
        w_action=0.05,
        w_smooth=0.1,
        w_subgoal=0.0,
    )

    # Evaluate zero action
    zero_actions = np.zeros((horizon, 2), dtype=np.float64)
    zero_rollout = rollout_action_sequence_mujoco(
        env=env,
        action_sequence=zero_actions,
        restore_state=True,
    )
    zero_cost = mujoco_oracle_rollout_cost(
        env=env,
        action_sequence=zero_actions,
        weights=weights,
        restore_state=True,
    )

    print("Zero action baseline:")
    print(f"  Zero cost: {zero_cost:.4f}")
    print(f"  Object displacement: {np.linalg.norm(zero_rollout.predicted_object_poses[-1, :2] - zero_rollout.predicted_object_poses[0, :2]):.6f} m")
    print(f"  Max contact: {np.max(zero_rollout.contact_flags):.2f}")
    print()

    # Run CEM-MPC
    planner = CEMMPC(
        horizon=horizon,
        action_dim=2,
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
        return mujoco_oracle_rollout_cost(
            env=env,
            action_sequence=action_sequence,
            weights=weights,
            restore_state=True,
        )

    print("Running CEM-MPC...")
    first_action, cem_result = planner.plan(cost_fn)
    print(f"  CEM iterations: {len(cem_result.cost_history)}")
    print(f"  Cost history: {[f'{c:.4f}' for c in cem_result.cost_history]}")
    print(f"  Best cost: {cem_result.best_cost:.4f}")
    print(f"  First action: {first_action}")
    print()

    # Evaluate planned action
    planned_rollout = rollout_action_sequence_mujoco(
        env=env,
        action_sequence=cem_result.action_sequence,
        restore_state=True,
    )

    print("Planned rollout analysis:")
    print(f"  Planned cost: {cem_result.best_cost:.4f}")
    print(f"  Cost improvement: {zero_cost - cem_result.best_cost:.6f}")
    print()

    # Analyze object movement
    object_displacement = np.linalg.norm(
        planned_rollout.predicted_object_poses[-1, :2] - planned_rollout.predicted_object_poses[0, :2]
    )
    print(f"  Object displacement: {object_displacement:.6f} m")
    print(f"  Object start: {planned_rollout.predicted_object_poses[0, :2]}")
    print(f"  Object end: {planned_rollout.predicted_object_poses[-1, :2]}")
    print()

    # Analyze distances
    distances_to_goal = np.linalg.norm(
        planned_rollout.predicted_object_poses[:, :2] - initial_goal_pose[:2],
        axis=1,
    )
    best_min_dist = float(np.min(distances_to_goal))
    final_dist = float(distances_to_goal[-1])

    print(f"  Initial distance to goal: {distances_to_goal[0]:.4f} m")
    print(f"  Best min distance to goal: {best_min_dist:.4f} m")
    print(f"  Final distance to goal: {final_dist:.4f} m")
    print(f"  Distance improvement: {distances_to_goal[0] - best_min_dist:.6f} m")
    print()

    # Analyze contact
    max_contact = float(np.max(planned_rollout.contact_flags))
    contact_steps = np.sum(planned_rollout.contact_flags > 0.5)
    print(f"  Max contact: {max_contact:.2f}")
    print(f"  Contact steps: {contact_steps}/{horizon}")
    print()

    # Analyze EE movement
    ee_displacement = np.linalg.norm(
        planned_rollout.ee_positions[-1] - planned_rollout.ee_positions[0]
    )
    print(f"  EE displacement: {ee_displacement:.4f} m")
    print(f"  EE start: {planned_rollout.ee_positions[0]}")
    print(f"  EE end: {planned_rollout.ee_positions[-1]}")
    print()

    # Analyze action sequence
    action_norms = np.linalg.norm(cem_result.action_sequence, axis=1)
    print(f"  Action norms (first 10): {action_norms[:10]}")
    print(f"  Mean action norm: {np.mean(action_norms):.4f}")
    print(f"  Max action norm: {np.max(action_norms):.4f}")
    print()

    # Diagnosis
    print("=" * 70)
    print("Diagnosis:")
    print("=" * 70)

    if object_displacement < 0.001:
        print("❌ Object barely moved (< 1mm)")
        print()
        print("Possible causes:")
        print("  1. CEM found actions that reduce cost without pushing")
        print("     (e.g., only reach/action shaping, no actual contact)")
        print("  2. Horizon too short to reach and push the object")
        print("  3. Cost weights favor reaching over pushing")
        print("  4. Initial EE-object distance too large")
        print()

        if max_contact < 0.5:
            print("  → No contact detected! CEM didn't find pushing actions.")
            print("    Check: w_reach, w_no_contact, initial EE-object distance")
        else:
            print(f"  → Contact detected ({contact_steps} steps), but object didn't move.")
            print("    Check: object mass, friction, pusher force")

    elif object_displacement < 0.01:
        print("⚠️  Object moved slightly (< 1cm)")
        print("  → CEM found some pushing, but not enough to reach goal")
        print("    Check: horizon length, cost weights (w_pos)")

    else:
        print("✓ Object moved significantly")
        if best_min_dist < distances_to_goal[0] - 0.01:
            print("✓ Distance to goal improved")
        else:
            print("⚠️  Object moved but distance didn't improve much")
            print("    Check: push direction, cost weights")


if __name__ == "__main__":
    main()
