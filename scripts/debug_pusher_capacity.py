#!/usr/bin/env python3
"""
Diagnostic script to test pusher physical capacity without MPC/CEM.

Tests whether the current MuJoCo pusher has sufficient control authority
and pushing capability under different max_speed settings.

Does NOT modify:
- MujocoPushEnv default XML
- Cost functions
- CEM/MPC planners
- Reset templates

Only tests raw pusher control capacity with direct commanded actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.mujoco_push_env import MujocoPushEnv
from src.interventions.reset_template_loader import load_reset_templates


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a 2D vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def evaluate_pusher_capacity(
    template: dict[str, Any],
    max_speed_mps: float,
    pre_contact_offset: float = 0.06,
    pre_contact_threshold: float = 0.005,
    phase_a_max_steps: int = 100,
    phase_b_steps: int = 300,
    success_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Test pusher capacity with direct commanded actions.

    Phase A: Move pusher to pre-contact position
    Phase B: Push object toward goal with constant commanded velocity

    Args:
        template: Reset template
        max_speed_mps: Maximum pusher speed
        pre_contact_offset: Distance behind object to start pushing (m)
        pre_contact_threshold: Distance threshold to consider pre-contact reached (m)
        phase_a_max_steps: Maximum steps for Phase A before timeout
        phase_b_steps: Number of steps to execute in Phase B
        success_threshold: Success distance threshold (m)

    Returns:
        Dictionary with diagnostic metrics
    """
    # Create environment with specified max_speed
    env = MujocoPushEnv(
        control_dt=0.1,
        max_speed_mps=max_speed_mps,
    )
    env.reset_from_template(template)

    # Get initial poses
    initial_object_pose = env.get_object_pose()
    initial_ee_pos = env.get_ee_pos()
    goal_pose = env.get_goal_pose()

    initial_dist = float(np.linalg.norm(initial_object_pose[:2] - goal_pose[:2]))

    # Calculate push direction (from object to goal)
    push_dir = normalize_vector(goal_pose[:2] - initial_object_pose[:2])

    # Calculate pre-contact position (behind object)
    pre_contact_pos = initial_object_pose[:2] - push_dir * pre_contact_offset

    # Tracking variables
    first_contact_step_global = -1
    first_contact_step_in_push_phase = -1
    time_to_first_contact = -1.0
    phase_a_actual_steps = 0
    phase_b_actual_steps = 0
    approach_timeout = False

    contact_steps = []
    object_displacements = []
    pusher_speeds_phase_a = []
    commanded_speeds_phase_a = []
    pusher_speeds_phase_b = []
    commanded_speeds_phase_b = []
    object_poses_log = [initial_object_pose.copy()]
    ee_positions_log = [initial_ee_pos.copy()]
    ee_object_distances = []
    ee_to_pre_contact_distances = []

    current_phase = "A"  # A: approach, B: push
    phase_b_start_step = -1

    # Phase A: Move to pre-contact position
    for step in range(phase_a_max_steps):
        current_ee_pos = env.get_ee_pos()
        current_object_pose = env.get_object_pose()
        contact_flag = env.get_contact_flag()

        # Record contact (for metrics only, not phase transition)
        if contact_flag:
            contact_steps.append(step)
            if first_contact_step_global < 0:
                first_contact_step_global = step
                time_to_first_contact = step * env.control_dt

        # Calculate distances
        ee_to_pre_contact_dist = np.linalg.norm(current_ee_pos - pre_contact_pos)
        ee_to_object_dist = np.linalg.norm(current_ee_pos - current_object_pose[:2])
        ee_to_pre_contact_distances.append(ee_to_pre_contact_dist)
        ee_object_distances.append(ee_to_object_dist)

        # Phase transition: switch to push phase when close to pre-contact position
        if ee_to_pre_contact_dist < pre_contact_threshold:
            phase_a_actual_steps = step
            phase_b_start_step = step
            break

        # Phase A: Move toward pre-contact position
        direction = normalize_vector(pre_contact_pos - current_ee_pos)
        action = direction  # Full speed in direction

        # Execute action
        commanded_speed = np.linalg.norm(action) * max_speed_mps
        commanded_speeds_phase_a.append(commanded_speed)

        env.step(action)

        # Record state after step
        new_ee_pos = env.get_ee_pos()
        new_object_pose = env.get_object_pose()

        ee_positions_log.append(new_ee_pos.copy())
        object_poses_log.append(new_object_pose.copy())

        # Calculate actual pusher speed
        pusher_displacement = np.linalg.norm(new_ee_pos - current_ee_pos)
        pusher_speed = pusher_displacement / env.control_dt
        pusher_speeds_phase_a.append(pusher_speed)

        # Calculate object displacement
        object_displacement = np.linalg.norm(
            new_object_pose[:2] - current_object_pose[:2]
        )
        object_displacements.append(object_displacement)
    else:
        # Phase A timeout
        phase_a_actual_steps = phase_a_max_steps
        phase_b_start_step = phase_a_max_steps
        approach_timeout = True

    # Phase B: Push along push direction
    for step in range(phase_b_steps):
        global_step = phase_b_start_step + step
        current_ee_pos = env.get_ee_pos()
        current_object_pose = env.get_object_pose()
        contact_flag = env.get_contact_flag()

        # Record contact
        if contact_flag:
            contact_steps.append(global_step)
            if first_contact_step_global < 0:
                first_contact_step_global = global_step
                time_to_first_contact = global_step * env.control_dt
            if first_contact_step_in_push_phase < 0:
                first_contact_step_in_push_phase = step

        # Calculate distances
        ee_to_pre_contact_dist = np.linalg.norm(current_ee_pos - pre_contact_pos)
        ee_to_object_dist = np.linalg.norm(current_ee_pos - current_object_pose[:2])
        ee_to_pre_contact_distances.append(ee_to_pre_contact_dist)
        ee_object_distances.append(ee_to_object_dist)

        # Phase B: Push along push_dir
        action = push_dir  # Full speed in push direction

        # Execute action
        commanded_speed = np.linalg.norm(action) * max_speed_mps
        commanded_speeds_phase_b.append(commanded_speed)

        env.step(action)

        # Record state after step
        new_ee_pos = env.get_ee_pos()
        new_object_pose = env.get_object_pose()

        ee_positions_log.append(new_ee_pos.copy())
        object_poses_log.append(new_object_pose.copy())

        # Calculate actual pusher speed
        pusher_displacement = np.linalg.norm(new_ee_pos - current_ee_pos)
        pusher_speed = pusher_displacement / env.control_dt
        pusher_speeds_phase_b.append(pusher_speed)

        # Calculate object displacement
        object_displacement = np.linalg.norm(
            new_object_pose[:2] - current_object_pose[:2]
        )
        object_displacements.append(object_displacement)

        # Check success
        current_dist = np.linalg.norm(new_object_pose[:2] - goal_pose[:2])
        if current_dist < success_threshold:
            phase_b_actual_steps = step + 1
            break
    else:
        phase_b_actual_steps = phase_b_steps

    # Final metrics
    final_object_pose = env.get_object_pose()
    final_ee_pos = env.get_ee_pos()
    final_dist = float(np.linalg.norm(final_object_pose[:2] - goal_pose[:2]))
    dist_delta = initial_dist - final_dist

    total_object_displacement = float(
        np.linalg.norm(final_object_pose[:2] - initial_object_pose[:2])
    )

    # Phase A metrics
    mean_pusher_speed_phase_a = (
        float(np.mean(pusher_speeds_phase_a)) if pusher_speeds_phase_a else 0.0
    )
    mean_commanded_speed_phase_a = (
        float(np.mean(commanded_speeds_phase_a)) if commanded_speeds_phase_a else 0.0
    )
    pusher_tracking_ratio_phase_a = (
        mean_pusher_speed_phase_a / mean_commanded_speed_phase_a
        if mean_commanded_speed_phase_a > 0
        else 0.0
    )

    # Phase B metrics
    mean_pusher_speed_phase_b = (
        float(np.mean(pusher_speeds_phase_b)) if pusher_speeds_phase_b else 0.0
    )
    mean_commanded_speed_phase_b = (
        float(np.mean(commanded_speeds_phase_b)) if commanded_speeds_phase_b else 0.0
    )
    pusher_tracking_ratio_phase_b = (
        mean_pusher_speed_phase_b / mean_commanded_speed_phase_b
        if mean_commanded_speed_phase_b > 0
        else 0.0
    )

    # Distance metrics
    final_ee_to_pre_contact_dist = float(
        ee_to_pre_contact_distances[-1] if ee_to_pre_contact_distances else 0.0
    )
    min_ee_object_dist = float(np.min(ee_object_distances)) if ee_object_distances else 0.0

    # Contact metrics
    contact_rate = (
        len(contact_steps) / (phase_a_actual_steps + phase_b_actual_steps)
        if (phase_a_actual_steps + phase_b_actual_steps) > 0
        else 0.0
    )

    success = final_dist < success_threshold

    return {
        "reset_template_id": template["reset_template_id"],
        "split": template["split"],
        "layout_family": template["layout_family"],
        "shape_family": template["shape_family"],
        "max_speed_mps": max_speed_mps,
        "initial_dist": initial_dist,
        "final_dist": final_dist,
        "dist_delta": dist_delta,
        "success": success,
        "phase_a_steps": phase_a_actual_steps,
        "phase_b_steps": phase_b_actual_steps,
        "total_steps": phase_a_actual_steps + phase_b_actual_steps,
        "approach_timeout": approach_timeout,
        "final_ee_to_pre_contact_dist": final_ee_to_pre_contact_dist,
        "min_ee_object_dist": min_ee_object_dist,
        "first_contact_step_global": first_contact_step_global,
        "first_contact_step_in_push_phase": first_contact_step_in_push_phase,
        "time_to_first_contact": time_to_first_contact,
        "contact_rate": contact_rate,
        "contact_steps_count": len(contact_steps),
        "total_object_displacement": total_object_displacement,
        "mean_pusher_speed_phase_a": mean_pusher_speed_phase_a,
        "mean_commanded_speed_phase_a": mean_commanded_speed_phase_a,
        "pusher_tracking_ratio_phase_a": pusher_tracking_ratio_phase_a,
        "mean_pusher_speed_phase_b": mean_pusher_speed_phase_b,
        "mean_commanded_speed_phase_b": mean_commanded_speed_phase_b,
        "pusher_tracking_ratio_phase_b": pusher_tracking_ratio_phase_b,
        "final_object_pose": final_object_pose.tolist(),
        "final_ee_pos": final_ee_pos.tolist(),
    }


def main() -> None:
    # Load templates
    template_path = Path("data/sim/metadata/reset_templates_v0.json")
    if not template_path.exists():
        raise FileNotFoundError(
            f"Reset template file not found: {template_path}\n"
            "Generate it first with:\n"
            "  PYTHONPATH=. python scripts/generate_reset_templates.py"
        )

    templates = load_reset_templates(template_path)

    # Select train_sim_id template 0
    train_templates = [t for t in templates if t["split"] == "train_sim_id"]
    if not train_templates:
        raise ValueError("No train_sim_id templates found")

    template = train_templates[0]

    print(f"Testing template: {template['reset_template_id']}")
    print(f"Layout: {template['layout_family']}")
    print(f"Shape: {template['shape_family']}")
    print()

    # Sweep max_speed_mps
    max_speed_sweep = [0.05, 0.10, 0.15, 0.20]

    results = []
    for max_speed in max_speed_sweep:
        print(f"Testing max_speed_mps = {max_speed:.2f} m/s ({max_speed*100:.0f} cm/s)...")
        result = evaluate_pusher_capacity(
            template=template,
            max_speed_mps=max_speed,
            pre_contact_offset=0.06,
            pre_contact_threshold=0.005,
            phase_a_max_steps=100,
            phase_b_steps=300,
            success_threshold=0.05,
        )
        results.append(result)
        print(f"  Phase A steps: {result['phase_a_steps']}")
        print(f"  Phase B steps: {result['phase_b_steps']}")
        print(f"  Approach timeout: {result['approach_timeout']}")
        print(f"  Final dist: {result['final_dist']:.4f} m")
        print(f"  Dist delta: {result['dist_delta']:.4f} m")
        print(f"  Success: {result['success']}")
        print(f"  First contact (global): step {result['first_contact_step_global']}")
        print(f"  First contact (push phase): step {result['first_contact_step_in_push_phase']}")
        print(f"  Contact rate: {result['contact_rate']*100:.1f}%")
        print(f"  Pusher tracking Phase A: {result['pusher_tracking_ratio_phase_a']*100:.1f}%")
        print(f"  Pusher tracking Phase B: {result['pusher_tracking_ratio_phase_b']*100:.1f}%")
        print()

    # Save results
    output_dir = Path("runs/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pusher_capacity.json"

    report = {
        "template": {
            "reset_template_id": template["reset_template_id"],
            "split": template["split"],
            "layout_family": template["layout_family"],
            "shape_family": template["shape_family"],
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

    # Print summary table
    print("=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    print(
        f"{'Speed':<10} {'Phase A':<10} {'Phase B':<10} {'Timeout':<10} "
        f"{'Final Dist':<12} {'Success':<10} {'Contact':<12} {'Track A%':<12} {'Track B%':<12}"
    )
    print(
        f"{'(cm/s)':<10} {'steps':<10} {'steps':<10} {'':<10} "
        f"{'(m)':<12} {'':<10} {'(push)':<12} {'':<12} {'':<12}"
    )
    print("-" * 120)

    for result in results:
        speed_cms = result["max_speed_mps"] * 100
        phase_a = result["phase_a_steps"]
        phase_b = result["phase_b_steps"]
        timeout = "Yes" if result["approach_timeout"] else "No"
        final_dist = result["final_dist"]
        success = "Yes" if result["success"] else "No"
        contact_push = result["first_contact_step_in_push_phase"]
        contact_str = str(contact_push) if contact_push >= 0 else "None"
        track_a = result["pusher_tracking_ratio_phase_a"] * 100
        track_b = result["pusher_tracking_ratio_phase_b"] * 100

        print(
            f"{speed_cms:<10.0f} {phase_a:<10} {phase_b:<10} {timeout:<10} "
            f"{final_dist:<12.4f} {success:<10} {contact_str:<12} {track_a:<12.1f} {track_b:<12.1f}"
        )

    print("=" * 120)


if __name__ == "__main__":
    main()
