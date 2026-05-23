from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.mujoco_push_env import MujocoPushEnv
from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import CostWeights, collision_cost, wrap_angle
from src.planners.mujoco_oracle_rollout import (
    mujoco_oracle_rollout_cost,
    rollout_action_sequence_mujoco,
)


def make_default_mujoco_cost_weights() -> CostWeights:
    """
    Cost weights for MuJoCo oracle-MPC capacity smoke test.

    These are initial weights for the MuJoCo scaffold.
    They may be tuned later for real layout-OOD experiments.
    """
    return CostWeights(
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


def compute_pose_success_metrics(
    final_pos_error: float, final_theta_error_deg: float
) -> dict[str, bool]:
    """Compute unified pose-level success metrics for paper reporting.

    All thresholds use <= (inclusive).
    Primary success = success_pose_1cm_5deg (pos<=1cm AND theta<=5deg).
    """
    # Position-only thresholds
    success_pos_5cm = bool(final_pos_error <= 0.05)
    success_pos_3cm = bool(final_pos_error <= 0.03)
    success_pos_2cm = bool(final_pos_error <= 0.02)
    success_pos_1p5cm = bool(final_pos_error <= 0.015)
    success_pos_1cm = bool(final_pos_error <= 0.01)
    success_pos_0p5cm = bool(final_pos_error <= 0.005)
    success_pos_0p15cm = bool(final_pos_error <= 0.0015)

    # Theta-only thresholds
    success_theta_15deg = bool(final_theta_error_deg <= 15.0)
    success_theta_10deg = bool(final_theta_error_deg <= 10.0)
    success_theta_5deg = bool(final_theta_error_deg <= 5.0)
    success_theta_3deg = bool(final_theta_error_deg <= 3.0)

    # Combined pose thresholds
    success_pose_5cm_15deg = bool(success_pos_5cm and success_theta_15deg)
    success_pose_3cm_15deg = bool(success_pos_3cm and success_theta_15deg)
    success_pose_2cm_15deg = bool(success_pos_2cm and success_theta_15deg)
    success_pose_2cm_10deg = bool(success_pos_2cm and success_theta_10deg)
    success_pose_1cm_15deg = bool(success_pos_1cm and success_theta_15deg)
    success_pose_1p5cm_10deg = bool(success_pos_1p5cm and success_theta_10deg)
    success_pose_1cm_10deg = bool(success_pos_1cm and success_theta_10deg)
    success_pose_1cm_5deg = bool(success_pos_1cm and success_theta_5deg)
    success_pose_0p5cm_5deg = bool(success_pos_0p5cm and success_theta_5deg)
    success_pose_0p15cm_3deg = bool(success_pos_0p15cm and success_theta_3deg)

    return {
        # Position-only
        "success_pos_5cm": success_pos_5cm,
        "success_pos_3cm": success_pos_3cm,
        "success_pos_2cm": success_pos_2cm,
        "success_pos_1p5cm": success_pos_1p5cm,
        "success_pos_1cm": success_pos_1cm,
        "success_pos_0p5cm": success_pos_0p5cm,
        "success_pos_0p15cm": success_pos_0p15cm,
        # Theta-only
        "success_theta_15deg": success_theta_15deg,
        "success_theta_10deg": success_theta_10deg,
        "success_theta_5deg": success_theta_5deg,
        "success_theta_3deg": success_theta_3deg,
        # Combined pose
        "success_pose_5cm_15deg": success_pose_5cm_15deg,
        "success_pose_3cm_15deg": success_pose_3cm_15deg,
        "success_pose_2cm_15deg": success_pose_2cm_15deg,
        "success_pose_2cm_10deg": success_pose_2cm_10deg,
        "success_pose_1cm_15deg": success_pose_1cm_15deg,
        "success_pose_1p5cm_10deg": success_pose_1p5cm_10deg,
        "success_pose_1cm_10deg": success_pose_1cm_10deg,
        "success_pose_1cm_5deg": success_pose_1cm_5deg,
        "success_pose_0p5cm_5deg": success_pose_0p5cm_5deg,
        "success_pose_0p15cm_3deg": success_pose_0p15cm_3deg,
        # Semantic aliases (paper naming)
        "primary_success": success_pose_1cm_5deg,
        "coarse_success": success_pose_2cm_15deg,
        "precision_success": success_pose_0p5cm_5deg,
        "strict_completion": success_pose_0p15cm_3deg,
        "legacy_pos_5cm": success_pos_5cm,
    }


def evaluate_one_template_mujoco_oracle_mpc(
    template: dict[str, Any],
    horizon: int = 80,
    num_samples: int = 1536,
    num_elites: int = 128,
    num_iterations: int = 7,
    seed: int = 42,
    success_dist_threshold: float = 0.05,
    pusher_radius: float = 0.010,
    pusher_halfheight: float = 0.014,
    pusher_z: float = 0.016,
) -> dict[str, Any]:
    """
    Evaluate one reset template with MujocoPushEnv + oracle rollout + CEM.

    Important:
        v0.1 MujocoPushEnv does not yet instantiate obstacles from templates.
        Therefore this is not yet a full layout-OOD planner-capacity result.
        It verifies that MuJoCo oracle rollout + CEM-MPC interface works.
    """
    env = MujocoPushEnv(
        control_dt=0.1,
        max_speed_mps=0.05,
        pusher_radius=pusher_radius,
        pusher_halfheight=pusher_halfheight,
        pusher_z=pusher_z,
    )
    env.reset_from_template(template)

    initial_object_pose = env.get_object_pose()
    initial_ee_pos = env.get_ee_pos()
    initial_goal_pose = env.get_goal_pose()
    initial_dist = float(
        np.linalg.norm(initial_object_pose[:2] - initial_goal_pose[:2])
    )

    initial_state = env.clone_state()

    weights = make_default_mujoco_cost_weights()

    # Evaluate zero action baseline
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

    # Compute zero action distances
    zero_distances_to_goal = np.linalg.norm(
        zero_rollout.predicted_object_poses[:, :2] - initial_goal_pose[:2],
        axis=1,
    )
    zero_min_dist = float(np.min(zero_distances_to_goal))
    zero_final_dist = float(zero_distances_to_goal[-1])

    planner = CEMMPC(
        horizon=horizon,
        action_dim=2,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
        init_std=0.8,
        smoothing=0.2,
        seed=seed,
    )

    def cost_fn(action_sequence: np.ndarray) -> float:
        return mujoco_oracle_rollout_cost(
            env=env,
            action_sequence=action_sequence,
            weights=weights,
            restore_state=True,
        )

    first_action, cem_result = planner.plan(cost_fn)

    planned_rollout = rollout_action_sequence_mujoco(
        env=env,
        action_sequence=cem_result.action_sequence,
        restore_state=True,
    )

    # Compute planned action distances
    planned_distances_to_goal = np.linalg.norm(
        planned_rollout.predicted_object_poses[:, :2] - initial_goal_pose[:2],
        axis=1,
    )
    planned_min_dist = float(np.min(planned_distances_to_goal))
    planned_final_dist = float(planned_distances_to_goal[-1])

    # Legacy names for backward compatibility
    best_min_dist = planned_min_dist
    final_dist = planned_final_dist

    planned_cost = mujoco_oracle_rollout_cost(
        env=env,
        action_sequence=cem_result.action_sequence,
        weights=weights,
        restore_state=True,
    )

    # Contact and collision
    max_contact = float(np.max(planned_rollout.contact_flags))
    max_collision = float(np.max(planned_rollout.collision_flags))
    planned_collision_any, planned_collision_count, planned_collision_rate = collision_cost(
        planned_rollout.collision_flags
    )

    # Object displacement
    planned_object_pose_start = planned_rollout.predicted_object_poses[0].copy()
    planned_object_pose_final = planned_rollout.predicted_object_poses[-1].copy()
    object_displacement = float(
        np.linalg.norm(planned_object_pose_final[:2] - planned_object_pose_start[:2])
    )
    object_moved = bool(object_displacement > 1e-4)

    # EE positions
    ee_start = planned_rollout.ee_positions[0].copy()
    ee_final = planned_rollout.ee_positions[-1].copy()

    # Action statistics
    action_norms = np.linalg.norm(cem_result.action_sequence, axis=1)
    best_action_norm_mean = float(np.mean(action_norms))
    best_action_norm_max = float(np.max(action_norms))

    # Deltas (high precision)
    dist_delta = float(initial_dist - planned_min_dist)
    cost_delta = float(zero_cost - planned_cost)

    improved_cost = bool(planned_cost < zero_cost)
    improved_dist = bool(best_min_dist < initial_dist)
    success = bool(
        improved_cost
        and improved_dist
        and final_dist < success_dist_threshold
        and max_collision < 0.5
    )

    final_env_state = env.clone_state()
    restored_ok = bool(
        np.allclose(final_env_state.qpos, initial_state.qpos)
        and np.allclose(final_env_state.qvel, initial_state.qvel)
        and np.allclose(final_env_state.ctrl, initial_state.ctrl)
        and np.allclose(final_env_state.goal_pose, initial_state.goal_pose)
        and final_env_state.step_count == initial_state.step_count
    )

    return {
        "reset_template_id": template["reset_template_id"],
        "split": template["split"],
        "layout_family": template["layout_family"],
        "shape_family": template["shape_family"],
        "seed": template.get("seed"),
        # Distance metrics
        "initial_dist": initial_dist,
        "best_min_dist": best_min_dist,  # Legacy name
        "final_dist": final_dist,  # Legacy name
        "zero_min_dist": zero_min_dist,
        "zero_final_dist": zero_final_dist,
        "planned_min_dist": planned_min_dist,
        "planned_final_dist": planned_final_dist,
        "dist_delta": dist_delta,
        # Cost metrics
        "zero_cost": float(zero_cost),
        "planned_cost": float(planned_cost),
        "best_cost": float(cem_result.best_cost),
        "cost_delta": cost_delta,
        # Contact and collision
        "max_contact": max_contact,
        "max_collision": max_collision,
        "planned_collision_any": planned_collision_any,
        "planned_collision_count": planned_collision_count,
        "planned_collision_rate": planned_collision_rate,
        # Object movement
        "object_displacement": object_displacement,
        "object_moved": object_moved,
        "planned_object_pose_start": planned_object_pose_start.tolist(),
        "planned_object_pose_final": planned_object_pose_final.tolist(),
        # EE positions
        "ee_start": ee_start.tolist(),
        "ee_final": ee_final.tolist(),
        # Goal pose
        "goal_pose": initial_goal_pose.tolist(),
        # Action statistics
        "best_action_norm_mean": best_action_norm_mean,
        "best_action_norm_max": best_action_norm_max,
        "first_action": first_action.tolist(),
        "cost_history": [float(x) for x in cem_result.cost_history],
        # Status flags
        "improved_cost": improved_cost,
        "improved_dist": improved_dist,
        "restored_ok": restored_ok,
        "success": success,
        "success_definition": "open_loop_compound_legacy",
    }

    # Add unified pose success metrics
    pose_metrics = compute_pose_success_metrics(
        final_pos_error=final_dist,  # open-loop uses final_dist as pos error
        final_theta_error_deg=0.0,   # open-loop does not track theta separately
    )
    result.update(pose_metrics)
    return result


def run_mujoco_oracle_mpc_capacity(
    templates: list[dict[str, Any]],
    horizon: int = 80,
    num_samples: int = 1536,
    num_elites: int = 128,
    num_iterations: int = 7,
    seed: int = 42,
    success_dist_threshold: float = 0.05,
    pusher_radius: float = 0.010,
    pusher_halfheight: float = 0.014,
    pusher_z: float = 0.016,
) -> dict[str, Any]:
    """
    Run MuJoCo oracle-MPC capacity smoke test over templates.

    This is a smoke test for:
        reset template → MujocoPushEnv → oracle rollout → CEM-MPC

    v0.1 does not yet instantiate obstacles, so this is not yet a full
    layout-OOD planner-capacity result.
    """
    if not templates:
        raise ValueError(
            "No templates provided for MuJoCo oracle-MPC capacity check."
        )

    results = []

    for i, template in enumerate(templates):
        result = evaluate_one_template_mujoco_oracle_mpc(
            template=template,
            horizon=horizon,
            num_samples=num_samples,
            num_elites=num_elites,
            num_iterations=num_iterations,
            seed=seed + i,
            success_dist_threshold=success_dist_threshold,
            pusher_radius=pusher_radius,
            pusher_halfheight=pusher_halfheight,
            pusher_z=pusher_z,
        )
        results.append(result)

    num_templates = len(results)
    num_success = sum(1 for r in results if r["success"])
    num_improved_cost = sum(1 for r in results if r["improved_cost"])
    num_improved_dist = sum(1 for r in results if r["improved_dist"])
    num_restored_ok = sum(1 for r in results if r["restored_ok"])

    summary = {
        "num_templates": num_templates,
        "num_success": num_success,
        "success_rate": float(num_success / num_templates),
        "success_rate_definition": "open_loop_compound_legacy",
        "primary_success_rate": float(sum(1 for r in results if r["primary_success"]) / num_templates),
        "coarse_success_rate": float(sum(1 for r in results if r["coarse_success"]) / num_templates),
        "precision_success_rate": float(sum(1 for r in results if r["precision_success"]) / num_templates),
        "strict_completion_rate": float(sum(1 for r in results if r["strict_completion"]) / num_templates),
        "legacy_pos_5cm_rate": float(sum(1 for r in results if r["legacy_pos_5cm"]) / num_templates),
        "num_improved_cost": num_improved_cost,
        "num_improved_dist": num_improved_dist,
        "num_restored_ok": num_restored_ok,
        "mean_initial_dist": float(np.mean([r["initial_dist"] for r in results])),
        "mean_best_min_dist": float(np.mean([r["best_min_dist"] for r in results])),
        "mean_final_dist": float(np.mean([r["final_dist"] for r in results])),
        "mean_zero_cost": float(np.mean([r["zero_cost"] for r in results])),
        "mean_planned_cost": float(np.mean([r["planned_cost"] for r in results])),
    }

    return {
        "summary": summary,
        "results": results,
    }


def save_mujoco_oracle_mpc_report(
    report: dict[str, Any], path: str | Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def evaluate_one_template_mujoco_oracle_mpc_closed_loop(
    template: dict[str, Any],
    planning_horizon: int = 80,
    num_samples: int = 1536,
    num_elites: int = 128,
    num_iterations: int = 7,
    execute_steps: int = 5,
    max_mpc_steps: int = 40,
    seed: int = 42,
    success_dist_threshold: float = 0.05,
    pusher_radius: float = 0.010,
    pusher_halfheight: float = 0.014,
    pusher_z: float = 0.016,
    disable_early_stop: bool = False,
    success_pos_threshold: float = 0.05,
    success_theta_threshold_deg: float = 180.0,
    max_speed_mps: float = 0.05,
) -> dict[str, Any]:
    """
    Evaluate one reset template with closed-loop MuJoCo Oracle-MPC.

    Repeatedly plans and executes actions until success or max_mpc_steps.

    Args:
        template: Reset template dict
        planning_horizon: CEM planning horizon
        num_samples: CEM number of samples
        num_elites: CEM number of elites
        num_iterations: CEM number of iterations
        execute_steps: Number of steps to execute from planned sequence
        max_mpc_steps: Maximum number of MPC replanning steps
        seed: Random seed
        success_dist_threshold: Success distance threshold in meters
        pusher_radius: Pusher radius in meters
        pusher_halfheight: Pusher half-height for cylinder
        pusher_z: Pusher body z position in meters

    Returns:
        Result dict with per-template metrics
    """
    # Print template start
    print(f"\n{'='*80}")
    print(f"Template: {template['reset_template_id']}")
    print(f"  split={template['split']}, layout={template['layout_family']}, shape={template['shape_family']}")
    print(f"{'='*80}")

    env = MujocoPushEnv(
        control_dt=0.1,
        max_speed_mps=max_speed_mps,
        pusher_radius=pusher_radius,
        pusher_halfheight=pusher_halfheight,
        pusher_z=pusher_z,
    )
    env.reset_from_template(template)

    initial_object_pose = env.get_object_pose()
    initial_goal_pose = env.get_goal_pose()
    initial_dist = float(
        np.linalg.norm(initial_object_pose[:2] - initial_goal_pose[:2])
    )

    strict_pose_early_stop = (
        success_pos_threshold < success_dist_threshold
        or success_theta_threshold_deg < 180.0
    )
    strict_pose_stop_active = strict_pose_early_stop
    print(f"Initial distance to goal: {initial_dist:.4f} m")
    print(f"Success threshold (dist): {success_dist_threshold:.4f} m")
    print(f"Strict pose early stop: pos<={success_pos_threshold*1000:.1f}mm AND theta<={success_theta_threshold_deg:.1f}deg  (active={strict_pose_early_stop})")
    print(f"Planning horizon: {planning_horizon}, Execute steps: {execute_steps}, Max MPC steps: {max_mpc_steps}")

    weights = make_default_mujoco_cost_weights()

    # Track trajectory
    distances = [initial_dist]
    contact_flags = []
    collision_flags = []
    num_mpc_steps = 0
    total_executed_steps = 0
    success = False
    strict_pose_early_stop_triggered = False
    strict_pose_stop_step: int | None = None
    strict_pose_stop_pos_error: float | None = None
    strict_pose_stop_theta_error_deg: float | None = None
    should_stop = False
    legacy_success_reached = False
    best_dist = initial_dist
    mpc_step_logs = []

    # Threshold milestone tracking (position-only thresholds)
    position_thresholds = {
        "5cm": 0.05,
        "3cm": 0.03,
        "2cm": 0.02,
        "1p5cm": 0.015,
        "1cm": 0.01,
        "0p5cm": 0.005,
    }
    threshold_first_reach = {
        k: {
            "reached": False,
            "total_executed_steps": None,
            "mpc_step": None,
            "env_step_in_chunk": None,
            "pos_error": None,
            "theta_error_deg": None,
            "pose_cost": None,
        }
        for k in position_thresholds
    }
    threshold_post_reach_trace: dict[str, list] = {k: [] for k in position_thresholds}
    # Pose-level joint threshold: 1.5mm + 3deg
    first_reach_1p5mm_3deg: dict = {
        "reached": False,
        "total_executed_steps": None,
        "mpc_step": None,
        "pos_error": None,
        "theta_error_deg": None,
    }
    mpc_step_error_trace: list[dict] = []
    initial_theta_error_rad = float(abs(wrap_angle(initial_object_pose[2] - initial_goal_pose[2])))
    initial_theta_error_deg = float(np.rad2deg(initial_theta_error_rad))
    best_pos_error = initial_dist
    best_pose_cost = (initial_dist / 0.01) ** 2 + (initial_theta_error_deg / 5.0) ** 2
    best_theta_error_deg_at_best_pos = initial_theta_error_deg
    best_pose_step = 0

    for mpc_step in range(max_mpc_steps):
        num_mpc_steps += 1

        # Get current distance before planning
        current_object_pose = env.get_object_pose()
        current_dist = float(
            np.linalg.norm(current_object_pose[:2] - initial_goal_pose[:2])
        )

        # Print MPC step start
        print(f"\n--- MPC Step {num_mpc_steps}/{max_mpc_steps} ---")
        print(f"  current_dist: {current_dist:.4f} m")
        print(f"  best_dist_so_far: {best_dist:.4f} m")

        # Create a fresh planner for each MPC step to avoid state carryover
        planner = CEMMPC(
            horizon=planning_horizon,
            action_dim=2,
            num_samples=num_samples,
            num_elites=num_elites,
            num_iterations=num_iterations,
            action_low=[-1.0, -1.0],
            action_high=[1.0, 1.0],
            init_std=0.8,
            smoothing=0.2,
            seed=seed + mpc_step,  # Different seed for each MPC step
        )

        # Define cost function for current state
        def cost_fn(action_sequence: np.ndarray) -> float:
            return mujoco_oracle_rollout_cost(
                env=env,
                action_sequence=action_sequence,
                weights=weights,
                restore_state=True,
            )

        # Plan from current state
        first_action, cem_result = planner.plan(cost_fn)

        # Print CEM result
        print(f"  CEM best_cost: {cem_result.best_cost:.4f}")
        print(f"  Planned first action: [{first_action[0]:+.3f}, {first_action[1]:+.3f}]")

        # ========== Planned-vs-Actual Diagnostic ==========
        # Run planned rollout for diagnosis (with restore)
        planned_rollout = rollout_action_sequence_mujoco(
            env=env,
            action_sequence=cem_result.action_sequence,
            restore_state=True,
        )

        # Calculate planned metrics
        contact_mask = planned_rollout.contact_flags > 0.5
        planned_contact_first_step = int(np.argmax(contact_mask)) if np.any(contact_mask) else -1
        planned_contact_rate = float(np.mean(contact_mask))

        execute_idx = min(execute_steps, len(planned_rollout.predicted_object_poses) - 1)
        planned_contact_within_execute_steps = bool(np.any(contact_mask[:execute_idx + 1]))

        # Planned distances
        planned_distances = np.linalg.norm(
            planned_rollout.predicted_object_poses[:, :2] - initial_goal_pose[:2],
            axis=1,
        )
        planned_min_dist = float(np.min(planned_distances))
        planned_final_dist = float(planned_distances[-1])
        planned_dist_after_execute_steps = float(planned_distances[execute_idx])

        # Planned object displacement
        planned_object_displacement = float(
            np.linalg.norm(
                planned_rollout.predicted_object_poses[-1, :2] - planned_rollout.predicted_object_poses[0, :2]
            )
        )
        planned_object_displacement_after_execute_steps = float(
            np.linalg.norm(
                planned_rollout.predicted_object_poses[execute_idx, :2] - planned_rollout.predicted_object_poses[0, :2]
            )
        )

        # Planned EE-object distances
        planned_ee_object_dist_start = float(
            np.linalg.norm(
                planned_rollout.ee_positions[0] - planned_rollout.predicted_object_poses[0, :2]
            )
        )
        planned_ee_object_dist_after_execute_steps = float(
            np.linalg.norm(
                planned_rollout.ee_positions[execute_idx] - planned_rollout.predicted_object_poses[execute_idx, :2]
            )
        )

        # Planned contact steps (first 10)
        planned_contact_steps_list = np.where(contact_mask)[0].tolist()[:10]

        # Planned collision metrics
        planned_collision_any, planned_collision_count, planned_collision_rate = collision_cost(
            planned_rollout.collision_flags
        )

        # Print planned trajectory
        print(f"\n  Planned trajectory:")
        print(f"    first_contact_step: {planned_contact_first_step}")
        print(f"    contact_within_execute_steps: {planned_contact_within_execute_steps}")
        print(f"    contact_rate: {planned_contact_rate:.3f}")
        print(f"    collision_count: {planned_collision_count:.0f}  collision_rate: {planned_collision_rate:.3f}")
        print(f"    planned_dist_after_execute_steps: {planned_dist_after_execute_steps:.4f} m")
        print(f"    planned_object_displacement_after_execute_steps: {planned_object_displacement_after_execute_steps:.4f} m")
        print(f"    planned_contact_steps_first10: {planned_contact_steps_list}")

        # Diagnosis
        if planned_contact_first_step > execute_steps:
            print(f"    → Diagnosis: planned contact happens after executed prefix.")
        # ========== End Planned-vs-Actual Diagnostic ==========

        # Execute first execute_steps actions (without restore)
        actions_to_execute = cem_result.action_sequence[:execute_steps]

        # Track execution metrics
        exec_start_pose = env.get_object_pose()
        actual_contact_flags_this_round = []
        actual_collision_flags_this_round = []

        for env_step_in_chunk, action in enumerate(actions_to_execute, start=1):
            env.step(action)
            total_executed_steps += 1

            # Record state
            current_object_pose = env.get_object_pose()
            current_dist = float(
                np.linalg.norm(current_object_pose[:2] - initial_goal_pose[:2])
            )
            current_theta_error_rad = float(abs(wrap_angle(current_object_pose[2] - initial_goal_pose[2])))
            current_theta_error_deg = float(np.rad2deg(current_theta_error_rad))
            current_pose_cost = (current_dist / 0.01) ** 2 + (current_theta_error_deg / 5.0) ** 2
            distances.append(current_dist)
            contact_flag = env.get_contact_flag()
            collision_flag = env.get_collision_flag()
            contact_flags.append(contact_flag)
            collision_flags.append(collision_flag)
            actual_contact_flags_this_round.append(contact_flag)
            actual_collision_flags_this_round.append(collision_flag)

            # Update best distance
            if current_dist < best_dist:
                best_dist = current_dist

            # Per-step best_pos_error / best_pose_cost update
            if current_dist < best_pos_error:
                best_pos_error = current_dist
                best_theta_error_deg_at_best_pos = current_theta_error_deg
            if current_pose_cost < best_pose_cost:
                best_pose_cost = current_pose_cost
                best_pose_step = total_executed_steps

            # Per-step threshold first-reach check
            for tname, tval in position_thresholds.items():
                if not threshold_first_reach[tname]["reached"] and current_dist < tval:
                    _step_pose_cost = (current_dist / 0.01) ** 2 + (current_theta_error_deg / 5.0) ** 2
                    threshold_first_reach[tname].update({
                        "reached": True,
                        "total_executed_steps": total_executed_steps,
                        "mpc_step": num_mpc_steps,
                        "env_step_in_chunk": env_step_in_chunk,
                        "pos_error": current_dist,
                        "theta_error_deg": current_theta_error_deg,
                        "pose_cost": _step_pose_cost,
                    })

            # Per-step joint pose threshold: 1.5mm + 3deg
            if (
                not first_reach_1p5mm_3deg["reached"]
                and current_dist <= 0.0015
                and current_theta_error_deg <= 3.0
            ):
                first_reach_1p5mm_3deg.update({
                    "reached": True,
                    "total_executed_steps": total_executed_steps,
                    "mpc_step": num_mpc_steps,
                    "pos_error": current_dist,
                    "theta_error_deg": current_theta_error_deg,
                })

            # Check strict pose early stop (joint pos + theta condition)
            _strict_pos_ok = current_dist <= success_pos_threshold
            _strict_theta_ok = current_theta_error_deg <= success_theta_threshold_deg
            if _strict_pos_ok and _strict_theta_ok:
                success = True
                strict_pose_early_stop_triggered = True
                strict_pose_stop_step = total_executed_steps
                strict_pose_stop_pos_error = current_dist
                strict_pose_stop_theta_error_deg = current_theta_error_deg
                if not disable_early_stop:
                    should_stop = True
                    break
            elif current_dist <= success_dist_threshold:
                # Legacy 5cm success
                if not strict_pose_stop_active:
                    success = True
                    if not disable_early_stop:
                        should_stop = True
                        break
                else:
                    # strict_pose_stop_active=True: record as metric only, no early stop
                    legacy_success_reached = True

        # Calculate actual metrics
        exec_end_pose = env.get_object_pose()
        exec_displacement = float(
            np.linalg.norm(exec_end_pose[:2] - exec_start_pose[:2])
        )
        actual_contact_any = bool(np.any(np.array(actual_contact_flags_this_round) > 0.5))
        actual_contact_steps = [i for i, c in enumerate(actual_contact_flags_this_round) if c > 0.5][:10]
        actual_collision_any = bool(np.any(np.array(actual_collision_flags_this_round) > 0.5))
        actual_collision_count = float(np.sum(np.array(actual_collision_flags_this_round)))
        actual_collision_rate = float(np.mean(np.array(actual_collision_flags_this_round)))

        # Print execution result
        print(f"\n  After execution:")
        print(f"    actual_contact_any: {actual_contact_any}")
        print(f"    actual_contact_steps_first10: {actual_contact_steps}")
        print(f"    actual_collision_count: {actual_collision_count:.0f}  actual_collision_rate: {actual_collision_rate:.3f}")
        print(f"    current_dist: {current_dist:.4f} m")
        print(f"    object_displacement: {exec_displacement:.4f} m")

        # Diagnosis
        if planned_contact_within_execute_steps and not actual_contact_any:
            print(f"    → Diagnosis: planned/actual mismatch, inspect rollout/execute consistency.")

        # Save MPC step log
        mpc_step_log = {
            "mpc_step": num_mpc_steps,
            "planned_contact_first_step": planned_contact_first_step,
            "planned_contact_rate": planned_contact_rate,
            "planned_contact_within_execute_steps": planned_contact_within_execute_steps,
            "planned_collision_any": planned_collision_any,
            "planned_collision_count": planned_collision_count,
            "planned_collision_rate": planned_collision_rate,
            "planned_min_dist": planned_min_dist,
            "planned_final_dist": planned_final_dist,
            "planned_dist_after_execute_steps": planned_dist_after_execute_steps,
            "planned_object_displacement": planned_object_displacement,
            "planned_object_displacement_after_execute_steps": planned_object_displacement_after_execute_steps,
            "planned_ee_object_dist_start": planned_ee_object_dist_start,
            "planned_ee_object_dist_after_execute_steps": planned_ee_object_dist_after_execute_steps,
            "planned_contact_steps_list": planned_contact_steps_list,
            "actual_contact_any": actual_contact_any,
            "actual_contact_steps": actual_contact_steps,
            "actual_collision_any": actual_collision_any,
            "actual_collision_count": actual_collision_count,
            "actual_collision_rate": actual_collision_rate,
            "actual_current_dist": current_dist,
            "actual_object_displacement": exec_displacement,
        }
        mpc_step_logs.append(mpc_step_log)

        # Chunk-end pose metrics
        chunk_end_pose = env.get_object_pose()
        chunk_end_pos_error = float(np.linalg.norm(chunk_end_pose[:2] - initial_goal_pose[:2]))
        chunk_end_theta_error_rad = float(abs(wrap_angle(chunk_end_pose[2] - initial_goal_pose[2])))
        chunk_end_theta_error_deg = float(np.rad2deg(chunk_end_theta_error_rad))
        chunk_end_pose_cost = (chunk_end_pos_error / 0.01) ** 2 + (chunk_end_theta_error_deg / 5.0) ** 2
        contact_rate_this_chunk = (
            float(np.mean(np.array(actual_contact_flags_this_round) > 0.5))
            if actual_contact_flags_this_round else None
        )

        # Update best tracking
        if chunk_end_pos_error < best_pos_error:
            best_pos_error = chunk_end_pos_error
            best_theta_error_deg_at_best_pos = chunk_end_theta_error_deg
        if chunk_end_pose_cost < best_pose_cost:
            best_pose_cost = chunk_end_pose_cost
            best_pose_step = total_executed_steps

        # Append to mpc_step_error_trace
        mpc_step_error_trace.append({
            "mpc_step": num_mpc_steps,
            "total_executed_steps": total_executed_steps,
            "pos_error": chunk_end_pos_error,
            "theta_error_deg": chunk_end_theta_error_deg,
            "pose_cost": chunk_end_pose_cost,
            "best_pos_error_so_far": best_pos_error,
            "best_pose_cost_so_far": best_pose_cost,
            "contact_rate_this_chunk": contact_rate_this_chunk,
        })

        # Update post-reach traces (first-reach is now detected per env.step above)
        for tname in position_thresholds:
            if threshold_first_reach[tname]["reached"]:
                first_reach_pos = threshold_first_reach[tname]["pos_error"]
                first_reach_cost = threshold_first_reach[tname]["pose_cost"]
                threshold_post_reach_trace[tname].append({
                    "mpc_step": num_mpc_steps,
                    "total_executed_steps": total_executed_steps,
                    "pos_error": chunk_end_pos_error,
                    "theta_error_deg": chunk_end_theta_error_deg,
                    "pose_cost": chunk_end_pose_cost,
                    "delta_pos_error_from_first_reach": chunk_end_pos_error - first_reach_pos,
                    "delta_pose_cost_from_first_reach": chunk_end_pose_cost - first_reach_cost,
                    "best_pos_error_so_far": best_pos_error,
                    "best_pose_cost_so_far": best_pose_cost,
                })

        if should_stop:
            if strict_pose_early_stop_triggered:
                print(f"\n✓ STRICT POSE EARLY STOP at MPC step {num_mpc_steps}! (pos<={success_pos_threshold*1000:.1f}mm AND theta<={success_theta_threshold_deg:.1f}deg)")
            else:
                print(f"\n✓ SUCCESS at MPC step {num_mpc_steps}!")
            break
        elif success and disable_early_stop:
            print(f"\n  success threshold reached, but disable_early_stop=True; continuing full budget")

    # Print threshold first reach summary
    print(f"\nThreshold first reach:")
    for tname in position_thresholds:
        fr = threshold_first_reach[tname]
        if fr["reached"]:
            print(f"  {tname}: reached at step {fr['total_executed_steps']}, pos={fr['pos_error']:.4f}, theta={fr['theta_error_deg']:.2f}deg")
        else:
            print(f"  {tname}: not reached")
    print(f"\nPost-reach trend:")
    for tname in position_thresholds:
        trace = threshold_post_reach_trace[tname]
        if trace:
            final_delta = trace[-1]["delta_pos_error_from_first_reach"]
            best_after = min(t["pos_error"] for t in trace)
            print(f"  after {tname}: final_delta_pos={final_delta:+.4f}, best_after_reach={best_after:.4f}")
        else:
            print(f"  after {tname}: no post-reach data")

    final_object_pose = env.get_object_pose()
    final_goal_pose = env.get_goal_pose()
    final_dist = float(
        np.linalg.norm(final_object_pose[:2] - final_goal_pose[:2])
    )

    # Calculate pose-level metrics
    final_pos_error = float(
        np.linalg.norm(final_object_pose[:2] - final_goal_pose[:2])
    )
    final_theta_error_rad = float(
        abs(wrap_angle(final_object_pose[2] - final_goal_pose[2]))
    )
    final_theta_error_deg = float(np.rad2deg(final_theta_error_rad))

    # When disable_early_stop=True, success is defined by final state only
    if disable_early_stop:
        success = bool(final_pos_error < success_dist_threshold)

    # Unified pose success metrics (all thresholds use <=)
    pose_metrics = compute_pose_success_metrics(final_pos_error, final_theta_error_deg)

    # Paper primary success = success_pose_1cm_5deg
    # success is now primary_success for all closed-loop modes
    success = pose_metrics["primary_success"]

    # Calculate metrics
    total_object_displacement = float(
        np.linalg.norm(final_object_pose[:2] - initial_object_pose[:2])
    )
    dist_delta = float(initial_dist - best_dist)
    contact_rate = float(np.mean(contact_flags)) if contact_flags else 0.0
    collision_rate = float(np.mean(collision_flags)) if collision_flags else 0.0
    collision_count = float(np.sum(collision_flags)) if collision_flags else 0.0
    collision_any = bool(collision_count > 0.5)

    # Print final summary
    print(f"\n{'='*80}")
    print(f"Template {template['reset_template_id']} completed:")
    print(f"  Success: {success} (primary=success_pose_1cm_5deg)  strict_pose_early_stop_triggered: {strict_pose_early_stop_triggered}")
    print(f"  Initial dist: {initial_dist:.4f} m → Final dist: {final_dist:.4f} m")
    print(f"  Best dist: {best_dist:.4f} m (delta: {dist_delta:.4f} m)")
    print(f"  Final pose errors:")
    print(f"    position: {final_pos_error:.6f} m ({final_pos_error*1000:.2f} mm)")
    print(f"    theta: {final_theta_error_deg:.2f} deg")
    print(f"  Pose-level success:")
    print(f"    success_pose_2cm_15deg: {success_pose_2cm_15deg}")
    print(f"    success_pose_1cm_5deg:  {success_pose_1cm_5deg}")
    print(f"    success_pose_0p5cm_5deg:{success_pose_0p5cm_5deg}")
    print(f"    success_pose_0p15cm_3deg:{success_pose_0p15cm_3deg}")
    if first_reach_1p5mm_3deg["reached"]:
        print(f"  first_reach_1p5mm_3deg: step={first_reach_1p5mm_3deg['total_executed_steps']}, pos={first_reach_1p5mm_3deg['pos_error']*1000:.2f}mm, theta={first_reach_1p5mm_3deg['theta_error_deg']:.2f}deg")
    else:
        print(f"  first_reach_1p5mm_3deg: not reached")
    print(f"  Total MPC steps: {num_mpc_steps}/{max_mpc_steps}")
    print(f"  Total executed steps: {total_executed_steps}")
    print(f"  Object displacement: {total_object_displacement:.4f} m")
    print(f"  Contact rate: {contact_rate:.3f}")
    print(f"  Collision count: {collision_count:.0f}  Collision rate: {collision_rate:.3f}")
    print(f"{'='*80}\n")

    return {
        "reset_template_id": template["reset_template_id"],
        "split": template["split"],
        "layout_family": template["layout_family"],
        "shape_family": template["shape_family"],
        "seed": template.get("seed"),
        "initial_dist": initial_dist,
        "best_dist": best_dist,
        "final_dist": final_dist,
        "dist_delta": dist_delta,
        "total_object_displacement": total_object_displacement,
        "num_mpc_steps": num_mpc_steps,
        "total_executed_steps": total_executed_steps,
        "contact_rate": contact_rate,
        "collision_rate": collision_rate,
        "collision_count": collision_count,
        "collision_any": collision_any,
        "success": success,
        "restored_ok": True,  # Not applicable for closed-loop
        "mpc_step_logs": mpc_step_logs,
        # Pose-level metrics
        "final_object_pose": final_object_pose.tolist(),
        "final_goal_pose": final_goal_pose.tolist(),
        "final_pos_error": final_pos_error,
        "final_theta_error_rad": final_theta_error_rad,
        "final_theta_error_deg": final_theta_error_deg,
        # Unified pose success metrics (from compute_pose_success_metrics)
        **pose_metrics,
        "success_definition": "success_pose_1cm_5deg",
        # Strict pose early stop tracking
        "strict_pose_early_stop_triggered": strict_pose_early_stop_triggered,
        "strict_pose_success": strict_pose_early_stop_triggered,
        "strict_pose_stop_step": strict_pose_stop_step,
        "strict_pose_stop_pos_error": strict_pose_stop_pos_error,
        "strict_pose_stop_theta_error_deg": strict_pose_stop_theta_error_deg,
        # Continuous best/final pose cost
        "final_pose_cost": (final_pos_error / 0.01) ** 2 + (final_theta_error_deg / 5.0) ** 2,
        "best_pos_error": best_pos_error,
        "best_pose_cost": best_pose_cost,
        "best_pose_step": best_pose_step,
        "best_theta_error_deg_at_best_pos": best_theta_error_deg_at_best_pos,
        # Threshold milestone tracking
        "threshold_first_reach": threshold_first_reach,
        "threshold_post_reach_trace": threshold_post_reach_trace,
        "mpc_step_error_trace": mpc_step_error_trace,
        "first_reach_1p5mm_3deg": first_reach_1p5mm_3deg,
    }


def run_mujoco_oracle_mpc_closed_loop_capacity(
    templates: list[dict[str, Any]],
    planning_horizon: int = 80,
    num_samples: int = 1536,
    num_elites: int = 128,
    num_iterations: int = 7,
    execute_steps: int = 5,
    max_mpc_steps: int = 40,
    seed: int = 42,
    success_dist_threshold: float = 0.05,
    pusher_radius: float = 0.010,
    pusher_halfheight: float = 0.014,
    pusher_z: float = 0.016,
    disable_early_stop: bool = False,
    success_pos_threshold: float = 0.05,
    success_theta_threshold_deg: float = 180.0,
) -> dict[str, Any]:
    """
    Run closed-loop MuJoCo oracle-MPC capacity test over templates.

    This verifies whether MuJoCo true dynamics + CEM-MPC can complete
    push-to-pose tasks through repeated replanning.
    """
    if not templates:
        raise ValueError(
            "No templates provided for MuJoCo oracle-MPC closed-loop capacity check."
        )

    results = []

    for i, template in enumerate(templates):
        result = evaluate_one_template_mujoco_oracle_mpc_closed_loop(
            template=template,
            planning_horizon=planning_horizon,
            num_samples=num_samples,
            num_elites=num_elites,
            num_iterations=num_iterations,
            execute_steps=execute_steps,
            max_mpc_steps=max_mpc_steps,
            seed=seed + i,
            success_dist_threshold=success_dist_threshold,
            pusher_radius=pusher_radius,
            pusher_halfheight=pusher_halfheight,
            pusher_z=pusher_z,
            disable_early_stop=disable_early_stop,
            success_pos_threshold=success_pos_threshold,
            success_theta_threshold_deg=success_theta_threshold_deg,
        )
        results.append(result)

    num_templates = len(results)
    num_success = sum(1 for r in results if r["success"])
    num_success_pose_2cm_15deg = sum(1 for r in results if r["success_pose_2cm_15deg"])
    num_success_pose_0p15cm_3deg = sum(1 for r in results if r.get("success_pose_0p15cm_3deg", False))
    num_strict_pose_early_stop = sum(1 for r in results if r.get("strict_pose_early_stop_triggered", False))
    _1p5mm_3deg_reached = [r for r in results if r.get("first_reach_1p5mm_3deg", {}).get("reached", False)]
    _1p5mm_3deg_steps = [r["first_reach_1p5mm_3deg"]["total_executed_steps"] for r in _1p5mm_3deg_reached]

    # Compute threshold-specific stats
    _tnames = ["5cm", "3cm", "2cm", "1p5cm", "1cm", "0p5cm"]
    _ts: dict[str, dict] = {}
    for _tn in _tnames:
        _reached = [r for r in results if r["threshold_first_reach"][_tn]["reached"]]
        _steps = [r["threshold_first_reach"][_tn]["total_executed_steps"] for r in _reached]
        _final_delta_pos = [
            r["threshold_post_reach_trace"][_tn][-1]["delta_pos_error_from_first_reach"]
            for r in _reached if r["threshold_post_reach_trace"][_tn]
        ]
        _final_delta_cost = [
            r["threshold_post_reach_trace"][_tn][-1]["delta_pose_cost_from_first_reach"]
            for r in _reached if r["threshold_post_reach_trace"][_tn]
        ]
        _best_pos = [
            min(t["pos_error"] for t in r["threshold_post_reach_trace"][_tn])
            for r in _reached if r["threshold_post_reach_trace"][_tn]
        ]
        _best_cost = [
            min(t["pose_cost"] for t in r["threshold_post_reach_trace"][_tn])
            for r in _reached if r["threshold_post_reach_trace"][_tn]
        ]
        _ts[_tn] = {
            "reach_rate": float(len(_reached) / num_templates),
            "mean_first_reach_steps": float(np.mean(_steps)) if _steps else float("nan"),
            "median_first_reach_steps": float(np.median(_steps)) if _steps else float("nan"),
            "mean_final_minus_first_reach_pos_error": float(np.mean(_final_delta_pos)) if _final_delta_pos else float("nan"),
            "mean_final_minus_first_reach_pose_cost": float(np.mean(_final_delta_cost)) if _final_delta_cost else float("nan"),
            "mean_best_after_reach_pos_error": float(np.mean(_best_pos)) if _best_pos else float("nan"),
            "mean_best_after_reach_pose_cost": float(np.mean(_best_cost)) if _best_cost else float("nan"),
        }

    summary = {
        "num_templates": num_templates,
        # success_rate = primary_success_rate = success_pose_1cm_5deg_rate
        "success_rate": float(num_success / num_templates),
        "success_rate_definition": "success_pose_1cm_5deg",
        "primary_success_rate": float(sum(1 for r in results if r["primary_success"]) / num_templates),
        "coarse_success_rate": float(sum(1 for r in results if r["coarse_success"]) / num_templates),
        "precision_success_rate": float(sum(1 for r in results if r["precision_success"]) / num_templates),
        "strict_completion_rate": float(sum(1 for r in results if r["strict_completion"]) / num_templates),
        "legacy_pos_5cm_rate": float(sum(1 for r in results if r["legacy_pos_5cm"]) / num_templates),
        "mean_initial_dist": float(np.mean([r["initial_dist"] for r in results])),
        "mean_best_dist": float(np.mean([r["best_dist"] for r in results])),
        "mean_final_dist": float(np.mean([r["final_dist"] for r in results])),
        "mean_dist_delta": float(np.mean([r["dist_delta"] for r in results])),
        "mean_total_object_displacement": float(
            np.mean([r["total_object_displacement"] for r in results])
        ),
        "mean_contact_rate": float(np.mean([r["contact_rate"] for r in results])),
        "mean_collision_rate": float(np.mean([r["collision_rate"] for r in results])),
        "mean_collision_count": float(np.mean([r["collision_count"] for r in results])),
        # Pose-level summary metrics
        "mean_final_pos_error": float(np.mean([r["final_pos_error"] for r in results])),
        "median_final_pos_error": float(np.median([r["final_pos_error"] for r in results])),
        "mean_final_theta_error_deg": float(np.mean([r["final_theta_error_deg"] for r in results])),
        "median_final_theta_error_deg": float(np.median([r["final_theta_error_deg"] for r in results])),
        # Multi-level position success rates
        "success_pos_5cm_rate": float(sum(1 for r in results if r["success_pos_5cm"]) / num_templates),
        "success_pos_3cm_rate": float(sum(1 for r in results if r["success_pos_3cm"]) / num_templates),
        "success_pos_2cm_rate": float(sum(1 for r in results if r["success_pos_2cm"]) / num_templates),
        "success_pos_1p5cm_rate": float(sum(1 for r in results if r["success_pos_1p5cm"]) / num_templates),
        "success_pos_1cm_rate": float(sum(1 for r in results if r["success_pos_1cm"]) / num_templates),
        "success_pos_0p5cm_rate": float(sum(1 for r in results if r["success_pos_0p5cm"]) / num_templates),
        # Multi-level pose success rates
        "success_pose_5cm_15deg_rate": float(sum(1 for r in results if r["success_pose_5cm_15deg"]) / num_templates),
        "success_pose_3cm_15deg_rate": float(sum(1 for r in results if r["success_pose_3cm_15deg"]) / num_templates),
        "success_pose_2cm_15deg_rate": float(num_success_pose_2cm_15deg / num_templates),
        "success_pose_2cm_10deg_rate": float(sum(1 for r in results if r["success_pose_2cm_10deg"]) / num_templates),
        "success_pose_1cm_15deg_rate": float(sum(1 for r in results if r["success_pose_1cm_15deg"]) / num_templates),
        "success_pose_1p5cm_10deg_rate": float(sum(1 for r in results if r["success_pose_1p5cm_10deg"]) / num_templates),
        "success_pose_1cm_10deg_rate": float(sum(1 for r in results if r["success_pose_1cm_10deg"]) / num_templates),
        "success_pose_1cm_5deg_rate": float(sum(1 for r in results if r["success_pose_1cm_5deg"]) / num_templates),
        "success_pose_0p5cm_5deg_rate": float(sum(1 for r in results if r["success_pose_0p5cm_5deg"]) / num_templates),
        "success_pose_0p15cm_3deg_rate": float(num_success_pose_0p15cm_3deg / num_templates),
        "num_strict_pose_early_stop": num_strict_pose_early_stop,
        "strict_pose_early_stop_rate": float(num_strict_pose_early_stop / num_templates),
        "strict_pose_success_rate": float(num_strict_pose_early_stop / num_templates),
        "mean_strict_pose_stop_step": float(np.mean([r["strict_pose_stop_step"] for r in results if r.get("strict_pose_stop_step") is not None])) if any(r.get("strict_pose_stop_step") is not None for r in results) else float("nan"),
        "reach_1p5mm_3deg_rate": float(len(_1p5mm_3deg_reached) / num_templates),
        "mean_first_reach_1p5mm_3deg_steps": float(np.mean(_1p5mm_3deg_steps)) if _1p5mm_3deg_steps else float("nan"),
        "median_first_reach_1p5mm_3deg_steps": float(np.median(_1p5mm_3deg_steps)) if _1p5mm_3deg_steps else float("nan"),
        # Continuous best/final pose cost
        "mean_best_pose_cost": float(np.mean([r["best_pose_cost"] for r in results])),
        "median_best_pose_cost": float(np.median([r["best_pose_cost"] for r in results])),
        "mean_final_pose_cost": float(np.mean([r["final_pose_cost"] for r in results])),
        "median_final_pose_cost": float(np.median([r["final_pose_cost"] for r in results])),
        "mean_best_pos_error": float(np.mean([r["best_pos_error"] for r in results])),
        "median_best_pos_error": float(np.median([r["best_pos_error"] for r in results])),
        "mean_best_theta_error_deg_at_best_pos": float(np.nanmean([r["best_theta_error_deg_at_best_pos"] for r in results])),
        # Threshold reach rates
        "reach_5cm_rate": _ts["5cm"]["reach_rate"],
        "reach_3cm_rate": _ts["3cm"]["reach_rate"],
        "reach_2cm_rate": _ts["2cm"]["reach_rate"],
        "reach_1p5cm_rate": _ts["1p5cm"]["reach_rate"],
        "reach_1cm_rate": _ts["1cm"]["reach_rate"],
        "reach_0p5cm_rate": _ts["0p5cm"]["reach_rate"],
        # Mean/median first reach steps
        "mean_first_reach_5cm_steps": _ts["5cm"]["mean_first_reach_steps"],
        "median_first_reach_5cm_steps": _ts["5cm"]["median_first_reach_steps"],
        "mean_first_reach_3cm_steps": _ts["3cm"]["mean_first_reach_steps"],
        "median_first_reach_3cm_steps": _ts["3cm"]["median_first_reach_steps"],
        "mean_first_reach_2cm_steps": _ts["2cm"]["mean_first_reach_steps"],
        "median_first_reach_2cm_steps": _ts["2cm"]["median_first_reach_steps"],
        "mean_first_reach_1p5cm_steps": _ts["1p5cm"]["mean_first_reach_steps"],
        "median_first_reach_1p5cm_steps": _ts["1p5cm"]["median_first_reach_steps"],
        "mean_first_reach_1cm_steps": _ts["1cm"]["mean_first_reach_steps"],
        "median_first_reach_1cm_steps": _ts["1cm"]["median_first_reach_steps"],
        "mean_first_reach_0p5cm_steps": _ts["0p5cm"]["mean_first_reach_steps"],
        "median_first_reach_0p5cm_steps": _ts["0p5cm"]["median_first_reach_steps"],
        # Post-reach degradation stats
        "mean_final_minus_first_reach_pos_error_5cm": _ts["5cm"]["mean_final_minus_first_reach_pos_error"],
        "mean_final_minus_first_reach_pose_cost_5cm": _ts["5cm"]["mean_final_minus_first_reach_pose_cost"],
        "mean_best_after_reach_pos_error_5cm": _ts["5cm"]["mean_best_after_reach_pos_error"],
        "mean_best_after_reach_pose_cost_5cm": _ts["5cm"]["mean_best_after_reach_pose_cost"],
        "mean_final_minus_first_reach_pos_error_3cm": _ts["3cm"]["mean_final_minus_first_reach_pos_error"],
        "mean_final_minus_first_reach_pose_cost_3cm": _ts["3cm"]["mean_final_minus_first_reach_pose_cost"],
        "mean_best_after_reach_pos_error_3cm": _ts["3cm"]["mean_best_after_reach_pos_error"],
        "mean_best_after_reach_pose_cost_3cm": _ts["3cm"]["mean_best_after_reach_pose_cost"],
        "mean_final_minus_first_reach_pos_error_2cm": _ts["2cm"]["mean_final_minus_first_reach_pos_error"],
        "mean_final_minus_first_reach_pose_cost_2cm": _ts["2cm"]["mean_final_minus_first_reach_pose_cost"],
        "mean_best_after_reach_pos_error_2cm": _ts["2cm"]["mean_best_after_reach_pos_error"],
        "mean_best_after_reach_pose_cost_2cm": _ts["2cm"]["mean_best_after_reach_pose_cost"],
        "mean_final_minus_first_reach_pos_error_1p5cm": _ts["1p5cm"]["mean_final_minus_first_reach_pos_error"],
        "mean_final_minus_first_reach_pose_cost_1p5cm": _ts["1p5cm"]["mean_final_minus_first_reach_pose_cost"],
        "mean_best_after_reach_pos_error_1p5cm": _ts["1p5cm"]["mean_best_after_reach_pos_error"],
        "mean_best_after_reach_pose_cost_1p5cm": _ts["1p5cm"]["mean_best_after_reach_pose_cost"],
        "mean_final_minus_first_reach_pos_error_1cm": _ts["1cm"]["mean_final_minus_first_reach_pos_error"],
        "mean_final_minus_first_reach_pose_cost_1cm": _ts["1cm"]["mean_final_minus_first_reach_pose_cost"],
        "mean_best_after_reach_pos_error_1cm": _ts["1cm"]["mean_best_after_reach_pos_error"],
        "mean_best_after_reach_pose_cost_1cm": _ts["1cm"]["mean_best_after_reach_pose_cost"],
        "mean_final_minus_first_reach_pos_error_0p5cm": _ts["0p5cm"]["mean_final_minus_first_reach_pos_error"],
        "mean_final_minus_first_reach_pose_cost_0p5cm": _ts["0p5cm"]["mean_final_minus_first_reach_pose_cost"],
        "mean_best_after_reach_pos_error_0p5cm": _ts["0p5cm"]["mean_best_after_reach_pos_error"],
        "mean_best_after_reach_pose_cost_0p5cm": _ts["0p5cm"]["mean_best_after_reach_pose_cost"],
    }

    return {
        "summary": summary,
        "results": results,
    }


def save_mujoco_oracle_mpc_closed_loop_report(
    report: dict[str, Any], path: str | Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
