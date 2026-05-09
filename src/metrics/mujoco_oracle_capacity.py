from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.mujoco_push_env import MujocoPushEnv
from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import CostWeights
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
        w_action=0.05,
        w_smooth=0.1,
        w_subgoal=0.0,
    )


def evaluate_one_template_mujoco_oracle_mpc(
    template: dict[str, Any],
    horizon: int = 80,
    num_samples: int = 1536,
    num_elites: int = 128,
    num_iterations: int = 7,
    seed: int = 42,
    success_dist_threshold: float = 0.05,
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
    }


def run_mujoco_oracle_mpc_capacity(
    templates: list[dict[str, Any]],
    horizon: int = 80,
    num_samples: int = 1536,
    num_elites: int = 128,
    num_iterations: int = 7,
    seed: int = 42,
    success_dist_threshold: float = 0.05,
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
