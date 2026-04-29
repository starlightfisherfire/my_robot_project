from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.toy_push_env import ToyPushEnv
from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import CostWeights
from src.planners.oracle_rollout import (
    oracle_rollout_cost,
    rollout_action_sequence,
)


def make_default_toy_cost_weights() -> CostWeights:
    """
    Cost weights for toy oracle-MPC capacity smoke test.

    These are not final MuJoCo / real-robot weights.
    They are chosen to make the toy planner interface easy to debug.
    """
    return CostWeights(
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


def evaluate_one_template_toy_oracle_mpc(
    template: dict[str, Any],
    horizon: int = 18,
    num_samples: int = 768,
    num_elites: int = 96,
    num_iterations: int = 6,
    seed: int = 42,
    success_dist_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Evaluate one reset template with ToyPushEnv + oracle rollout + CEM.

    Important:
        ToyPushEnv ignores obstacles. Therefore this is not a real layout-OOD
        planner-capacity result. It only verifies that reset templates can be
        converted into an environment state and optimized through the oracle
        rollout interface.
    """
    env = ToyPushEnv()
    env.reset_from_template(template)

    state = env.clone_state()
    initial_dist = float(np.linalg.norm(state.object_pose[:2] - state.goal_pose[:2]))

    weights = make_default_toy_cost_weights()

    zero_actions = np.zeros((horizon, 2), dtype=np.float64)
    zero_cost = oracle_rollout_cost(
        env=env,
        action_sequence=zero_actions,
        weights=weights,
        restore_state=True,
    )

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
        return oracle_rollout_cost(
            env=env,
            action_sequence=action_sequence,
            weights=weights,
            restore_state=True,
        )

    first_action, cem_result = planner.plan(cost_fn)

    planned_rollout = rollout_action_sequence(
        env=env,
        action_sequence=cem_result.action_sequence,
        restore_state=True,
    )

    final_object_pose = planned_rollout.predicted_object_poses[-1]
    goal_pose = env.clone_state().goal_pose
    final_dist = float(np.linalg.norm(final_object_pose[:2] - goal_pose[:2]))

    planned_cost = oracle_rollout_cost(
        env=env,
        action_sequence=cem_result.action_sequence,
        weights=weights,
        restore_state=True,
    )

    max_contact = float(np.max(planned_rollout.contact_flags))
    max_collision = float(np.max(planned_rollout.collision_flags))

    improved_cost = bool(planned_cost < zero_cost)
    improved_dist = bool(final_dist < initial_dist)
    success = bool(
        improved_cost
        and improved_dist
        and final_dist < success_dist_threshold
        and max_collision < 0.5
    )

    # Check env was restored after candidate evaluations.
    final_env_state = env.clone_state()
    restored_ok = bool(
        np.allclose(final_env_state.object_pose, state.object_pose)
        and np.allclose(final_env_state.goal_pose, state.goal_pose)
        and np.allclose(final_env_state.ee_pos, state.ee_pos)
        and final_env_state.step_count == state.step_count
    )

    return {
        "reset_template_id": template["reset_template_id"],
        "split": template["split"],
        "layout_family": template["layout_family"],
        "shape_family": template["shape_family"],
        "seed": template.get("seed"),
        "initial_dist": initial_dist,
        "final_dist": final_dist,
        "zero_cost": float(zero_cost),
        "planned_cost": float(planned_cost),
        "best_cost": float(cem_result.best_cost),
        "first_action": first_action.tolist(),
        "cost_history": [float(x) for x in cem_result.cost_history],
        "max_contact": max_contact,
        "max_collision": max_collision,
        "improved_cost": improved_cost,
        "improved_dist": improved_dist,
        "restored_ok": restored_ok,
        "success": success,
    }


def run_toy_oracle_mpc_capacity(
    templates: list[dict[str, Any]],
    horizon: int = 18,
    num_samples: int = 768,
    num_elites: int = 96,
    num_iterations: int = 6,
    seed: int = 42,
    success_dist_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Run toy oracle-MPC capacity smoke test over a list of templates.

    This is only a smoke test for:
        reset template → ToyPushEnv → oracle rollout → CEM-MPC

    It is not a MuJoCo planner-capacity result and should not be interpreted
    as real layout-OOD performance.
    """
    if not templates:
        raise ValueError("No templates provided for toy oracle-MPC capacity check.")

    results = []

    for i, template in enumerate(templates):
        result = evaluate_one_template_toy_oracle_mpc(
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
        "mean_final_dist": float(np.mean([r["final_dist"] for r in results])),
        "mean_zero_cost": float(np.mean([r["zero_cost"] for r in results])),
        "mean_planned_cost": float(np.mean([r["planned_cost"] for r in results])),
    }

    return {
        "summary": summary,
        "results": results,
    }


def save_toy_oracle_mpc_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)