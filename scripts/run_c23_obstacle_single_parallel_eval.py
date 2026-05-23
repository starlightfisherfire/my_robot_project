#!/usr/bin/env python3
"""
Single-template parallel CEM-MPC evaluation.

WARNING: This is a debug/capacity-acceleration runner, NOT a replacement for
the serial evaluator.  Results should be checked against the serial runner on
a small smoke case before being used formally.  For official results, use
run_c23_obstacle_sixpack_sweep.py (serial) and verify parallel output matches.

Parallelization strategy:
  - Each CEM iteration distributes num_samples action sequences across
    num_workers subprocesses via ProcessPoolExecutor.
  - Each worker process initializes ONE MujocoPushEnv via initializer,
    then reuses it for all chunks (no per-chunk env creation).
  - Cost ordering is guaranteed by index-based dispatch: each worker
    receives (indices, state_payload, action_sequences) and returns
    (indices, costs).  Main process uses all_costs[indices] = costs.
  - The main process handles CEM iteration logic (sampling, elite selection,
    mean/std update) and closed-loop execution.

Usage (standard eval):
  PYTHONPATH=. python scripts/run_c23_obstacle_single_parallel_eval.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --split test_sim_layout_ood_blocking_easy \
    --template-index 0 \
    --budget-name c23_strict1000 \
    --num-workers 8 \
    --out runs/debug/parallel_eval/blocking_easy_000.json

Usage (quick compare-only, no full eval):
  PYTHONPATH=. python scripts/run_c23_obstacle_single_parallel_eval.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --split test_sim_layout_ood_blocking_easy \
    --template-index 0 \
    --debug-serial-compare --compare-only --compare-num-samples 64
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


# ─── Budget definitions ─────────────────────────────────────────────────────

BUDGETS = {
    "c23_strict600": {
        "horizon": 80, "execute_steps": 20, "max_mpc_steps": 30,
        "num_samples": 1024, "num_elites": 96, "num_iterations": 5,
        "success_pos_threshold": 0.0015, "success_theta_threshold_deg": 3.0,
    },
    "c23_strict800": {
        "horizon": 80, "execute_steps": 20, "max_mpc_steps": 40,
        "num_samples": 1024, "num_elites": 96, "num_iterations": 5,
        "success_pos_threshold": 0.0015, "success_theta_threshold_deg": 3.0,
    },
    "c23_strict1000": {
        "horizon": 80, "execute_steps": 20, "max_mpc_steps": 50,
        "num_samples": 1024, "num_elites": 96, "num_iterations": 5,
        "success_pos_threshold": 0.0015, "success_theta_threshold_deg": 3.0,
    },
}


# ─── State serialization ────────────────────────────────────────────────────

def _state_to_payload(env) -> dict:
    """Serialize env state to a plain dict for cross-process transfer."""
    state = env.clone_state()
    return {
        "qpos": state.qpos.copy(),
        "qvel": state.qvel.copy(),
        "ctrl": state.ctrl.copy(),
        "goal_pose": state.goal_pose.copy(),
        "step_count": int(state.step_count),
        "last_contact": bool(state.last_contact),
        "last_collision": bool(state.last_collision),
    }


# ─── Worker initializer and evaluation ─────────────────────────────────────

_WORKER_ENV = None
_WORKER_WEIGHTS = None


def _init_worker(template: dict, env_config: dict, weights_dict: dict) -> None:
    """Initializer: create ONE MujocoPushEnv per worker process."""
    global _WORKER_ENV, _WORKER_WEIGHTS
    from src.envs.mujoco_push_env import MujocoPushEnv
    from src.planners.cost_functions import CostWeights
    _WORKER_ENV = MujocoPushEnv(
        control_dt=env_config.get("control_dt", 0.1),
        max_speed_mps=env_config.get("max_speed_mps", 0.05),
        pusher_radius=env_config.get("pusher_radius", 0.010),
        pusher_halfheight=env_config.get("pusher_halfheight", 0.014),
        pusher_z=env_config.get("pusher_z", 0.016),
    )
    _WORKER_ENV.reset_from_template(template)
    _WORKER_WEIGHTS = CostWeights(**weights_dict)


def _eval_chunk(args_tuple: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a chunk of action sequences.  Uses global _WORKER_ENV.

    Args:
        args_tuple: (indices, state_payload, action_sequences)

    Returns:
        (indices, costs)
    """
    indices, state_payload, action_sequences = args_tuple

    from src.envs.mujoco_push_env import MujocoPushState
    from src.planners.cost_functions import rollout_cost
    from src.planners.mujoco_oracle_rollout import rollout_action_sequence_mujoco

    env = _WORKER_ENV
    weights = _WORKER_WEIGHTS

    restored_state = MujocoPushState(
        qpos=np.array(state_payload["qpos"], dtype=np.float64),
        qvel=np.array(state_payload["qvel"], dtype=np.float64),
        ctrl=np.array(state_payload["ctrl"], dtype=np.float64),
        goal_pose=np.array(state_payload["goal_pose"], dtype=np.float64),
        step_count=state_payload["step_count"],
        last_contact=state_payload["last_contact"],
        last_collision=state_payload["last_collision"],
    )
    goal_pose = env.get_goal_pose()

    costs = []
    for seq in action_sequences:
        env.restore_state(restored_state)
        result = rollout_action_sequence_mujoco(
            env=env,
            action_sequence=seq,
            restore_state=False,
        )
        cost = rollout_cost(
            predicted_object_poses=result.predicted_object_poses,
            ee_positions=result.ee_positions,
            action_sequence=seq,
            goal_pose=goal_pose,
            weights=weights,
            contact_flags=result.contact_flags,
            collision_flags=result.collision_flags,
        )
        costs.append(cost)

    return indices, np.asarray(costs, dtype=np.float64)


# ─── Parallel CEM planner with index-based cost ordering ────────────────────

class ParallelCEMPlanner:
    """CEM planner that evaluates candidates in parallel subprocesses.

    Uses index-based dispatch to guarantee cost ordering matches samples.
    """

    def __init__(
        self,
        horizon: int = 80,
        action_dim: int = 2,
        num_samples: int = 1024,
        num_elites: int = 96,
        num_iterations: int = 5,
        action_low: float = -1.0,
        action_high: float = 1.0,
        init_std: float = 0.8,
        smoothing: float = 0.2,
        seed: int = 42,
        num_workers: int = 8,
        mp_start_method: str = "spawn",
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.action_low = action_low
        self.action_high = action_high
        self.init_std = init_std
        self.smoothing = smoothing
        self.seed = seed
        self.num_workers = num_workers
        self.mp_start_method = mp_start_method
        self.rng = np.random.default_rng(seed)

        assert num_workers >= 1, f"num_workers must be >= 1, got {num_workers}"
        assert num_elites <= num_samples, (
            f"num_elites={num_elites} > num_samples={num_samples}"
        )

        # Pre-split indices for stable chunk assignment
        all_indices = np.arange(num_samples)
        index_chunks = np.array_split(all_indices, num_workers)
        self._index_chunks = [c for c in index_chunks if len(c) > 0]

    def plan(
        self,
        state_payload: dict,
        template: dict,
        env_config: dict,
        weights_dict: dict,
        init_mean: np.ndarray | None = None,
        init_std: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Run parallel CEM optimization.

        Returns:
            first_action: [action_dim]
            result_dict: {action_sequence, best_cost, mean, std, cost_history}
        """
        if init_mean is None:
            mean = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        else:
            mean = np.asarray(init_mean, dtype=np.float64).copy()

        if init_std is None:
            std = np.full(
                (self.horizon, self.action_dim), self.init_std, dtype=np.float64
            )
        else:
            std = np.asarray(init_std, dtype=np.float64).copy()

        best_sequence = mean.copy()
        best_cost = float("inf")
        cost_history = []

        low = self.action_low
        high = self.action_high

        ctx = mp.get_context(self.mp_start_method)
        actual_workers = len(self._index_chunks)

        with ProcessPoolExecutor(
            max_workers=actual_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(template, env_config, weights_dict),
        ) as executor:
            for iteration in range(self.num_iterations):
                # Sample
                samples = self.rng.normal(
                    loc=mean[None, :, :],
                    scale=std[None, :, :],
                    size=(self.num_samples, self.horizon, self.action_dim),
                )
                samples = np.clip(samples, low, high)

                # Dispatch with indices
                worker_args = [
                    (idx_chunk, state_payload, samples[idx_chunk])
                    for idx_chunk in self._index_chunks
                ]

                all_costs = np.empty(self.num_samples, dtype=np.float64)
                for indices, chunk_costs in executor.map(_eval_chunk, worker_args):
                    all_costs[indices] = chunk_costs

                # Validate
                assert all_costs.shape == (self.num_samples,)
                assert np.isfinite(all_costs).all(), "Non-finite costs detected"

                # Select elites
                elite_idx = np.argsort(all_costs)[:self.num_elites]
                elites = samples[elite_idx]

                elite_mean = elites.mean(axis=0)
                elite_std = elites.std(axis=0) + 1e-6

                mean = self.smoothing * mean + (1.0 - self.smoothing) * elite_mean
                std = self.smoothing * std + (1.0 - self.smoothing) * elite_std

                iter_best_idx = int(np.argmin(all_costs))
                iter_best_cost = float(all_costs[iter_best_idx])

                if iter_best_cost < best_cost:
                    best_cost = iter_best_cost
                    best_sequence = samples[iter_best_idx].copy()

                cost_history.append(best_cost)
                print(f"    CEM iter {iteration+1}/{self.num_iterations}: best_cost={best_cost:.4f}")

        return best_sequence[0].copy(), {
            "action_sequence": best_sequence,
            "best_cost": best_cost,
            "mean": mean,
            "std": std,
            "cost_history": cost_history,
        }


# ─── Serial vs parallel comparison ─────────────────────────────────────────

def debug_serial_compare(
    template: dict,
    horizon: int,
    num_samples: int,
    seed: int,
    num_workers: int,
    env_config: dict,
    weights_dict: dict,
    mp_start_method: str = "spawn",
    max_speed_mps: float = 0.05,
    pusher_radius: float = 0.010,
    pusher_halfheight: float = 0.014,
    pusher_z: float = 0.016,
) -> None:
    """Compare serial and parallel CEM cost on the same samples."""
    from src.envs.mujoco_push_env import MujocoPushEnv
    from src.metrics.mujoco_oracle_capacity import make_default_mujoco_cost_weights
    from src.planners.cost_functions import CostWeights
    from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost

    print(f"\n{'='*60}")
    print(f"DEBUG: Serial vs Parallel comparison ({num_samples} samples)")
    print(f"{'='*60}")

    # Create env for serial eval
    env = MujocoPushEnv(
        control_dt=0.1,
        max_speed_mps=max_speed_mps,
        pusher_radius=pusher_radius,
        pusher_halfheight=pusher_halfheight,
        pusher_z=pusher_z,
    )
    env.reset_from_template(template)

    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=0.0, scale=0.8, size=(num_samples, horizon, 2))
    samples = np.clip(samples, -1.0, 1.0)

    state_payload = _state_to_payload(env)
    weights = make_default_mujoco_cost_weights()

    # Serial
    t0 = time.time()
    serial_costs = []
    for seq in samples:
        cost = mujoco_oracle_rollout_cost(
            env=env, action_sequence=seq, weights=weights, restore_state=True,
        )
        serial_costs.append(cost)
    serial_costs = np.asarray(serial_costs, dtype=np.float64)
    serial_time = time.time() - t0

    # Parallel
    t0 = time.time()
    all_indices = np.arange(num_samples)
    index_chunks = np.array_split(all_indices, num_workers)
    index_chunks = [c for c in index_chunks if len(c) > 0]
    actual_workers = len(index_chunks)

    ctx = mp.get_context(mp_start_method)
    parallel_costs = np.empty(num_samples, dtype=np.float64)
    with ProcessPoolExecutor(
        max_workers=actual_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(template, env_config, weights_dict),
    ) as executor:
        worker_args = [
            (idx_chunk, state_payload, samples[idx_chunk])
            for idx_chunk in index_chunks
        ]
        for indices, chunk_costs in executor.map(_eval_chunk, worker_args):
            parallel_costs[indices] = chunk_costs
    parallel_time = time.time() - t0

    max_abs_diff = float(np.max(np.abs(serial_costs - parallel_costs)))
    mean_abs_diff = float(np.mean(np.abs(serial_costs - parallel_costs)))

    print(f"  Serial:   {serial_time:.3f}s  costs range=[{serial_costs.min():.4f}, {serial_costs.max():.4f}]")
    print(f"  Parallel: {parallel_time:.3f}s  costs range=[{parallel_costs.min():.4f}, {parallel_costs.max():.4f}]")
    print(f"  Max abs diff:  {max_abs_diff:.2e}")
    print(f"  Mean abs diff: {mean_abs_diff:.2e}")
    ok = max_abs_diff <= 1e-6
    print(f"  Result: {'PASS' if ok else 'FAIL'} (threshold=1e-6)")
    if not ok:
        worst_idx = int(np.argmax(np.abs(serial_costs - parallel_costs)))
        print(f"  Worst sample #{worst_idx}: serial={serial_costs[worst_idx]:.8f} parallel={parallel_costs[worst_idx]:.8f}")
    print(f"{'='*60}\n")


# ─── Closed-loop evaluation ─────────────────────────────────────────────────

def evaluate_single_template_parallel(
    template: dict,
    budget_name: str,
    seed: int = 42,
    num_workers: int = 8,
    max_speed_mps: float = 0.05,
    pusher_radius: float = 0.010,
    pusher_halfheight: float = 0.014,
    pusher_z: float = 0.016,
    mp_start_method: str = "spawn",
) -> dict[str, Any]:
    """Run closed-loop CEM-MPC with parallel CEM evaluation for one template."""
    from src.envs.mujoco_push_env import MujocoPushEnv
    from src.planners.cost_functions import CostWeights, wrap_angle
    from src.metrics.mujoco_oracle_capacity import (
        make_default_mujoco_cost_weights,
        compute_pose_success_metrics,
    )

    config = BUDGETS[budget_name]
    horizon = config["horizon"]
    execute_steps = config["execute_steps"]
    max_mpc_steps = config["max_mpc_steps"]
    num_samples = config["num_samples"]
    num_elites = config["num_elites"]
    num_iterations = config["num_iterations"]

    success_pos_threshold = config.get("success_pos_threshold", 0.0015)
    success_theta_threshold_deg = config.get("success_theta_threshold_deg", 3.0)
    success_dist_threshold = 0.05

    # Create env in main process
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
    initial_dist = float(np.linalg.norm(initial_object_pose[:2] - initial_goal_pose[:2]))

    weights = make_default_mujoco_cost_weights()
    env_config = {
        "control_dt": 0.1,
        "max_speed_mps": max_speed_mps,
        "pusher_radius": pusher_radius,
        "pusher_halfheight": pusher_halfheight,
        "pusher_z": pusher_z,
    }
    weights_dict = {
        "w_pos": weights.w_pos,
        "w_theta": weights.w_theta,
        "w_reach": weights.w_reach,
        "w_no_contact": weights.w_no_contact,
        "w_push_alignment": weights.w_push_alignment,
        "w_collision": weights.w_collision,
        "w_collision_step": weights.w_collision_step,
        "w_action": weights.w_action,
        "w_smooth": weights.w_smooth,
        "w_subgoal": weights.w_subgoal,
    }

    # Trajectory tracking
    distances = [initial_dist]
    contact_flags = []
    collision_flags = []
    num_mpc_steps = 0
    total_executed_steps = 0
    success = False
    best_dist = initial_dist
    best_pos_error = initial_dist
    best_theta_error_deg_at_best_pos = float("inf")
    best_pose_cost = float("inf")
    best_pose_step = 0
    mpc_step_logs = []

    print(f"\n{'='*80}")
    print(f"Template: {template['reset_template_id']}  budget={budget_name}")
    print(f"  Parallel CEM: {num_workers} workers, mp_start={mp_start_method}")
    print(f"  Initial dist: {initial_dist:.4f} m")
    print(f"{'='*80}\n")

    planner = ParallelCEMPlanner(
        horizon=horizon,
        action_dim=2,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
        action_low=-1.0,
        action_high=1.0,
        init_std=0.8,
        smoothing=0.2,
        seed=seed,
        num_workers=num_workers,
        mp_start_method=mp_start_method,
    )

    init_mean = None
    init_std = None
    t_start = time.time()

    for mpc_step in range(1, max_mpc_steps + 1):
        num_mpc_steps = mpc_step
        current_object_pose = env.get_object_pose()
        current_dist = float(np.linalg.norm(current_object_pose[:2] - initial_goal_pose[:2]))

        print(f"MPC Step {mpc_step}/{max_mpc_steps}, dist={current_dist:.4f}m")

        # Parallel CEM plan
        state_payload = _state_to_payload(env)
        first_action, cem_result = planner.plan(
            state_payload=state_payload,
            template=template,
            env_config=env_config,
            weights_dict=weights_dict,
            init_mean=init_mean,
            init_std=init_std,
        )

        print(f"  CEM best_cost: {cem_result['best_cost']:.4f}")

        # Execute first execute_steps actions
        actions_to_execute = cem_result["action_sequence"][:execute_steps]
        actual_contact_this_round = []
        actual_collision_this_round = []

        for env_step_in_chunk, action in enumerate(actions_to_execute, start=1):
            env.step(action)
            total_executed_steps += 1

            current_object_pose = env.get_object_pose()
            current_dist = float(np.linalg.norm(current_object_pose[:2] - initial_goal_pose[:2]))
            current_theta_error_rad = float(abs(wrap_angle(current_object_pose[2] - initial_goal_pose[2])))
            current_theta_error_deg = float(np.rad2deg(current_theta_error_rad))
            current_pose_cost = (current_dist / 0.01) ** 2 + (current_theta_error_deg / 5.0) ** 2

            distances.append(current_dist)
            cf = env.get_contact_flag()
            clf = env.get_collision_flag()
            contact_flags.append(cf)
            collision_flags.append(clf)
            actual_contact_this_round.append(cf)
            actual_collision_this_round.append(clf)

            if current_dist < best_dist:
                best_dist = current_dist
            if current_dist < best_pos_error:
                best_pos_error = current_dist
                best_theta_error_deg_at_best_pos = current_theta_error_deg
            if current_pose_cost < best_pose_cost:
                best_pose_cost = current_pose_cost
                best_pose_step = total_executed_steps

            # Strict pose early stop
            if current_dist <= success_pos_threshold and current_theta_error_deg <= success_theta_threshold_deg:
                success = True
                print(f"  STRICT POSE EARLY STOP at step {total_executed_steps}")
                break

        # Log
        collision_count_r = float(np.sum(actual_collision_this_round))
        collision_rate_r = float(np.mean(actual_collision_this_round))
        contact_rate_r = float(np.mean(actual_contact_this_round))

        mpc_step_logs.append({
            "mpc_step": mpc_step,
            "total_executed_steps": total_executed_steps,
            "best_cost": cem_result["best_cost"],
            "collision_count": collision_count_r,
            "collision_rate": collision_rate_r,
            "contact_rate": contact_rate_r,
            "pos_error": current_dist,
        })

        # Warm start for next CEM iteration
        init_mean = cem_result["mean"]
        init_std = cem_result["std"]

        if success:
            break

    t_elapsed = time.time() - t_start

    # Final metrics
    final_object_pose = env.get_object_pose()
    final_pos_error = float(np.linalg.norm(final_object_pose[:2] - initial_goal_pose[:2]))
    final_theta_error_rad = float(abs(wrap_angle(final_object_pose[2] - initial_goal_pose[2])))
    final_theta_error_deg = float(np.rad2deg(final_theta_error_rad))

    pose_metrics = compute_pose_success_metrics(final_pos_error, final_theta_error_deg)

    contact_rate = float(np.mean(contact_flags)) if contact_flags else 0.0
    collision_rate = float(np.mean(collision_flags)) if collision_flags else 0.0
    collision_count = float(np.sum(collision_flags)) if collision_flags else 0.0

    print(f"\n{'='*80}")
    print(f"Done: success={pose_metrics['primary_success']}  time={t_elapsed:.1f}s")
    print(f"  final_pos_error={final_pos_error*1000:.2f}mm  theta={final_theta_error_deg:.2f}deg")
    print(f"  collision_count={collision_count:.0f}  collision_rate={collision_rate:.3f}")
    print(f"{'='*80}\n")

    return {
        "reset_template_id": template["reset_template_id"],
        "split": template["split"],
        "layout_family": template["layout_family"],
        "shape_family": template["shape_family"],
        "budget_name": budget_name,
        "seed": seed,
        "num_workers": num_workers,
        "mp_start_method": mp_start_method,
        "max_speed_mps": max_speed_mps,
        "final_pos_error": final_pos_error,
        "final_theta_error_deg": final_theta_error_deg,
        "best_pos_error": best_pos_error,
        "best_theta_error_deg_at_best_pos": best_theta_error_deg_at_best_pos,
        "total_executed_steps": total_executed_steps,
        "num_mpc_steps": num_mpc_steps,
        "contact_rate": contact_rate,
        "collision_rate": collision_rate,
        "collision_count": collision_count,
        "collision_any": bool(collision_count > 0.5),
        "success": success,
        "elapsed_seconds": t_elapsed,
        "mpc_step_logs": mpc_step_logs,
        **pose_metrics,
        "success_definition": "success_pose_1cm_5deg",
    }


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-template parallel CEM-MPC eval (debug/capacity acceleration)."
    )
    p.add_argument("--templates", required=True, help="Template JSON path.")
    p.add_argument("--split", required=True, help="Split name.")
    p.add_argument("--template-index", type=int, default=0, help="Template index.")
    p.add_argument("--reset-template-id", default=None, help="Override template ID.")
    p.add_argument("--budget-name", default="c23_strict1000", choices=list(BUDGETS.keys()))
    p.add_argument("--num-workers", type=int, default=8, help="Parallel CEM workers.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-speed-mps", type=float, default=0.05,
                   help="Pusher max speed m/s (default: 0.05).")
    p.add_argument("--pusher-radius", type=float, default=0.010)
    p.add_argument("--pusher-halfheight", type=float, default=0.014)
    p.add_argument("--pusher-z", type=float, default=0.016)
    p.add_argument("--mp-start-method", type=str, default="spawn",
                   choices=["spawn", "forkserver", "fork"],
                   help="Multiprocessing start method (default: spawn).")
    p.add_argument("--out", default=None, help="Output JSON path.")
    # Debug compare
    p.add_argument("--debug-serial-compare", action="store_true", default=False,
                   help="Run serial vs parallel cost comparison.")
    p.add_argument("--compare-num-samples", type=int, default=64,
                   help="Number of samples for debug comparison (default: 64).")
    p.add_argument("--compare-only", action="store_true", default=False,
                   help="Exit after serial/parallel compare (no full eval).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load template
    with open(args.templates, encoding="utf-8") as f:
        all_templates = json.load(f)

    templates_in_split = [t for t in all_templates if t["split"] == args.split]
    templates_in_split.sort(key=lambda t: t["reset_template_id"])

    if args.reset_template_id:
        matches = [t for t in templates_in_split if t["reset_template_id"] == args.reset_template_id]
        if not matches:
            print(f"ERROR: template {args.reset_template_id} not found in split {args.split}")
            return
        template = matches[0]
    else:
        if args.template_index < 0 or args.template_index >= len(templates_in_split):
            print(f"ERROR: template-index {args.template_index} out of range [0, {len(templates_in_split)-1}]")
            return
        template = templates_in_split[args.template_index]

    print(f"Template: {template['reset_template_id']}")
    print(f"Budget: {args.budget_name}  Workers: {args.num_workers}")
    print(f"Max speed: {args.max_speed_mps} m/s  mp_start: {args.mp_start_method}")

    # Debug compare mode
    if args.debug_serial_compare:
        weights_obj = __import__("src.metrics.mujoco_oracle_capacity", fromlist=["make_default_mujoco_cost_weights"]).make_default_mujoco_cost_weights()
        env_config = {
            "control_dt": 0.1,
            "max_speed_mps": args.max_speed_mps,
            "pusher_radius": args.pusher_radius,
            "pusher_halfheight": args.pusher_halfheight,
            "pusher_z": args.pusher_z,
        }
        weights_dict = {
            "w_pos": weights_obj.w_pos, "w_theta": weights_obj.w_theta,
            "w_reach": weights_obj.w_reach, "w_no_contact": weights_obj.w_no_contact,
            "w_push_alignment": weights_obj.w_push_alignment,
            "w_collision": weights_obj.w_collision, "w_collision_step": weights_obj.w_collision_step,
            "w_action": weights_obj.w_action, "w_smooth": weights_obj.w_smooth,
            "w_subgoal": weights_obj.w_subgoal,
        }
        debug_serial_compare(
            template=template,
            horizon=80,
            num_samples=args.compare_num_samples,
            seed=args.seed,
            num_workers=args.num_workers,
            env_config=env_config,
            weights_dict=weights_dict,
            mp_start_method=args.mp_start_method,
            max_speed_mps=args.max_speed_mps,
            pusher_radius=args.pusher_radius,
            pusher_halfheight=args.pusher_halfheight,
            pusher_z=args.pusher_z,
        )
        if args.compare_only:
            print("Compare-only mode: exiting after serial/parallel comparison.")
            return

    # Full evaluation
    result = evaluate_single_template_parallel(
        template=template,
        budget_name=args.budget_name,
        seed=args.seed,
        num_workers=args.num_workers,
        max_speed_mps=args.max_speed_mps,
        pusher_radius=args.pusher_radius,
        pusher_halfheight=args.pusher_halfheight,
        pusher_z=args.pusher_z,
        mp_start_method=args.mp_start_method,
    )

    # Save
    if args.out:
        out_path = Path(args.out)
    else:
        tid = template["reset_template_id"]
        out_path = Path("runs/debug/parallel_eval") / f"{tid}_{args.budget_name}_parallel.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Result saved: {out_path}")


if __name__ == "__main__":
    main()
