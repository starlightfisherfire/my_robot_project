#!/usr/bin/env python3
"""
run_learned_mpc_eval.py — Learned rollout + CEM-MPC internal smoke test.

This script uses learned rollout for both planning and state update.
It is NOT MuJoCo closed-loop evaluation.

Usage:
    python scripts/run_learned_mpc_eval.py \
        --checkpoint runs/train_state16_poc/flat/checkpoints/best.pt \
        --model-type flat \
        --normalizer runs/train_state16_poc/normalizer_state16.json \
        --split train_sim_id \
        --max-templates 3 \
        --out runs/learned_mpc_eval/flat_smoke.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.planners.cem_mpc import CEMMPC
from src.planners.rollout_model import LearnedRolloutModel, load_learned_rollout_model
from src.planners.cost_functions import CostWeights


def load_state16_episode(npz_path: str, history_len: int = 6) -> np.ndarray:
    """Load a state16 episode and return the initial history window."""
    data = np.load(npz_path, allow_pickle=True)
    states = data["states"]  # [T, N, D]
    if len(states) < history_len:
        pad = np.tile(states[0:1], (history_len - len(states), 1, 1))
        states = np.concatenate([pad, states], axis=0)
    return states[:history_len]  # [H, N, D]


def get_goal_pose(npz_path: str) -> np.ndarray:
    """Load goal pose from episode."""
    data = np.load(npz_path, allow_pickle=True)
    return data["goal_pose"]  # [3]


def run_single_template(
    rollout_model: LearnedRolloutModel,
    cem: CEMMPC,
    state16_path: str,
    goal_pose: np.ndarray,
    max_mpc_steps: int = 30,
    pos_threshold: float = 0.02,
    theta_threshold_deg: float = 10.0,
) -> dict:
    """Run learned MPC on a single template/episode."""
    initial_state = load_state16_episode(state16_path)
    current_state = initial_state.copy()

    best_dist = float("inf")
    final_dist = float("inf")
    theta_err = 0.0
    mpc_steps_taken = 0
    collision_count = 0

    first_zero_cost = None
    first_planned_cost = None

    for step in range(max_mpc_steps):
        # Rebuild cost_fn with current_state (fix: was using stale initial_state)
        cost_fn = rollout_model.make_cost_fn_for_cem(
            initial_state=current_state,
            goal_pose=goal_pose,
            weights=CostWeights(),
        )

        # Compute zero-action baseline cost
        zero_seq = np.zeros((cem.horizon, cem.action_dim), dtype=np.float64)
        zero_cost = cost_fn(zero_seq)

        # Plan
        first_action, cem_result = cem.plan(cost_fn)
        planned_cost = cem_result.best_cost

        # Record first step costs
        if step == 0:
            first_zero_cost = float(zero_cost)
            first_planned_cost = float(planned_cost)

        # Execute: update state based on predicted dynamics
        step_result = rollout_model.forward_step(current_state, first_action)

        # Update current state
        current_state = rollout_model._update_state(
            current_state[np.newaxis],
            step_result.pred_object_pose[np.newaxis],
            (current_state[-1, 0, :2] + first_action)[np.newaxis],
        )[0]

        # Compute distance to goal
        pred_pose = step_result.pred_object_pose
        dist = float(np.sqrt(np.sum((pred_pose[:2] - goal_pose[:2]) ** 2)))
        theta_err = float(abs(np.arctan2(
            np.sin(pred_pose[2] - goal_pose[2]),
            np.cos(pred_pose[2] - goal_pose[2])
        )))

        best_dist = min(best_dist, dist)
        final_dist = dist
        mpc_steps_taken = step + 1

        # Check success
        if dist < pos_threshold and np.degrees(theta_err) < theta_threshold_deg:
            break

    # Build result
    success = bool(final_dist < pos_threshold and np.degrees(theta_err) < theta_threshold_deg)

    return {
        "final_pos_dist_m": float(final_dist),
        "best_pos_dist_m": float(best_dist),
        "final_theta_error_deg": float(np.degrees(theta_err)),
        "mpc_steps": mpc_steps_taken,
        "success": success,
        "collision_count": collision_count,
        "first_zero_cost": first_zero_cost,
        "first_planned_cost": first_planned_cost,
        "first_cost_improvement": (first_zero_cost - first_planned_cost)
            if (first_zero_cost is not None and first_planned_cost is not None)
            else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Learned rollout + CEM-MPC internal smoke (NOT MuJoCo closed-loop)"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--model-type", required=True,
                        choices=["flat", "object_centric", "causality_aware"])
    parser.add_argument("--normalizer", default=None, help="Path to normalizer .json")
    parser.add_argument("--split", default=None, help="Split filter (e.g. train_sim_id)")
    parser.add_argument("--dataset-dir", default=None, help="Dataset directory (default: data/sim/layout_ood_state16_v0)")
    parser.add_argument("--max-templates", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--num-elites", type=int, default=32)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--init-std", type=float, default=0.01,
                        help="CEM init std (default 0.01, matches action range ±0.02)")
    parser.add_argument("--max-mpc-steps", type=int, default=30)
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    rollout_model = load_learned_rollout_model(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device="cpu",
        normalizer_path=args.normalizer,
    )

    # Build CEM planner
    cem = CEMMPC(
        horizon=args.horizon,
        action_dim=2,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_iterations=args.num_iterations,
        action_low=-0.02,
        action_high=0.02,
        init_std=args.init_std,
        smoothing=0.2,
        seed=42,
    )

    # Find episodes
    if args.dataset_dir:
        data_dir = Path(args.dataset_dir)
    else:
        data_dir = REPO_ROOT / "data" / "sim" / "layout_ood_state16_v0"
    ep_dir = data_dir / "episodes"
    meta_path = data_dir / "metadata" / "episodes.jsonl"

    if not meta_path.exists():
        print(f"No episodes found at {meta_path}")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "eval_type": "learned_planning_internal_smoke",
            "note": "This script uses learned rollout for both planning and state update; "
                    "it is not MuJoCo closed-loop evaluation.",
            "smoke_pass": False,
            "checkpoint": args.checkpoint,
            "model_type": args.model_type,
            "status": "no_data",
            "message": "No state16 episodes found. Run data collection first.",
        }
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {out_path}")
        sys.exit(1)

    with open(meta_path) as f:
        all_episodes = [json.loads(line) for line in f if line.strip()]

    # Filter by split
    requested_split = args.split
    if requested_split:
        has_split = any("split_name" in ep for ep in all_episodes)
        if has_split:
            episodes = [ep for ep in all_episodes if ep.get("split_name") == requested_split]
            if not episodes:
                print(f"WARNING: No episodes with split_name='{requested_split}', "
                      f"using all {len(all_episodes)} episodes")
                episodes = all_episodes
        else:
            print(f"WARNING: No split_name field in metadata, using all {len(all_episodes)} episodes")
            episodes = all_episodes
    else:
        episodes = all_episodes

    episodes = episodes[:args.max_templates]
    print(f"Evaluating {len(episodes)} episodes (requested_split={requested_split})...")

    results = []
    for i, ep in enumerate(episodes):
        ep_id = ep["episode_id"]
        npz_path = str(ep_dir / f"{ep_id}.npz")

        if not Path(npz_path).exists():
            print(f"  [{i+1}] {ep_id}: NPZ not found, skipping")
            continue

        goal_pose = get_goal_pose(npz_path)

        t0 = time.time()
        result = run_single_template(
            rollout_model=rollout_model,
            cem=cem,
            state16_path=npz_path,
            goal_pose=goal_pose,
            max_mpc_steps=args.max_mpc_steps,
        )
        elapsed = time.time() - t0

        result["episode_id"] = ep_id
        result["family"] = ep.get("family", "unknown")
        result["runtime_sec"] = elapsed
        results.append(result)

        status = "✅" if result["success"] else "❌"
        cost_info = ""
        if result.get("first_cost_improvement") is not None:
            cost_info = (f" zero={result['first_zero_cost']:.4f}"
                         f" planned={result['first_planned_cost']:.4f}"
                         f" improve={result['first_cost_improvement']:.4f}")
        print(f"  [{i+1}] {ep_id}: {status} dist={result['final_pos_dist_m']:.4f}m "
              f"best={result['best_pos_dist_m']:.4f}m steps={result['mpc_steps']} "
              f"time={elapsed:.1f}s{cost_info}")

    # Summary
    if results:
        success_count = sum(1 for r in results if r["success"])
        final_dists = [r["final_pos_dist_m"] for r in results]
        best_dists = [r["best_pos_dist_m"] for r in results]
        cost_improvements = [r["first_cost_improvement"] for r in results
                             if r.get("first_cost_improvement") is not None]

        # Smoke pass criteria
        checkpoint_loaded = True
        found_episode = len(results) > 0
        cem_planned = all(r.get("first_planned_cost") is not None for r in results)
        costs_finite = all(
            np.isfinite(r.get("first_planned_cost", float("nan")))
            and np.isfinite(r.get("first_zero_cost", float("nan")))
            for r in results
        )
        cost_improved = any(
            (r.get("first_cost_improvement") or 0) > 0 for r in results
        )

        smoke_pass = all([
            checkpoint_loaded,
            found_episode,
            cem_planned,
            costs_finite,
            cost_improved,
        ])

        summary = {
            "eval_type": "learned_planning_internal_smoke",
            "note": "This script uses learned rollout for both planning and state update; "
                    "it is not MuJoCo closed-loop evaluation.",
            "smoke_pass": smoke_pass,
            "checkpoint": args.checkpoint,
            "model_type": args.model_type,
            "requested_split": requested_split,
            "actual_num_episodes": len(results),
            "num_episodes": len(results),
            "success_count": success_count,
            "success_rate": success_count / len(results),
            "mean_final_dist": float(np.mean(final_dists)),
            "mean_best_dist": float(np.mean(best_dists)),
            "min_best_dist": float(np.min(best_dists)),
            "mean_first_cost_improvement": float(np.mean(cost_improvements))
                if cost_improvements else None,
            "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in results])),
            "cem_config": {
                "horizon": args.horizon,
                "num_samples": args.num_samples,
                "num_elites": args.num_elites,
                "num_iterations": args.num_iterations,
                "action_low": -0.02,
                "action_high": 0.02,
                "init_std": args.init_std,
            },
        }
    else:
        summary = {
            "eval_type": "learned_planning_internal_smoke",
            "note": "This script uses learned rollout for both planning and state update; "
                    "it is not MuJoCo closed-loop evaluation.",
            "smoke_pass": False,
            "checkpoint": args.checkpoint,
            "model_type": args.model_type,
            "status": "no_results",
        }

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": summary,
        "episodes": results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {out_path}")
    print(f"smoke_pass = {summary.get('smoke_pass')}")


if __name__ == "__main__":
    main()
