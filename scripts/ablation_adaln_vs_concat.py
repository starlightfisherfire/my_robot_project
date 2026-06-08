#!/usr/bin/env python3
"""
Multi-template ablation: AdaLN vs Concat dynamics head.

Uses batched learned rollout for speed.

Run: conda activate lerobot && python3 scripts/ablation_adaln_vs_concat.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.models.rig_world import RIGWorldModel
from src.planners.mppi import MPPI
from src.planners.batched_rollout_cost import BatchedLearnedRolloutCostFn
from src.planners.cost_functions import CostWeights
from src.planners.cost_modes import rollout_cost_with_mode
from src.planners.rollout_model import LearnedRolloutModel
from src.envs.toy_push_env import ToyPushEnv

HEAD_TYPES = ["concat", "adaln", "adaln_e2e"]
CHECKPOINT_DIR = "runs/adaln_head_comparison"
E2E_CHECKPOINT = "runs/adaln_e2e/best.pt"
MPPI_CFG = {"temperature": 0.2, "num_samples": 256, "horizon": 20, "init_std": 0.5, "smoothing": 0.2}
COST_MODES = ["current", "staged_full"]
SUCCESS_THRESH = 0.10
MAX_STEPS = 40
SEED = 42

SCENARIOS = [
    {"obj": [0.20, 0.18, 0.0], "goal": [0.50, 0.18, 0.0], "ee": [0.10, 0.18], "desc": "open_straight"},
    {"obj": [0.20, 0.18, 0.0], "goal": [0.50, 0.40, 0.0], "ee": [0.10, 0.18], "desc": "open_diagonal"},
    {"obj": [0.10, 0.18, 0.0], "goal": [0.55, 0.18, 0.0], "ee": [0.02, 0.18], "desc": "open_long"},
    {"obj": [0.40, 0.30, 0.0], "goal": [0.10, 0.10, 0.0], "ee": [0.50, 0.30], "desc": "far_push_reverse"},
    {"obj": [0.20, 0.20, 0.0], "goal": [0.20, 0.50, 0.0], "ee": [0.10, 0.20], "desc": "open_lateral"},
    {"obj": [0.20, 0.20, 0.0], "goal": [0.45, 0.40, 0.5], "ee": [0.10, 0.20], "desc": "open_rotation"},
]


def build_state_history(obj_pose, ee_pos):
    """Build minimal state history [H=6, N=6, D=16] from current poses."""
    history = np.zeros((6, 6, 16), dtype=np.float32)
    for h in range(6):
        history[h, 0, 0] = obj_pose[0]
        history[h, 0, 1] = obj_pose[1]
        history[h, 1, 0] = ee_pos[0]
        history[h, 1, 1] = ee_pos[1]
    return history


def run_episode_batched(
    model, device, env, cost_mode, obj_init, goal, ee_init, max_steps=MAX_STEPS,
):
    """Run episode using batched MPPI with learned rollout."""
    env.reset(
        object_pose=np.array(obj_init, dtype=np.float64),
        goal_pose=np.array(goal, dtype=np.float64),
        ee_pos=np.array(ee_init, dtype=np.float64),
    )
    goal_arr = np.array(goal, dtype=np.float64)
    init_obj = env.clone_state().object_pose.copy()

    contact_flags = []
    obj_poses = [init_obj.copy()]

    planner = MPPI(
        horizon=MPPI_CFG["horizon"],
        num_samples=MPPI_CFG["num_samples"],
        temperature=MPPI_CFG["temperature"],
        init_std=MPPI_CFG["init_std"],
        smoothing=MPPI_CFG["smoothing"],
        seed=SEED,
    )

    for step in range(max_steps):
        state = env.clone_state()
        history = build_state_history(state.object_pose, state.ee_pos)

        if cost_mode == "current":
            weights = CostWeights()
        else:
            weights = CostWeights(
                w_no_contact=0, w_reach=2, w_push_alignment=0,
                w_early_contact=4, w_persistent_contact=3,
            )

        cost_fn = BatchedLearnedRolloutCostFn(
            model=model, normalizer=None, initial_state=history,
            goal_pose=goal_arr, device=device, weights=weights,
        )

        result = planner.optimize(cost_fn)
        action = result.action_sequence[0]
        state = env.step(action)

        obj_poses.append(state.object_pose.copy())
        contact_flags.append(float(state.last_contact))

        if np.linalg.norm(state.object_pose[:2] - goal_arr[:2]) < SUCCESS_THRESH:
            break

    obj_poses = np.array(obj_poses)
    contact_flags = np.array(contact_flags)

    init_dist = float(np.linalg.norm(init_obj[:2] - goal_arr[:2]))
    final_dist = float(np.linalg.norm(obj_poses[-1, :2] - goal_arr[:2]))
    best_dist = float(np.min(np.linalg.norm(obj_poses[:, :2] - goal_arr[:2], axis=-1)))
    cr = float(np.mean(contact_flags > 0.5)) if len(contact_flags) > 0 else 0.0
    ci = np.where(contact_flags > 0.5)[0]

    return {
        "contact_rate": round(cr, 4),
        "first_contact_step": int(ci[0]) if len(ci) > 0 else -1,
        "object_progress": round(init_dist - final_dist, 4),
        "drift": round(max(0.0, final_dist - best_dist), 4),
        "final_success": bool(final_dist < SUCCESS_THRESH),
        "final_dist": round(final_dist, 4),
        "n_steps": len(contact_flags),
    }


def main():
    run_dir = Path(__file__).resolve().parent.parent / "runs" / f"adaln_ablation_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ToyPushEnv()

    all_results = []
    total = len(HEAD_TYPES) * len(COST_MODES) * len(SCENARIOS)  # 3*2*6=36
    trial = 0

    for head_type in HEAD_TYPES:
        print(f"\n{'='*60}")
        print(f"Loading {head_type} model...")

        if head_type == "adaln_e2e":
            ckpt = torch.load(E2E_CHECKPOINT, map_location=device, weights_only=False)
            model = RIGWorldModel(model_type="flat", dynamics_head_type="adaln")
        else:
            ckpt = torch.load(f"{CHECKPOINT_DIR}/{head_type}/best.pt", map_location=device, weights_only=False)
            model = RIGWorldModel(model_type="flat", dynamics_head_type=head_type)

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)
        model.eval()

        for cost_mode in COST_MODES:
            for scenario in SCENARIOS:
                trial += 1
                desc = scenario["desc"]
                print(f"[{trial}/{total}] {head_type}/{cost_mode}/{desc} ...", end=" ", flush=True)

                try:
                    metrics = run_episode_batched(
                        model, device, env, cost_mode,
                        scenario["obj"], scenario["goal"], scenario["ee"],
                    )
                    result = {"trial": trial, "head_type": head_type, "cost_mode": cost_mode,
                              "scenario": desc, **metrics}
                    all_results.append(result)
                    s = "✅" if metrics["final_success"] else "❌"
                    print(f"{s} p={metrics['object_progress']:.3f} c={metrics['contact_rate']:.2f} d={metrics['drift']:.3f}")
                except Exception as e:
                    print(f"❌ {e}")
                    import traceback; traceback.print_exc()
                    all_results.append({"trial": trial, "head_type": head_type, "cost_mode": cost_mode,
                                        "scenario": desc, "error": str(e)})

    # Save
    json_path = run_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    valid = [r for r in all_results if "error" not in r]
    if valid:
        csv_path = run_dir / "ablation_results.csv"
        with open(csv_path, "w") as f:
            keys = list(valid[0].keys())
            f.write(",".join(keys) + "\n")
            for r in valid:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    # Summary
    print(f"\n{'='*70}")
    print("ADALN vs CONCAT ABLATION SUMMARY")
    print(f"{'='*70}")

    for ht in HEAD_TYPES:
        for cm in COST_MODES:
            mr = [r for r in valid if r.get("head_type") == ht and r.get("cost_mode") == cm]
            if not mr:
                continue
            print(f"\n  {ht} + {cm}:")
            print(f"    trials: {len(mr)}")
            print(f"    success_rate: {np.mean([r['final_success'] for r in mr]):.2%}")
            print(f"    avg_progress: {np.mean([r['object_progress'] for r in mr]):.4f}")
            print(f"    avg_contact:  {np.mean([r['contact_rate'] for r in mr]):.4f}")
            print(f"    avg_drift:    {np.mean([r['drift'] for r in mr]):.4f}")

    # Per-scenario comparison
    print(f"\n{'─'*70}")
    print(f"{'scenario':<20} {'cost':<14} {'concat':>9} {'adaln':>9} {'delta':>9}")
    print(f"{'─'*70}")
    for scenario in SCENARIOS:
        desc = scenario["desc"]
        for cm in COST_MODES:
            cc = next((r for r in valid if r["head_type"]=="concat" and r["cost_mode"]==cm and r["scenario"]==desc), None)
            ac = next((r for r in valid if r["head_type"]=="adaln" and r["cost_mode"]==cm and r["scenario"]==desc), None)
            if cc and ac:
                d = ac["object_progress"] - cc["object_progress"]
                m = "↑" if d > 0 else ("↓" if d < 0 else "─")
                print(f"  {desc:<18} {cm:<14} {cc['object_progress']:>9.4f} {ac['object_progress']:>9.4f} {d:>+9.4f} {m}")

    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
