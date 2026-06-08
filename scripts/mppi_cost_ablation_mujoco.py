#!/usr/bin/env python3
"""
MPPI best config + new cost modes on low-success templates.

Templates from obstacle_difficulty_v0 (previous sweep showed low success):
  - ph07 (passage_direct_narrow:7) — 0% success
  - ph04 (passage_direct_narrow:4) — 33% success
  - bh00 (blocking_hard:0) — 57% success
  - bh01 (blocking_hard:1) — 57% success

MPPI best config: T=0.2, N=2048, H=120, speed=0.50 m/s

Run: conda activate lerobot && python3 scripts/mppi_cost_ablation_mujoco.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.envs.mujoco_push_env import MujocoPushEnv
from src.planners.mppi import MPPI
from src.planners.mujoco_oracle_rollout import rollout_action_sequence_mujoco
from src.planners.cost_modes import rollout_cost_with_mode

# ── MPPI best config ──
MPPI_CFG = {
    "temperature": 0.2,
    "num_samples": 512,
    "horizon": 60,
    "init_std": 0.5,
    "smoothing": 0.2,
    "max_speed_mps": 0.50,
}

# ── Templates (low success from previous sweep) ──
TEMPLATES_JSON = "data/sim/metadata/reset_templates_obstacle_difficulty_v0.json"
CORE_TEMPLATES = [
    ("bh00", "test_sim_layout_ood_blocking_hard", 0),
    ("bh01", "test_sim_layout_ood_blocking_hard", 1),
    ("ph04", "test_sim_layout_ood_passage_direct_narrow", 4),
    ("ph07", "test_sim_layout_ood_passage_direct_narrow", 7),
]

COST_MODES = ["current", "staged_full"]
SUCCESS_THRESH = 0.10
MAX_MPC_STEPS = 100
EXECUTE_STEPS = 10
SEED = 42


def load_template(split, idx):
    """Load one template from the difficulty json."""
    with open(TEMPLATES_JSON) as f:
        data = json.load(f)
    templates = [t for t in data if t["split"] == split]
    if idx >= len(templates):
        raise IndexError(f"Template index {idx} out of range for split {split} (len={len(templates)})")
    return templates[idx]


def extract_obstacle_info(template):
    """Extract obstacle positions and radii from template for cost function."""
    obs = template.get("obstacles", [])
    if not obs:
        return None, None
    positions = []
    radii = []
    for o in obs:
        pose = o.get("pose", o)  # handle nested pose or flat format
        positions.append([pose["x"], pose["y"]])
        radii.append(max(o.get("size_x", 0.05), o.get("size_y", 0.05)) / 2.0)
    return np.array(positions), np.array(radii)


def run_episode(env, template, cost_mode, mppi_cfg):
    """Run one closed-loop MPPI episode on MuJoCo."""
    env.reset_from_template(template)
    goal = env.get_goal_pose()
    init_obj = env.get_object_pose().copy()

    obs_pos, obs_rad = extract_obstacle_info(template)

    planner = MPPI(
        horizon=mppi_cfg["horizon"],
        num_samples=mppi_cfg["num_samples"],
        temperature=mppi_cfg["temperature"],
        init_std=mppi_cfg["init_std"],
        smoothing=mppi_cfg["smoothing"],
        seed=SEED,
    )

    contact_flags_all = []
    collision_flags_all = []
    obj_poses = [init_obj.copy()]
    ee_positions = [env.get_ee_pos().copy()]
    steps_executed = 0

    for mpc_step in range(MAX_MPC_STEPS // EXECUTE_STEPS):
        def cost_fn(action_seq):
            rollout = rollout_action_sequence_mujoco(env, action_seq, restore_state=True)
            return rollout_cost_with_mode(
                predicted_object_poses=rollout.predicted_object_poses,
                ee_positions=rollout.ee_positions,
                action_sequence=action_seq,
                goal_pose=np.asarray(goal, dtype=np.float64),
                cost_mode=cost_mode,
                contact_flags=rollout.contact_flags,
                collision_flags=rollout.collision_flags,
                obstacle_positions=obs_pos,
                obstacle_radii=obs_rad,
            )

        result = planner.optimize(cost_fn)

        for i in range(min(EXECUTE_STEPS, len(result.action_sequence))):
            action = result.action_sequence[i]
            env.step(action)
            steps_executed += 1

            obj_poses.append(env.get_object_pose().copy())
            ee_positions.append(env.get_ee_pos().copy())
            contact_flags_all.append(env.get_contact_flag())
            collision_flags_all.append(env.get_collision_flag())

            cur_obj = env.get_object_pose()
            dist = np.linalg.norm(cur_obj[:2] - goal[:2])
            if dist < SUCCESS_THRESH:
                break

        if np.linalg.norm(env.get_object_pose()[:2] - goal[:2]) < SUCCESS_THRESH:
            break

    # Metrics
    obj_poses = np.array(obj_poses)
    contact_flags = np.array(contact_flags_all)
    collision_flags = np.array(collision_flags_all)

    init_dist = float(np.linalg.norm(init_obj[:2] - goal[:2]))
    final_dist = float(np.linalg.norm(obj_poses[-1, :2] - goal[:2]))
    best_dist = float(np.min(np.linalg.norm(obj_poses[:, :2] - goal[:2], axis=-1)))

    contact_rate = float(np.mean(contact_flags > 0.5)) if len(contact_flags) > 0 else 0.0
    contact_indices = np.where(contact_flags > 0.5)[0]
    first_contact_step = int(contact_indices[0]) if len(contact_indices) > 0 else -1

    # Success step (first step where dist < threshold)
    dists = np.linalg.norm(obj_poses[:, :2] - goal[:2], axis=-1)
    success_indices = np.where(dists < SUCCESS_THRESH)[0]
    success_step = int(success_indices[0]) if len(success_indices) > 0 else -1

    return {
        "contact_rate": round(contact_rate, 4),
        "first_contact_step": first_contact_step,
        "object_progress": round(init_dist - final_dist, 4),
        "drift": round(max(0.0, final_dist - best_dist), 4),
        "final_success": bool(final_dist < SUCCESS_THRESH),
        "success_step": success_step,
        "init_dist": round(init_dist, 4),
        "final_dist": round(final_dist, 4),
        "best_dist": round(best_dist, 4),
        "n_steps": steps_executed,
        "collision_rate": round(float(np.mean(collision_flags > 0.5)), 4) if len(collision_flags) > 0 else 0.0,
    }


def main():
    run_dir = Path(__file__).resolve().parent.parent / "runs" / f"mppi_cost_ablation_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = MujocoPushEnv(max_speed_mps=MPPI_CFG["max_speed_mps"], pusher_mass=0.300)

    all_results = []
    total = len(CORE_TEMPLATES) * len(COST_MODES)
    trial = 0

    for label, split, idx in CORE_TEMPLATES:
        template = load_template(split, idx)
        print(f"\n{'='*60}")
        print(f"Template: {label} ({split}:{idx})")
        print(f"{'='*60}")

        for cost_mode in COST_MODES:
            trial += 1
            print(f"[{trial}/{total}] {label}/{cost_mode} ...", flush=True)

            try:
                metrics = run_episode(env, template, cost_mode, MPPI_CFG)
                result = {"trial": trial, "label": label, "split": split, "template_idx": idx,
                          "cost_mode": cost_mode, **metrics}
                all_results.append(result)

                status = "✅" if metrics["final_success"] else "❌"
                print(f"  {status} progress={metrics['object_progress']:.3f} "
                      f"contact={metrics['contact_rate']:.2f} "
                      f"1st_contact={metrics['first_contact_step']} "
                      f"drift={metrics['drift']:.3f} "
                      f"success_step={metrics['success_step']} "
                      f"collision={metrics['collision_rate']:.2f}")
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                import traceback; traceback.print_exc()
                all_results.append({"trial": trial, "label": label, "cost_mode": cost_mode, "error": str(e)})

    # Save
    json_path = run_dir / "mppi_cost_ablation.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    valid = [r for r in all_results if "error" not in r]
    if valid:
        csv_path = run_dir / "mppi_cost_ablation.csv"
        with open(csv_path, "w") as f:
            keys = list(valid[0].keys())
            f.write(",".join(keys) + "\n")
            for r in valid:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    # Summary
    print(f"\n{'='*70}")
    print("MPPI COST ABLATION SUMMARY")
    print(f"{'='*70}")

    for mode in COST_MODES:
        mr = [r for r in all_results if r.get("cost_mode") == mode and "error" not in r]
        if not mr:
            continue
        print(f"\n  {mode}:")
        print(f"    trials:          {len(mr)}")
        print(f"    success_rate:    {np.mean([r['final_success'] for r in mr]):.2%}")
        print(f"    avg_contact:     {np.mean([r['contact_rate'] for r in mr]):.4f}")
        print(f"    avg_progress:    {np.mean([r['object_progress'] for r in mr]):.4f}")
        print(f"    avg_drift:       {np.mean([r['drift'] for r in mr]):.4f}")
        fc = [r['first_contact_step'] for r in mr if r['first_contact_step'] >= 0]
        print(f"    avg_1st_contact: {np.mean(fc):.1f} steps" if fc else "    avg_1st_contact: N/A")
        ss = [r['success_step'] for r in mr if r['success_step'] >= 0]
        print(f"    avg_success_step:{np.mean(ss):.1f} steps" if ss else "    avg_success_step: N/A")

    # Per-template
    print(f"\n{'─'*70}")
    print(f"{'label':<8} {'mode':<14} {'success':>8} {'contact':>8} {'1st_c':>6} {'progress':>9} {'drift':>7} {'s_step':>7}")
    print(f"{'─'*70}")
    for r in valid:
        print(f"  {r['label']:<6} {r['cost_mode']:<14} {str(r['final_success']):>8} "
              f"{r['contact_rate']:>8.2f} {r['first_contact_step']:>6} "
              f"{r['object_progress']:>9.3f} {r['drift']:>7.3f} {r['success_step']:>7}")

    print(f"\nResults: {run_dir}")


if __name__ == "__main__":
    main()
