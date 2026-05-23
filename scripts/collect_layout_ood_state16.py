#!/usr/bin/env python3
"""Data collection for layout_ood_state16_v0 experiment.

Modes:
    --smoke:     10 random episodes, weak MPPI (fast verification)
    --small-main: Fixed templates, strong MPPI, labeled train/val/test splits
    (default):   Random templates, main MPPI (large-scale)

Usage:
    PYTHONPATH=. python scripts/collect_layout_ood_state16.py --smoke
    PYTHONPATH=. python scripts/collect_layout_ood_state16.py --small-main
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from src.data.episode_writer import EpisodeWriter
from src.data.template_generator import generate_template, is_template_valid
from src.envs.mujoco_push_env import MujocoPushEnv
from src.planners.cost_functions import CostWeights
from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost
from src.planners.mppi import MPPI

# ---- Config defaults --------------------------------------------------------

DEFAULT_CONFIG = Path("configs/experiments/layout_ood_state16_v0.yaml")
TEMPLATE_FILE = "data/sim/metadata/reset_templates_obstacle_10family_v0.json"

DEFAULT_CONFIG_DICT = {
    "experiment": {"name": "layout_ood_state16_v0", "seed": 0},
    "collect": {
        "output_dir": "data/sim/layout_ood_state16_v0",
        "max_steps_per_episode": 250,
        "control_dt": 0.1,
        "noise": {
            "zero_noise_prob": 0.5, "low_noise_prob": 0.3, "high_noise_prob": 0.2,
        },
        "cost_weights": {
            "w_pos": 10.0, "w_theta": 2.0, "w_reach": 5.0, "w_no_contact": 2.0,
            "w_push_alignment": 1.0, "w_collision": 20.0, "w_collision_step": 1.0,
            "w_proximity": 5.0, "w_action": 0.05, "w_smooth": 0.1, "w_subgoal": 0.0,
        },
    },
}

SMOKE_MPPI = {"horizon": 8, "num_samples": 128, "num_iterations": 2,
             "init_std": 0.5, "temperature": 0.1, "smoothing": 0.2}
SMALL_MAIN_MPPI = {"horizon": 12, "num_samples": 512, "num_iterations": 3,
                   "init_std": 0.5, "temperature": 0.1, "smoothing": 0.2}

# ---- Split definitions for small-main ---------------------------------------

SMALL_MAIN_SPLITS = {
    "train": {
        "open": list(range(0, 5)),
        "blocking_easy": list(range(0, 5)),
        "blocking_medium": list(range(0, 5)),
        "blocking_hard": list(range(0, 5)),
        "episodes_per_template": 2,
    },
    "val_single": {
        "blocking_easy": [6],
        "blocking_medium": [6],
        "blocking_hard": [6],
        "episodes_per_template": 1,
    },
    "test_single_heldout": {
        "blocking_easy": [7, 8, 9],
        "blocking_medium": [7, 8, 9],
        "blocking_hard": [7, 8, 9],
        "episodes_per_template": 1,
    },
    "test_dual_layout_ood": {
        "passage_direct_wide": [0, 1, 2, 3],
        "passage_direct_medium": [0, 1, 2, 3],
        "passage_direct_narrow": [0, 1, 2, 3],
        "episodes_per_template": 1,
    },
}

# ---- Helpers -----------------------------------------------------------------

def _check_success(env: MujocoPushEnv, pos_threshold: float = 0.05) -> bool:
    obj = env.get_object_pose()
    goal = env.get_goal_pose()
    return float(np.linalg.norm(obj[:2] - goal[:2])) < pos_threshold


def _extract_structured_state(env: MujocoPushEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    ee_p = env.get_ee_pos()
    obj_p = env.get_object_pose()
    goal_p = env.get_goal_pose()
    contact = env.get_contact_flag()
    collision = env.get_collision_flag()

    tokens = np.zeros((6, 16), dtype=np.float32)
    tokens[0, 0] = ee_p[0]; tokens[0, 1] = ee_p[1]; tokens[0, 3] = 1.0
    tokens[0, 14] = float(contact); tokens[0, 15] = 1.0
    tokens[1, 0] = obj_p[0]; tokens[1, 1] = obj_p[1]
    tokens[1, 2] = np.sin(obj_p[2]); tokens[1, 3] = np.cos(obj_p[2])
    tokens[1, 7] = 0.048; tokens[1, 8] = 0.048; tokens[1, 9] = 1.0
    tokens[1, 12] = 0.038; tokens[1, 13] = 0.8; tokens[1, 15] = 1.0
    tokens[2, 0] = goal_p[0]; tokens[2, 1] = goal_p[1]
    tokens[2, 2] = np.sin(goal_p[2]); tokens[2, 3] = np.cos(goal_p[2])
    tokens[2, 7] = 0.048; tokens[2, 8] = 0.048; tokens[2, 9] = 1.0
    tokens[2, 12] = 0.038; tokens[2, 13] = 0.8; tokens[2, 15] = 1.0
    return tokens, obj_p, ee_p, contact, collision


def _fill_obstacle_tokens(tokens: np.ndarray, template: dict) -> np.ndarray:
    for oi, obs in enumerate(template.get("obstacles", [])[:3]):
        idx = 3 + oi
        tokens[idx, 0] = obs["pose"]["x"]; tokens[idx, 1] = obs["pose"]["y"]
        tokens[idx, 3] = 1.0
        tokens[idx, 7] = obs["size_x"]; tokens[idx, 8] = obs["size_y"]
        tokens[idx, 12] = 0.5; tokens[idx, 13] = 0.8; tokens[idx, 15] = 1.0
    return tokens


def _obstacle_features_from_template(template: dict) -> np.ndarray:
    feat = np.zeros(18, dtype=np.float32)
    for oi, obs in enumerate(template.get("obstacles", [])[:3]):
        base = oi * 6
        feat[base:base+6] = [obs["pose"]["x"], obs["pose"]["y"], 0.0,
                              obs["size_x"], obs["size_y"], 1.0]
    return feat


# ---- Episode runner ----------------------------------------------------------

def run_one_episode(
    env: MujocoPushEnv,
    template: dict,
    mppi_cfg: dict,
    cost_w: CostWeights,
    noise_std: float,
    ep_seed: int,
    max_steps: int,
    writer: EpisodeWriter,
    split_name: str = "train",
) -> dict:
    """Run one episode. Returns stats dict."""
    np_rng = np.random.default_rng(ep_seed)
    env.reset_from_template(template)
    goal_pose = env.get_goal_pose().copy()
    num_obstacles = len(template.get("obstacles", []))
    family = template.get("family", "unknown")

    mppi = MPPI(
        horizon=mppi_cfg["horizon"], action_dim=2,
        num_samples=mppi_cfg["num_samples"],
        num_iterations=mppi_cfg["num_iterations"],
        init_std=mppi_cfg["init_std"],
        temperature=mppi_cfg["temperature"],
        smoothing=mppi_cfg["smoothing"],
        seed=ep_seed,
    )

    ep_success = False
    best_dist = 999.0
    coll_cnt = 0
    contact_cnt = 0
    prev_ee = env.get_ee_pos().copy()
    prev_obj = env.get_object_pose().copy()
    obs_feat = _obstacle_features_from_template(template)

    for step in range(max_steps):
        def cost_fn(seq):
            return mujoco_oracle_rollout_cost(env, seq, goal_pose, cost_w, restore_state=True)

        try:
            action, _ = mppi.plan(cost_fn)
        except Exception:
            action = np.zeros(2)

        if noise_std > 0:
            action = np.clip(action + np_rng.normal(0, noise_std, 2), -1.0, 1.0)

        action_phys = action * env.max_speed_mps

        tokens, obj_p, ee_p, contact, collision = _extract_structured_state(env)
        tokens = _fill_obstacle_tokens(tokens, template)

        ee_v = (ee_p - prev_ee) / 0.1
        obj_v = (obj_p - prev_obj) / 0.1
        tokens[1, 4] = obj_v[0]; tokens[1, 5] = obj_v[1]; tokens[1, 6] = obj_v[2]
        prev_ee = ee_p.copy(); prev_obj = obj_p.copy()

        env.step(action)
        next_tokens, nobj_p, nee_p, ncontact, ncollision = _extract_structured_state(env)
        next_tokens = _fill_obstacle_tokens(next_tokens, template)

        writer.add_step(tokens, action, action_phys, next_tokens,
                        obj_p, nobj_p, ee_p, nee_p,
                        ncontact, ncollision, ee_v, obj_v)

        dist = float(np.linalg.norm(nobj_p[:2] - goal_pose[:2]))
        if dist < best_dist: best_dist = dist
        if ncollision: coll_cnt += 1
        if ncontact: contact_cnt += 1

        if dist < 0.05:
            ep_success = True
            break

    episode_id = writer.save_episode(
        goal_pose=goal_pose,
        obstacle_features=obs_feat.copy(),
        metadata={
            "family": family,
            "split_name": split_name,
            "template_source": template.get("template_source_file", "random"),
            "template_id": template.get("reset_template_id", ""),
            "num_obstacles": num_obstacles,
            "success": ep_success,
            "best_dist": round(best_dist, 6),
            "collision_count": coll_cnt,
            "contact_count": contact_cnt,
            "noise_std": noise_std,
            "seed": ep_seed,
        },
    )

    return {
        "success": ep_success,
        "best_dist": best_dist,
        "collision_count": coll_cnt,
        "contact_count": contact_cnt,
        "family": family,
        "split_name": split_name,
    }


# ---- Small-main mode ---------------------------------------------------------

def run_small_main(output_dir: str, mppi_cfg: dict, noise_cfg: dict,
                   cost_w: CostWeights, max_steps: int, control_dt: float):
    """Collect small-main dataset using fixed templates + labeled splits."""
    data_path = Path(output_dir)
    writer = EpisodeWriter(output_dir=data_path, run_root="layout_ood_state16_v0")
    env = MujocoPushEnv(shape_type="T", control_dt=control_dt)

    # Load templates
    with open(TEMPLATE_FILE) as f:
        all_templates = json.load(f)

    # Index templates by family
    templates_by_family: dict[str, list[dict]] = {}
    for t in all_templates:
        fam = t["family"]
        templates_by_family.setdefault(fam, []).append(t)

    # Sort within each family by reset_template_id for deterministic ordering
    for fam in templates_by_family:
        templates_by_family[fam].sort(key=lambda t: t["reset_template_id"])

    # Build job list
    jobs: list[tuple[str, str, dict]] = []  # (split_name, family, template)
    for split_name, split_cfg in SMALL_MAIN_SPLITS.items():
        eps_per = split_cfg["episodes_per_template"]
        for family, indices in split_cfg.items():
            if family == "episodes_per_template":
                continue
            if family not in templates_by_family:
                print(f"  [WARN] family '{family}' not found in templates, skipping")
                continue
            fam_templates = templates_by_family[family]
            for idx in indices:
                if idx >= len(fam_templates):
                    print(f"  [WARN] family '{family}' index {idx} out of range (max {len(fam_templates)-1})")
                    continue
                for _ in range(eps_per):
                    jobs.append((split_name, family, fam_templates[idx]))

    total_eps = len(jobs)
    print(f"=== small-main: {total_eps} episodes ===")
    # Summarize
    split_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for s, f, _ in jobs:
        split_counts[s] = split_counts.get(s, 0) + 1
        family_counts[f] = family_counts.get(f, 0) + 1
    for s in sorted(split_counts):
        print(f"  {s}: {split_counts[s]} episodes")
    for f in sorted(family_counts):
        print(f"    {f}: {family_counts[f]}")
    print()

    rng = random.Random(0)
    stats = {"total": 0, "success": 0, "failure": 0, "by_family": {}, "by_split": {},
             "total_steps": 0}
    t_start = time.time()

    for job_idx, (split_name, family, template) in enumerate(jobs):
        ep_seed = 1000 + job_idx
        nc = rng.random()
        if nc < noise_cfg["zero_noise_prob"]:
            noise_std = 0.0
        elif nc < noise_cfg["zero_noise_prob"] + noise_cfg["low_noise_prob"]:
            noise_std = 0.05
        else:
            noise_std = 0.15

        ep_stats = run_one_episode(
            env=env, template=template, mppi_cfg=mppi_cfg, cost_w=cost_w,
            noise_std=noise_std, ep_seed=ep_seed, max_steps=max_steps,
            writer=writer, split_name=split_name,
        )

        stats["total"] += 1
        if ep_stats["success"]: stats["success"] += 1
        else: stats["failure"] += 1
        stats["total_steps"] += writer.step_count
        stats["by_family"].setdefault(family, {"total": 0, "success": 0})
        stats["by_family"][family]["total"] += 1
        if ep_stats["success"]: stats["by_family"][family]["success"] += 1
        stats["by_split"].setdefault(split_name, {"total": 0, "success": 0})
        stats["by_split"][split_name]["total"] += 1
        if ep_stats["success"]: stats["by_split"][split_name]["success"] += 1

        # Progress
        if (job_idx + 1) % 5 == 0 or job_idx == 0 or job_idx == total_eps - 1:
            elapsed = time.time() - t_start
            rate = (job_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total_eps - job_idx - 1) / rate if rate > 0 else 0
            print(f"  [{job_idx+1}/{total_eps}] ok={stats['success']}/{stats['total']} "
                  f"({elapsed:.0f}s, ~{rate:.2f} ep/min, ETA {eta:.0f}s) "
                  f"| {family} | {split_name}")

    elapsed = time.time() - t_start
    print(f"\n=== small-main complete ===")
    print(f"  Total: {stats['total']} episodes, {stats['total_steps']} transitions")
    print(f"  Success: {stats['success']} ({100*stats['success']/max(1,stats['total']):.1f}%)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)\n")
    print("  By split:")
    for s in ["train", "val_single", "test_single_heldout", "test_dual_layout_ood"]:
        if s in stats["by_split"]:
            ss = stats["by_split"][s]
            print(f"    {s}: {ss['success']}/{ss['total']} "
                  f"({100*ss['success']/max(1,ss['total']):.1f}%)")
    print()
    print("  By family:")
    for f in sorted(stats["by_family"]):
        fs = stats["by_family"][f]
        print(f"    {f}: {fs['success']}/{fs['total']} "
              f"({100*fs['success']/max(1,fs['total']):.1f}%)")

    # Save manifest
    manifest = {
        "mode": "small_main",
        "config": {"mppi": mppi_cfg, "noise": noise_cfg},
        "stats": {
            "total_episodes": stats["total"],
            "total_transitions": stats["total_steps"],
            "success": stats["success"],
            "failure": stats["failure"],
            "elapsed_seconds": elapsed,
            "by_split": {k: dict(v) for k, v in stats["by_split"].items()},
            "by_family": {k: dict(v) for k, v in stats["by_family"].items()},
        },
    }
    manifest_path = data_path / "metadata" / "small_main_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest: {manifest_path}")


# ---- Smoke mode --------------------------------------------------------------

def run_smoke(output_dir: str, mppi_cfg: dict, noise_cfg: dict,
              cost_w: CostWeights, max_steps: int, control_dt: float):
    """Run 10 random-template episodes for smoke test."""
    data_path = Path(output_dir)
    writer = EpisodeWriter(output_dir=data_path, run_root="layout_ood_state16_v0")
    env = MujocoPushEnv(shape_type="T", control_dt=control_dt)

    families = ["open", "blocking", "passage"]
    probs = [0.15, 0.55, 0.30]
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)

    total = 10
    stats = {"total": 0, "success": 0, "by_family": {}}
    t_start = time.time()

    for ep in range(total):
        ep_seed = ep
        ep_rng = random.Random(ep_seed)
        family = ep_rng.choices(families, weights=probs, k=1)[0]

        for _ in range(100):
            tpl = generate_template(family, ep_rng)
            if is_template_valid(tpl):
                break
        else:
            continue

        nc = ep_rng.random()
        if nc < noise_cfg["zero_noise_prob"]: noise_std = 0.0
        elif nc < noise_cfg["zero_noise_prob"] + noise_cfg["low_noise_prob"]: noise_std = 0.05
        else: noise_std = 0.15

        ep_stats = run_one_episode(
            env=env, template=tpl, mppi_cfg=mppi_cfg, cost_w=cost_w,
            noise_std=noise_std, ep_seed=ep_seed, max_steps=max_steps,
            writer=writer, split_name="smoke",
        )

        stats["total"] += 1
        if ep_stats["success"]: stats["success"] += 1
        stats["by_family"].setdefault(family, {"total": 0, "success": 0})
        stats["by_family"][family]["total"] += 1
        if ep_stats["success"]: stats["by_family"][family]["success"] += 1

        elapsed = time.time() - t_start
        print(f"[{ep+1}/{total}] {family:10s} | ok={ep_stats['success']} "
              f"| d={ep_stats['best_dist']:.3f} | {elapsed:.0f}s")

    elapsed = time.time() - t_start
    print(f"\n=== smoke done: {stats['success']}/{stats['total']} | {elapsed:.0f}s")
    for f in sorted(stats["by_family"]):
        s = stats["by_family"][f]
        print(f"  {f}: {s['success']}/{s['total']}")


# ---- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect layout_ood_state16_v0 data")
    parser.add_argument("--smoke", action="store_true", help="10 random episodes (fast)")
    parser.add_argument("--small-main", action="store_true",
                        help="Small-scale fixed-template collection with labeled splits")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    # Load config
    config = DEFAULT_CONFIG_DICT
    config_path = Path(args.config)
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except (ImportError, Exception):
            pass

    cfg = config["collect"]
    output_dir = cfg["output_dir"]
    max_steps = cfg["max_steps_per_episode"]
    control_dt = cfg["control_dt"]
    noise_cfg = cfg["noise"]
    cost_w = CostWeights(**cfg["cost_weights"])

    if args.smoke:
        print("=== SMOKE MODE ===")
        run_smoke(output_dir, SMOKE_MPPI, noise_cfg, cost_w, max_steps, control_dt)

    elif args.small_main:
        print("=== SMALL-MAIN MODE ===")
        run_small_main(output_dir, SMALL_MAIN_MPPI, noise_cfg, cost_w, max_steps, control_dt)

    else:
        print("No mode selected. Use --smoke or --small-main")
        sys.exit(1)


if __name__ == "__main__":
    main()
