#!/usr/bin/env python3
"""Collect canonical state16 data for layout_ood_state16_v1.

Uses existing collector functions but with Stage 2C MPPI config.

Usage:
    PYTHONPATH=. python scripts/collect_layout_ood_state16_v1.py --dry-run
    PYTHONPATH=. python scripts/collect_layout_ood_state16_v1.py --smoke
    PYTHONPATH=. python scripts/collect_layout_ood_state16_v1.py --mini
    PYTHONPATH=. python scripts/collect_layout_ood_state16_v1.py --full
"""
from __future__ import annotations

import argparse, json, random, sys, time, os
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.episode_writer import EpisodeWriter
from src.envs.mujoco_push_env import MujocoPushEnv
from src.planners.cost_functions import CostWeights
from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost
from src.planners.mppi import MPPI
from scripts.collect_layout_ood_state16 import (
    _extract_structured_state,
    _fill_obstacle_tokens,
    _obstacle_features_from_template,
    run_one_episode,
)

TEMPLATE_FILE = "data/sim/metadata/reset_templates_obstacle_10family_v0.json"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_templates(template_file: str) -> dict[str, list[dict]]:
    with open(template_file) as f:
        all_templates = json.load(f)
    by_family: dict[str, list[dict]] = {}
    for t in all_templates:
        fam = t["family"]
        by_family.setdefault(fam, []).append(t)
    for fam in by_family:
        by_family[fam].sort(key=lambda t: t.get("reset_template_id", ""))
    return by_family


def build_jobs(splits_cfg: dict, templates_by_family: dict) -> list[dict]:
    """Build job list from splits config."""
    jobs = []
    for split_name, family_counts in splits_cfg.items():
        for family, target_count in family_counts.items():
            if family not in templates_by_family:
                print(f"  [WARN] family '{family}' not in templates, skipping")
                continue
            fam_templates = templates_by_family[family]
            if not fam_templates:
                continue
            n_templates = len(fam_templates)
            # Distribute episodes across templates
            eps_per_template = max(1, target_count // n_templates)
            remainder = target_count - eps_per_template * n_templates
            for t_idx, tpl in enumerate(fam_templates):
                n = eps_per_template + (1 if t_idx < remainder else 0)
                for _ in range(n):
                    jobs.append({
                        "split_name": split_name,
                        "family": family,
                        "template": tpl,
                        "template_index": t_idx,
                    })
    return jobs


def dry_run(jobs: list[dict]):
    """Print job summary without running."""
    split_counts = {}
    family_counts = {}
    train_families = set()
    ood_families = set()

    for j in jobs:
        s, f = j["split_name"], j["family"]
        split_counts[s] = split_counts.get(s, 0) + 1
        family_counts[f] = family_counts.get(f, 0) + 1
        if "train" in s:
            train_families.add(f)
        if "ood" in s:
            ood_families.add(f)

    print(f"=== DRY RUN: {len(jobs)} total episodes ===\n")
    print("By split:")
    for s in sorted(split_counts):
        print(f"  {s}: {split_counts[s]}")
    print("\nBy family:")
    for f in sorted(family_counts):
        print(f"  {f}: {family_counts[f]}")

    print(f"\nTrain families: {sorted(train_families)}")
    print(f"OOD families: {sorted(ood_families)}")

    # Check: train should not have passage
    train_has_passage = any("passage" in f for f in train_families)
    print(f"\nTrain contains passage: {train_has_passage}")
    if train_has_passage:
        print("  ⚠️ LEAKAGE RISK: passage in train!")

    # Check: OOD should not have train-only families
    ood_has_blocking = any("blocking" in f for f in ood_families)
    print(f"OOD contains blocking: {ood_has_blocking}")


def collect_episodes(
    jobs: list[dict],
    output_dir: str,
    mppi_cfg: dict,
    noise_cfg: dict,
    cost_w: CostWeights,
    max_steps: int,
    control_dt: float,
    smoke: bool = False,
    resume: bool = True,
):
    """Run collection."""
    data_path = Path(output_dir)
    writer = EpisodeWriter(output_dir=data_path, run_root="layout_ood_state16_v1")
    env = MujocoPushEnv(shape_type="T", control_dt=control_dt, max_speed_mps=mppi_cfg.get("max_speed_mps", 0.3))

    # Check for existing episodes (resume support)
    existing_ids = set()
    if resume:
        meta_path = data_path / "metadata" / "episodes.jsonl"
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    d = json.loads(line)
                    existing_ids.add(d["episode_id"])
            if existing_ids:
                print(f"[RESUME] Found {len(existing_ids)} existing episodes, will skip them")

    rng = random.Random(0)
    stats = {"total": 0, "success": 0, "failure": 0, "skipped": 0,
             "by_family": {}, "by_split": {}, "total_steps": 0}
    t_start = time.time()
    consecutive_failures = 0

    for job_idx, job in enumerate(jobs):
        split_name = job["split_name"]
        family = job["family"]
        template = job["template"]
        template_index = job["template_index"]

        ep_seed = 1000 + job_idx
        episode_id = f"v1_{split_name}_{family}_{template_index:03d}_{ep_seed}"

        # Resume skip
        if episode_id in existing_ids:
            stats["skipped"] += 1
            continue

        # Noise
        nc = rng.random()
        if nc < noise_cfg["zero_noise_prob"]:
            noise_std = 0.0
        elif nc < noise_cfg["zero_noise_prob"] + noise_cfg["low_noise_prob"]:
            noise_std = 0.05
        else:
            noise_std = 0.15

        try:
            ep_stats = run_one_episode(
                env=env, template=template, mppi_cfg=mppi_cfg, cost_w=cost_w,
                noise_std=noise_std, ep_seed=ep_seed, max_steps=max_steps,
                writer=writer, split_name=split_name,
            )
            consecutive_failures = 0
        except Exception as e:
            print(f"  [ERROR] {family}/{split_name} ep {job_idx}: {e}")
            consecutive_failures += 1
            if consecutive_failures >= 10:
                print("  [ABORT] 10 consecutive failures, stopping")
                break
            continue

        stats["total"] += 1
        if ep_stats["success"]:
            stats["success"] += 1
        else:
            stats["failure"] += 1
        stats["total_steps"] += writer.step_count
        stats["by_family"].setdefault(family, {"total": 0, "success": 0})
        stats["by_family"][family]["total"] += 1
        if ep_stats["success"]:
            stats["by_family"][family]["success"] += 1
        stats["by_split"].setdefault(split_name, {"total": 0, "success": 0})
        stats["by_split"][split_name]["total"] += 1
        if ep_stats["success"]:
            stats["by_split"][split_name]["success"] += 1

        # Progress
        total_target = len(jobs) - stats["skipped"]
        done = stats["total"]
        if done % 10 == 0 or done == total_target:
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_target - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total_target}] ok={stats['success']}/{stats['total']} "
                  f"({elapsed:.0f}s, ~{rate:.2f} ep/min, ETA {eta:.0f}s) "
                  f"| {family} | {split_name}")

        if smoke and done >= 10:
            break

    elapsed = time.time() - t_start
    print(f"\n=== Collection complete ===")
    print(f"  Total: {stats['total']} episodes, {stats['total_steps']} transitions")
    print(f"  Success: {stats['success']} ({100*stats['success']/max(1,stats['total']):.1f}%)")
    print(f"  Skipped (resume): {stats['skipped']}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)\n")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/collect_layout_ood_state16_v1.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    planner_cfg = cfg["planner"]
    noise_cfg = cfg["noise"]
    episode_cfg = cfg["episode"]
    splits_cfg = cfg["splits"]

    # Build MPPI config dict for run_one_episode
    mppi_cfg = {
        "horizon": planner_cfg["horizon"],
        "num_samples": planner_cfg["num_samples"],
        "num_iterations": planner_cfg["num_iterations"],
        "init_std": planner_cfg["init_std"],
        "temperature": planner_cfg["temperature"],
        "smoothing": planner_cfg["smoothing"],
        "max_speed_mps": planner_cfg["max_speed_mps"],
    }

    cost_w = CostWeights(
        w_pos=10.0, w_theta=2.0, w_reach=5.0, w_no_contact=2.0,
        w_push_alignment=1.0, w_collision=20.0, w_collision_step=1.0,
        w_proximity=5.0, w_action=0.05, w_smooth=0.1, w_subgoal=0.0,
    )

    templates_by_family = load_templates(TEMPLATE_FILE)
    jobs = build_jobs(splits_cfg, templates_by_family)

    if args.dry_run:
        print("=== DRY RUN MODE ===\n")
        print(f"MPPI config: {json.dumps(mppi_cfg, indent=2)}")
        print(f"Speed: {planner_cfg['max_speed_mps']} mps")
        print()
        dry_run(jobs)
        return

    output_dir = cfg["experiment"]["output_dir"]

    if args.smoke:
        # Smoke: 1 episode per family, max 10 episodes, light MPPI
        smoke_splits = {}
        smoke_families = list(set(f for split in splits_cfg.values() for f in split))
        for fam in smoke_families[:10]:  # max 10 families
            smoke_splits["smoke"] = {fam: 1}
        jobs = build_jobs(smoke_splits, templates_by_family)
        output_dir = output_dir + "_smoke"
        # Use lighter MPPI for smoke
        mppi_cfg = {"horizon": 30, "num_samples": 256, "num_iterations": 2,
                    "init_std": 0.5, "temperature": 0.1, "smoothing": 0.2,
                    "max_speed_mps": 0.3}
        episode_cfg["max_steps"] = 50
        print("=== SMOKE MODE ===")
        print(f"Jobs: {len(jobs)}, light MPPI, max_steps=50")
        print()

    elif args.mini:
        # Mini: 2 episodes per family
        mini_splits = {}
        for split, families in splits_cfg.items():
            mini_splits[split] = {f: 2 for f in families}
        jobs = build_jobs(mini_splits, templates_by_family)
        output_dir = output_dir + "_mini"
        print("=== MINI MODE ===\n")

    print(f"Output: {output_dir}")
    print(f"Total jobs: {len(jobs)}")

    stats = collect_episodes(
        jobs=jobs,
        output_dir=output_dir,
        mppi_cfg=mppi_cfg,
        noise_cfg=noise_cfg,
        cost_w=cost_w,
        max_steps=episode_cfg["max_steps"],
        control_dt=0.1,
        smoke=args.smoke,
        resume=not args.smoke,
    )

    print(f"\n[DONE] Output: {output_dir}")


if __name__ == "__main__":
    main()
