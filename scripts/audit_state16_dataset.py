#!/usr/bin/env python3
"""
audit_state16_dataset.py — Audit state16 dataset for Paper 1.

Scans data/sim/layout_ood_state16_* directories and reports:
    - Episode count by split/family/shape/success
    - Sample count per dataset
    - Which datasets can be used for training smoke
    - Which datasets cannot be used for main results

Usage:
    python scripts/audit_state16_dataset.py [--max-print 20]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def scan_dataset(base_dir: Path) -> dict:
    """Scan a state16 dataset directory."""
    meta_path = base_dir / "metadata" / "episodes.jsonl"
    ep_dir = base_dir / "episodes"

    if not meta_path.exists():
        return {"path": str(base_dir), "error": "no episodes.jsonl"}

    with open(meta_path) as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    # Count .npz files
    npz_files = list(ep_dir.glob("*.npz")) if ep_dir.exists() else []

    # Statistics
    families = Counter(ep.get("family", "unknown") for ep in episodes)
    splits = Counter(ep.get("split_name", "unknown") for ep in episodes)
    successes = Counter(ep.get("success", False) for ep in episodes)
    obstacles = Counter(ep.get("num_obstacles", 0) for ep in episodes)

    # Sample count (approximate)
    total_transitions = sum(ep.get("num_transitions", 0) for ep in episodes)

    return {
        "path": str(base_dir),
        "episodes": len(episodes),
        "npz_files": len(npz_files),
        "families": dict(families),
        "splits": dict(splits),
        "success_true": successes.get(True, 0),
        "success_false": successes.get(False, 0),
        "obstacles": dict(obstacles),
        "total_transitions": total_transitions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-print", type=int, default=20)
    args = parser.parse_args()

    # Find all state16 datasets
    sim_dir = REPO_ROOT / "data" / "sim"
    datasets = sorted(sim_dir.glob("layout_ood_state16_*"))

    if not datasets:
        print("No layout_ood_state16_* datasets found in data/sim/")
        return

    print("=" * 80)
    print("STATE16 DATASET AUDIT")
    print("=" * 80)

    for ds_dir in datasets:
        info = scan_dataset(ds_dir)
        print(f"\n📁 {ds_dir.name}")
        print(f"   Episodes: {info.get('episodes', 0)}")
        print(f"   NPZ files: {info.get('npz_files', 0)}")
        print(f"   Transitions: {info.get('total_transitions', 0)}")

        if "error" in info:
            print(f"   ❌ Error: {info['error']}")
            continue

        print(f"   Families: {info['families']}")
        print(f"   Splits: {info['splits']}")
        print(f"   Success: True={info['success_true']}, False={info['success_false']}")
        print(f"   Obstacles: {info['obstacles']}")

        # Usability assessment
        n_ep = info["episodes"]
        n_success = info["success_true"]

        if n_ep >= 50 and n_success > 0:
            print(f"   ✅ Usable for training + eval")
        elif n_ep >= 10:
            print(f"   ⚠️ Usable for smoke test only ({n_ep} episodes, {n_success} success)")
        else:
            print(f"   ❌ Too small for meaningful use ({n_ep} episodes)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_eps = 0
    total_success = 0
    for ds_dir in datasets:
        info = scan_dataset(ds_dir)
        if "error" not in info:
            total_eps += info["episodes"]
            total_success += info["success_true"]

    print(f"  Total episodes: {total_eps}")
    print(f"  Total success: {total_success}")
    print(f"  Datasets: {len(datasets)}")

    if total_eps < 100:
        print(f"\n  ⚠️ DATA INSUFFICIENT for main results")
        print(f"  Need: ~500+ episodes per split for meaningful training")
        print(f"  Have: {total_eps} episodes total")

    if total_success == 0:
        print(f"\n  ⚠️ NO SUCCESSFUL EPISODES")
        print(f"  All data has success=False")
        print(f"  Training on failed episodes may still learn dynamics,")
        print(f"  but cannot evaluate task success.")


if __name__ == "__main__":
    main()
