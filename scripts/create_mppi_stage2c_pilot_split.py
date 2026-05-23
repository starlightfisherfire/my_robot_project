#!/usr/bin/env python3
"""
create_mppi_stage2c_pilot_split.py — Create train/val/test splits for mppi_stage2c_state16 pilot.

Outputs: configs/splits/mppi_stage2c_state16_pilot.yaml

Usage:
    PYTHONPATH=. python scripts/create_mppi_stage2c_pilot_split.py
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
META_PATH = REPO_ROOT / "data" / "sim" / "mppi_stage2c_state16" / "metadata" / "episodes.jsonl"
OUT_PATH = REPO_ROOT / "configs" / "splits" / "mppi_stage2c_state16_pilot.yaml"


def main():
    with open(META_PATH) as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    print(f"Total episodes: {len(episodes)}")

    # Group by family
    by_family = defaultdict(list)
    for ep in episodes:
        by_family[ep.get("family", "unknown")].append(ep["episode_id"])

    print(f"Families: {dict((k, len(v)) for k, v in by_family.items())}")

    # A. Random episode split (stratified by family)
    rng = random.Random(42)
    train_ids, val_ids, test_ids = [], [], []

    for fam, ep_ids in sorted(by_family.items()):
        rng.shuffle(ep_ids)
        n = len(ep_ids)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))
        train_ids.extend(ep_ids[:n_train])
        val_ids.extend(ep_ids[n_train:n_train + n_val])
        test_ids.extend(ep_ids[n_train + n_val:])

    # Verify no overlap
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0

    print(f"\nRandom split:")
    print(f"  train: {len(train_ids)}")
    print(f"  val:   {len(val_ids)}")
    print(f"  test:  {len(test_ids)}")

    # B. Family holdout split
    train_fams = ["blocking_hard", "passage_direct_narrow"]
    test_ood_fams = ["passage_bypass_wide", "passage_bypass_medium", "passage_bypass_narrow"]

    holdout_train = []
    holdout_val = []
    holdout_test_ood = []

    for fam in train_fams:
        ids = by_family.get(fam, [])
        rng.shuffle(ids)
        n_train = max(1, int(len(ids) * 0.85))
        holdout_train.extend(ids[:n_train])
        holdout_val.extend(ids[n_train:])

    for fam in test_ood_fams:
        holdout_test_ood.extend(by_family.get(fam, []))

    print(f"\nFamily holdout split:")
    print(f"  train (blocking_hard + passage_direct_narrow): {len(holdout_train)}")
    print(f"  val (from train families): {len(holdout_val)}")
    print(f"  test_ood (passage_bypass): {len(holdout_test_ood)}")

    # Build YAML
    split_cfg = {
        "version": "0.1",
        "source": "mppi_stage2c_state16",
        "description": "Pilot splits for Paper 1 state16 representation learning",
        "splits": {
            "random_episode_split": {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            },
            "family_holdout_split": {
                "train": holdout_train,
                "val": holdout_val,
                "test_ood": holdout_test_ood,
            },
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        yaml.dump(split_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved to: {OUT_PATH}")

    # Print distributions
    ep_map = {ep["episode_id"]: ep for ep in episodes}
    for split_name, split_data in split_cfg["splits"].items():
        print(f"\n=== {split_name} ===")
        for part, ids in split_data.items():
            fams = Counter(ep_map[i].get("family", "?") for i in ids if i in ep_map)
            succ = sum(1 for i in ids if i in ep_map and ep_map[i].get("success", False))
            print(f"  {part}: {len(ids)} episodes, success={succ}, families={dict(fams)}")


if __name__ == "__main__":
    main()
