#!/usr/bin/env python3
"""Gate 2: Dataset smoke test."""

import argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import yaml

from src.data.episode_loader import State16Dataset
from src.data.state_normalizer import StateNormalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/train_state16_poc.yaml")
    parser.add_argument("--limit-samples", type=int, default=64)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]
    nc = cfg["normalizer"]
    seed = cfg["experiment"]["seed"]

    print("=== Gate 2: Dataset Smoke ===\n")

    # 1. Read metadata
    meta_path = Path(dc["metadata"])
    if not meta_path.exists():
        print(f"[FAIL] Metadata not found: {meta_path}")
        sys.exit(1)
    print(f"[PASS] Metadata exists: {meta_path}")

    # 2. Create datasets
    train_ds = State16Dataset(
        metadata_path=dc["metadata"],
        episode_root=f"{dc['root']}/episodes",
        history_len=dc["history_len"],
        token_count=dc["token_count"],
        state_dim=dc["state_dim"],
        split="train",
        train_ratio=dc["train_ratio_if_no_split"],
        val_ratio=dc["val_ratio_if_no_split"],
        limit_samples=args.limit_samples,
        seed=seed,
    )
    val_ds = State16Dataset(
        metadata_path=dc["metadata"],
        episode_root=f"{dc['root']}/episodes",
        history_len=dc["history_len"],
        token_count=dc["token_count"],
        state_dim=dc["state_dim"],
        split="val",
        train_ratio=dc["train_ratio_if_no_split"],
        val_ratio=dc["val_ratio_if_no_split"],
        seed=seed,
    )

    if len(train_ds) == 0:
        print("[FAIL] No train samples")
        sys.exit(1)
    print(f"[PASS] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 3. Check sample shapes
    sample = train_ds[0]
    H, N, D = dc["history_len"], dc["token_count"], dc["state_dim"]
    
    checks = [
        ("history shape [H,N,D]", sample["history"].shape, (H, N, D)),
        ("action shape [2]", sample["action"].shape, (2,)),
        ("dynamics_target shape [3]", sample["dynamics_target"].shape, (3,)),
        ("subgoal_target shape [3]", sample["subgoal_target"].shape, (3,)),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            print(f"[FAIL] {name}: got {actual}, expected {expected}")
            sys.exit(1)
        print(f"[PASS] {name}: {tuple(actual)}")

    # 4. Check finite
    for key in ["history", "action", "dynamics_target", "subgoal_target"]:
        v = sample[key]
        if not torch_isfinite(v):
            print(f"[FAIL] {key} has NaN/Inf")
            sys.exit(1)
    print("[PASS] All values finite")

    # 5. Normalizer fit on train only
    all_train_states = []
    for i in range(min(len(train_ds), 200)):
        all_train_states.append(train_ds[i]["history"].numpy())
    all_train_states = np.array(all_train_states).reshape(-1, D)

    normalizer = StateNormalizer()
    valid = all_train_states[:, normalizer.valid_flag_index] > 0.5
    if valid.sum() == 0:
        print("[FAIL] No valid train tokens for normalizer")
        sys.exit(1)
    normalizer.fit(all_train_states)
    print("[PASS] Normalizer fit on train only")

    # 6. Val transform
    val_sample = val_ds[0]["history"].numpy()
    transformed = normalizer.transform(val_sample)
    assert transformed.shape == val_sample.shape
    assert np.isfinite(transformed).all()
    print("[PASS] Val transform successful")

    print(f"\n=== Gate 2: PASS ===")
    print(f"  Train episodes: {len(train_ds.episodes)}")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val episodes: {len(val_ds.episodes)}")
    print(f"  Val samples: {len(val_ds)}")


def torch_isfinite(x):
    import torch
    return torch.isfinite(x).all().item()


if __name__ == "__main__":
    main()
