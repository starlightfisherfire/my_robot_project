#!/usr/bin/env python3
"""Audit visual_state_v2 dataset."""
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--schema", default="configs/state_schema/visual_structured_state_v2.yaml")
    args = p.parse_args()
    ds = Path(args.dataset)
    meta = ds / "metadata" / "episodes.jsonl"
    eps_dir = ds / "episodes"
    if not meta.exists():
        print(f"WARN: No episodes.jsonl at {meta}")
        return
    with open(meta) as f:
        eps = [json.loads(l) for l in f if l.strip()]
    print(f"Episodes: {len(eps)}")
    valid = 0; nan = 0; missing = 0
    for ep in eps:
        npz = eps_dir / f"{ep['episode_id']}.npz"
        if not npz.exists():
            missing += 1; continue
        d = np.load(npz, allow_pickle=True)
        has_nan = False
        for k in d.keys():
            if isinstance(d[k], np.ndarray) and d[k].dtype.kind in 'fi':
                if np.any(np.isnan(d[k])): has_nan = True
        if has_nan: nan += 1
        else: valid += 1
    print(f"Valid: {valid}, NaN: {nan}, Missing: {missing}")
if __name__ == "__main__": main()
