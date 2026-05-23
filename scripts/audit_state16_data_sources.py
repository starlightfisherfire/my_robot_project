#!/usr/bin/env python3
"""Audit all data/sim npz files for canonical state16 training readiness."""

import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FIELDS = ["states", "actions_norm", "actions_physical", "object_poses", "next_object_poses", "goal_pose"]


def classify_npz(path: Path) -> dict:
    """Classify a single npz file."""
    info = {
        "schema_type": "unknown_or_invalid",
        "T": 0, "N": 0, "D": 0,
        "has_states": False, "has_next_states": False,
        "has_actions_norm": False, "has_actions_physical": False,
        "has_object_poses": False, "has_next_object_poses": False,
        "has_goal_pose": False,
        "split_name": "", "family": "", "template_id": "",
        "shape_type": "", "obstacle_count": 0,
        "success": "", "policy_source": "",
        "state16_training_ready": False,
        "reason_if_not_ready": "",
    }
    try:
        # Make path relative to REPO_ROOT for reporting
        try:
            rel_path = path.resolve().relative_to(REPO_ROOT)
        except ValueError:
            rel_path = path
        info["episode_npz_path"] = str(rel_path)
        info["source_dir"] = str(rel_path.parent)

        d = np.load(str(path), allow_pickle=True)
        keys = set(d.keys())
        for f in REQUIRED_FIELDS:
            info[f"has_{f}"] = f in keys

        if "states" in keys:
            s = np.array(d["states"])
            info["T"] = s.shape[0]
            if s.ndim >= 2:
                info["N"] = s.shape[-2] if s.ndim >= 3 else 0
                info["D"] = s.shape[-1]
            if s.ndim == 3 and s.shape[1:] == (6, 16):
                info["schema_type"] = "canonical_state16"
                # Check next_states
                if "next_states" in keys:
                    ns = np.array(d["next_states"])
                    if ns.shape == s.shape:
                        info["has_next_states"] = True
                # Check actions
                if "actions_norm" in keys and np.array(d["actions_norm"]).shape[-1] == 2:
                    info["has_actions_norm"] = True
                if "actions_physical" in keys and np.array(d["actions_physical"]).shape[-1] == 2:
                    info["has_actions_physical"] = True
                # Check poses
                if "object_poses" in keys and np.array(d["object_poses"]).ndim >= 1:
                    info["has_object_poses"] = True
                if "next_object_poses" in keys:
                    info["has_next_object_poses"] = True
                if "goal_pose" in keys:
                    info["has_goal_pose"] = True

                # Ready?
                ready = (info["has_states"] and info["has_next_states"] and
                         info["has_actions_norm"] and info["has_actions_physical"] and
                         info["has_object_poses"] and info["has_next_object_poses"] and
                         info["has_goal_pose"] and info["T"] >= 20)
                info["state16_training_ready"] = ready
                if not ready:
                    missing = [f for f in ["states","next_states","actions_norm","actions_physical",
                                           "object_poses","next_object_poses","goal_pose"]
                               if not info[f"has_{f}"]]
                    if info["T"] < 20:
                        missing.append(f"T={info['T']}<20")
                    info["reason_if_not_ready"] = "; ".join(missing) if missing else "shape_mismatch"
            elif s.ndim == 2:
                info["schema_type"] = "compact_planner_rollout"
                info["reason_if_not_ready"] = "compact_state_2d"
            else:
                info["schema_type"] = "compact_planner_rollout"
                info["reason_if_not_ready"] = f"unexpected_shape_{tuple(s.shape)}"
        else:
            info["schema_type"] = "video_or_log_only"
            info["reason_if_not_ready"] = "no_states"
    except Exception as e:
        info["schema_type"] = "unknown_or_invalid"
        info["reason_if_not_ready"] = str(e)[:100]

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(REPO_ROOT / "data/sim"))
    parser.add_argument("--out", default=str(REPO_ROOT / "docs/layout_ood_state16_data_source_audit.md"))
    parser.add_argument("--csv", default=str(REPO_ROOT / "runs/_index/state16_source_inventory.csv"))
    args = parser.parse_args()

    source_root = Path(args.source_root)
    npz_files = sorted(source_root.rglob("*.npz"))
    print(f"Scanning {len(npz_files)} npz files in {source_root}...")

    rows = []
    for i, f in enumerate(npz_files):
        rows.append(classify_npz(f))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(npz_files)}")

    # Stats
    canonical = [r for r in rows if r["schema_type"] == "canonical_state16"]
    ready = [r for r in canonical if r["state16_training_ready"]]
    compact = [r for r in rows if r["schema_type"] == "compact_planner_rollout"]
    invalid = [r for r in rows if r["schema_type"] in ("unknown_or_invalid", "video_or_log_only")]

    # Family distribution (canonical only)
    family_dist = {}
    for r in canonical:
        f = r.get("family", "unknown") or "unknown"
        family_dist[f] = family_dist.get(f, 0) + 1

    # Source dir distribution
    src_dist = {}
    for r in canonical:
        d = r["source_dir"]
        src_dist[d] = src_dist.get(d, 0) + 1

    # Write CSV
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {csv_path}")

    # Write markdown
    md_path = Path(args.out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write("# Data Source Audit\n\n")
        f.write(f"- Total npz: {len(rows)}\n")
        f.write(f"- canonical_state16: {len(canonical)}\n")
        f.write(f"- state16_training_ready: {len(ready)}\n")
        f.write(f"- compact_planner_rollout: {len(compact)}\n")
        f.write(f"- invalid/video: {len(invalid)}\n\n")

        f.write("## Family distribution (canonical_state16)\n\n")
        for k, v in sorted(family_dist.items()):
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## Source dir distribution (canonical_state16)\n\n")
        for k, v in sorted(src_dist.items()):
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## Training-ready details\n\n")
        ready_families = {}
        for r in ready:
            rf = r.get("family", "unknown") or "unknown"
            ready_families[rf] = ready_families.get(rf, 0) + 1
        f.write(f"Total training-ready: {len(ready)}\n\n")
        for k, v in sorted(ready_families.items()):
            f.write(f"- {k}: {v}\n")

    print(f"Markdown: {md_path}")
    print(f"\n=== SUMMARY ===")
    print(f"Total: {len(rows)}, Canonical: {len(canonical)}, Ready: {len(ready)}, Compact: {len(compact)}, Invalid: {len(invalid)}")


if __name__ == "__main__":
    main()
