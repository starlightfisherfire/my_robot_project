#!/usr/bin/env python3
"""
audit_state16_transitions.py — Audit state16 dataset transition quality.

Scans data/sim/layout_ood_state16_* npz files and checks:
- Total episodes / transitions
- Success episodes
- Contact transitions
- Object moved transitions
- Zero movement transitions
- Near-goal-failed episodes
- By family / split / source counts
- Key validation: states/actions alignment, state dim = 16

Output: runs/self_check/state16_transition_audit.json

Usage:
    PYTHONPATH=. python scripts/audit_state16_transitions.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "runs" / "self_check" / "state16_transition_audit.json"


def audit_npz(npz_path: str) -> dict:
    """Audit a single npz episode file."""
    data = np.load(npz_path, allow_pickle=True)

    result = {"valid": True, "errors": []}

    # Check required keys
    required = ["states", "actions_physical", "object_poses", "next_object_poses", "goal_pose"]
    missing = [k for k in required if k not in data]
    if missing:
        result["valid"] = False
        result["errors"].append(f"missing_keys: {missing}")
        return result

    states = data["states"]
    actions = data["actions_physical"]
    obj_poses = data["object_poses"]
    next_obj_poses = data["next_object_poses"]
    goal_pose = data["goal_pose"]

    # State shape
    if states.ndim != 3 or states.shape[1] != 6 or states.shape[2] != 16:
        result["valid"] = False
        result["errors"].append(f"states_shape: {states.shape}, expected [T, 6, 16]")

    # Alignment
    T = states.shape[0] if states.ndim >= 1 else 0
    if len(actions) != T:
        result["errors"].append(f"actions_length_mismatch: states={T}, actions={len(actions)}")
    if len(obj_poses) != T:
        result["errors"].append(f"obj_poses_length_mismatch: states={T}, obj_poses={len(obj_poses)}")
    if len(next_obj_poses) != T:
        result["errors"].append(f"next_obj_poses_length_mismatch: states={T}, next_obj_poses={len(next_obj_poses)}")

    result["num_transitions"] = T

    # Contact transitions
    if states.ndim == 3 and states.shape[2] > 14:
        contact_flags = states[:, 0, 14]  # EE contact flag
        result["contact_transitions"] = int(np.sum(contact_flags > 0.5))

    # Object moved
    if len(obj_poses) > 1 and len(next_obj_poses) > 1:
        deltas = np.linalg.norm(next_obj_poses[:len(obj_poses), :2] - obj_poses[:, :2], axis=-1)
        result["object_moved_transitions"] = int(np.sum(deltas > 0.001))
        result["zero_movement_transitions"] = int(np.sum(deltas <= 0.001))

    # Goal distance
    goal_xy = goal_pose[:2]
    if len(obj_poses) > 0:
        final_dist = float(np.linalg.norm(obj_poses[-1, :2] - goal_xy))
        result["final_dist_to_goal"] = final_dist
        result["near_goal_failed"] = final_dist < 0.05  # within 5cm

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-print", type=int, default=20)
    args = parser.parse_args()

    sim_dir = REPO_ROOT / "data" / "sim"
    datasets = sorted(sim_dir.glob("layout_ood_state16_*"))

    report = {
        "datasets": {},
        "total_episodes": 0,
        "total_transitions": 0,
        "total_valid": 0,
        "total_invalid": 0,
        "total_success": 0,
        "total_contact_transitions": 0,
        "total_object_moved": 0,
        "total_zero_movement": 0,
        "total_near_goal_failed": 0,
        "by_family": {},
        "by_split": {},
    }

    for ds_dir in datasets:
        meta_path = ds_dir / "metadata" / "episodes.jsonl"
        ep_dir = ds_dir / "episodes"

        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            episodes = [json.loads(line) for line in f if line.strip()]

        ds_info = {
            "num_episodes": len(episodes),
            "num_transitions": 0,
            "valid": 0,
            "invalid": 0,
            "success": 0,
            "contact_transitions": 0,
            "object_moved": 0,
            "zero_movement": 0,
            "near_goal_failed": 0,
            "by_family": {},
            "by_split": {},
        }

        for ep in episodes:
            ep_id = ep["episode_id"]
            family = ep.get("family", "unknown")
            split = ep.get("split_name", "unknown")
            success = ep.get("success", False)
            num_t = ep.get("num_transitions", 0)

            npz_path = ep_dir / f"{ep_id}.npz"
            if not npz_path.exists():
                ds_info["invalid"] += 1
                continue

            audit = audit_npz(str(npz_path))

            ds_info["num_transitions"] += audit.get("num_transitions", num_t)
            if audit["valid"]:
                ds_info["valid"] += 1
            else:
                ds_info["invalid"] += 1

            if success:
                ds_info["success"] += 1

            ds_info["contact_transitions"] += audit.get("contact_transitions", 0)
            ds_info["object_moved"] += audit.get("object_moved_transitions", 0)
            ds_info["zero_movement"] += audit.get("zero_movement_transitions", 0)
            if audit.get("near_goal_failed", False):
                ds_info["near_goal_failed"] += 1

            ds_info["by_family"][family] = ds_info["by_family"].get(family, 0) + 1
            ds_info["by_split"][split] = ds_info["by_split"].get(split, 0) + 1

        report["datasets"][ds_dir.name] = ds_info
        report["total_episodes"] += ds_info["num_episodes"]
        report["total_transitions"] += ds_info["num_transitions"]
        report["total_valid"] += ds_info["valid"]
        report["total_invalid"] += ds_info["invalid"]
        report["total_success"] += ds_info["success"]
        report["total_contact_transitions"] += ds_info["contact_transitions"]
        report["total_object_moved"] += ds_info["object_moved"]
        report["total_zero_movement"] += ds_info["zero_movement"]
        report["total_near_goal_failed"] += ds_info["near_goal_failed"]

        for fam, cnt in ds_info["by_family"].items():
            report["by_family"][fam] = report["by_family"].get(fam, 0) + cnt
        for spl, cnt in ds_info["by_split"].items():
            report["by_split"][spl] = report["by_split"].get(spl, 0) + cnt

    # Quality assessment
    n = report["total_episodes"]
    if n > 0:
        report["quality"] = {
            "valid_rate": report["total_valid"] / n,
            "success_rate": report["total_success"] / n,
            "contact_rate": report["total_contact_transitions"] / max(report["total_transitions"], 1),
            "object_moved_rate": report["total_object_moved"] / max(report["total_transitions"], 1),
            "zero_movement_rate": report["total_zero_movement"] / max(report["total_transitions"], 1),
            "near_goal_failed_rate": report["total_near_goal_failed"] / n,
        }

        q = report["quality"]
        if q["success_rate"] < 0.1:
            report["recommendation"] = "RE_COLLECT: success_rate too low for meaningful training"
        elif q["zero_movement_rate"] > 0.5:
            report["recommendation"] = "INVESTIGATE: too many zero-movement transitions"
        elif n < 100:
            report["recommendation"] = "COLLECT_MORE: need more episodes for training"
        else:
            report["recommendation"] = "READY_FOR_TRAINING"
    else:
        report["recommendation"] = "NO_DATA"

    # Print summary
    print("=" * 60)
    print("STATE16 TRANSITION AUDIT")
    print("=" * 60)
    print(f"  Total episodes: {report['total_episodes']}")
    print(f"  Total transitions: {report['total_transitions']}")
    print(f"  Valid: {report['total_valid']}, Invalid: {report['total_invalid']}")
    print(f"  Success: {report['total_success']}")
    print(f"  Contact transitions: {report['total_contact_transitions']}")
    print(f"  Object moved: {report['total_object_moved']}")
    print(f"  Zero movement: {report['total_zero_movement']}")
    print(f"  Near-goal failed: {report['total_near_goal_failed']}")
    print(f"  By family: {report['by_family']}")
    print(f"  By split: {report['by_split']}")
    if "quality" in report:
        print(f"\n  Quality:")
        for k, v in report["quality"].items():
            print(f"    {k}: {v:.3f}")
    print(f"\n  Recommendation: {report['recommendation']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
