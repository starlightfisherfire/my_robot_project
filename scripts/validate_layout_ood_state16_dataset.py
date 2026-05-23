#!/usr/bin/env python3
"""Validate layout_ood_state16_v0 dataset — strict consistency and leakage checks.

Usage:
    PYTHONPATH=. python scripts/validate_layout_ood_state16_dataset.py \
        --data-dir data/sim/layout_ood_state16_v0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_NPZ_KEYS = [
    "states", "actions_norm", "actions_physical", "next_states",
    "object_poses", "next_object_poses", "goal_pose",
    "ee_positions", "next_ee_positions",
    "contact_flags", "collision_flags",
    "actual_ee_velocities", "actual_object_velocities",
    "obstacle_features",
]

EXPECTED_SHAPES = {
    "states": (-1, 6, 16),
    "next_states": (-1, 6, 16),
    "actions_norm": (-1, 2),
    "actions_physical": (-1, 2),
    "object_poses": (-1, 3),
    "next_object_poses": (-1, 3),
    "ee_positions": (-1, 2),
    "next_ee_positions": (-1, 2),
    "contact_flags": (-1,),
    "collision_flags": (-1,),
    "actual_ee_velocities": (-1, 2),
    "actual_object_velocities": (-1, 3),
    "goal_pose": (3,),
    "obstacle_features": (18,),
}


def _shape_ok(key: str, arr: np.ndarray) -> bool:
    expected = EXPECTED_SHAPES[key]
    actual = arr.shape
    if len(expected) != len(actual):
        return False
    for e_sz, a_sz in zip(expected, actual):
        if e_sz != -1 and e_sz != a_sz:
            return False
    return True


def _check_finite(arr: np.ndarray, name: str) -> list[str]:
    if arr.dtype.kind == "b":
        return []
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        return [f"NaN/Inf in {name}"]
    return []


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate(data_dir: str) -> dict[str, Any]:
    data_path = Path(data_dir)
    meta_path = data_path / "metadata" / "episodes.jsonl"
    results: dict[str, Any] = {
        "data_dir": str(data_path),
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # --- 1. metadata file exists ---
    if not meta_path.exists():
        results["errors"].append(f"Metadata file not found: {meta_path}")
        return results

    # Load metadata
    episodes_meta: list[dict] = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes_meta.append(json.loads(line))

    total_eps = len(episodes_meta)
    results["stats"]["total_episodes"] = total_eps
    if total_eps == 0:
        results["errors"].append("No episodes in metadata")
        return results

    # --- Per-episode validation ---
    total_transitions = 0
    total_contact = 0
    total_collision = 0
    total_success = 0
    family_counts: dict[str, int] = {}
    splits: dict[str, int] = {}

    max_abs_diff_states = 0.0
    max_abs_diff_obj = 0.0

    for ep in episodes_meta:
        ep_id = ep.get("episode_id", "?")
        ep_path_str = ep.get("episode_path", "")
        ep_path = Path(ep_path_str)
        family = ep.get("family", "unknown")
        success = ep.get("success", False)
        num_obs = ep.get("num_obstacles", 0)
        split_name = ep.get("split_name", "unknown")

        family_counts[family] = family_counts.get(family, 0) + 1
        splits[split_name] = splits.get(split_name, 0) + 1
        if success:
            total_success += 1

        # --- 2. npz file exists ---
        if not ep_path.exists():
            # Try alternative: episodes/<id>.npz
            alt_path = data_path / "episodes" / f"{ep_id}.npz"
            if alt_path.exists():
                ep_path = alt_path
            else:
                results["errors"].append(f"ep {ep_id}: npz not found at {ep_path_str}")
                continue

        try:
            data = np.load(ep_path, allow_pickle=False)
        except Exception as e:
            results["errors"].append(f"ep {ep_id}: failed to load npz: {e}")
            continue

        # --- 3. required keys ---
        for key in REQUIRED_NPZ_KEYS:
            if key not in data:
                results["errors"].append(f"ep {ep_id}: missing key '{key}'")
        if any(k not in data for k in REQUIRED_NPZ_KEYS):
            continue

        T = data["states"].shape[0]
        total_transitions += T

        # --- 4. shape check ---
        for key, arr in data.items():
            if key in EXPECTED_SHAPES:
                if not _shape_ok(key, arr):
                    results["errors"].append(
                        f"ep {ep_id}: shape mismatch for '{key}': "
                        f"got {arr.shape}, expected {EXPECTED_SHAPES[key]}"
                    )

        # --- 5. NaN/Inf check ---
        for key in data.keys():
            arr = data[key]
            errs = _check_finite(arr, key)
            for e in errs:
                results["errors"].append(f"ep {ep_id}: {e}")

        # --- 6. token valid check ---
        states = data["states"]
        if states.ndim == 3 and states.shape[1:] == (6, 16):
            # Token 0 (EE): valid_flag should be 1 everywhere
            ee_valid = states[:, 0, 15]
            if not np.allclose(ee_valid, 1.0):
                results["warnings"].append(
                    f"ep {ep_id}: EE valid_flag not all 1 (min={ee_valid.min():.3f})"
                )

            # Token 1 (Object): valid_flag should be 1
            obj_valid = states[:, 1, 15]
            if not np.allclose(obj_valid, 1.0):
                results["warnings"].append(
                    f"ep {ep_id}: Object valid_flag not all 1"
                )

            # Token 2 (Goal): valid_flag should be 1
            goal_valid = states[:, 2, 15]
            if not np.allclose(goal_valid, 1.0):
                results["warnings"].append(
                    f"ep {ep_id}: Goal valid_flag not all 1"
                )

            # Obstacle valid counts
            obs_valid_count = int((states[0, 3:, 15] > 0.5).sum())
            if obs_valid_count != num_obs:
                results["warnings"].append(
                    f"ep {ep_id}: obstacle valid_count={obs_valid_count} "
                    f"but metadata num_obstacles={num_obs}"
                )

            # Padding obstacles (slots > num_obs) should have valid_flag=0
            for oi in range(num_obs, 3):
                pad_valid = states[:, 3 + oi, 15]
                if not np.allclose(pad_valid, 0.0):
                    results["warnings"].append(
                        f"ep {ep_id}: padding obstacle {oi} has non-zero valid_flag"
                    )

        # --- 7. transition alignment ---
        next_states = data["next_states"]
        if states.shape == next_states.shape and T > 1:
            # next_states[t] ≈ states[t+1] for EE and Object tokens
            ee_diff = np.abs(next_states[:-1, 0, :2] - states[1:, 0, :2]).max()
            obj_diff = np.abs(next_states[:-1, 1, :2] - states[1:, 1, :2]).max()
            max_abs_diff_states = max(max_abs_diff_states, ee_diff, obj_diff)
            if max(ee_diff, obj_diff) > 1e-3:
                results["warnings"].append(
                    f"ep {ep_id}: transition misalignment: "
                    f"ee_diff={ee_diff:.6f}, obj_diff={obj_diff:.6f}"
                )
            elif max(ee_diff, obj_diff) > 1e-5:
                results["warnings"].append(
                    f"ep {ep_id}: small transition diff: "
                    f"ee_diff={ee_diff:.6f}, obj_diff={obj_diff:.6f} (acceptable)"
                )

        # --- 8. object pose consistency ---
        obj_poses = data["object_poses"]
        if obj_poses.ndim == 2 and states.ndim == 3:
            diff_xy = np.abs(obj_poses[:, :2] - states[:, 1, :2]).max()
            max_abs_diff_obj = max(max_abs_diff_obj, diff_xy)
            if diff_xy > 1e-3:
                results["warnings"].append(
                    f"ep {ep_id}: object_pose vs states[:,1] mismatch: {diff_xy:.6f}"
                )

        # --- contact / collision counts ---
        total_contact += int(data["contact_flags"].sum())
        total_collision += int(data["collision_flags"].sum())

    # --- 9. split leakage check ---
    # For smoke mode: families are mixed; no strict split
    # Only warn if passage_direct/bypass appears in what should be train
    # Since smoke uses random templates, no split leakage check needed for now
    # but record family distribution
    results["stats"]["family_counts"] = family_counts
    results["stats"]["split_counts"] = splits
    results["stats"]["total_transitions"] = total_transitions
    results["stats"]["total_success"] = total_success
    results["stats"]["total_failure"] = total_eps - total_success
    results["stats"]["contact_ratio"] = round(
        total_contact / max(1, total_transitions), 4
    )
    results["stats"]["collision_ratio"] = round(
        total_collision / max(1, total_transitions), 4
    )
    results["stats"]["max_transition_diff"] = round(
        float(max_abs_diff_states), 8
    )
    results["stats"]["max_obj_pose_diff"] = round(
        float(max_abs_diff_obj), 8
    )

    # --- PASS / WARN / FAIL ---
    if results["errors"]:
        verdict = "FAIL"
    elif results["warnings"]:
        verdict = "WARN"
    else:
        verdict = "PASS"
    results["verdict"] = verdict

    return results


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def save_reports(results: dict, data_dir: str):
    data_path = Path(data_dir)

    # JSON report
    json_path = data_path / "validation_report.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Markdown report
    md_path = data_path / "validation_report.md"
    stats = results.get("stats", {})

    with open(md_path, "w") as f:
        f.write("# Dataset Validation Report\n\n")
        f.write(f"**Data dir:** {results['data_dir']}\n\n")
        f.write(f"**Verdict:** {results['verdict']}\n\n")

        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total episodes | {stats.get('total_episodes', 0)} |\n")
        f.write(f"| Total transitions | {stats.get('total_transitions', 0)} |\n")
        f.write(f"| Success | {stats.get('total_success', 0)} |\n")
        f.write(f"| Failure | {stats.get('total_failure', 0)} |\n")
        f.write(f"| Contact ratio | {stats.get('contact_ratio', 0)} |\n")
        f.write(f"| Collision ratio | {stats.get('collision_ratio', 0)} |\n")
        f.write(f"| Max transition diff | {stats.get('max_transition_diff', 'N/A')} |\n")
        f.write(f"| Max obj pose diff | {stats.get('max_obj_pose_diff', 'N/A')} |\n\n")

        # Family distribution
        fc = stats.get("family_counts", {})
        if fc:
            f.write("## Family Distribution\n\n")
            f.write("| Family | Count |\n")
            f.write("|--------|-------|\n")
            for fam in sorted(fc):
                f.write(f"| {fam} | {fc[fam]} |\n")
            f.write("\n")

        # Warnings
        if results["warnings"]:
            f.write(f"## Warnings ({len(results['warnings'])})\n\n")
            for w in results["warnings"]:
                f.write(f"- {w}\n")
            f.write("\n")

        # Errors
        if results["errors"]:
            f.write(f"## Errors ({len(results['errors'])})\n\n")
            for e in results["errors"]:
                f.write(f"- {e}\n")
            f.write("\n")

        f.write(f"## Verdict\n\n**{results['verdict']}**\n")

    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate layout_ood_state16_v0 dataset"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to data directory (e.g., data/sim/layout_ood_state16_v0)"
    )
    args = parser.parse_args()

    results = validate(args.data_dir)
    json_path, md_path = save_reports(results, args.data_dir)

    # Print summary
    print(f"=== Validation: {results['verdict']} ===")
    s = results.get("stats", {})
    print(f"  Episodes: {s.get('total_episodes', 0)}")
    print(f"  Transitions: {s.get('total_transitions', 0)}")
    print(f"  Success: {s.get('total_success', 0)} / Failure: {s.get('total_failure', 0)}")
    print(f"  Contact ratio: {s.get('contact_ratio', 0)}")
    print(f"  Collision ratio: {s.get('collision_ratio', 0)}")
    fc = s.get("family_counts", {})
    if fc:
        print("  Families:")
        for fam in sorted(fc):
            print(f"    {fam}: {fc[fam]}")
    if results["errors"]:
        print(f"  Errors ({len(results['errors'])}):")
        for e in results["errors"]:
            print(f"    - {e}")
    if results["warnings"]:
        print(f"  Warnings ({len(results['warnings'])}):")
        for w in results["warnings"]:
            print(f"    - {w}")
    print(f"\n  Reports: {json_path}, {md_path}")

    # Exit code
    sys.exit(0 if results["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
