#!/usr/bin/env python3
"""
audit_runs_summary.py — Audit existing runs for Paper 1.

Scans runs/ directory and reports:
    - Available run types (oracle_mpc, training, mppi, etc.)
    - Key metrics from manifest.csv / summary files
    - Whether each run can serve as a main result or just debug

Usage:
    python scripts/audit_runs_summary.py [--max-print 20]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def classify_run(run_dir: Path) -> str:
    """Classify run type based on directory name."""
    name = run_dir.name.lower()
    if "mppi" in name:
        return "mppi_sweep"
    if "horizon140" in name:
        return "oracle_mpc_sweep"
    if "speed_horizon" in name:
        return "oracle_mpc_sweep"
    if "dual_obstacle" in name:
        return "oracle_mpc_sweep"
    if "heavy_pusher" in name:
        return "oracle_mpc_sweep"
    if "best_bh" in name or "best_config" in name:
        return "oracle_mpc_sweep"
    if "planner_trio" in name:
        return "oracle_mpc_sweep"
    if "train_state16" in name:
        return "training"
    if "layout_ood" in name:
        return "data_collection"
    if "obstacle" in name:
        return "obstacle_debug"
    if "video" in name or "render" in name:
        return "visualization"
    if "debug" in name:
        return "debug"
    return "unknown"


def read_manifest(run_dir: Path) -> list[dict]:
    """Read manifest.csv if it exists."""
    manifest_path = run_dir / "manifest.csv"
    if not manifest_path.exists():
        return []

    rows = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize_manifest(rows: list[dict]) -> dict:
    """Compute summary stats from manifest rows."""
    if not rows:
        return {}

    n = len(rows)
    successes = sum(1 for r in rows if str(r.get("status", "")).lower() == "true"
                    or str(r.get("success", "")).lower() == "true")

    # Try to get best_dist
    best_dists = []
    for r in rows:
        try:
            d = float(r.get("best_dist", r.get("best_pos_dist_m", "nan")))
            if d == d:  # not nan
                best_dists.append(d)
        except (ValueError, TypeError):
            pass

    return {
        "total_runs": n,
        "successes": successes,
        "success_rate": successes / n if n > 0 else 0,
        "best_dist_min": min(best_dists) if best_dists else None,
        "best_dist_mean": sum(best_dists) / len(best_dists) if best_dists else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-print", type=int, default=20)
    args = parser.parse_args()

    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        print("No runs/ directory found.")
        return

    # Collect all run directories
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    print("=" * 80)
    print("RUNS AUDIT SUMMARY")
    print("=" * 80)

    # Group by type
    by_type = defaultdict(list)
    for rd in run_dirs:
        run_type = classify_run(rd)
        by_type[run_type].append(rd)

    for run_type, dirs in sorted(by_type.items()):
        print(f"\n📁 {run_type} ({len(dirs)} runs)")

        for rd in dirs[:args.max_print]:
            manifest = read_manifest(rd)
            summary = summarize_manifest(manifest)

            line = f"  {rd.name}"
            if summary:
                line += f" | runs={summary['total_runs']}"
                line += f" | success_rate={summary['success_rate']:.1%}"
                if summary["best_dist_min"] is not None:
                    line += f" | best_dist_min={summary['best_dist_min']:.4f}"
            else:
                # Check for other files
                has_csv = any(rd.glob("*.csv"))
                has_json = any(rd.glob("*.json"))
                has_log = any(rd.glob("*.log"))
                extras = []
                if has_csv:
                    extras.append("csv")
                if has_json:
                    extras.append("json")
                if has_log:
                    extras.append("log")
                if extras:
                    line += f" | files: {','.join(extras)}"

            print(line)

        if len(dirs) > args.max_print:
            print(f"  ... and {len(dirs) - args.max_print} more")

    # Usability assessment
    print("\n" + "=" * 80)
    print("USABILITY ASSESSMENT")
    print("=" * 80)

    print("\n  Can serve as Oracle-MPC capacity evidence:")
    for rd in by_type.get("oracle_mpc_sweep", []):
        manifest = read_manifest(rd)
        summary = summarize_manifest(manifest)
        if summary and summary["success_rate"] > 0:
            print(f"    ✅ {rd.name} ({summary['success_rate']:.0%} success)")

    print("\n  Can serve as training pipeline evidence:")
    for rd in by_type.get("training", []):
        gate_report = rd / "gate_report.md"
        if gate_report.exists():
            print(f"    ✅ {rd.name} (gate report exists)")

    print("\n  Can serve as MPPI planner evidence:")
    for rd in by_type.get("mppi_sweep", []):
        manifest = read_manifest(rd)
        summary = summarize_manifest(manifest)
        if summary and summary["success_rate"] > 0.5:
            print(f"    ✅ {rd.name} ({summary['success_rate']:.0%} success)")

    print("\n  ⚠️ NOT usable as main results (debug only):")
    for run_type in ["debug", "visualization", "obstacle_debug", "data_collection"]:
        for rd in by_type.get(run_type, []):
            print(f"    ❌ {rd.name}")


if __name__ == "__main__":
    main()
