#!/usr/bin/env python3
"""
Analyze Speed × Horizon Ablation results.

Usage:
  python scripts/analyze_speed_horizon_ablation.py <run_root>

Reads manifest.csv and produces:
  1. Summary table (speed × horizon × mode)
  2. Success rate heatmap
  3. Best distance distribution
  4. Runtime comparison
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_speed_horizon_ablation.py <run_root>")
        sys.exit(1)

    run_root = Path(sys.argv[1])
    manifest_path = run_root / "manifest.csv"

    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found")
        sys.exit(1)

    # ── Parse manifest ──
    rows = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} results from {manifest_path}\n")

    # ── Group by (speed, horizon, mode) ──
    groups = defaultdict(list)
    for r in rows:
        key = (r["speed"], r["horizon"], r["mode"])
        groups[key].append(r)

    # ── Summary Table ──
    print("=" * 100)
    print("SUMMARY: Speed × Horizon × Mode")
    print("=" * 100)
    print(f"{'Speed':>6} {'Horizon':>7} {'Mode':<20} {'Succ%':>6} {'AvgDist':>8} {'MinDist':>8} {'Collisions':>10} {'AvgCost':>10} {'Runtime':>8}")
    print("-" * 100)

    for key in sorted(groups.keys()):
        speed, horizon, mode = key
        runs = groups[key]
        n = len(runs)
        n_succ = sum(1 for r in runs if r["status"] == "True")
        succ_pct = n_succ / n * 100 if n > 0 else 0

        dists = []
        costs = []
        colls = []
        runtimes = []
        for r in runs:
            try:
                dists.append(float(r["best_dist"]))
            except (ValueError, KeyError):
                pass
            try:
                costs.append(float(r["avg_cost"]))
            except (ValueError, KeyError):
                pass
            try:
                colls.append(int(r["collision_count"]))
            except (ValueError, KeyError):
                pass
            try:
                runtimes.append(float(r["runtime_sec"]))
            except (ValueError, KeyError):
                pass

        avg_dist = f"{sum(dists)/len(dists):.4f}" if dists else "N/A"
        min_dist = f"{min(dists):.4f}" if dists else "N/A"
        total_coll = sum(colls) if colls else 0
        avg_cost = f"{sum(costs)/len(costs):.4f}" if costs else "N/A"
        avg_rt = f"{sum(runtimes)/len(runtimes):.1f}s" if runtimes else "N/A"

        print(f"{speed:>6} {horizon:>7} {mode:<20} {succ_pct:>5.0f}% {avg_dist:>8} {min_dist:>8} {total_coll:>10} {avg_cost:>10} {avg_rt:>8}")

    # ── Speed Comparison (aggregated across horizons and modes) ──
    print()
    print("=" * 80)
    print("SPEED COMPARISON (aggregated)")
    print("=" * 80)

    speed_groups = defaultdict(list)
    for r in rows:
        speed_groups[r["speed"]].append(r)

    print(f"{'Speed':>6} {'N':>4} {'Succ%':>6} {'AvgDist':>8} {'MinDist':>8} {'TotalColl':>10}")
    print("-" * 50)
    for speed in sorted(speed_groups.keys(), key=float):
        runs = speed_groups[speed]
        n = len(runs)
        n_succ = sum(1 for r in runs if r["status"] == "True")
        succ_pct = n_succ / n * 100 if n > 0 else 0
        dists = [float(r["best_dist"]) for r in runs if r["best_dist"] != "N/A"]
        colls = [int(r["collision_count"]) for r in runs if r["collision_count"] != "N/A"]
        avg_dist = f"{sum(dists)/len(dists):.4f}" if dists else "N/A"
        min_dist = f"{min(dists):.4f}" if dists else "N/A"
        total_coll = sum(colls) if colls else 0
        print(f"{speed:>6} {n:>4} {succ_pct:>5.0f}% {avg_dist:>8} {min_dist:>8} {total_coll:>10}")

    # ── Horizon Comparison ──
    print()
    print("=" * 80)
    print("HORIZON COMPARISON (aggregated)")
    print("=" * 80)

    horizon_groups = defaultdict(list)
    for r in rows:
        horizon_groups[r["horizon"]].append(r)

    print(f"{'Horizon':>7} {'N':>4} {'Succ%':>6} {'AvgDist':>8} {'MinDist':>8} {'TotalColl':>10}")
    print("-" * 50)
    for horizon in sorted(horizon_groups.keys(), key=int):
        runs = horizon_groups[horizon]
        n = len(runs)
        n_succ = sum(1 for r in runs if r["status"] == "True")
        succ_pct = n_succ / n * 100 if n > 0 else 0
        dists = [float(r["best_dist"]) for r in runs if r["best_dist"] != "N/A"]
        colls = [int(r["collision_count"]) for r in runs if r["collision_count"] != "N/A"]
        avg_dist = f"{sum(dists)/len(dists):.4f}" if dists else "N/A"
        min_dist = f"{min(dists):.4f}" if dists else "N/A"
        total_coll = sum(colls) if colls else 0
        print(f"{horizon:>7} {n:>4} {succ_pct:>5.0f}% {avg_dist:>8} {min_dist:>8} {total_coll:>10}")

    # ── Per-template details ──
    print()
    print("=" * 100)
    print("PER-TEMPLATE DETAILS")
    print("=" * 100)
    print(f"{'Config':<30} {'Speed':>6} {'Hz':>4} {'Mode':<18} {'Tpl':>3} {'Status':>8} {'Dist':>8} {'Cost':>10} {'Coll':>5} {'RT':>8}")
    print("-" * 100)
    for r in rows:
        print(f"{r['config']:<30} {r['speed']:>6} {r['horizon']:>4} {r['mode']:<18} {r['template_idx']:>3} {r['status']:>8} {r['best_dist']:>8} {r.get('avg_cost','N/A'):>10} {r['collision_count']:>5} {r.get('runtime_sec','N/A'):>8}")


if __name__ == "__main__":
    main()
