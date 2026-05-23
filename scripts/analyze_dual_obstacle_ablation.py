#!/usr/bin/env python3
"""
Analyze Dual-Obstacle Speed Ablation results.

Usage:
  python scripts/analyze_dual_obstacle_ablation.py <run_root>

Reads manifest.csv and produces summary tables + comparisons.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_dual_obstacle_ablation.py <run_root>")
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

    # ── Group by (speed, mode) ──
    groups = defaultdict(list)
    for r in rows:
        key = (r["speed"], r["mode"])
        groups[key].append(r)

    # ── Summary Table ──
    print("=" * 105)
    print("SUMMARY: Speed × Mode (Dual-Obstacle, horizon=80)")
    print("=" * 105)
    print(f"{'Speed':>6} {'Mode':<22} {'N':>3} {'Succ%':>6} {'AvgDist':>9} {'MinDist':>9} {'Collisions':>10} {'AvgCost':>10} {'AvgRT':>8}")
    print("-" * 105)

    for key in sorted(groups.keys(), key=lambda x: (float(x[0]), x[1])):
        speed, mode = key
        runs = groups[key]
        n = len(runs)
        n_succ = sum(1 for r in runs if r["status"] == "True")
        succ_pct = n_succ / n * 100 if n > 0 else 0

        dists = [float(r["best_dist"]) for r in runs if r["best_dist"] != "N/A"]
        costs = [float(r["avg_cost"]) for r in runs if r.get("avg_cost", "N/A") != "N/A"]
        colls = [int(r["collision_count"]) for r in runs if r["collision_count"] != "N/A"]
        runtimes = [float(r["runtime_sec"]) for r in runs if r.get("runtime_sec", "N/A") != "N/A"]

        avg_dist = f"{sum(dists)/len(dists):.4f}" if dists else "N/A"
        min_dist = f"{min(dists):.4f}" if dists else "N/A"
        total_coll = sum(colls) if colls else 0
        avg_cost = f"{sum(costs)/len(costs):.4f}" if costs else "N/A"
        avg_rt = f"{sum(runtimes)/len(runtimes):.1f}s" if runtimes else "N/A"

        print(f"{speed:>6} {mode:<22} {n:>3} {succ_pct:>5.0f}% {avg_dist:>9} {min_dist:>9} {total_coll:>10} {avg_cost:>10} {avg_rt:>8}")

    # ── Speed Comparison (aggregated across modes) ──
    print()
    print("=" * 80)
    print("SPEED COMPARISON (aggregated across all passage modes)")
    print("=" * 80)

    speed_groups = defaultdict(list)
    for r in rows:
        speed_groups[r["speed"]].append(r)

    print(f"{'Speed':>6} {'N':>4} {'Succ%':>6} {'AvgDist':>9} {'MinDist':>9} {'TotalColl':>10} {'AvgCost':>10}")
    print("-" * 65)
    for speed in sorted(speed_groups.keys(), key=float):
        runs = speed_groups[speed]
        n = len(runs)
        n_succ = sum(1 for r in runs if r["status"] == "True")
        succ_pct = n_succ / n * 100 if n > 0 else 0
        dists = [float(r["best_dist"]) for r in runs if r["best_dist"] != "N/A"]
        costs = [float(r["avg_cost"]) for r in runs if r.get("avg_cost", "N/A") != "N/A"]
        colls = [int(r["collision_count"]) for r in runs if r["collision_count"] != "N/A"]
        avg_dist = f"{sum(dists)/len(dists):.4f}" if dists else "N/A"
        min_dist = f"{min(dists):.4f}" if dists else "N/A"
        total_coll = sum(colls) if colls else 0
        avg_cost = f"{sum(costs)/len(costs):.4f}" if costs else "N/A"
        print(f"{speed:>6} {n:>4} {succ_pct:>5.0f}% {avg_dist:>9} {min_dist:>9} {total_coll:>10} {avg_cost:>10}")

    # ── Mode Comparison (aggregated across speeds) ──
    print()
    print("=" * 80)
    print("MODE COMPARISON (aggregated across all speeds)")
    print("=" * 80)

    mode_groups = defaultdict(list)
    for r in rows:
        mode_groups[r["mode"]].append(r)

    print(f"{'Mode':<22} {'N':>4} {'Succ%':>6} {'AvgDist':>9} {'MinDist':>9} {'TotalColl':>10}")
    print("-" * 65)
    for mode in sorted(mode_groups.keys()):
        runs = mode_groups[mode]
        n = len(runs)
        n_succ = sum(1 for r in runs if r["status"] == "True")
        succ_pct = n_succ / n * 100 if n > 0 else 0
        dists = [float(r["best_dist"]) for r in runs if r["best_dist"] != "N/A"]
        colls = [int(r["collision_count"]) for r in runs if r["collision_count"] != "N/A"]
        avg_dist = f"{sum(dists)/len(dists):.4f}" if dists else "N/A"
        min_dist = f"{min(dists):.4f}" if dists else "N/A"
        total_coll = sum(colls) if colls else 0
        print(f"{mode:<22} {n:>4} {succ_pct:>5.0f}% {avg_dist:>9} {min_dist:>9} {total_coll:>10}")

    # ── Per-template details ──
    print()
    print("=" * 115)
    print("PER-TEMPLATE DETAILS")
    print("=" * 115)
    print(f"{'Config':<35} {'Speed':>6} {'Hz':>4} {'Mode':<20} {'Tpl':>3} {'Status':>8} {'Dist':>9} {'Cost':>10} {'Coll':>5} {'RT':>8}")
    print("-" * 115)
    for r in rows:
        print(f"{r['config']:<35} {r['speed']:>6} {r['horizon']:>4} {r['mode']:<20} {r['template_idx']:>3} {r['status']:>8} {r['best_dist']:>9} {r.get('avg_cost','N/A'):>10} {r['collision_count']:>5} {r.get('runtime_sec','N/A'):>8}")


if __name__ == "__main__":
    main()
