#!/usr/bin/env python3
"""
Compare Horizon 100 vs 120 vs 140 single-obstacle results.

Usage:
  python scripts/analyze_horizon_comparison.py <h100_manifest> <h120_manifest> <h140_manifest>
"""

import csv
import sys
from collections import defaultdict


def load_manifest(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_stats(rows, group_key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)

    results = {}
    for key, runs in groups.items():
        n = len(runs)
        n_succ = sum(1 for r in runs if r["status"] == "True")
        dists = [float(r["best_dist"]) for r in runs if r["best_dist"] != "N/A"]
        costs = [float(r["avg_cost"]) for r in runs if r.get("avg_cost", "N/A") != "N/A"]
        colls = [int(r["collision_count"]) for r in runs if r["collision_count"] != "N/A"]
        results[key] = {
            "n": n,
            "succ": n_succ,
            "succ_pct": n_succ / n * 100 if n > 0 else 0,
            "avg_dist": sum(dists) / len(dists) if dists else None,
            "min_dist": min(dists) if dists else None,
            "avg_cost": sum(costs) / len(costs) if costs else None,
            "total_coll": sum(colls),
        }
    return results


def main():
    if len(sys.argv) < 4:
        print("Usage: python analyze_horizon_comparison.py <h100_manifest> <h120_manifest> <h140_manifest>")
        print("Example:")
        print("  python scripts/analyze_horizon_comparison.py \\")
        print("    runs/speed_horizon_ablation_*/manifest.csv \\")
        print("    runs/speed_horizon_ablation_*/manifest.csv \\")
        print("    runs/horizon140_sweep_*/manifest.csv")
        sys.exit(1)

    h100 = load_manifest(sys.argv[1])
    h120 = load_manifest(sys.argv[2])
    h140 = load_manifest(sys.argv[3])

    # Filter by horizon
    h100 = [r for r in h100 if r["horizon"] == "100"]
    h120 = [r for r in h120 if r["horizon"] == "120"]
    h140 = [r for r in h140 if r["horizon"] == "140"]

    all_rows = h100 + h120 + h140

    print(f"Loaded: h100={len(h100)}, h120={len(h120)}, h140={len(h140)}")

    # ── Horizon × Speed ──
    print()
    print("=" * 90)
    print("HORIZON × SPEED COMPARISON")
    print("=" * 90)
    print(f"{'Horizon':>7} {'Speed':>6} {'N':>3} {'Succ%':>6} {'AvgDist':>9} {'MinDist':>9} {'AvgCost':>10} {'Coll':>5}")
    print("-" * 90)

    for horizon in ["100", "120", "140"]:
        rows = [r for r in all_rows if r["horizon"] == horizon]
        speed_stats = compute_stats(rows, "speed")
        for speed in sorted(speed_stats.keys(), key=float):
            s = speed_stats[speed]
            avg_d = f"{s['avg_dist']:.4f}" if s["avg_dist"] is not None else "N/A"
            min_d = f"{s['min_dist']:.4f}" if s["min_dist"] is not None else "N/A"
            avg_c = f"{s['avg_cost']:.4f}" if s["avg_cost"] is not None else "N/A"
            print(f"{horizon:>7} {speed:>6} {s['n']:>3} {s['succ_pct']:>5.0f}% {avg_d:>9} {min_d:>9} {avg_c:>10} {s['total_coll']:>5}")
        print()

    # ── Horizon × Mode ──
    print("=" * 90)
    print("HORIZON × MODE COMPARISON")
    print("=" * 90)
    print(f"{'Horizon':>7} {'Mode':<20} {'N':>3} {'Succ%':>6} {'AvgDist':>9} {'MinDist':>9} {'AvgCost':>10} {'Coll':>5}")
    print("-" * 90)

    for horizon in ["100", "120", "140"]:
        rows = [r for r in all_rows if r["horizon"] == horizon]
        mode_stats = compute_stats(rows, "mode")
        for mode in sorted(mode_stats.keys()):
            s = mode_stats[mode]
            avg_d = f"{s['avg_dist']:.4f}" if s["avg_dist"] is not None else "N/A"
            min_d = f"{s['min_dist']:.4f}" if s["min_dist"] is not None else "N/A"
            avg_c = f"{s['avg_cost']:.4f}" if s["avg_cost"] is not None else "N/A"
            print(f"{horizon:>7} {mode:<20} {s['n']:>3} {s['succ_pct']:>5.0f}% {avg_d:>9} {min_d:>9} {avg_c:>10} {s['total_coll']:>5}")
        print()

    # ── Overall Horizon comparison ──
    print("=" * 70)
    print("OVERALL HORIZON COMPARISON")
    print("=" * 70)
    print(f"{'Horizon':>7} {'N':>3} {'Succ%':>6} {'AvgDist':>9} {'MinDist':>9} {'AvgCost':>10} {'Coll':>5}")
    print("-" * 70)

    for horizon in ["100", "120", "140"]:
        rows = [r for r in all_rows if r["horizon"] == horizon]
        stats = compute_stats(rows, "horizon")
        s = stats[horizon]
        avg_d = f"{s['avg_dist']:.4f}" if s["avg_dist"] is not None else "N/A"
        min_d = f"{s['min_dist']:.4f}" if s["min_dist"] is not None else "N/A"
        avg_c = f"{s['avg_cost']:.4f}" if s["avg_cost"] is not None else "N/A"
        print(f"{horizon:>7} {s['n']:>3} {s['succ_pct']:>5.0f}% {avg_d:>9} {min_d:>9} {avg_c:>10} {s['total_coll']:>5}")

    # ── Trend analysis ──
    print()
    print("=" * 70)
    print("TREND: Does increasing horizon reduce avg_dist?")
    print("=" * 70)
    horizons_data = {}
    for horizon in ["100", "120", "140"]:
        rows = [r for r in all_rows if r["horizon"] == horizon]
        dists = [float(r["best_dist"]) for r in rows if r["best_dist"] != "N/A"]
        horizons_data[horizon] = sum(dists) / len(dists) if dists else 0

    prev = None
    for h in ["100", "120", "140"]:
        d = horizons_data[h]
        if prev is not None:
            delta = d - prev
            pct = (delta / prev * 100) if prev > 0 else 0
            direction = "↓ BETTER" if delta < 0 else "↑ WORSE"
            print(f"  Horizon {h}: avg_dist = {d:.4f}  (vs prev: {delta:+.4f}, {pct:+.1f}% {direction})")
        else:
            print(f"  Horizon {h}: avg_dist = {d:.4f}")
        prev = d


if __name__ == "__main__":
    main()
