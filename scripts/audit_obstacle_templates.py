#!/usr/bin/env python3
"""
Audit obstacle difficulty templates.

Reads the template JSON, computes per-template metrics, and outputs
CSV + TXT reports.  Does NOT import MuJoCo.  Does NOT modify templates.

Handles both old-format (passage_gap = center distance) and new-format
(passage_gap = effective inner wall gap, passage_gap_definition = "inner_wall_gap")
templates.

Usage:
  PYTHONPATH=. python scripts/audit_obstacle_templates.py \
    --templates data/sim/metadata/reset_templates_obstacle_difficulty_v0.json \
    --out-csv runs/debug/template_previews/obstacle_difficulty_v0_audit.csv \
    --out-txt runs/debug/template_previews/obstacle_difficulty_v0_audit.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Shortest distance from point (px,py) to line segment (ax,ay)->(bx,by)."""
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0,
        ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    ))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


# ---------------------------------------------------------------------------
# passage gap helpers
# ---------------------------------------------------------------------------

def _get_effective_passage_gap(template: dict) -> float:
    """Get effective passage gap, handling both old and new formats."""
    # New format: explicit effective_passage_gap
    if "effective_passage_gap" in template:
        return float(template["effective_passage_gap"])

    # Old format: passage_gap was center distance
    passage_gap = template.get("passage_gap")
    if passage_gap is None:
        return float("nan")

    gap_def = template.get("passage_gap_definition", "")
    if gap_def == "inner_wall_gap":
        # New format: passage_gap IS the effective gap
        return float(passage_gap)

    # Old format: passage_gap is center distance, subtract obstacle_size_y
    obs_size_y = template.get("obstacle_size_y", 0.06)
    obstacles = template.get("obstacles", [])
    if obstacles:
        obs_size_y = obstacles[0].get("size_y", obs_size_y)
    return float(passage_gap) - float(obs_size_y)


def _get_passage_center_distance(template: dict) -> float:
    """Get passage center distance."""
    if "passage_center_distance" in template:
        return float(template["passage_center_distance"])

    passage_gap = template.get("passage_gap")
    if passage_gap is None:
        return float("nan")

    gap_def = template.get("passage_gap_definition", "")
    if gap_def == "inner_wall_gap":
        obs_size_y = template.get("obstacle_size_y", 0.06)
        obstacles = template.get("obstacles", [])
        if obstacles:
            obs_size_y = obstacles[0].get("size_y", obs_size_y)
        return float(passage_gap) + float(obs_size_y)

    # Old format: passage_gap IS the center distance
    return float(passage_gap)


def _feasibility_label(effective_gap: float) -> str:
    """Label based on effective passage gap relative to T-shape footprint (48mm)."""
    if math.isnan(effective_gap):
        return "no_passage"
    gap_mm = effective_gap * 1000.0
    if gap_mm < 48.0:
        return "impossible_for_unrotated_object"
    elif gap_mm < 60.0:
        return "extreme_5cm_boundary"
    elif gap_mm < 80.0:
        return "medium_orientation_sensitive"
    elif gap_mm < 110.0:
        return "wide_feasible"
    else:
        return "very_wide"


# ---------------------------------------------------------------------------
# per-template metrics
# ---------------------------------------------------------------------------

def _compute_record(template: dict) -> dict:
    obj = template["object_initial_pose"]
    goal = template["goal_pose"]

    obj_goal_dist = math.hypot(
        goal["x"] - obj["x"], goal["y"] - obj["y"]
    )

    obstacles = template.get("obstacles", [])
    n_obs = len(obstacles)

    obs_sizes_str = "; ".join(
        f"{o['size_x']:.3f}x{o['size_y']:.3f}" for o in obstacles
    )

    # Minimum distance from any obstacle centre to the object->goal segment
    min_obs_line_dist = float("nan")
    if obstacles:
        best = float("inf")
        for o in obstacles:
            d = _point_to_segment_distance(
                o["pose"]["x"], o["pose"]["y"],
                obj["x"], obj["y"],
                goal["x"], goal["y"],
            )
            if d < best:
                best = d
        min_obs_line_dist = best

    passage_gap_raw = template.get("passage_gap", float("nan"))
    effective_gap = _get_effective_passage_gap(template)
    center_dist = _get_passage_center_distance(template)
    gap_def = template.get("passage_gap_definition", "")
    obs_size_y = template.get("obstacle_size_y", float("nan"))
    if obstacles and math.isnan(obs_size_y):
        obs_size_y = obstacles[0].get("size_y", float("nan"))
    blocking_diff = template.get("blocking_difficulty", "")

    feasibility = _feasibility_label(effective_gap) if n_obs == 2 else ""

    return {
        "reset_template_id": template["reset_template_id"],
        "split": template["split"],
        "layout_family": template["layout_family"],
        "num_obstacles": n_obs,
        "obj_goal_distance_m": round(obj_goal_dist, 4),
        "obstacle_sizes": obs_sizes_str,
        "passage_gap_m": round(passage_gap_raw, 4)
            if not math.isnan(passage_gap_raw) else "",
        "effective_passage_gap_m": round(effective_gap, 4)
            if not math.isnan(effective_gap) else "",
        "passage_center_distance_m": round(center_dist, 4)
            if not math.isnan(center_dist) else "",
        "passage_gap_definition": gap_def,
        "obstacle_size_y_m": round(obs_size_y, 4)
            if not math.isnan(obs_size_y) else "",
        "blocking_difficulty": blocking_diff,
        "min_obs_line_distance_m": round(min_obs_line_dist, 4)
            if not math.isnan(min_obs_line_dist) else "",
        "feasibility": feasibility,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit obstacle difficulty templates (no MuJoCo).",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json",
        help="Path to the obstacle difficulty templates JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (legacy). If set, writes obstacle_template_audit.csv/txt there.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output CSV path. Overrides --out-dir.",
    )
    parser.add_argument(
        "--out-txt",
        type=str,
        default=None,
        help="Output TXT path. Overrides --out-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    templates_path = Path(args.templates)
    if not templates_path.exists():
        raise FileNotFoundError(f"Templates file not found: {templates_path}")

    with open(templates_path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    if not isinstance(templates, list):
        raise ValueError(
            f"Expected a list of templates, got {type(templates)}"
        )

    # Determine output paths
    if args.out_csv:
        csv_path = Path(args.out_csv)
    elif args.out_dir:
        csv_path = Path(args.out_dir) / "obstacle_template_audit.csv"
    else:
        csv_path = Path("runs/debug/template_previews") / "obstacle_template_audit.csv"

    if args.out_txt:
        txt_path = Path(args.out_txt)
    elif args.out_dir:
        txt_path = Path(args.out_dir) / "obstacle_template_audit.txt"
    else:
        txt_path = Path("runs/debug/template_previews") / "obstacle_template_audit.txt"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    # --- build records ---
    records = [_compute_record(t) for t in templates]

    # --- CSV ---
    fieldnames = [
        "reset_template_id",
        "split",
        "layout_family",
        "num_obstacles",
        "obj_goal_distance_m",
        "obstacle_sizes",
        "passage_gap_m",
        "effective_passage_gap_m",
        "passage_center_distance_m",
        "passage_gap_definition",
        "obstacle_size_y_m",
        "blocking_difficulty",
        "min_obs_line_distance_m",
        "feasibility",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV  written: {csv_path}")

    # --- TXT ---
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("Obstacle Difficulty Template Audit\n")
        f.write(f"Source: {templates_path}\n")
        f.write(f"Total templates: {len(templates)}\n")
        f.write("=" * 72 + "\n\n")

        # Summary by split
        by_split: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_split[r["split"]].append(r)

        f.write("--- By Split ---\n\n")
        for split in sorted(by_split):
            recs = by_split[split]
            n = len(recs)
            obj_goal_dists = [r["obj_goal_distance_m"] for r in recs]
            effective_gaps = [
                r["effective_passage_gap_m"] for r in recs
                if r["effective_passage_gap_m"] != ""
            ]
            center_dists = [
                r["passage_center_distance_m"] for r in recs
                if r["passage_center_distance_m"] != ""
            ]
            f.write(f"  {split}: {n} templates\n")
            f.write(
                f"    obj->goal dist: "
                f"min={min(obj_goal_dists):.3f}  "
                f"max={max(obj_goal_dists):.3f}  "
                f"mean={sum(obj_goal_dists)/n:.3f}\n"
            )
            if effective_gaps:
                f.write(
                    f"    effective_passage_gap: "
                    f"min={min(effective_gaps):.3f}  "
                    f"max={max(effective_gaps):.3f}  "
                    f"mean={sum(effective_gaps)/len(effective_gaps):.3f}\n"
                )
            if center_dists:
                f.write(
                    f"    passage_center_distance: "
                    f"min={min(center_dists):.3f}  "
                    f"max={max(center_dists):.3f}  "
                    f"mean={sum(center_dists)/len(center_dists):.3f}\n"
                )

            # Feasibility distribution
            feas_counts: dict[str, int] = {}
            for r in recs:
                fl = r.get("feasibility", "")
                if fl:
                    feas_counts[fl] = feas_counts.get(fl, 0) + 1
            if feas_counts:
                f.write(f"    feasibility: {feas_counts}\n")

            f.write("\n")

        # Per-template details
        f.write("\n--- Per-Template Details ---\n\n")
        for r in records:
            f.write(f"{r['reset_template_id']}\n")
            f.write(
                f"  split={r['split']}  "
                f"layout={r['layout_family']}  "
                f"num_obs={r['num_obstacles']}\n"
            )
            f.write(f"  obj->goal={r['obj_goal_distance_m']} m\n")
            if r["obstacle_sizes"]:
                f.write(f"  obstacle sizes: {r['obstacle_sizes']}\n")
            if r["passage_gap_m"] != "":
                f.write(f"  passage_gap={r['passage_gap_m']} m\n")
            if r["effective_passage_gap_m"] != "":
                f.write(f"  effective_passage_gap={r['effective_passage_gap_m']} m\n")
            if r["passage_center_distance_m"] != "":
                f.write(f"  passage_center_distance={r['passage_center_distance_m']} m\n")
            if r["passage_gap_definition"]:
                f.write(f"  passage_gap_definition={r['passage_gap_definition']}\n")
            if r["obstacle_size_y_m"] != "":
                f.write(f"  obstacle_size_y={r['obstacle_size_y_m']} m\n")
            if r["blocking_difficulty"]:
                f.write(f"  blocking_difficulty={r['blocking_difficulty']}\n")
            if r["min_obs_line_distance_m"] != "":
                f.write(
                    f"  min obs->line dist={r['min_obs_line_distance_m']} m\n"
                )
            if r["feasibility"]:
                f.write(f"  feasibility={r['feasibility']}\n")
            f.write("\n")

    print(f"TXT  written: {txt_path}")
    print("Audit complete.")


if __name__ == "__main__":
    main()
