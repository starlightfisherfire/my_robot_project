#!/usr/bin/env python3
"""
Select 6 representative templates (sixpack) from the 60-template obstacle
difficulty file.

One template per condition:
  - blocking_easy
  - blocking_medium
  - blocking_hard
  - passage_direct_wide
  - passage_direct_medium
  - passage_direct_narrow

Strategy:
  - first: pick the 000000 template from each condition
  - median: pick the template closest to the group median initial distance

Does NOT call MuJoCo.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CONDITIONS = [
    "blocking_easy",
    "blocking_medium",
    "blocking_hard",
    "passage_direct_wide",
    "passage_direct_medium",
    "passage_direct_narrow",
]


def _initial_dist(t: dict) -> float:
    obj = t["object_initial_pose"]
    goal = t["goal_pose"]
    return math.hypot(goal["x"] - obj["x"], goal["y"] - obj["y"])


def select_sixpack(
    templates: list[dict],
    strategy: str = "first",
) -> list[dict]:
    """Select one template per condition."""
    by_family: dict[str, list[dict]] = {}
    for t in templates:
        fam = t["layout_family"]
        by_family.setdefault(fam, []).append(t)

    selected = []
    for cond in CONDITIONS:
        group = by_family.get(cond, [])
        if not group:
            raise ValueError(f"No templates found for condition: {cond}")

        if strategy == "first":
            # Pick the 000000 template
            match = [t for t in group if t["reset_template_id"].endswith("__000000")]
            if match:
                selected.append(match[0])
            else:
                selected.append(group[0])
        elif strategy == "median":
            # Pick template closest to group median initial distance
            dists = [_initial_dist(t) for t in group]
            median_d = sorted(dists)[len(dists) // 2]
            best = min(range(len(group)), key=lambda i: abs(dists[i] - median_d))
            selected.append(group[best])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select 6-template obstacle sixpack from difficulty file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json",
        help="Source template JSON (60 templates).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/sim/metadata/reset_templates_obstacle_sixpack_v0.json",
        help="Output sixpack JSON (6 templates).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["first", "median"],
        default="first",
        help="Selection strategy: 'first' picks 000000, 'median' picks closest to median.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    with open(source_path, "r", encoding="utf-8") as f:
        templates = json.load(f)

    print(f"Loaded {len(templates)} templates from {source_path}")

    sixpack = select_sixpack(templates, strategy=args.strategy)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sixpack, f, indent=2, ensure_ascii=False)

    print(f"\nSelected {len(sixpack)} sixpack templates:")
    for t in sixpack:
        fam = t["layout_family"]
        n_obs = len(t["obstacles"])
        sizes = [(o["size_x"], o["size_y"]) for o in t["obstacles"]]
        egap = t.get("effective_passage_gap", "")
        cdist = t.get("passage_center_distance", "")
        diff = t.get("blocking_difficulty", "")
        egap_str = f"  effective_gap={egap:.3f}m" if egap != "" else ""
        cdist_str = f"  center_dist={cdist:.3f}m" if cdist != "" else ""
        diff_str = f"  difficulty={diff}" if diff else ""
        print(f"  {t['reset_template_id']}")
        print(f"    layout={fam}  obs={n_obs}  sizes={sizes}{egap_str}{cdist_str}{diff_str}")

    print(f"\nSaved: {out_path}")
    print("Sixpack selection ok.")


if __name__ == "__main__":
    main()
