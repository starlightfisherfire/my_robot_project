"""
Generate reset templates for Paper 1 simulation experiments.

This script does not call MuJoCo.

It only creates structured reset conditions for:

- train_sim_id
- val_sim_id
- test_sim_id
- test_sim_layout_ood_blocking
- test_sim_layout_ood_narrow_passage
- test_sim_layout_ood_edge_goal
- test_sim_shape_ood_L

Output:
    data/sim/metadata/reset_templates_v0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.interventions.sampling_rules import (
    build_reset_templates,
    save_reset_templates,
    summarize_reset_templates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-per-split",
        type=int,
        default=20,
        help="Number of reset templates generated for each split.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="data/sim/metadata/reset_templates_v0.json",
        help="Output JSON path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    templates = build_reset_templates(
        num_per_split=args.num_per_split,
        base_seed=args.seed,
    )

    out_path = Path(args.out)
    save_reset_templates(templates, out_path)

    summary = summarize_reset_templates(templates)

    print("generated reset templates:", summary["num_templates"])
    print("by_split:")
    print(json.dumps(summary["by_split"], indent=2, ensure_ascii=False))
    print("by_layout_family:")
    print(json.dumps(summary["by_layout_family"], indent=2, ensure_ascii=False))
    print("by_shape_family:")
    print(json.dumps(summary["by_shape_family"], indent=2, ensure_ascii=False))
    print("save path:", out_path)
    print("reset template generation ok")


if __name__ == "__main__":
    main()