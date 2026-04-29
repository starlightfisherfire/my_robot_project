"""
Planner capacity check utilities for Paper 1.

Current implemented mode:

    state_sanity

This mode does not call MuJoCo and does not run CEM-MPC yet.
It only checks whether reset templates are geometrically reasonable enough
to be used later by Oracle-MPC / MuJoCo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.interventions.reset_template_loader import load_reset_templates
from src.metrics.planner_capacity import (
    DEFAULT_THRESHOLDS,
    DEFAULT_WORKSPACE_BOUNDS,
    run_state_sanity,
    save_state_sanity_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="state_sanity",
        choices=["state_sanity"],
        help="Planner capacity check mode.",
    )

    parser.add_argument(
        "--templates",
        type=str,
        default="data/sim/metadata/reset_templates_v0.json",
        help="Path to reset template JSON file.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="runs/debug/planner_capacity_state_sanity.json",
        help="Path to save state sanity report.",
    )

    parser.add_argument(
        "--workspace-x-min",
        type=float,
        default=DEFAULT_WORKSPACE_BOUNDS["x_min"],
    )
    parser.add_argument(
        "--workspace-x-max",
        type=float,
        default=DEFAULT_WORKSPACE_BOUNDS["x_max"],
    )
    parser.add_argument(
        "--workspace-y-min",
        type=float,
        default=DEFAULT_WORKSPACE_BOUNDS["y_min"],
    )
    parser.add_argument(
        "--workspace-y-max",
        type=float,
        default=DEFAULT_WORKSPACE_BOUNDS["y_max"],
    )

    parser.add_argument(
        "--max-print",
        type=int,
        default=10,
        help="Maximum number of templates with warnings/errors to print.",
    )

    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="If set, warnings also cause a non-zero exit.",
    )

    return parser.parse_args()


def print_problem_examples(report: dict, max_print: int) -> None:
    printed = 0

    for result in report["results"]:
        has_problem = bool(result["errors"]) or bool(result["warnings"])

        if not has_problem:
            continue

        print("-" * 80)
        print("reset_template_id:", result["reset_template_id"])
        print("split:", result["split"])
        print("layout_family:", result["layout_family"])
        print("shape_family:", result["shape_family"])
        print("object_goal_dist:", f"{result['object_goal_dist']:.4f}")
        print("ee_object_dist:", f"{result['ee_object_dist']:.4f}")
        print("num_obstacles:", result["num_obstacles"])

        if result["errors"]:
            print("errors:")
            for msg in result["errors"]:
                print("  -", msg)

        if result["warnings"]:
            print("warnings:")
            for msg in result["warnings"]:
                print("  -", msg)

        printed += 1
        if printed >= max_print:
            break


def run_state_sanity_mode(args: argparse.Namespace) -> None:
    template_path = Path(args.templates)

    if not template_path.exists():
        raise FileNotFoundError(
            f"Reset template file does not exist: {template_path}\n"
            "Generate it first with:\n"
            "  PYTHONPATH=. python scripts/generate_reset_templates.py"
        )

    workspace_bounds = {
        "x_min": args.workspace_x_min,
        "x_max": args.workspace_x_max,
        "y_min": args.workspace_y_min,
        "y_max": args.workspace_y_max,
    }

    templates = load_reset_templates(template_path)

    report = run_state_sanity(
        templates=templates,
        workspace_bounds=workspace_bounds,
        thresholds=DEFAULT_THRESHOLDS,
    )

    save_state_sanity_report(report, args.out)

    summary = report["summary"]

    print("mode: state_sanity")
    print("templates:", template_path)
    print("num_templates:", summary["num_templates"])
    print("num_ok_templates:", summary["num_ok_templates"])
    print("num_error_messages:", summary["num_error_messages"])
    print("num_warning_messages:", summary["num_warning_messages"])
    print("by_split:")
    print(json.dumps(summary["by_split"], indent=2, ensure_ascii=False))
    print("by_layout_family:")
    print(json.dumps(summary["by_layout_family"], indent=2, ensure_ascii=False))
    print("by_shape_family:")
    print(json.dumps(summary["by_shape_family"], indent=2, ensure_ascii=False))
    print("report path:", args.out)

    if summary["num_error_messages"] > 0 or summary["num_warning_messages"] > 0:
        print_problem_examples(report, max_print=args.max_print)

    if summary["num_error_messages"] > 0:
        raise SystemExit("state sanity failed: errors found")

    if args.strict_warnings and summary["num_warning_messages"] > 0:
        raise SystemExit("state sanity failed: warnings found under --strict-warnings")

    if summary["num_warning_messages"] > 0:
        print("state sanity ok with warnings")
    else:
        print("state sanity ok")


def main() -> None:
    args = parse_args()

    if args.mode == "state_sanity":
        run_state_sanity_mode(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()