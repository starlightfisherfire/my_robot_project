"""
Planner capacity check utilities for Paper 1.

Implemented modes:

    state_sanity
        Does not call MuJoCo and does not run CEM-MPC.
        Checks whether reset templates are geometrically reasonable.

    toy_oracle_mpc
        Does not call MuJoCo.
        Uses ToyPushEnv + true toy dynamics + CEM-MPC to verify the oracle
        rollout and planner interface over reset templates.

    mujoco_oracle_mpc
        Uses MujocoPushEnv + MuJoCo true dynamics + CEM-MPC.
        Smoke test for MuJoCo oracle rollout + CEM-MPC interface.
        v0.1 does not yet instantiate obstacles from templates.
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
from src.metrics.toy_oracle_capacity import (
    run_toy_oracle_mpc_capacity,
    save_toy_oracle_mpc_report,
)
from src.metrics.mujoco_oracle_capacity import (
    run_mujoco_oracle_mpc_capacity,
    save_mujoco_oracle_mpc_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="state_sanity",
        choices=["state_sanity", "toy_oracle_mpc", "mujoco_oracle_mpc"],
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
        default=None,
        help="Path to save report. If omitted, a mode-specific debug path is used.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train_sim_id",
        help=(
            "Split used by toy_oracle_mpc mode. "
            "Use 'all' to evaluate all templates."
        ),
    )

    parser.add_argument(
        "--max-templates",
        type=int,
        default=10,
        help="Maximum number of templates to evaluate in toy_oracle_mpc mode.",
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

    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="CEM planning horizon (default: mode-specific, toy=18, mujoco=80).",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="CEM number of samples (default: mode-specific, toy=768, mujoco=1536).",
    )

    parser.add_argument(
        "--num-elites",
        type=int,
        default=None,
        help="CEM number of elites (default: mode-specific, toy=96, mujoco=128).",
    )

    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="CEM number of iterations (default: mode-specific, toy=6, mujoco=7).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for toy_oracle_mpc mode.",
    )

    parser.add_argument(
        "--success-dist-threshold",
        type=float,
        default=0.05,
        help="Success distance threshold for toy_oracle_mpc mode.",
    )

    return parser.parse_args()


def _default_out_path(mode: str) -> str:
    if mode == "state_sanity":
        return "runs/debug/planner_capacity_state_sanity.json"

    if mode == "toy_oracle_mpc":
        return "runs/debug/planner_capacity_toy_oracle_mpc.json"

    if mode == "mujoco_oracle_mpc":
        return "runs/debug/planner_capacity_mujoco_oracle_mpc.json"

    raise ValueError(f"Unknown mode={mode}")


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

    out_path = args.out or _default_out_path(args.mode)
    save_state_sanity_report(report, out_path)

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
    print("report path:", out_path)

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


def run_toy_oracle_mpc_mode(args: argparse.Namespace) -> None:
    template_path = Path(args.templates)

    if not template_path.exists():
        raise FileNotFoundError(
            f"Reset template file does not exist: {template_path}\n"
            "Generate it first with:\n"
            "  PYTHONPATH=. python scripts/generate_reset_templates.py"
        )

    templates = load_reset_templates(template_path)

    if args.split != "all":
        templates = [t for t in templates if t["split"] == args.split]

    if not templates:
        raise ValueError(f"No templates selected for split={args.split}")

    if args.max_templates <= 0:
        raise ValueError(f"max_templates must be positive, got {args.max_templates}")

    templates = templates[: args.max_templates]

    # Mode-specific defaults for toy_oracle_mpc
    horizon = args.horizon if args.horizon is not None else 18
    num_samples = args.num_samples if args.num_samples is not None else 768
    num_elites = args.num_elites if args.num_elites is not None else 96
    num_iterations = args.num_iterations if args.num_iterations is not None else 6

    report = run_toy_oracle_mpc_capacity(
        templates=templates,
        horizon=horizon,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
        seed=args.seed,
        success_dist_threshold=args.success_dist_threshold,
    )

    out_path = args.out or _default_out_path(args.mode)
    save_toy_oracle_mpc_report(report, out_path)

    summary = report["summary"]

    print("mode: toy_oracle_mpc")
    print("templates:", template_path)
    print("split:", args.split)
    print("num_templates:", summary["num_templates"])
    print("num_success:", summary["num_success"])
    print("success_rate:", f"{summary['success_rate']:.3f}")
    print("num_improved_cost:", summary["num_improved_cost"])
    print("num_improved_dist:", summary["num_improved_dist"])
    print("num_restored_ok:", summary["num_restored_ok"])
    print("mean_initial_dist:", f"{summary['mean_initial_dist']:.4f}")
    print("mean_final_dist:", f"{summary['mean_final_dist']:.4f}")
    print("mean_zero_cost:", f"{summary['mean_zero_cost']:.4f}")
    print("mean_planned_cost:", f"{summary['mean_planned_cost']:.4f}")
    print("report path:", out_path)

    for result in report["results"][: min(5, len(report["results"]))]:
        print("-" * 80)
        print("reset_template_id:", result["reset_template_id"])
        print("initial_dist:", f"{result['initial_dist']:.4f}")
        print("final_dist:", f"{result['final_dist']:.4f}")
        print("zero_cost:", f"{result['zero_cost']:.4f}")
        print("planned_cost:", f"{result['planned_cost']:.4f}")
        print("success:", result["success"])
        print("restored_ok:", result["restored_ok"])

    if summary["num_restored_ok"] != summary["num_templates"]:
        raise SystemExit("toy oracle mpc failed: some rollouts did not restore env state")

    if summary["num_improved_cost"] != summary["num_templates"]:
        raise SystemExit("toy oracle mpc failed: not all templates improved cost")

    print("toy oracle mpc debug ok")


def run_mujoco_oracle_mpc_mode(args: argparse.Namespace) -> None:
    template_path = Path(args.templates)

    if not template_path.exists():
        raise FileNotFoundError(
            f"Reset template file does not exist: {template_path}\n"
            "Generate it first with:\n"
            "  PYTHONPATH=. python scripts/generate_reset_templates.py"
        )

    templates = load_reset_templates(template_path)

    if args.split != "all":
        templates = [t for t in templates if t["split"] == args.split]

    if not templates:
        raise ValueError(f"No templates selected for split={args.split}")

    if args.max_templates <= 0:
        raise ValueError(f"max_templates must be positive, got {args.max_templates}")

    templates = templates[: args.max_templates]

    # Mode-specific defaults for mujoco_oracle_mpc
    horizon = args.horizon if args.horizon is not None else 80
    num_samples = args.num_samples if args.num_samples is not None else 1536
    num_elites = args.num_elites if args.num_elites is not None else 128
    num_iterations = args.num_iterations if args.num_iterations is not None else 7

    report = run_mujoco_oracle_mpc_capacity(
        templates=templates,
        horizon=horizon,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
        seed=args.seed,
        success_dist_threshold=args.success_dist_threshold,
    )

    out_path = args.out or _default_out_path(args.mode)
    save_mujoco_oracle_mpc_report(report, out_path)

    summary = report["summary"]

    print("mode: mujoco_oracle_mpc")
    print("templates:", template_path)
    print("split:", args.split)
    print("num_templates:", summary["num_templates"])
    print("num_success:", summary["num_success"])
    print("success_rate:", f"{summary['success_rate']:.3f}")
    print("num_improved_cost:", summary["num_improved_cost"])
    print("num_improved_dist:", summary["num_improved_dist"])
    print("num_restored_ok:", summary["num_restored_ok"])
    print("mean_initial_dist:", f"{summary['mean_initial_dist']:.4f}")
    print("mean_best_min_dist:", f"{summary['mean_best_min_dist']:.4f}")
    print("mean_final_dist:", f"{summary['mean_final_dist']:.4f}")
    print("mean_zero_cost:", f"{summary['mean_zero_cost']:.4f}")
    print("mean_planned_cost:", f"{summary['mean_planned_cost']:.4f}")
    print("report path:", out_path)

    for result in report["results"][: min(5, len(report["results"]))]:
        print("-" * 80)
        print("reset_template_id:", result["reset_template_id"])
        print("initial_dist:", f"{result['initial_dist']:.4f}")
        print("best_min_dist:", f"{result['best_min_dist']:.4f}")
        print("final_dist:", f"{result['final_dist']:.4f}")
        print("zero_cost:", f"{result['zero_cost']:.4f}")
        print("planned_cost:", f"{result['planned_cost']:.4f}")
        print("best_cost:", f"{result['best_cost']:.4f}")
        print("improved_cost:", result["improved_cost"])
        print("improved_dist:", result["improved_dist"])
        print("success:", result["success"])
        print("restored_ok:", result["restored_ok"])

        # Enhanced diagnostic fields
        print(f"dist_delta: {result.get('dist_delta', 0.0):.8f} m")
        print(f"cost_delta: {result.get('cost_delta', 0.0):.6f}")
        print(f"object_displacement: {result.get('object_displacement', 0.0):.8f} m")
        print(f"object_moved: {result.get('object_moved', False)}")
        print(f"max_contact: {result.get('max_contact', 0.0):.2f}")
        print(f"best_action_norm_mean: {result.get('best_action_norm_mean', 0.0):.4f}")
        print(f"best_action_norm_max: {result.get('best_action_norm_max', 0.0):.4f}")

    if summary["num_restored_ok"] != summary["num_templates"]:
        raise SystemExit(
            "mujoco oracle mpc failed: some rollouts did not restore env state"
        )

    if summary["num_improved_cost"] == 0:
        raise SystemExit(
            "mujoco oracle mpc failed: no templates improved cost"
        )

    if summary["num_improved_dist"] == 0:
        raise SystemExit(
            "mujoco oracle mpc failed: no templates improved distance"
        )

    print("mujoco oracle mpc capacity check ok")


def main() -> None:
    args = parse_args()

    if args.mode == "state_sanity":
        run_state_sanity_mode(args)
    elif args.mode == "toy_oracle_mpc":
        run_toy_oracle_mpc_mode(args)
    elif args.mode == "mujoco_oracle_mpc":
        run_mujoco_oracle_mpc_mode(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()