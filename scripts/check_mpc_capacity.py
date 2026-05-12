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
    run_mujoco_oracle_mpc_closed_loop_capacity,
    save_mujoco_oracle_mpc_closed_loop_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="state_sanity",
        choices=["state_sanity", "toy_oracle_mpc", "mujoco_oracle_mpc", "mujoco_oracle_mpc_closed_loop"],
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
        help="Success distance threshold for oracle_mpc modes.",
    )

    parser.add_argument(
        "--execute-steps",
        type=int,
        default=5,
        help="Number of steps to execute per MPC iteration (closed-loop mode only).",
    )

    parser.add_argument(
        "--max-mpc-steps",
        type=int,
        default=40,
        help="Maximum number of MPC replanning steps (closed-loop mode only).",
    )

    parser.add_argument(
        "--pusher-radius",
        type=float,
        default=None,
        help="Pusher radius in meters (default: 0.010).",
    )

    parser.add_argument(
        "--pusher-halfheight",
        type=float,
        default=None,
        help="Pusher halfheight in meters (default: 0.014).",
    )

    parser.add_argument(
        "--pusher-z",
        type=float,
        default=None,
        help="Pusher z position in meters (default: 0.016).",
    )

    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        default=False,
        help="If set, run full max_mpc_steps even after reaching success threshold.",
    )

    parser.add_argument(
        "--success-pos-threshold",
        type=float,
        default=None,
        help="Strict pose early stop: position threshold in meters. If set, overrides --success-dist-threshold for early stop.",
    )

    parser.add_argument(
        "--stop-pos-threshold",
        type=float,
        default=None,
        dest="success_pos_threshold",
        help="Alias for --success-pos-threshold.",
    )

    parser.add_argument(
        "--success-theta-threshold-deg",
        type=float,
        default=180.0,
        help="Strict pose early stop: theta threshold in degrees. Default 180 = disabled.",
    )

    parser.add_argument(
        "--stop-theta-threshold-deg",
        type=float,
        default=None,
        dest="success_theta_threshold_deg",
        help="Alias for --success-theta-threshold-deg.",
    )

    parser.add_argument(
        "--strict-pose-stop",
        action="store_true",
        default=False,
        help="Enable strict pose early stop with --stop-pos-threshold and --stop-theta-threshold-deg.",
    )

    return parser.parse_args()


def _default_out_path(mode: str) -> str:
    if mode == "state_sanity":
        return "runs/debug/planner_capacity_state_sanity.json"
    if mode == "toy_oracle_mpc":
        return "runs/debug/planner_capacity_toy_oracle_mpc.json"
    if mode == "mujoco_oracle_mpc":
        return "runs/debug/planner_capacity_mujoco_oracle_mpc.json"
    if mode == "mujoco_oracle_mpc_closed_loop":
        return "runs/debug/planner_capacity_mujoco_oracle_mpc_closed_loop.json"
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

    # Pusher geometry defaults
    pusher_radius = args.pusher_radius if args.pusher_radius is not None else 0.010
    pusher_halfheight = args.pusher_halfheight if args.pusher_halfheight is not None else 0.014
    pusher_z = args.pusher_z if args.pusher_z is not None else 0.016

    report = run_mujoco_oracle_mpc_capacity(
        templates=templates,
        horizon=horizon,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
        seed=args.seed,
        success_dist_threshold=args.success_dist_threshold,
        pusher_radius=pusher_radius,
        pusher_halfheight=pusher_halfheight,
        pusher_z=pusher_z,
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


def run_mujoco_oracle_mpc_closed_loop_mode(args: argparse.Namespace) -> None:
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

    # Mode-specific defaults for mujoco_oracle_mpc_closed_loop
    planning_horizon = args.horizon if args.horizon is not None else 80
    num_samples = args.num_samples if args.num_samples is not None else 1536
    num_elites = args.num_elites if args.num_elites is not None else 128
    num_iterations = args.num_iterations if args.num_iterations is not None else 7

    # Pusher geometry defaults
    pusher_radius = args.pusher_radius if args.pusher_radius is not None else 0.010
    pusher_halfheight = args.pusher_halfheight if args.pusher_halfheight is not None else 0.014
    pusher_z = args.pusher_z if args.pusher_z is not None else 0.016

    report = run_mujoco_oracle_mpc_closed_loop_capacity(
        templates=templates,
        planning_horizon=planning_horizon,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
        execute_steps=args.execute_steps,
        max_mpc_steps=args.max_mpc_steps,
        seed=args.seed,
        success_dist_threshold=args.success_dist_threshold,
        pusher_radius=pusher_radius,
        pusher_halfheight=pusher_halfheight,
        pusher_z=pusher_z,
        disable_early_stop=args.disable_early_stop,
        success_pos_threshold=args.success_pos_threshold if args.success_pos_threshold is not None else args.success_dist_threshold,
        success_theta_threshold_deg=args.success_theta_threshold_deg,
    )

    out_path = args.out or _default_out_path(args.mode)
    save_mujoco_oracle_mpc_closed_loop_report(report, out_path)

    summary = report["summary"]

    print("mode: mujoco_oracle_mpc_closed_loop")
    print("templates:", template_path)
    print("split:", args.split)
    print("disable_early_stop:", args.disable_early_stop)
    print("num_templates:", summary["num_templates"])
    print("success_rate:", f"{summary['success_rate']:.3f}")
    print("mean_initial_dist:", f"{summary['mean_initial_dist']:.4f}")
    print("mean_best_dist:", f"{summary['mean_best_dist']:.4f}")
    print("mean_final_dist:", f"{summary['mean_final_dist']:.4f}")
    print("mean_dist_delta:", f"{summary['mean_dist_delta']:.4f}")
    print("mean_total_object_displacement:", f"{summary['mean_total_object_displacement']:.4f}")
    print("mean_contact_rate:", f"{summary['mean_contact_rate']:.3f}")

    # Pose-level diagnostics
    if "mean_final_pos_error" in summary:
        print("mean_final_pos_error:", f"{summary['mean_final_pos_error']:.4f}")
    if "mean_final_theta_error_deg" in summary:
        print("mean_final_theta_error_deg:", f"{summary['mean_final_theta_error_deg']:.2f}")
    if "success_pose_2cm_15deg_rate" in summary:
        print("success_pose_2cm_15deg_rate:", f"{summary['success_pose_2cm_15deg_rate']:.3f}")
    if "mean_best_pose_cost" in summary:
        print("mean_best_pose_cost:", f"{summary['mean_best_pose_cost']:.4f}")
    if "mean_final_pose_cost" in summary:
        print("mean_final_pose_cost:", f"{summary['mean_final_pose_cost']:.4f}")
    if "mean_best_pos_error" in summary:
        print("mean_best_pos_error:", f"{summary['mean_best_pos_error']:.4f}")
    if "mean_best_theta_error_deg_at_best_pos" in summary:
        print("mean_best_theta_error_deg_at_best_pos:", f"{summary['mean_best_theta_error_deg_at_best_pos']:.2f}")

    # Threshold reach rates
    for tname in ["5cm", "2cm", "1p5cm", "1cm", "0p5cm"]:
        key = f"reach_{tname}_rate"
        if key in summary:
            steps_key = f"mean_first_reach_{tname}_steps"
            steps_val = summary.get(steps_key, float("nan"))
            steps_str = f"{steps_val:.1f}" if steps_val == steps_val else "nan"
            print(f"  reach_{tname}_rate: {summary[key]:.3f}  mean_first_reach_steps: {steps_str}")

    # Post-reach degradation
    for tname in ["5cm", "2cm", "1p5cm", "1cm", "0p5cm"]:
        key = f"mean_final_minus_first_reach_pos_error_{tname}"
        if key in summary and summary[key] == summary[key]:
            best_key = f"mean_best_after_reach_pos_error_{tname}"
            best_val = summary.get(best_key, float("nan"))
            best_str = f"{best_val:.4f}" if best_val == best_val else "nan"
            print(f"  post_reach_{tname}: final_delta={summary[key]:+.4f}  best_after={best_str}")

    print("report path:", out_path)

    for result in report["results"][: min(5, len(report["results"]))]:
        print("-" * 80)
        print("reset_template_id:", result["reset_template_id"])
        print("initial_dist:", f"{result['initial_dist']:.4f}")
        print("best_dist:", f"{result['best_dist']:.4f}")
        print("final_dist:", f"{result['final_dist']:.4f}")
        print("dist_delta:", f"{result['dist_delta']:.4f}")
        print("total_object_displacement:", f"{result['total_object_displacement']:.4f}")
        print("num_mpc_steps:", result["num_mpc_steps"])
        print("total_executed_steps:", result["total_executed_steps"])
        print("contact_rate:", f"{result['contact_rate']:.3f}")
        print("success:", result["success"])

        # Pose-level diagnostics
        if "final_pos_error" in result:
            print("final_pos_error:", f"{result['final_pos_error']:.4f}")
        if "final_theta_error_deg" in result:
            print("final_theta_error_deg:", f"{result['final_theta_error_deg']:.2f}")
        if "success_pose_2cm_15deg" in result:
            print("success_pose_2cm_15deg:", result["success_pose_2cm_15deg"])

    if summary["success_rate"] == 0:
        print("WARNING: success_rate=0, no templates reached goal")

    print("mujoco oracle mpc closed-loop capacity check ok")


def main() -> None:
    args = parse_args()

    if args.mode == "state_sanity":
        run_state_sanity_mode(args)
    elif args.mode == "toy_oracle_mpc":
        run_toy_oracle_mpc_mode(args)
    elif args.mode == "mujoco_oracle_mpc":
        run_mujoco_oracle_mpc_mode(args)
    elif args.mode == "mujoco_oracle_mpc_closed_loop":
        run_mujoco_oracle_mpc_closed_loop_mode(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()