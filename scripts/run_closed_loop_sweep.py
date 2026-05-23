#!/usr/bin/env python3
"""
Sweep script for mujoco_oracle_mpc_closed_loop mode.

Performs config-level parallel parameter sweep to establish baseline
performance under default cost function.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def generate_sweep_configs(total_executed_budget: int, preset: str) -> List[Dict]:
    """Generate all parameter combinations for sweep."""
    if preset == "coarse":
        horizons = [80, 120]
        execute_steps_list = [10, 20, 30, 40]
        num_samples_list = [512, 1024]
        num_iterations_list = [5, 7]
    else:
        raise ValueError(f"Unknown preset: {preset}")

    configs = []
    config_id = 0

    for horizon in horizons:
        for execute_steps in execute_steps_list:
            for num_samples in num_samples_list:
                for num_iterations in num_iterations_list:
                    # Calculate max_mpc_steps based on budget
                    max_mpc_steps = total_executed_budget // execute_steps

                    # Skip if budget too small
                    if max_mpc_steps < 1:
                        continue

                    # Determine num_elites based on num_samples
                    if num_samples == 512:
                        num_elites = 48
                    elif num_samples == 1024:
                        num_elites = 96
                    elif num_samples == 1536:
                        num_elites = 128
                    else:
                        raise ValueError(f"Unexpected num_samples: {num_samples}")

                    configs.append({
                        "config_id": config_id,
                        "horizon": horizon,
                        "execute_steps": execute_steps,
                        "num_samples": num_samples,
                        "num_elites": num_elites,
                        "num_iterations": num_iterations,
                        "max_mpc_steps": max_mpc_steps,
                    })
                    config_id += 1

    return configs


def run_single_config(
    config: Dict,
    output_dir: Path,
    split: str,
    max_templates: int,
    timeout_sec: int,
) -> Tuple[Dict, bool, float, str, str]:
    """Run a single configuration using subprocess.

    Returns:
        config: The configuration dict
        success: Whether the run succeeded
        runtime_sec: Runtime in seconds
        status: Status string (success/timeout/failed/exception)
        failure_reason: Reason for failure (if failed)
    """
    import time

    config_id = config["config_id"]

    # Create output paths
    report_path = output_dir / "reports" / f"config_{config_id:03d}.json"
    log_path = output_dir / "logs" / f"config_{config_id:03d}.log"

    # Build command
    cmd = [
        sys.executable,
        "scripts/check_mpc_capacity.py",
        "--mode", "mujoco_oracle_mpc_closed_loop",
        "--split", split,
        "--max-templates", str(max_templates),
        "--horizon", str(config["horizon"]),
        "--execute-steps", str(config["execute_steps"]),
        "--max-mpc-steps", str(config["max_mpc_steps"]),
        "--num-samples", str(config["num_samples"]),
        "--num-elites", str(config["num_elites"]),
        "--num-iterations", str(config["num_iterations"]),
        "--out", str(report_path),
    ]

    print(f"[Config {config_id}] Starting: horizon={config['horizon']}, "
          f"execute_steps={config['execute_steps']}, "
          f"samples={config['num_samples']}, iters={config['num_iterations']}")

    # Set up environment with explicit PYTHONPATH
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")

    # Start timing
    start_time = time.time()

    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                env=env,
            )

        runtime_sec = time.time() - start_time
        success = result.returncode == 0

        if success:
            print(f"[Config {config_id}] Completed successfully in {runtime_sec:.1f}s")
            return config, True, runtime_sec, "success", ""
        else:
            print(f"[Config {config_id}] Failed with return code {result.returncode} after {runtime_sec:.1f}s")
            return config, False, runtime_sec, "failed", "returncode_nonzero"

    except subprocess.TimeoutExpired:
        runtime_sec = time.time() - start_time
        print(f"[Config {config_id}] Timeout after {timeout_sec} seconds")
        return config, False, runtime_sec, "timeout", "timeout"
    except Exception as e:
        runtime_sec = time.time() - start_time
        print(f"[Config {config_id}] Error: {e}")
        return config, False, runtime_sec, "exception", "exception"


def extract_metrics_from_report(report_path: Path, config: Dict, runtime_sec: float, status: str) -> Dict:
    """Extract metrics from a single JSON report."""
    try:
        with open(report_path, "r") as f:
            data = json.load(f)

        metrics = {
            "config_id": config["config_id"],
            "horizon": config["horizon"],
            "execute_steps": config["execute_steps"],
            "num_samples": config["num_samples"],
            "num_elites": config["num_elites"],
            "num_iterations": config["num_iterations"],
            "max_mpc_steps": config["max_mpc_steps"],
            "runtime_sec": runtime_sec,
            "status": status,
        }

        # Extract summary metrics
        summary = data.get("summary", {})
        metrics["success_rate"] = summary.get("success_rate", 0.0)
        metrics["mean_initial_dist"] = summary.get("mean_initial_dist", float("nan"))
        metrics["mean_best_dist"] = summary.get("mean_best_dist", float("nan"))
        metrics["mean_final_dist"] = summary.get("mean_final_dist", float("nan"))
        metrics["mean_dist_delta"] = summary.get("mean_dist_delta", float("nan"))
        metrics["mean_total_object_displacement"] = summary.get(
            "mean_total_object_displacement", float("nan")
        )
        metrics["mean_contact_rate"] = summary.get("mean_contact_rate", float("nan"))

        # Pose-level diagnostics
        metrics["mean_final_pos_error"] = summary.get("mean_final_pos_error", float("nan"))
        metrics["mean_final_theta_error_deg"] = summary.get("mean_final_theta_error_deg", float("nan"))
        metrics["success_pose_2cm_15deg_rate"] = summary.get("success_pose_2cm_15deg_rate", 0.0)

        # Semantic success rates (paper naming)
        metrics["success_rate_definition"] = summary.get("success_rate_definition", "success_pose_1cm_5deg")
        metrics["primary_success_rate"] = summary.get("primary_success_rate", 0.0)
        metrics["coarse_success_rate"] = summary.get("coarse_success_rate", 0.0)
        metrics["precision_success_rate"] = summary.get("precision_success_rate", 0.0)
        metrics["strict_completion_rate"] = summary.get("strict_completion_rate", 0.0)
        metrics["legacy_pos_5cm_rate"] = summary.get("legacy_pos_5cm_rate", 0.0)

        # Extract mpc_step_logs metrics if available
        template_results = data.get("results", data.get("template_results", []))
        first_contact_steps = []
        planned_contact_within_execute_steps_count = 0
        total_mpc_steps = 0

        for template_result in template_results:
            mpc_step_logs = template_result.get("mpc_step_logs", [])
            for log in mpc_step_logs:
                if "planned_contact_first_step" in log:
                    first_step = log["planned_contact_first_step"]
                    if first_step >= 0:
                        first_contact_steps.append(first_step)

                if "planned_contact_within_execute_steps" in log:
                    if log["planned_contact_within_execute_steps"]:
                        planned_contact_within_execute_steps_count += 1
                    total_mpc_steps += 1

        if first_contact_steps:
            metrics["mean_first_contact_step"] = sum(first_contact_steps) / len(first_contact_steps)
        else:
            metrics["mean_first_contact_step"] = float("nan")

        if total_mpc_steps > 0:
            metrics["mean_planned_contact_within_execute_steps_rate"] = (
                planned_contact_within_execute_steps_count / total_mpc_steps
            )
        else:
            metrics["mean_planned_contact_within_execute_steps_rate"] = float("nan")

        return metrics

    except Exception as e:
        print(f"Error extracting metrics from {report_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Sweep mujoco_oracle_mpc_closed_loop parameters"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["coarse"],
        default="coarse",
        help="Sweep preset (default: coarse)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Number of parallel jobs (default: 8)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train_sim_id",
        help="Split to evaluate (default: train_sim_id)",
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        default=3,
        help="Max templates per split (default: 3)",
    )
    parser.add_argument(
        "--total-executed-budget",
        type=int,
        default=240,
        help="Total executed steps budget (default: 240)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Timeout per config in seconds (default: 1800)",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Max configs to run for smoke test (default: None, run all)",
    )
    parser.add_argument(
        "--shuffle-configs",
        action="store_true",
        help="Shuffle configs before running (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / "sweeps" / f"default_cost_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Preset: {args.preset}")
    print(f"Parallel jobs: {args.jobs}")
    print(f"Total executed budget: {args.total_executed_budget}")
    print(f"Timeout per config: {args.timeout_sec} seconds")
    print()

    # Generate sweep configurations
    configs = generate_sweep_configs(args.total_executed_budget, args.preset)
    print(f"Generated {len(configs)} configurations")

    # Shuffle configs if requested (before max-configs limit)
    if args.shuffle_configs:
        import random
        random.seed(args.seed)
        random.shuffle(configs)
        print(f"Shuffled configurations with seed={args.seed}")

    # Apply max-configs limit for smoke test
    if args.max_configs is not None:
        configs = configs[:args.max_configs]
        print(f"Limited to {len(configs)} configurations for smoke test")
    print()

    # Save all configs to JSON
    configs_json_path = output_dir / "configs.json"
    with open(configs_json_path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"Saved configurations to: {configs_json_path}")
    print()

    # Run all configurations in parallel
    successful_configs = []
    failed_configs = []
    config_results = {}  # config_id -> (runtime_sec, status, failure_reason)

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_single_config,
                config,
                output_dir,
                args.split,
                args.max_templates,
                args.timeout_sec,
            ): config
            for config in configs
        }

        for future in as_completed(futures):
            config, success, runtime_sec, status, failure_reason = future.result()
            config_results[config["config_id"]] = (runtime_sec, status, failure_reason)

            if success:
                successful_configs.append(config)
            else:
                # Add failure info to config
                failed_config = config.copy()
                failed_config["failure_reason"] = failure_reason
                failed_config["log_path"] = str(output_dir / "logs" / f"config_{config['config_id']:03d}.log")
                failed_configs.append(failed_config)

    print()
    print(f"Completed: {len(successful_configs)} successful, {len(failed_configs)} failed")
    print()

    # Save failed configs if any
    if failed_configs:
        failed_configs_path = output_dir / "failed_configs.json"
        with open(failed_configs_path, "w") as f:
            json.dump(failed_configs, f, indent=2)
        print(f"Saved failed configurations to: {failed_configs_path}")
        print()

    # Extract metrics from all successful runs
    all_metrics = []
    for config in successful_configs:
        report_path = output_dir / "reports" / f"config_{config['config_id']:03d}.json"
        config_id = config["config_id"]
        runtime_sec, status, _ = config_results.get(config_id, (0.0, "unknown", ""))
        metrics = extract_metrics_from_report(report_path, config, runtime_sec, status)
        if metrics is not None:
            all_metrics.append(metrics)

    if not all_metrics:
        print("No successful runs with valid metrics. Exiting.")
        return

    # Create DataFrame and sort
    df = pd.DataFrame(all_metrics)

    # Sort by priority:
    # 1. success_pose_2cm_15deg_rate desc
    # 2. success_rate desc
    # 3. mean_final_pos_error asc
    # 4. mean_final_theta_error_deg asc
    # 5. mean_final_dist asc
    df_sorted = df.sort_values(
        by=[
            "success_pose_2cm_15deg_rate",
            "success_rate",
            "mean_final_pos_error",
            "mean_final_theta_error_deg",
            "mean_final_dist",
        ],
        ascending=[False, False, True, True, True],
    )

    # Save full summary
    summary_csv_path = output_dir / "summary.csv"
    df_sorted.to_csv(summary_csv_path, index=False)
    print(f"Saved summary to: {summary_csv_path}")

    # Save top results
    top_n = min(10, len(df_sorted))
    summary_top_path = output_dir / "summary_top.txt"
    with open(summary_top_path, "w") as f:
        f.write(f"Top {top_n} Configurations\n")
        f.write("=" * 80 + "\n\n")

        for idx, row in df_sorted.head(top_n).iterrows():
            f.write(f"Rank {df_sorted.index.get_loc(idx) + 1}\n")
            f.write(f"Config ID: {int(row['config_id'])}\n")
            f.write(f"  horizon={int(row['horizon'])}, execute_steps={int(row['execute_steps'])}, ")
            f.write(f"samples={int(row['num_samples'])}, elites={int(row['num_elites'])}, ")
            f.write(f"iters={int(row['num_iterations'])}, max_mpc_steps={int(row['max_mpc_steps'])}\n")
            f.write(f"  success_pose_2cm_15deg_rate: {row['success_pose_2cm_15deg_rate']:.3f}\n")
            f.write(f"  success_rate: {row['success_rate']:.3f}\n")
            if not pd.isna(row.get("mean_final_pos_error")):
                f.write(f"  mean_final_pos_error: {row['mean_final_pos_error']:.4f}\n")
            if not pd.isna(row.get("mean_final_theta_error_deg")):
                f.write(f"  mean_final_theta_error_deg: {row['mean_final_theta_error_deg']:.2f}\n")
            f.write(f"  mean_final_dist: {row['mean_final_dist']:.4f}\n")
            f.write(f"  mean_dist_delta: {row['mean_dist_delta']:.4f}\n")
            f.write(f"  mean_contact_rate: {row['mean_contact_rate']:.3f}\n")
            if not pd.isna(row.get("mean_first_contact_step")):
                f.write(f"  mean_first_contact_step: {row['mean_first_contact_step']:.1f}\n")
            if not pd.isna(row.get("mean_planned_contact_within_execute_steps_rate")):
                f.write(f"  mean_planned_contact_within_execute_steps_rate: {row['mean_planned_contact_within_execute_steps_rate']:.3f}\n")
            f.write("\n")

    print(f"Saved top results to: {summary_top_path}")
    print()
    print("Top 3 configurations:")
    for idx, row in df_sorted.head(3).iterrows():
        print(f"  [{int(row['config_id'])}] h={int(row['horizon'])}, e={int(row['execute_steps'])}, "
              f"s={int(row['num_samples'])}, i={int(row['num_iterations'])} -> "
              f"PSR={row['success_pose_2cm_15deg_rate']:.3f}, SR={row['success_rate']:.3f}, "
              f"PE={row['mean_final_pos_error']:.4f}, FD={row['mean_final_dist']:.4f}")


if __name__ == "__main__":
    main()
