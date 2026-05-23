#!/usr/bin/env python3
"""
Wide overnight MuJoCo Oracle-MPC closed-loop parameter sweep.

Orchestrates large-scale parameter sweep via subprocess calls to:
- scripts/check_mpc_capacity.py (numerical evaluation)
- scripts/render_closed_loop_rollout.py (best config video)

Does not modify any core files (env, cost, CEM, planner, templates).
No pandas dependency - uses Python standard library only.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def is_nan(value) -> bool:
    """Check if value is NaN."""
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def safe_float(value, default=float("nan")):
    """Safely convert to float, return default if invalid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_sweep_configs(preset: str) -> List[Dict]:
    """Generate all parameter combinations for sweep."""
    if preset == "wide_overnight_v2":
        horizons = [60, 80, 100, 120, 140, 160, 200]
        execute_steps_list = [10, 15, 20, 30, 40]
        max_mpc_steps_list = [10, 15, 20]
        num_samples_list = [512, 1024, 1536]
        num_iterations_list = [3, 5, 7, 9]
    elif preset == "boundary_refine_v1":
        horizons = [60, 80, 100, 120]
        execute_steps_list = [20, 30]
        max_mpc_steps_list = [20, 25]
        num_samples_list = [512, 1024]
        num_iterations_list = [3, 5]
    else:
        raise ValueError(f"Unknown preset: {preset}")

    configs = []
    config_id = 0

    for horizon in horizons:
        for execute_steps in execute_steps_list:
            for max_mpc_steps in max_mpc_steps_list:
                for num_samples in num_samples_list:
                    for num_iterations in num_iterations_list:
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
                            "max_mpc_steps": max_mpc_steps,
                            "num_samples": num_samples,
                            "num_elites": num_elites,
                            "num_iterations": num_iterations,
                        })
                        config_id += 1

    return configs


def run_single_config(
    config: Dict,
    output_dir: Path,
    max_templates: int,
    timeout_sec: int,
    resume: bool,
    disable_early_stop: bool = False,
) -> Tuple[Dict, bool, float, str, str]:
    """Run a single configuration using subprocess.

    Returns:
        config: The configuration dict
        success: Whether the run succeeded
        runtime_sec: Runtime in seconds
        status: Status string (success/timeout/failed/exception/skipped)
        failure_reason: Reason for failure (if failed)
    """
    config_id = config["config_id"]

    # Create output paths
    report_path = output_dir / "reports" / f"config_{config_id:06d}.json"
    log_path = output_dir / "logs" / f"config_{config_id:06d}_check.log"

    # Check if resume and report exists
    if resume and report_path.exists():
        print(f"[Config {config_id}] Skipping (report exists)")
        return config, True, 0.0, "skipped", ""

    # Build command
    cmd = [
        sys.executable,
        "scripts/check_mpc_capacity.py",
        "--mode", "mujoco_oracle_mpc_closed_loop",
        "--split", "train_sim_id",
        "--max-templates", str(max_templates),
        "--max-mpc-steps", str(config["max_mpc_steps"]),
        "--execute-steps", str(config["execute_steps"]),
        "--horizon", str(config["horizon"]),
        "--num-samples", str(config["num_samples"]),
        "--num-elites", str(config["num_elites"]),
        "--num-iterations", str(config["num_iterations"]),
        "--out", str(report_path),
    ]
    if disable_early_stop:
        cmd.append("--disable-early-stop")

    print(f"[Config {config_id}] Starting: h={config['horizon']}, "
          f"e={config['execute_steps']}, m={config['max_mpc_steps']}, "
          f"s={config['num_samples']}, i={config['num_iterations']}")

    # Set up environment with explicit PYTHONPATH
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
            print(f"[Config {config_id}] Completed in {runtime_sec:.1f}s")
            return config, True, runtime_sec, "success", ""
        else:
            print(f"[Config {config_id}] Failed (rc={result.returncode}) after {runtime_sec:.1f}s")
            return config, False, runtime_sec, "failed", "returncode_nonzero"

    except subprocess.TimeoutExpired:
        runtime_sec = time.time() - start_time
        print(f"[Config {config_id}] Timeout after {timeout_sec}s")
        return config, False, runtime_sec, "timeout", "timeout"
    except Exception as e:
        runtime_sec = time.time() - start_time
        print(f"[Config {config_id}] Exception: {e}")
        return config, False, runtime_sec, "exception", str(e)


def extract_metrics_from_report(
    report_path: Path,
    config: Dict,
    runtime_sec: float,
    status: str,
    max_templates: int,
) -> Optional[Dict]:
    """Extract metrics from a single JSON report."""
    try:
        with open(report_path, "r") as f:
            data = json.load(f)

        summary = data.get("summary", {})

        metrics = {
            "config_id": config["config_id"],
            "horizon": config["horizon"],
            "execute_steps": config["execute_steps"],
            "max_mpc_steps": config["max_mpc_steps"],
            "total_planned_horizon_steps": config["horizon"],
            "total_execution_budget": config["execute_steps"] * config["max_mpc_steps"],
            "num_samples": config["num_samples"],
            "num_elites": config["num_elites"],
            "num_iterations": config["num_iterations"],
            "max_templates": max_templates,
            "status": status,
            "runtime_sec": runtime_sec,
            "report_path": str(report_path),
            "success_rate": summary.get("success_rate", 0.0),
            "mean_final_dist": summary.get("mean_final_dist", float("nan")),
            "mean_dist_delta": summary.get("mean_dist_delta", float("nan")),
            "mean_total_object_displacement": summary.get("mean_total_object_displacement", float("nan")),
            "mean_contact_rate": summary.get("mean_contact_rate", float("nan")),
            "mean_final_pos_error": summary.get("mean_final_pos_error", float("nan")),
            "mean_final_theta_error_deg": summary.get("mean_final_theta_error_deg", float("nan")),
            "success_pose_2cm_15deg_rate": summary.get("success_pose_2cm_15deg_rate", 0.0),
            "mean_best_pose_cost": summary.get("mean_best_pose_cost", float("nan")),
            "mean_final_pose_cost": summary.get("mean_final_pose_cost", float("nan")),
            "mean_best_pos_error": summary.get("mean_best_pos_error", float("nan")),
            "mean_best_theta_error_deg_at_best_pos": summary.get("mean_best_theta_error_deg_at_best_pos", float("nan")),
            "reach_5cm_rate": summary.get("reach_5cm_rate", float("nan")),
            "reach_2cm_rate": summary.get("reach_2cm_rate", float("nan")),
            "reach_1p5cm_rate": summary.get("reach_1p5cm_rate", float("nan")),
            "reach_1cm_rate": summary.get("reach_1cm_rate", float("nan")),
            "reach_0p5cm_rate": summary.get("reach_0p5cm_rate", float("nan")),
            "mean_first_reach_5cm_steps": summary.get("mean_first_reach_5cm_steps", float("nan")),
            "mean_first_reach_2cm_steps": summary.get("mean_first_reach_2cm_steps", float("nan")),
            "mean_first_reach_1cm_steps": summary.get("mean_first_reach_1cm_steps", float("nan")),
            "median_final_pos_error": summary.get("median_final_pos_error", float("nan")),
            "median_final_theta_error_deg": summary.get("median_final_theta_error_deg", float("nan")),
            "median_best_pose_cost": summary.get("median_best_pose_cost", float("nan")),
            "median_final_pose_cost": summary.get("median_final_pose_cost", float("nan")),
            "median_best_pos_error": summary.get("median_best_pos_error", float("nan")),
            "success_pos_5cm_rate": summary.get("success_pos_5cm_rate", float("nan")),
            "success_pos_3cm_rate": summary.get("success_pos_3cm_rate", float("nan")),
            "success_pos_2cm_rate": summary.get("success_pos_2cm_rate", float("nan")),
            "success_pos_1p5cm_rate": summary.get("success_pos_1p5cm_rate", float("nan")),
            "success_pos_1cm_rate": summary.get("success_pos_1cm_rate", float("nan")),
            "success_pos_0p5cm_rate": summary.get("success_pos_0p5cm_rate", float("nan")),
            "success_pose_5cm_15deg_rate": summary.get("success_pose_5cm_15deg_rate", float("nan")),
            "success_pose_3cm_15deg_rate": summary.get("success_pose_3cm_15deg_rate", float("nan")),
            "success_pose_2cm_10deg_rate": summary.get("success_pose_2cm_10deg_rate", float("nan")),
            "success_pose_1cm_15deg_rate": summary.get("success_pose_1cm_15deg_rate", float("nan")),
            "reach_3cm_rate": summary.get("reach_3cm_rate", float("nan")),
            "mean_first_reach_3cm_steps": summary.get("mean_first_reach_3cm_steps", float("nan")),
            "median_first_reach_3cm_steps": summary.get("median_first_reach_3cm_steps", float("nan")),
            "mean_first_reach_1p5cm_steps": summary.get("mean_first_reach_1p5cm_steps", float("nan")),
            "mean_first_reach_0p5cm_steps": summary.get("mean_first_reach_0p5cm_steps", float("nan")),
            "mean_final_minus_first_reach_pos_error_5cm": summary.get("mean_final_minus_first_reach_pos_error_5cm", float("nan")),
            "mean_final_minus_first_reach_pos_error_3cm": summary.get("mean_final_minus_first_reach_pos_error_3cm", float("nan")),
            "mean_final_minus_first_reach_pos_error_2cm": summary.get("mean_final_minus_first_reach_pos_error_2cm", float("nan")),
            "mean_best_after_reach_pos_error_5cm": summary.get("mean_best_after_reach_pos_error_5cm", float("nan")),
            "mean_best_after_reach_pos_error_3cm": summary.get("mean_best_after_reach_pos_error_3cm", float("nan")),
            "mean_best_after_reach_pos_error_2cm": summary.get("mean_best_after_reach_pos_error_2cm", float("nan")),
            "mean_best_after_reach_pos_error_1cm": summary.get("mean_best_after_reach_pos_error_1cm", float("nan")),
            "success_pose_1p5cm_10deg_rate": summary.get("success_pose_1p5cm_10deg_rate", float("nan")),
            "success_pose_1cm_10deg_rate": summary.get("success_pose_1cm_10deg_rate", float("nan")),
            "success_pose_1cm_5deg_rate": summary.get("success_pose_1cm_5deg_rate", float("nan")),
            "success_pose_0p5cm_5deg_rate": summary.get("success_pose_0p5cm_5deg_rate", float("nan")),
        }

        return metrics

    except Exception as e:
        print(f"Error extracting metrics from {report_path}: {e}")
        return None


def sort_metrics(metrics_list: List[Dict]) -> List[Dict]:
    """Sort metrics by priority order."""
    def sort_key(m):
        def _v(key):
            v = m.get(key, float("nan"))
            return float("inf") if (v != v or v is None) else v  # NaN → inf
        return (
            _v("mean_best_pose_cost"),
            _v("mean_final_pose_cost"),
            _v("mean_best_pos_error"),
            _v("mean_final_pos_error"),
            _v("mean_best_theta_error_deg_at_best_pos"),
            _v("mean_final_theta_error_deg"),
            _v("runtime_sec"),
        )

    return sorted(metrics_list, key=sort_key)


def save_summary_csv(metrics_list: List[Dict], output_path: Path) -> None:
    """Save metrics to CSV file with dynamic fieldnames (union of all keys)."""
    if not metrics_list:
        return

    # Build ordered fieldnames: fixed prefix keys first, then any extras
    prefix_keys = [
        "config_id", "horizon", "execute_steps", "max_mpc_steps",
        "total_planned_horizon_steps", "total_execution_budget",
        "num_samples", "num_elites", "num_iterations", "max_templates",
        "status", "runtime_sec", "report_path",
    ]
    all_keys: set = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    extra_keys = sorted(k for k in all_keys if k not in prefix_keys)
    fieldnames = prefix_keys + extra_keys

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow(metrics)


def save_summary_top_txt(metrics_list: List[Dict], output_path: Path, top_n: int = 30) -> None:
    """Save top N results to text file."""
    with open(output_path, "w") as f:
        f.write(f"Top {min(top_n, len(metrics_list))} Configurations\n")
        f.write("=" * 80 + "\n\n")

        for rank, metrics in enumerate(metrics_list[:top_n], start=1):
            f.write(f"Rank {rank}\n")
            f.write(f"Config ID: {metrics['config_id']}\n")
            f.write(f"  horizon={metrics['horizon']}, execute_steps={metrics['execute_steps']}, ")
            f.write(f"max_mpc_steps={metrics['max_mpc_steps']}\n")
            f.write(f"  samples={metrics['num_samples']}, elites={metrics['num_elites']}, ")
            f.write(f"iters={metrics['num_iterations']}\n")
            f.write(f"  success_pose_2cm_15deg_rate: {metrics.get('success_pose_2cm_15deg_rate', float('nan')):.3f}\n")
            f.write(f"  success_rate: {metrics.get('success_rate', float('nan')):.3f}\n")
            for _k in ["success_pos_5cm_rate", "success_pos_3cm_rate", "success_pos_2cm_rate",
                       "success_pos_1p5cm_rate", "success_pos_1cm_rate", "success_pos_0p5cm_rate"]:
                _v = metrics.get(_k, float("nan"))
                if not is_nan(_v):
                    f.write(f"  {_k}: {_v:.3f}\n")
            for _k in ["success_pose_5cm_15deg_rate", "success_pose_3cm_15deg_rate",
                       "success_pose_2cm_10deg_rate", "success_pose_1cm_15deg_rate",
                       "success_pose_1p5cm_10deg_rate", "success_pose_1cm_10deg_rate",
                       "success_pose_1cm_5deg_rate", "success_pose_0p5cm_5deg_rate"]:
                _v = metrics.get(_k, float("nan"))
                if not is_nan(_v):
                    f.write(f"  {_k}: {_v:.3f}\n")
            for _k in ["mean_best_pose_cost", "median_best_pose_cost",
                       "mean_final_pose_cost", "median_final_pose_cost"]:
                _v = metrics.get(_k, float("nan"))
                if not is_nan(_v):
                    f.write(f"  {_k}: {_v:.4f}\n")
            for _k in ["mean_best_pos_error", "median_best_pos_error",
                       "mean_final_pos_error", "median_final_pos_error"]:
                _v = metrics.get(_k, float("nan"))
                if not is_nan(_v):
                    f.write(f"  {_k}: {_v:.4f}\n")
            for _k in ["mean_final_theta_error_deg", "median_final_theta_error_deg"]:
                _v = metrics.get(_k, float("nan"))
                if not is_nan(_v):
                    f.write(f"  {_k}: {_v:.2f}\n")
            for _tn in ["5cm", "3cm", "2cm", "1cm", "0p5cm"]:
                _rk = f"reach_{_tn}_rate"
                _sk = f"mean_first_reach_{_tn}_steps"
                _rv = metrics.get(_rk, float("nan"))
                if not is_nan(_rv):
                    _sv = metrics.get(_sk, float("nan"))
                    _ss = f"{_sv:.1f}" if not is_nan(_sv) else "nan"
                    f.write(f"  reach_{_tn}_rate: {_rv:.3f}  mean_first_reach_steps: {_ss}\n")
            for _tn in ["5cm", "3cm", "2cm"]:
                _dk = f"mean_final_minus_first_reach_pos_error_{_tn}"
                _bk = f"mean_best_after_reach_pos_error_{_tn}"
                _dv = metrics.get(_dk, float("nan"))
                _bv = metrics.get(_bk, float("nan"))
                if not is_nan(_dv):
                    _bs = f"{_bv:.4f}" if not is_nan(_bv) else "nan"
                    f.write(f"  post_reach_{_tn}: final_delta={_dv:+.4f}  best_after={_bs}\n")
            f.write(f"  mean_final_dist: {metrics.get('mean_final_dist', float('nan')):.4f}\n")
            f.write(f"  mean_dist_delta: {metrics.get('mean_dist_delta', float('nan')):.4f}\n")
            f.write(f"  mean_contact_rate: {metrics.get('mean_contact_rate', float('nan')):.3f}\n")
            f.write(f"  runtime_sec: {metrics.get('runtime_sec', float('nan')):.1f}\n")
            f.write("\n")


def generate_watchlist(metrics_list: List[Dict], output_path: Path) -> None:
    """Generate watchlist CSV with interesting subsets of configs."""
    watchlist_entries = []

    # Top 30 by strict pose success / final pos error
    sorted_by_pose = sorted(
        [m for m in metrics_list if not is_nan(m.get("mean_final_pos_error"))],
        key=lambda m: m["mean_final_pos_error"]
    )
    for m in sorted_by_pose[:30]:
        watchlist_entries.append({
            "config_id": m["config_id"],
            "category": "top_pose_accuracy",
            "success_pose_2cm_15deg_rate": m["success_pose_2cm_15deg_rate"],
            "mean_final_pos_error": m["mean_final_pos_error"],
            "mean_final_theta_error_deg": m.get("mean_final_theta_error_deg", float("nan")),
            "runtime_sec": m["runtime_sec"],
        })

    # Top 30 by 5cm success_rate
    sorted_by_5cm = sorted(metrics_list, key=lambda m: -m["success_rate"])
    for m in sorted_by_5cm[:30]:
        watchlist_entries.append({
            "config_id": m["config_id"],
            "category": "top_5cm_success",
            "success_rate": m["success_rate"],
            "mean_final_dist": m["mean_final_dist"],
            "runtime_sec": m["runtime_sec"],
        })

    # Worst 30 by final pos error
    sorted_by_pose_worst = sorted(
        [m for m in metrics_list if not is_nan(m.get("mean_final_pos_error"))],
        key=lambda m: -m["mean_final_pos_error"]
    )
    for m in sorted_by_pose_worst[:30]:
        watchlist_entries.append({
            "config_id": m["config_id"],
            "category": "worst_pose_accuracy",
            "mean_final_pos_error": m["mean_final_pos_error"],
            "success_rate": m["success_rate"],
        })

    # High success_rate but pose_success=0
    high_5cm_zero_pose = [
        m for m in metrics_list
        if m["success_rate"] > 0.5 and m["success_pose_2cm_15deg_rate"] == 0.0
    ]
    for m in high_5cm_zero_pose[:30]:
        watchlist_entries.append({
            "config_id": m["config_id"],
            "category": "high_5cm_zero_pose",
            "success_rate": m["success_rate"],
            "success_pose_2cm_15deg_rate": m["success_pose_2cm_15deg_rate"],
            "mean_final_pos_error": m.get("mean_final_pos_error", float("nan")),
        })

    # Fastest 30
    sorted_by_runtime = sorted(metrics_list, key=lambda m: m["runtime_sec"])
    for m in sorted_by_runtime[:30]:
        watchlist_entries.append({
            "config_id": m["config_id"],
            "category": "fastest",
            "runtime_sec": m["runtime_sec"],
            "success_pose_2cm_15deg_rate": m["success_pose_2cm_15deg_rate"],
            "success_rate": m["success_rate"],
        })

    # Failed/timeout configs
    failed = [m for m in metrics_list if m["status"] in ["failed", "timeout"]]
    for m in failed:
        watchlist_entries.append({
            "config_id": m["config_id"],
            "category": f"status_{m['status']}",
            "status": m["status"],
            "runtime_sec": m["runtime_sec"],
        })

    # Save watchlist
    if watchlist_entries:
        # Collect all unique fieldnames from all entries
        all_fieldnames = set()
        for entry in watchlist_entries:
            all_fieldnames.update(entry.keys())
        fieldnames = sorted(list(all_fieldnames))

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for entry in watchlist_entries:
                writer.writerow(entry)


def render_best_config_video(
    best_config: Dict,
    output_dir: Path,
    video_template_index: int,
    video_width: int,
    video_height: int,
    fps: int,
) -> Dict:
    """Render video for the best configuration.

    Returns:
        manifest: Dict with video metadata
    """
    config_id = best_config["config_id"]
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    video_filename = f"best_config_{config_id:06d}_t{video_template_index:03d}.mp4"
    video_path = video_dir / video_filename

    cmd = [
        sys.executable,
        "scripts/render_closed_loop_rollout.py",
        "--split", "train_sim_id",
        "--template-index", str(video_template_index),
        "--execute-steps", str(best_config["execute_steps"]),
        "--max-mpc-steps", str(best_config["max_mpc_steps"]),
        "--horizon", str(best_config["horizon"]),
        "--num-samples", str(best_config["num_samples"]),
        "--num-elites", str(best_config["num_elites"]),
        "--num-iterations", str(best_config["num_iterations"]),
        "--width", str(video_width),
        "--height", str(video_height),
        "--fps", str(fps),
        "--out-video", str(video_path),
    ]

    print(f"\nRendering best config video...")
    print(f"Config ID: {config_id}")
    print(f"Video path: {video_path}")

    # Set up environment with MUJOCO_GL=egl
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    env["MUJOCO_GL"] = "egl"

    log_path = output_dir / "logs" / f"best_config_{config_id:06d}_video.log"

    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600,  # 1 hour timeout for video rendering
                env=env,
            )

        if result.returncode == 0:
            print(f"✓ Video rendered successfully: {video_path}")
            manifest = {
                "config_id": config_id,
                "video_path": str(video_path),
                "template_index": video_template_index,
                "width": video_width,
                "height": video_height,
                "fps": fps,
                "config": best_config,
                "status": "success",
            }
        else:
            print(f"✗ Video rendering failed (rc={result.returncode})")
            manifest = {
                "config_id": config_id,
                "status": "failed",
                "log_path": str(log_path),
            }

    except subprocess.TimeoutExpired:
        print(f"✗ Video rendering timeout")
        manifest = {
            "config_id": config_id,
            "status": "timeout",
            "log_path": str(log_path),
        }
    except Exception as e:
        print(f"✗ Video rendering exception: {e}")
        manifest = {
            "config_id": config_id,
            "status": "exception",
            "error": str(e),
            "log_path": str(log_path),
        }

    # Save manifest
    manifest_path = output_dir / "best_video_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved video manifest to: {manifest_path}")

    return manifest


def append_to_jsonl(metrics: Dict, jsonl_path: Path) -> None:
    """Append metrics to JSONL file."""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def rebuild_summaries(output_dir: Path) -> None:
    """Rebuild summary files from existing reports."""
    print("Rebuilding summaries from existing reports...")

    # Load configs
    configs_path = output_dir / "configs.json"
    if not configs_path.exists():
        print(f"Error: {configs_path} not found")
        return

    with open(configs_path, "r") as f:
        configs = json.load(f)

    # Extract metrics from all existing reports
    all_metrics = []
    reports_dir = output_dir / "reports"

    for config in configs:
        config_id = config["config_id"]
        report_path = reports_dir / f"config_{config_id:06d}.json"

        if report_path.exists():
            # Determine status and runtime from report
            try:
                with open(report_path, "r") as f:
                    data = json.load(f)
                status = "success"
                runtime_sec = data.get("runtime_sec", 0.0)
            except:
                status = "unknown"
                runtime_sec = 0.0

            metrics = extract_metrics_from_report(
                report_path, config, runtime_sec, status,
                config.get("max_templates", 5)
            )
            if metrics:
                all_metrics.append(metrics)

    if not all_metrics:
        print("No valid metrics found")
        return

    # Sort metrics
    sorted_metrics = sort_metrics(all_metrics)

    # Save all outputs
    save_summary_csv(sorted_metrics, output_dir / "summary.csv")
    print(f"Saved summary.csv ({len(sorted_metrics)} configs)")

    save_summary_top_txt(sorted_metrics, output_dir / "summary_top.txt")
    print(f"Saved summary_top.txt")

    # Save all_results.jsonl
    jsonl_path = output_dir / "all_results.jsonl"
    with open(jsonl_path, "w") as f:
        for metrics in all_metrics:
            f.write(json.dumps(metrics) + "\n")
    print(f"Saved all_results.jsonl")

    # Generate watchlist
    generate_watchlist(sorted_metrics, output_dir / "watchlist.csv")
    print(f"Saved watchlist.csv")

    print(f"\nTop 5 configurations:")
    for rank, m in enumerate(sorted_metrics[:5], start=1):
        print(f"  {rank}. [ID={m['config_id']}] "
              f"h={m['horizon']}, e={m['execute_steps']}, m={m['max_mpc_steps']}, "
              f"s={m['num_samples']}, i={m['num_iterations']} -> "
              f"PSR={m['success_pose_2cm_15deg_rate']:.3f}, SR={m['success_rate']:.3f}, "
              f"PE={m.get('mean_final_pos_error', float('nan')):.4f}, RT={m['runtime_sec']:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Wide overnight MuJoCo Oracle-MPC closed-loop parameter sweep"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["wide_overnight_v2", "boundary_refine_v1"],
        default="wide_overnight_v2",
        help="Sweep preset (default: wide_overnight_v2)",
    )
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        default=False,
        help="Run full max_mpc_steps even after reaching success threshold.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=24,
        help="Number of parallel jobs for numerical evaluation (default: 24)",
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        default=5,
        help="Max templates per config (default: 5)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=36000,
        help="Timeout per config in seconds (default: 36000 = 10 hours)",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="runs/sweeps",
        help="Output root directory (default: runs/sweeps)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run directory for resume (default: None, create new)",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configs with existing reports (default: False)",
    )
    parser.add_argument(
        "--rebuild-summary-only",
        action="store_true",
        help="Only rebuild summary files from existing reports (default: False)",
    )
    parser.add_argument(
        "--render-best-video",
        action="store_true",
        help="Render video for best config after sweep (default: False)",
    )
    parser.add_argument(
        "--render-best-video-only",
        action="store_true",
        help="Only render video for best config from existing summary (default: False)",
    )
    parser.add_argument(
        "--video-template-index",
        type=int,
        default=0,
        help="Template index for best config video (default: 0)",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1280,
        help="Video width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=720,
        help="Video height in pixels (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video frames per second (default: 10)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.run_dir:
        output_dir = Path(args.run_dir)
        if not output_dir.exists():
            print(f"Error: --run-dir {output_dir} does not exist")
            return
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.out_root) / f"{args.preset}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Handle rebuild-summary-only mode
    if args.rebuild_summary_only:
        rebuild_summaries(output_dir)
        return

    # Handle render-best-video-only mode
    if args.render_best_video_only:
        summary_csv_path = output_dir / "summary.csv"
        if not summary_csv_path.exists():
            print(f"Error: {summary_csv_path} not found")
            return

        # Read summary CSV to get best config
        with open(summary_csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print("Error: summary.csv is empty")
            return

        best_row = rows[0]
        best_config = {
            "config_id": int(best_row["config_id"]),
            "horizon": int(best_row["horizon"]),
            "execute_steps": int(best_row["execute_steps"]),
            "max_mpc_steps": int(best_row["max_mpc_steps"]),
            "num_samples": int(best_row["num_samples"]),
            "num_elites": int(best_row["num_elites"]),
            "num_iterations": int(best_row["num_iterations"]),
        }

        render_best_config_video(
            best_config,
            output_dir,
            args.video_template_index,
            args.video_width,
            args.video_height,
            args.fps,
        )
        return

    # Check resume mode
    if args.resume and not args.run_dir:
        print("WARNING: --resume only works with --run-dir")
        print("         Resume will work within current output dir")

    print("=" * 80)
    print("Wide MuJoCo Oracle-MPC Closed-Loop Parameter Sweep")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Preset: {args.preset}")
    print(f"Parallel jobs: {args.jobs}")
    print(f"Max templates per config: {args.max_templates}")
    print(f"Timeout per config: {args.timeout_sec} seconds")
    print(f"Resume mode: {args.resume}")
    print(f"Render best video: {args.render_best_video}")
    print()

    # Generate sweep configurations
    configs = generate_sweep_configs(args.preset)
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

    # Initialize JSONL file
    jsonl_path = output_dir / "all_results.jsonl"

    # Run all configurations in parallel
    print("Starting numerical evaluations...")
    print()
    successful_configs = []
    failed_configs = []
    config_results = {}  # config_id -> (runtime_sec, status, failure_reason)
    all_metrics = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_single_config,
                config,
                output_dir,
                args.max_templates,
                args.timeout_sec,
                args.resume,
                args.disable_early_stop,
            ): config
            for config in configs
        }

        for future in as_completed(futures):
            config, success, runtime_sec, status, failure_reason = future.result()
            config_results[config["config_id"]] = (runtime_sec, status, failure_reason)

            if success:
                successful_configs.append(config)

                # Extract metrics immediately (including skipped configs with existing reports)
                report_path = output_dir / "reports" / f"config_{config['config_id']:06d}.json"
                if report_path.exists():
                    metrics = extract_metrics_from_report(
                        report_path, config, runtime_sec, status, args.max_templates
                    )
                    if metrics:
                        all_metrics.append(metrics)
                        # Append to JSONL immediately (skip if already skipped to avoid duplicates)
                        if status != "skipped":
                            append_to_jsonl(metrics, jsonl_path)

                        completed_count += 1
                        # Rebuild summaries every 10 configs
                        if completed_count % 10 == 0:
                            sorted_metrics = sort_metrics(all_metrics)
                            save_summary_csv(sorted_metrics, output_dir / "summary.csv")
                            save_summary_top_txt(sorted_metrics, output_dir / "summary_top.txt")
                            print(f"[Progress] Updated summaries ({completed_count} configs completed)")
            else:
                # Add failure info to config
                failed_config = config.copy()
                failed_config["failure_reason"] = failure_reason
                failed_config["log_path"] = str(
                    output_dir / "logs" / f"config_{config['config_id']:06d}_check.log"
                )
                failed_configs.append(failed_config)

    print()
    print("=" * 80)
    print(f"Numerical evaluation completed:")
    print(f"  Successful: {len(successful_configs)}")
    print(f"  Failed: {len(failed_configs)}")
    print("=" * 80)
    print()

    # Save failed configs (always, even if empty)
    failed_configs_path = output_dir / "failed_configs.json"
    with open(failed_configs_path, "w") as f:
        json.dump(failed_configs, f, indent=2)
    print(f"Saved failed configurations to: {failed_configs_path}")
    print()

    if not all_metrics:
        print("No successful runs with valid metrics. Exiting.")
        return

    # Final sort and save
    sorted_metrics = sort_metrics(all_metrics)

    # Save full summary
    summary_csv_path = output_dir / "summary.csv"
    save_summary_csv(sorted_metrics, summary_csv_path)
    print(f"Saved summary to: {summary_csv_path}")

    # Save top results
    summary_top_path = output_dir / "summary_top.txt"
    save_summary_top_txt(sorted_metrics, summary_top_path)
    print(f"Saved top results to: {summary_top_path}")

    # Generate watchlist
    watchlist_path = output_dir / "watchlist.csv"
    generate_watchlist(sorted_metrics, watchlist_path)
    print(f"Saved watchlist to: {watchlist_path}")

    print()
    print("Top 5 configurations:")
    for rank, m in enumerate(sorted_metrics[:5], start=1):
        print(f"  {rank}. [ID={m['config_id']}] "
              f"h={m['horizon']}, e={m['execute_steps']}, m={m['max_mpc_steps']}, "
              f"s={m['num_samples']}, i={m['num_iterations']} -> "
              f"PSR={m['success_pose_2cm_15deg_rate']:.3f}, SR={m['success_rate']:.3f}, "
              f"PE={m.get('mean_final_pos_error', float('nan')):.4f}, RT={m['runtime_sec']:.1f}s")

    # Render best config video if requested
    if args.render_best_video and len(sorted_metrics) > 0:
        print()
        print("=" * 80)
        best_metrics = sorted_metrics[0]
        best_config = {
            "config_id": best_metrics["config_id"],
            "horizon": best_metrics["horizon"],
            "execute_steps": best_metrics["execute_steps"],
            "max_mpc_steps": best_metrics["max_mpc_steps"],
            "num_samples": best_metrics["num_samples"],
            "num_elites": best_metrics["num_elites"],
            "num_iterations": best_metrics["num_iterations"],
        }

        render_best_config_video(
            best_config,
            output_dir,
            args.video_template_index,
            args.video_width,
            args.video_height,
            args.fps,
        )
        print("=" * 80)

    print()
    print("Sweep completed successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()



