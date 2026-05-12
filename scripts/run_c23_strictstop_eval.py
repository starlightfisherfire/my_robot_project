#!/usr/bin/env python3
"""
Config23 Confirmatory Evaluation with Strict Pose Early Stop (600-step budget).

Config23 / c23_precise:
  horizon=80, execute_steps=20, max_mpc_steps=30 (600 total env steps)
  num_samples=1024, num_elites=96, num_iterations=5
  strict stop: pos<=1.5mm AND theta<=3.0deg (joint condition)

Templates: 3 open_space + 3 mild_offset from train_sim_id
  NOTE: train_sim_id only has open_space and mild_offset layouts.
  mild_offset is used as the "constrained" category (closest available).

Usage:
  MUJOCO_GL=egl PYTHONPATH=. python scripts/run_c23_strictstop_eval.py
  MUJOCO_GL=osmesa PYTHONPATH=. python scripts/run_c23_strictstop_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.mujoco_push_env import MujocoPushEnv
from src.interventions.reset_template_loader import load_reset_templates
from src.metrics.mujoco_oracle_capacity import (
    evaluate_one_template_mujoco_oracle_mpc_closed_loop,
)

# ─── Config23 parameters ────────────────────────────────────────────────────

C23_CONFIG: dict[str, Any] = {
    "horizon": 80,
    "execute_steps": 20,
    "max_mpc_steps": 30,          # 30 * 20 = 600 total env steps
    "num_samples": 1024,
    "num_elites": 96,
    "num_iterations": 5,
    "success_pos_threshold": 0.0015,   # 1.5 mm
    "success_theta_threshold_deg": 3.0,
    "success_dist_threshold": 0.05,    # legacy 5cm threshold (for backward-compat metrics)
    "pusher_radius": 0.010,
    "pusher_halfheight": 0.014,
    "pusher_z": 0.016,
    "seed": 42,
}

TEMPLATES_PATH = "data/sim/metadata/reset_templates_v0.json"
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 10


# ─── Template selection ──────────────────────────────────────────────────────

def select_templates(
    templates_path: str,
    n_open: int = 3,
    n_constrained: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    Select n_open open_space and n_constrained mild_offset templates.

    NOTE: train_sim_id only has open_space and mild_offset layouts.
    mild_offset is used as the 'constrained' category (closest available).
    There are no blocking/narrow_passage/edge_goal templates in train_sim_id.
    """
    templates = load_reset_templates(templates_path)
    train = [t for t in templates if t["split"] == "train_sim_id"]
    open_t = [t for t in train if t["layout_family"] == "open_space"]
    constrained_t = [t for t in train if t["layout_family"] == "mild_offset"]
    return open_t[:n_open], constrained_t[:n_constrained]


# ─── Evaluation ─────────────────────────────────────────────────────────────

def run_one_eval(
    template: dict,
    config: dict,
    seed: int,
    log_path: Path,
) -> dict[str, Any]:
    """Run closed-loop evaluation for one template, capturing stdout to log."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        result = evaluate_one_template_mujoco_oracle_mpc_closed_loop(
            template=template,
            planning_horizon=config["horizon"],
            num_samples=config["num_samples"],
            num_elites=config["num_elites"],
            num_iterations=config["num_iterations"],
            execute_steps=config["execute_steps"],
            max_mpc_steps=config["max_mpc_steps"],
            seed=seed,
            success_dist_threshold=config["success_dist_threshold"],
            pusher_radius=config["pusher_radius"],
            pusher_halfheight=config["pusher_halfheight"],
            pusher_z=config["pusher_z"],
            disable_early_stop=False,
            success_pos_threshold=config["success_pos_threshold"],
            success_theta_threshold_deg=config["success_theta_threshold_deg"],
        )
    log_text = buf.getvalue()
    # Also print to real stdout
    print(log_text, end="")
    log_path.write_text(log_text, encoding="utf-8")
    return result


# ─── Report generation ───────────────────────────────────────────────────────

def build_report(result: dict, config: dict, category: str) -> dict:
    """Build a flat JSON report from evaluation result."""
    tid = result["reset_template_id"]
    fr = result.get("threshold_first_reach", {})
    fr1p5 = result.get("first_reach_1p5mm_3deg", {})

    def _steps(tname: str):
        return fr.get(tname, {}).get("total_executed_steps")

    def _delta(tname: str):
        trace = result.get("threshold_post_reach_trace", {}).get(tname, [])
        if not trace:
            return None
        first_pos = fr.get(tname, {}).get("pos_error")
        if first_pos is None:
            return None
        return trace[-1]["pos_error"] - first_pos

    report = {
        "template_id": tid,
        "template_name": tid,
        "split": result["split"],
        "layout_family": result["layout_family"],
        "shape_family": result["shape_family"],
        "category": category,
        # Config
        "config_horizon": config["horizon"],
        "config_execute_steps": config["execute_steps"],
        "config_max_mpc_steps": config["max_mpc_steps"],
        "config_num_samples": config["num_samples"],
        "config_num_elites": config["num_elites"],
        "config_num_iterations": config["num_iterations"],
        "config_success_pos_threshold_m": config["success_pos_threshold"],
        "config_success_theta_threshold_deg": config["success_theta_threshold_deg"],
        # Distance
        "initial_distance": result["initial_dist"],
        "final_pos_error": result["final_pos_error"],
        "final_theta_error_deg": result["final_theta_error_deg"],
        "best_pos_error": result["best_pos_error"],
        "best_theta_error_deg_at_best_pos": result["best_theta_error_deg_at_best_pos"],
        "final_pose_cost": result["final_pose_cost"],
        "best_pose_cost": result["best_pose_cost"],
        # Steps
        "total_env_steps": result["total_executed_steps"],
        "total_mpc_steps_used": result["num_mpc_steps"],
        "strict_pose_success": result.get("strict_pose_success", False),
        "strict_pose_stop_step": result.get("strict_pose_stop_step"),
        "strict_pose_stop_pos_error": result.get("strict_pose_stop_pos_error"),
        "strict_pose_stop_theta_error_deg": result.get("strict_pose_stop_theta_error_deg"),
        # Threshold first-reach steps
        "first_reach_5cm_steps": _steps("5cm"),
        "first_reach_3cm_steps": _steps("3cm"),
        "first_reach_2cm_steps": _steps("2cm"),
        "first_reach_1p5cm_steps": _steps("1p5cm"),
        "first_reach_1cm_steps": _steps("1cm"),
        "first_reach_0p5cm_steps": _steps("0p5cm"),
        "first_reach_1p5mm_3deg_steps": fr1p5.get("total_executed_steps"),
        # Post-reach degradation
        "final_minus_first_reach_pos_error_5cm": _delta("5cm"),
        "final_minus_first_reach_pos_error_2cm": _delta("2cm"),
        "final_minus_first_reach_pos_error_1cm": _delta("1cm"),
        "final_minus_first_reach_pos_error_0p5cm": _delta("0p5cm"),
        # Position success flags
        "success_pos_5cm": result.get("success_pos_5cm"),
        "success_pos_3cm": result.get("success_pos_3cm"),
        "success_pos_2cm": result.get("success_pos_2cm"),
        "success_pos_1p5cm": result.get("success_pos_1p5cm"),
        "success_pos_1cm": result.get("success_pos_1cm"),
        "success_pos_0p5cm": result.get("success_pos_0p5cm"),
        # Pose success flags
        "success_pose_2cm_15deg": result.get("success_pose_2cm_15deg"),
        "success_pose_1cm_5deg": result.get("success_pose_1cm_5deg"),
        "success_pose_0p5cm_5deg": result.get("success_pose_0p5cm_5deg"),
        "success_pose_0p15cm_3deg": result.get("success_pose_0p15cm_3deg"),
        # Reach flags
        "reach_5cm": fr.get("5cm", {}).get("reached", False),
        "reach_3cm": fr.get("3cm", {}).get("reached", False),
        "reach_2cm": fr.get("2cm", {}).get("reached", False),
        "reach_1p5cm": fr.get("1p5cm", {}).get("reached", False),
        "reach_1cm": fr.get("1cm", {}).get("reached", False),
        "reach_0p5cm": fr.get("0p5cm", {}).get("reached", False),
        # Other
        "contact_rate": result["contact_rate"],
        "object_displacement": result["total_object_displacement"],
    }
    return report


# ─── Summary helpers ─────────────────────────────────────────────────────────

def _mean(vals: list) -> float:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")

def _rate(flags: list) -> float:
    flags = [bool(f) for f in flags if f is not None]
    return float(np.mean(flags)) if flags else float("nan")

def summarize_reports(reports: list[dict]) -> dict:
    """Compute aggregate stats over a list of per-template reports."""
    if not reports:
        return {}
    n = len(reports)
    return {
        "n": n,
        "mean_initial_distance": _mean([r["initial_distance"] for r in reports]),
        "mean_final_pos_error": _mean([r["final_pos_error"] for r in reports]),
        "mean_final_theta_error_deg": _mean([r["final_theta_error_deg"] for r in reports]),
        "mean_best_pos_error": _mean([r["best_pos_error"] for r in reports]),
        "mean_best_theta_error_deg_at_best_pos": _mean([r["best_theta_error_deg_at_best_pos"] for r in reports]),
        "mean_final_pose_cost": _mean([r["final_pose_cost"] for r in reports]),
        "mean_best_pose_cost": _mean([r["best_pose_cost"] for r in reports]),
        "mean_total_env_steps": _mean([r["total_env_steps"] for r in reports]),
        "mean_total_mpc_steps_used": _mean([r["total_mpc_steps_used"] for r in reports]),
        "strict_pose_success_rate": _rate([r["strict_pose_success"] for r in reports]),
        "mean_strict_pose_stop_step": _mean([r["strict_pose_stop_step"] for r in reports]),
        "mean_first_reach_5cm_steps": _mean([r["first_reach_5cm_steps"] for r in reports]),
        "mean_first_reach_1cm_steps": _mean([r["first_reach_1cm_steps"] for r in reports]),
        "mean_first_reach_0p5cm_steps": _mean([r["first_reach_0p5cm_steps"] for r in reports]),
        "mean_first_reach_1p5mm_3deg_steps": _mean([r["first_reach_1p5mm_3deg_steps"] for r in reports]),
        "success_pos_5cm_rate": _rate([r["success_pos_5cm"] for r in reports]),
        "success_pos_2cm_rate": _rate([r["success_pos_2cm"] for r in reports]),
        "success_pos_1cm_rate": _rate([r["success_pos_1cm"] for r in reports]),
        "success_pos_0p5cm_rate": _rate([r["success_pos_0p5cm"] for r in reports]),
        "success_pose_2cm_15deg_rate": _rate([r["success_pose_2cm_15deg"] for r in reports]),
        "success_pose_1cm_5deg_rate": _rate([r["success_pose_1cm_5deg"] for r in reports]),
        "success_pose_0p5cm_5deg_rate": _rate([r["success_pose_0p5cm_5deg"] for r in reports]),
        "success_pose_0p15cm_3deg_rate": _rate([r["success_pose_0p15cm_3deg"] for r in reports]),
        "reach_5cm_rate": _rate([r["reach_5cm"] for r in reports]),
        "reach_1cm_rate": _rate([r["reach_1cm"] for r in reports]),
        "reach_0p5cm_rate": _rate([r["reach_0p5cm"] for r in reports]),
        "mean_contact_rate": _mean([r["contact_rate"] for r in reports]),
        "mean_object_displacement": _mean([r["object_displacement"] for r in reports]),
    }


def fmt(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def write_compact_summary(
    run_dir: Path,
    config: dict,
    open_reports: list[dict],
    obstacle_reports: list[dict],
    all_reports: list[dict],
    open_summary: dict,
    obstacle_summary: dict,
    all_summary: dict,
) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("Config23 Confirmatory Evaluation — Strict Pose Stop 600-step Budget")
    lines.append("=" * 72)
    lines.append("")
    lines.append("## 1. Experiment Configuration")
    lines.append(f"  horizon          = {config['horizon']}")
    lines.append(f"  execute_steps    = {config['execute_steps']}")
    lines.append(f"  max_mpc_steps    = {config['max_mpc_steps']}  (total budget = {config['execute_steps']*config['max_mpc_steps']} env steps)")
    lines.append(f"  num_samples      = {config['num_samples']}")
    lines.append(f"  num_elites       = {config['num_elites']}")
    lines.append(f"  num_iterations   = {config['num_iterations']}")
    lines.append(f"  strict stop      = pos <= {config['success_pos_threshold']*1000:.1f} mm  AND  theta <= {config['success_theta_threshold_deg']:.1f} deg")
    lines.append(f"  templates        = 3 open_space + 3 mild_offset (train_sim_id)")
    lines.append(f"  NOTE: train_sim_id has no blocking/narrow_passage templates.")
    lines.append(f"        mild_offset is used as the 'constrained' category.")
    lines.append("")

    def _section(label: str, reports: list[dict], summary: dict) -> None:
        lines.append(f"## {label}")
        for r in reports:
            stop_str = "EARLY_STOP" if r["strict_pose_success"] else f"full_budget({r['total_mpc_steps_used']}/{config['max_mpc_steps']})"
            lines.append(f"  {r['template_id']}")
            lines.append(f"    layout={r['layout_family']}  shape={r['shape_family']}")
            lines.append(f"    init_dist={fmt(r['initial_distance'])}m  final_pos={fmt(r['final_pos_error']*1000,2)}mm  final_theta={fmt(r['final_theta_error_deg'],2)}deg")
            lines.append(f"    best_pos={fmt(r['best_pos_error']*1000,2)}mm  best_theta@best_pos={fmt(r['best_theta_error_deg_at_best_pos'],2)}deg")
            lines.append(f"    steps={r['total_env_steps']}  {stop_str}")
            lines.append(f"    success_pose_0p15cm_3deg={r['success_pose_0p15cm_3deg']}  success_pose_1cm_5deg={r['success_pose_1cm_5deg']}")
            fr_str = fmt(r['first_reach_1p5mm_3deg_steps']) if r['first_reach_1p5mm_3deg_steps'] else "not_reached"
            lines.append(f"    first_reach_1p5mm_3deg_steps={fr_str}")
        lines.append(f"  --- Averages ({summary['n']} templates) ---")
        lines.append(f"  mean_final_pos_error        = {fmt(summary['mean_final_pos_error']*1000,2)} mm")
        lines.append(f"  mean_final_theta_error_deg  = {fmt(summary['mean_final_theta_error_deg'],2)} deg")
        lines.append(f"  mean_best_pos_error         = {fmt(summary['mean_best_pos_error']*1000,2)} mm")
        lines.append(f"  success_pose_0p15cm_3deg_rate = {fmt(summary['success_pose_0p15cm_3deg_rate'],3)}")
        lines.append(f"  success_pose_1cm_5deg_rate    = {fmt(summary['success_pose_1cm_5deg_rate'],3)}")
        lines.append(f"  strict_pose_early_stop_rate   = {fmt(summary['strict_pose_early_stop_rate'],3)}")
        lines.append(f"  mean_first_reach_1p5mm_3deg_steps = {fmt(summary['mean_first_reach_1p5mm_3deg_steps'],1)}")
        lines.append(f"  mean_total_env_steps          = {fmt(summary['mean_total_env_steps'],1)}")
        lines.append("")

    _section("2. Open / Easy Templates (open_space)", open_reports, open_summary)
    _section("3. Constrained Templates (mild_offset)", obstacle_reports, obstacle_summary)
    _section("4. All 6 Templates Combined", all_reports, all_summary)

    lines.append("## 5. Key Findings")
    lines.append(f"  success_pose_0p15cm_3deg_rate (all)      = {fmt(all_summary['success_pose_0p15cm_3deg_rate'],3)}")
    lines.append(f"  mean_final_pos_error (all)               = {fmt(all_summary['mean_final_pos_error']*1000,2)} mm")
    lines.append(f"  mean_final_theta_error_deg (all)         = {fmt(all_summary['mean_final_theta_error_deg'],2)} deg")
    lines.append(f"  mean_best_pos_error (all)                = {fmt(all_summary['mean_best_pos_error']*1000,2)} mm")
    lines.append(f"  mean_first_reach_0p5cm_steps (all)       = {fmt(all_summary['mean_first_reach_0p5cm_steps'],1)}")
    lines.append(f"  mean_first_reach_1p5mm_3deg_steps (all)  = {fmt(all_summary['mean_first_reach_1p5mm_3deg_steps'],1)}")
    lines.append(f"  mean_total_env_steps (all)               = {fmt(all_summary['mean_total_env_steps'],1)}")
    lines.append("")

    early_stop_cases = [r["template_id"] for r in all_reports if r["strict_pose_success"]]
    full_budget_cases = [r["template_id"] for r in all_reports if not r["strict_pose_success"]]
    lines.append(f"  Early stop triggered: {early_stop_cases if early_stop_cases else 'none'}")
    lines.append(f"  Used full budget:     {full_budget_cases if full_budget_cases else 'none'}")
    lines.append("=" * 72)

    txt = "\n".join(lines)
    (run_dir / "compact_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


# ─── Video rendering ─────────────────────────────────────────────────────────

def render_video_for_template(
    template: dict,
    config: dict,
    seed: int,
    out_video: Path,
    category: str,
) -> bool:
    """Call render_closed_loop_rollout.py as subprocess to render video."""
    tid = template["reset_template_id"]
    split = template["split"]
    # Find template index within split
    templates = load_reset_templates(TEMPLATES_PATH)
    split_templates = [t for t in templates if t["split"] == split]
    idx = next((i for i, t in enumerate(split_templates) if t["reset_template_id"] == tid), None)
    if idx is None:
        print(f"  ERROR: template {tid} not found in split {split}", file=sys.stderr)
        return False

    cmd = [
        sys.executable, "scripts/render_closed_loop_rollout.py",
        "--templates", TEMPLATES_PATH,
        "--split", split,
        "--template-index", str(idx),
        "--horizon", str(config["horizon"]),
        "--execute-steps", str(config["execute_steps"]),
        "--max-mpc-steps", str(config["max_mpc_steps"]),
        "--num-samples", str(config["num_samples"]),
        "--num-elites", str(config["num_elites"]),
        "--num-iterations", str(config["num_iterations"]),
        "--success-pos-threshold", str(config["success_pos_threshold"]),
        "--success-theta-threshold-deg", str(config["success_theta_threshold_deg"]),
        "--success-dist-threshold", str(config["success_dist_threshold"]),
        "--width", str(VIDEO_WIDTH),
        "--height", str(VIDEO_HEIGHT),
        "--fps", str(VIDEO_FPS),
        "--seed", str(seed),
        "--out-video", str(out_video),
        "--pusher-radius", str(config["pusher_radius"]),
        "--pusher-halfheight", str(config["pusher_halfheight"]),
        "--pusher-z", str(config["pusher_z"]),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    print(f"\n  Rendering video: {out_video.name}")
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        print(f"  WARNING: render returned code {ret.returncode}", file=sys.stderr)
        return False
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config23 confirmatory eval with strict pose stop")
    p.add_argument("--templates", default=TEMPLATES_PATH)
    p.add_argument("--skip-video", action="store_true", help="Skip video rendering")
    p.add_argument("--run-dir", default=None, help="Override run directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = C23_CONFIG.copy()

    # ── Create run directory ──────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path(f"runs/video_sweeps/c23_precise_strictstop600_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    print(f"\n{'='*72}")
    print(f"Config23 Confirmatory Evaluation — Strict Pose Stop 600-step Budget")
    print(f"Run directory: {run_dir}")
    print(f"{'='*72}\n")

    # ── Select templates ──────────────────────────────────────────────────
    open_templates, constrained_templates = select_templates(args.templates)
    all_templates = open_templates + constrained_templates
    categories = ["open"] * 3 + ["obstacle"] * 3

    print("Selected templates:")
    print("  Open / easy (open_space):")
    for t in open_templates:
        print(f"    {t['reset_template_id']}  layout={t['layout_family']}  shape={t['shape_family']}")
    print("  Constrained (mild_offset — closest available, no blocking/narrow_passage in train_sim_id):")
    for t in constrained_templates:
        print(f"    {t['reset_template_id']}  layout={t['layout_family']}  shape={t['shape_family']}")
    print()

    # ── Run evaluations ───────────────────────────────────────────────────
    all_reports = []
    all_results = []
    video_paths = []

    for i, (template, category) in enumerate(zip(all_templates, categories)):
        tid = template["reset_template_id"]
        seed = config["seed"] + i
        print(f"\n{'─'*72}")
        print(f"[{i+1}/6] Evaluating: {tid}  category={category}  seed={seed}")
        print(f"{'─'*72}")

        log_path = run_dir / "logs" / f"{tid}.log"
        result = run_one_eval(template, config, seed, log_path)
        all_results.append(result)

        report = build_report(result, config, category)
        all_reports.append(report)

        report_path = run_dir / "reports" / f"{tid}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Report saved: {report_path.name}")

        # ── Render video ──────────────────────────────────────────────────
        if not args.skip_video:
            cat_short = "open" if category == "open" else "obstacle"
            video_name = f"c23_strict600_{cat_short}_{tid[-6:]}.mp4"
            video_path = run_dir / "videos" / video_name
            ok = render_video_for_template(template, config, seed, video_path, category)
            if ok:
                video_paths.append(str(video_path))
                print(f"  Video saved: {video_name}")
            else:
                print(f"  Video FAILED for {tid}")
        else:
            print(f"  Video skipped (--skip-video)")

    # ── Generate summaries ────────────────────────────────────────────────
    open_reports = [r for r in all_reports if r["category"] == "open"]
    obstacle_reports = [r for r in all_reports if r["category"] == "obstacle"]
    open_summary = summarize_reports(open_reports)
    obstacle_summary = summarize_reports(obstacle_reports)
    all_summary = summarize_reports(all_reports)

    # summary.csv
    if all_reports:
        csv_path = run_dir / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_reports[0].keys()))
            writer.writeheader()
            writer.writerows(all_reports)

        open_csv = run_dir / "summary_open.csv"
        with open_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_reports[0].keys()))
            writer.writeheader()
            writer.writerows(open_reports)

        obstacle_csv = run_dir / "summary_obstacle.csv"
        with obstacle_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_reports[0].keys()))
            writer.writeheader()
            writer.writerows(obstacle_reports)

    # summary_top.txt
    top_lines = [
        f"run_dir: {run_dir}",
        f"timestamp: {ts}",
        f"n_templates: {len(all_reports)}",
        f"success_pose_0p15cm_3deg_rate (all):      {fmt(all_summary.get('success_pose_0p15cm_3deg_rate'),3)}",
        f"success_pose_1cm_5deg_rate (all):         {fmt(all_summary.get('success_pose_1cm_5deg_rate'),3)}",
        f"mean_final_pos_error (all):               {fmt(all_summary.get('mean_final_pos_error',0)*1000,2)} mm",
        f"mean_final_theta_error_deg (all):         {fmt(all_summary.get('mean_final_theta_error_deg'),2)} deg",
        f"mean_best_pos_error (all):                {fmt(all_summary.get('mean_best_pos_error',0)*1000,2)} mm",
        f"strict_pose_early_stop_rate (all):        {fmt(all_summary.get('strict_pose_early_stop_rate'),3)}",
        f"mean_total_env_steps (all):               {fmt(all_summary.get('mean_total_env_steps'),1)}",
        f"videos: {video_paths}",
    ]
    (run_dir / "summary_top.txt").write_text("\n".join(top_lines), encoding="utf-8")

    # manifest.json
    manifest = {
        "run_dir": str(run_dir),
        "timestamp": ts,
        "config": config,
        "templates": [
            {"id": t["reset_template_id"], "category": c, "layout": t["layout_family"], "shape": t["shape_family"]}
            for t, c in zip(all_templates, categories)
        ],
        "reports": [str(run_dir / "reports" / f"{r['template_id']}.json") for r in all_reports],
        "videos": video_paths,
        "logs": [str(run_dir / "logs" / f"{r['template_id']}.log") for r in all_reports],
        "summary_csv": str(run_dir / "summary.csv"),
        "summary_open_csv": str(run_dir / "summary_open.csv"),
        "summary_obstacle_csv": str(run_dir / "summary_obstacle.csv"),
        "summary_top_txt": str(run_dir / "summary_top.txt"),
        "compact_summary_txt": str(run_dir / "compact_summary.txt"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # compact_summary.txt
    write_compact_summary(run_dir, config, open_reports, obstacle_reports, all_reports,
                          open_summary, obstacle_summary, all_summary)

    print(f"\n{'='*72}")
    print(f"Run complete. Directory: {run_dir}")
    print(f"Videos: {len(video_paths)}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()

