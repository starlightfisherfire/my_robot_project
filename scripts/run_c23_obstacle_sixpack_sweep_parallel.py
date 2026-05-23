#!/usr/bin/env python3
"""
Config23 Obstacle Sixpack Sweep — Parallel version.

Same as run_c23_obstacle_sixpack_sweep.py but runs evaluations in parallel
using ProcessPoolExecutor.  Each evaluation gets its own process with its
own MujocoPushEnv, so there is no shared-state problem.

Usage (12 workers for 6 templates x 3 budgets = 18 evals):
  MUJOCO_GL=egl PYTHONPATH=. python scripts/run_c23_obstacle_sixpack_sweep_parallel.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --output-root runs/obstacle_sweeps \
    --run-name c23_obstacle_sixpack \
    --seed 42 \
    --num-workers 12

Usage (speed ablation — capacity ablation, not physics override):
  MUJOCO_GL=egl PYTHONPATH=. python scripts/run_c23_obstacle_sixpack_sweep_parallel.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --budgets c23_strict1000 \
    --max-speed-mps 0.075 \
    --num-workers 12
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# ─── Budget definitions (must be picklable → module-level dict) ──────────────

BUDGETS = {
    "c23_strict600": {
        "horizon": 80,
        "execute_steps": 20,
        "max_mpc_steps": 30,
        "num_samples": 1024,
        "num_elites": 96,
        "num_iterations": 5,
        "success_pos_threshold": 0.0015,
        "success_theta_threshold_deg": 3.0,
        "success_dist_threshold": 0.05,
        "pusher_radius": 0.010,
        "pusher_halfheight": 0.014,
        "pusher_z": 0.016,
    },
    "c23_strict800": {
        "horizon": 80,
        "execute_steps": 20,
        "max_mpc_steps": 40,
        "num_samples": 1024,
        "num_elites": 96,
        "num_iterations": 5,
        "success_pos_threshold": 0.0015,
        "success_theta_threshold_deg": 3.0,
        "success_dist_threshold": 0.05,
        "pusher_radius": 0.010,
        "pusher_halfheight": 0.014,
        "pusher_z": 0.016,
    },
    "c23_strict1000": {
        "horizon": 80,
        "execute_steps": 20,
        "max_mpc_steps": 50,
        "num_samples": 1024,
        "num_elites": 96,
        "num_iterations": 5,
        "success_pos_threshold": 0.0015,
        "success_theta_threshold_deg": 3.0,
        "success_dist_threshold": 0.05,
        "pusher_radius": 0.010,
        "pusher_halfheight": 0.014,
        "pusher_z": 0.016,
    },
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _mean(vals: list) -> float:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _median(vals: list) -> float:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(vals)) if vals else float("nan")


def _rate(flags: list) -> float:
    flags = [bool(f) for f in flags if f is not None]
    return float(np.mean(flags)) if flags else float("nan")


def fmt(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


# ─── Worker function (runs in subprocess) ────────────────────────────────────

def _run_one_task(args_tuple: tuple) -> dict:
    """Run one evaluation in a subprocess.  Returns a dict with 'report' and 'log'.

    This function is the top-level callable for ProcessPoolExecutor, so it
    must be picklable — all imports happen inside the function.
    """
    template, config, budget_name, run_name, seed, log_path_str, max_speed_mps = args_tuple
    log_path = Path(log_path_str)

    import io
    from contextlib import redirect_stdout

    from src.metrics.mujoco_oracle_capacity import (
        evaluate_one_template_mujoco_oracle_mpc_closed_loop,
    )

    buf = io.StringIO()
    try:
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
                max_speed_mps=max_speed_mps,
            )

        log_text = buf.getvalue()
        log_path.write_text(log_text, encoding="utf-8")

        report = _build_report(result, template, config, budget_name, run_name)
        return {"status": "ok", "report": report, "log_path": str(log_path)}

    except Exception as e:
        log_text = buf.getvalue()
        log_path.write_text(log_text, encoding="utf-8")

        tid = template["reset_template_id"]
        report = {
            "run_name": run_name,
            "budget_name": budget_name,
            "reset_template_id": tid,
            "split": template["split"],
            "layout_family": template["layout_family"],
            "shape_family": template["shape_family"],
            "num_obstacles": len(template.get("obstacles", [])),
            "obstacle_sizes": "",
            "passage_gap": template.get("passage_gap"),
            "effective_passage_gap": template.get("effective_passage_gap"),
            "passage_center_distance": template.get("passage_center_distance"),
            "passage_gap_definition": template.get("passage_gap_definition"),
            "blocking_difficulty": template.get("blocking_difficulty"),
            # Success flags (all False on failure)
            "success": False,
            "success_definition": "success_pose_1cm_5deg",
            "primary_success": False,
            "coarse_success": False,
            "precision_success": False,
            "strict_completion": False,
            "legacy_pos_5cm": False,
            "strict_pose_success": False,
            "success_pos_5cm": False,
            "success_pos_3cm": False,
            "success_pos_2cm": False,
            "success_pos_1p5cm": False,
            "success_pos_1cm": False,
            "success_pos_0p5cm": False,
            "success_pos_0p15cm": False,
            "success_pose_2cm_15deg": False,
            "success_pose_1cm_5deg": False,
            "success_pose_0p5cm_5deg": False,
            "success_pose_0p15cm_3deg": False,
            # Continuous metrics (None on failure)
            "initial_dist": None,
            "final_pos_error": None,
            "final_theta_error_deg": None,
            "best_pos_error": None,
            "best_theta_error_deg_at_best_pos": None,
            "final_pose_cost": None,
            "best_pose_cost": None,
            # Steps
            "total_executed_steps": 0,
            "num_mpc_steps": 0,
            "strict_pose_stop_step": None,
            "strict_pose_stop_pos_error": None,
            "strict_pose_stop_theta_error_deg": None,
            # Contact / collision
            "contact_rate": None,
            "collision_rate": None,
            "collision_count": None,
            "collision_any": None,
            "object_obstacle_collision_rate": None,
            "object_displacement": None,
            "failure_reason": str(e),
        }
        return {"status": "error", "report": report, "error": str(e), "log_path": str(log_path)}


# ─── Report building (must be in module scope for pickling) ──────────────────

def _build_report(
    result: dict,
    template: dict,
    config: dict,
    budget_name: str,
    run_name: str,
) -> dict:
    tid = result["reset_template_id"]
    fr = result.get("threshold_first_reach", {})

    obstacles = template.get("obstacles", [])
    n_obs = len(obstacles)
    obs_sizes = "; ".join(
        f"{o['size_x']:.3f}x{o['size_y']:.3f}" for o in obstacles
    )

    return {
        "run_name": run_name,
        "budget_name": budget_name,
        "reset_template_id": tid,
        "split": result["split"],
        "layout_family": result["layout_family"],
        "shape_family": result["shape_family"],
        "num_obstacles": n_obs,
        "obstacle_sizes": obs_sizes,
        "passage_gap": template.get("passage_gap"),
        "effective_passage_gap": template.get("effective_passage_gap"),
        "passage_center_distance": template.get("passage_center_distance"),
        "passage_gap_definition": template.get("passage_gap_definition"),
        "blocking_difficulty": template.get("blocking_difficulty"),
        "initial_dist": result["initial_dist"],
        "final_pos_error": result["final_pos_error"],
        "final_theta_error_deg": result["final_theta_error_deg"],
        "best_pos_error": result["best_pos_error"],
        "best_theta_error_deg_at_best_pos": result["best_theta_error_deg_at_best_pos"],
        "final_pose_cost": result["final_pose_cost"],
        "best_pose_cost": result["best_pose_cost"],
        "total_executed_steps": result["total_executed_steps"],
        "num_mpc_steps": result["num_mpc_steps"],
        "strict_pose_success": result.get("strict_pose_success", False),
        "strict_pose_stop_step": result.get("strict_pose_stop_step"),
        "strict_pose_stop_pos_error": result.get("strict_pose_stop_pos_error"),
        "strict_pose_stop_theta_error_deg": result.get("strict_pose_stop_theta_error_deg"),
        "success_pos_5cm": result.get("success_pos_5cm"),
        "success_pos_3cm": result.get("success_pos_3cm"),
        "success_pos_2cm": result.get("success_pos_2cm"),
        "success_pos_1p5cm": result.get("success_pos_1p5cm"),
        "success_pos_1cm": result.get("success_pos_1cm"),
        "success_pos_0p5cm": result.get("success_pos_0p5cm"),
        "success_pose_2cm_15deg": result.get("success_pose_2cm_15deg"),
        "success_pose_1cm_5deg": result.get("success_pose_1cm_5deg"),
        "success_pose_0p5cm_5deg": result.get("success_pose_0p5cm_5deg"),
        "success_pose_0p15cm_3deg": result.get("success_pose_0p15cm_3deg"),
        # Semantic aliases (paper naming)
        "primary_success": result.get("primary_success"),
        "coarse_success": result.get("coarse_success"),
        "precision_success": result.get("precision_success"),
        "strict_completion": result.get("strict_completion"),
        "legacy_pos_5cm": result.get("legacy_pos_5cm"),
        "success_definition": result.get("success_definition", "success_pose_1cm_5deg"),
        "threshold_first_reach": fr,
        "contact_rate": result["contact_rate"],
        "object_displacement": result.get("total_object_displacement"),
        "collision_rate": result.get("collision_rate"),
        "collision_count": result.get("collision_count"),
        "collision_any": result.get("collision_any"),
        "object_obstacle_collision_rate": result.get("object_obstacle_collision_rate"),
        "success": result.get("success"),
        "failure_reason": result.get("failure_reason", ""),
    }


# ─── Summary (identical to serial version) ──────────────────────────────────

def summarize_by_budget(reports: list[dict]) -> list[dict]:
    by_budget: dict[str, list[dict]] = {}
    for r in reports:
        by_budget.setdefault(r["budget_name"], []).append(r)

    rows = []
    for bname in sorted(by_budget):
        recs = by_budget[bname]
        n = len(recs)
        rows.append({
            "group": "budget",
            "budget_name": bname,
            "num_templates": n,
            "strict_pose_success_rate": _rate([r.get("strict_pose_success") for r in recs]),
            "mean_strict_pose_stop_step": _mean([r.get("strict_pose_stop_step") for r in recs]),
            "mean_final_pos_error": _mean([r.get("final_pos_error") for r in recs]),
            "median_final_pos_error": _median([r.get("final_pos_error") for r in recs]),
            "mean_best_pos_error": _mean([r.get("best_pos_error") for r in recs]),
            "median_best_pos_error": _median([r.get("best_pos_error") for r in recs]),
            "mean_final_theta_error_deg": _mean([r.get("final_theta_error_deg") for r in recs]),
            "mean_best_theta_error_deg_at_best_pos": _mean([r.get("best_theta_error_deg_at_best_pos") for r in recs]),
            "mean_final_pose_cost": _mean([r.get("final_pose_cost") for r in recs]),
            "mean_best_pose_cost": _mean([r.get("best_pose_cost") for r in recs]),
            "success_pose_0p15cm_3deg_rate": _rate([r.get("success_pose_0p15cm_3deg") for r in recs]),
            "success_pose_0p5cm_5deg_rate": _rate([r.get("success_pose_0p5cm_5deg") for r in recs]),
            "success_pose_1cm_5deg_rate": _rate([r.get("success_pose_1cm_5deg") for r in recs]),
            # Semantic success rates (paper naming)
            "success_rate": _rate([r.get("primary_success") for r in recs]),
            "success_rate_definition": "success_pose_1cm_5deg",
            "primary_success_rate": _rate([r.get("primary_success") for r in recs]),
            "coarse_success_rate": _rate([r.get("coarse_success") for r in recs]),
            "precision_success_rate": _rate([r.get("precision_success") for r in recs]),
            "strict_completion_rate": _rate([r.get("strict_completion") for r in recs]),
            "legacy_pos_5cm_rate": _rate([r.get("legacy_pos_5cm") for r in recs]),
            "mean_total_executed_steps": _mean([r.get("total_executed_steps") for r in recs]),
            "mean_contact_rate": _mean([r.get("contact_rate") for r in recs]),
            "mean_collision_rate": _mean([r.get("collision_rate") for r in recs]),
            "mean_collision_count": _mean([r.get("collision_count") for r in recs]),
            "median_collision_count": _median([r.get("collision_count") for r in recs]),
            "max_collision_count": max(
                (r.get("collision_count") for r in recs),
                default=float("nan"),
            ),
        })
    return rows


def summarize_by_layout(reports: list[dict]) -> list[dict]:
    by_key: dict[tuple, list[dict]] = {}
    for r in reports:
        key = (r["budget_name"], r["layout_family"])
        by_key.setdefault(key, []).append(r)

    rows = []
    for (bname, lfamily) in sorted(by_key):
        recs = by_key[(bname, lfamily)]
        n = len(recs)
        rows.append({
            "group": "layout",
            "budget_name": bname,
            "layout_family": lfamily,
            "num_templates": n,
            "strict_pose_success_rate": _rate([r.get("strict_pose_success") for r in recs]),
            "mean_final_pos_error": _mean([r.get("final_pos_error") for r in recs]),
            "mean_final_theta_error_deg": _mean([r.get("final_theta_error_deg") for r in recs]),
            "mean_best_pos_error": _mean([r.get("best_pos_error") for r in recs]),
            "success_pose_0p5cm_5deg_rate": _rate([r.get("success_pose_0p5cm_5deg") for r in recs]),
            "success_pose_0p15cm_3deg_rate": _rate([r.get("success_pose_0p15cm_3deg") for r in recs]),
            # Semantic success rates
            "success_rate": _rate([r.get("primary_success") for r in recs]),
            "success_rate_definition": "success_pose_1cm_5deg",
            "primary_success_rate": _rate([r.get("primary_success") for r in recs]),
            "coarse_success_rate": _rate([r.get("coarse_success") for r in recs]),
            "precision_success_rate": _rate([r.get("precision_success") for r in recs]),
            "strict_completion_rate": _rate([r.get("strict_completion") for r in recs]),
            "legacy_pos_5cm_rate": _rate([r.get("legacy_pos_5cm") for r in recs]),
            "mean_collision_count": _mean([r.get("collision_count") for r in recs]),
            "mean_collision_rate": _mean([r.get("collision_rate") for r in recs]),
        })
    return rows


def write_summary_csv(reports: list[dict], path: Path) -> None:
    if not reports:
        return
    fieldnames = list(reports[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)


def write_compact_summary(
    run_dir: Path,
    budgets: dict[str, dict],
    reports: list[dict],
) -> None:
    budget_summary = summarize_by_budget(reports)
    layout_summary = summarize_by_layout(reports)

    lines = []
    lines.append("=" * 72)
    lines.append("Config23 Obstacle Sixpack Sweep — Compact Summary (parallel)")
    lines.append("=" * 72)
    lines.append("")

    for bs in budget_summary:
        bname = bs["budget_name"]
        cfg = budgets[bname]
        lines.append(f"## Budget: {bname}")
        lines.append(f"  horizon={cfg['horizon']}  execute_steps={cfg['execute_steps']}  "
                      f"max_mpc_steps={cfg['max_mpc_steps']}  "
                      f"total_budget={cfg['execute_steps']*cfg['max_mpc_steps']}")
        lines.append(f"  strict stop: pos<={cfg['success_pos_threshold']*1000:.1f}mm  "
                      f"theta<={cfg['success_theta_threshold_deg']:.1f}deg")
        lines.append(f"  num_templates={bs['num_templates']}")
        lines.append(f"  success_rate_definition        = success_pose_1cm_5deg")
        lines.append(f"  primary_success_rate           = {fmt(bs['primary_success_rate'],3)}")
        lines.append(f"  coarse_success_rate            = {fmt(bs['coarse_success_rate'],3)}")
        lines.append(f"  precision_success_rate         = {fmt(bs['precision_success_rate'],3)}")
        lines.append(f"  strict_completion_rate         = {fmt(bs['strict_completion_rate'],3)}")
        lines.append(f"  strict_pose_success_rate       = {fmt(bs['strict_pose_success_rate'],3)}  (strict stop, not paper primary)")
        lines.append(f"  legacy_pos_5cm_rate            = {fmt(bs['legacy_pos_5cm_rate'],3)}  (debug only)")
        lines.append(f"  mean_final_pos_error           = {fmt(bs['mean_final_pos_error']*1000,2)} mm")
        lines.append(f"  median_final_pos_error         = {fmt(bs['median_final_pos_error']*1000,2)} mm")
        lines.append(f"  mean_best_pos_error            = {fmt(bs['mean_best_pos_error']*1000,2)} mm")
        lines.append(f"  median_best_pos_error          = {fmt(bs['median_best_pos_error']*1000,2)} mm")
        lines.append(f"  mean_final_theta_error_deg     = {fmt(bs['mean_final_theta_error_deg'],2)} deg")
        lines.append(f"  mean_final_pose_cost           = {fmt(bs['mean_final_pose_cost'],4)}")
        lines.append(f"  mean_best_pose_cost            = {fmt(bs['mean_best_pose_cost'],4)}")
        lines.append(f"  success_pose_0p15cm_3deg_rate  = {fmt(bs['success_pose_0p15cm_3deg_rate'],3)}")
        lines.append(f"  success_pose_0p5cm_5deg_rate   = {fmt(bs['success_pose_0p5cm_5deg_rate'],3)}")
        lines.append(f"  success_pose_1cm_5deg_rate     = {fmt(bs['success_pose_1cm_5deg_rate'],3)}")
        lines.append(f"  mean_total_executed_steps      = {fmt(bs['mean_total_executed_steps'],1)}")
        lines.append(f"  mean_contact_rate              = {fmt(bs['mean_contact_rate'],3)}")
        if not (isinstance(bs.get('mean_collision_rate', float('nan')), float) and math.isnan(bs.get('mean_collision_rate', float('nan')))):
            lines.append(f"  mean_collision_rate            = {fmt(bs['mean_collision_rate'],3)}")
            lines.append(f"  mean_collision_count           = {fmt(bs['mean_collision_count'],1)}")
            lines.append(f"  median_collision_count         = {fmt(bs['median_collision_count'],1)}")
            lines.append(f"  max_collision_count            = {fmt(bs['max_collision_count'],0)}")
        lines.append("")

    for bname in sorted(budgets):
        brecs = [r for r in reports if r["budget_name"] == bname]
        lines.append(f"--- {bname}: Per-Template Results ---")
        for r in brecs:
            stop_str = "EARLY_STOP" if r["strict_pose_success"] else f"full({r['num_mpc_steps']})"
            egap = r.get("effective_passage_gap")
            egap_str = f"  eff_gap={egap:.3f}m" if egap is not None else ""
            cc = r.get('collision_count')
            cc_str = f"  coll={cc:.0f}" if cc is not None else ""
            lines.append(
                f"  {r['reset_template_id']}  "
                f"layout={r['layout_family']}  "
                f"final_pos={fmt(r['final_pos_error']*1000,2)}mm  "
                f"final_theta={fmt(r['final_theta_error_deg'],2)}deg  "
                f"best_pos={fmt(r['best_pos_error']*1000,2)}mm  "
                f"steps={r['total_executed_steps']}  "
                f"{stop_str}{egap_str}{cc_str}"
            )
        lines.append("")

    lines.append("--- By Layout Family ---")
    for ls in layout_summary:
        lines.append(
            f"  {ls['budget_name']:20s}  {ls['layout_family']:25s}  "
            f"n={ls['num_templates']}  "
            f"primary={fmt(ls['primary_success_rate'],3)}  "
            f"coarse={fmt(ls['coarse_success_rate'],3)}  "
            f"precision={fmt(ls['precision_success_rate'],3)}  "
            f"mean_pos={fmt(ls['mean_final_pos_error']*1000,2)}mm  "
            f"mean_theta={fmt(ls['mean_final_theta_error_deg'],2)}deg"
        )
    lines.append("")
    lines.append("=" * 72)

    txt = "\n".join(lines)
    (run_dir / "compact_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Config23 obstacle sixpack sweep (parallel): 6 templates x N budgets",
    )
    p.add_argument(
        "--templates",
        default="data/sim/metadata/reset_templates_obstacle_sixpack_v0.json",
    )
    p.add_argument("--output-root", default="runs/obstacle_sweeps")
    p.add_argument("--run-name", default="c23_obstacle_sixpack")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=12,
                   help="Number of parallel workers (default: 12).")
    p.add_argument("--budgets", nargs="+", default=None)
    p.add_argument("--max-speed-mps", type=float, default=0.05,
                   help="Pusher max speed in m/s (default: 0.05).")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Late imports (main process only)
    from src.interventions.reset_template_loader import load_reset_templates

    template_path = Path(args.templates)
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    templates = load_reset_templates(template_path)
    print(f"Loaded {len(templates)} templates from {template_path}")

    for t in templates:
        print(f"  {t['reset_template_id']}  layout={t['layout_family']}")

    budget_names = args.budgets if args.budgets else list(BUDGETS.keys())
    for bn in budget_names:
        if bn not in BUDGETS:
            raise ValueError(f"Unknown budget: {bn}. Available: {list(BUDGETS.keys())}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{args.run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)

    total_evals = len(templates) * len(budget_names)
    num_workers = min(args.num_workers, total_evals)

    print(f"\n{'='*72}")
    print(f"Config23 Obstacle Sixpack Sweep (PARALLEL)")
    print(f"Run directory:  {run_dir}")
    print(f"Budgets:        {budget_names}")
    print(f"Templates:      {len(templates)}")
    print(f"Total evals:    {total_evals}")
    print(f"Workers:        {num_workers}")
    print(f"{'='*72}\n")

    # Build task list
    tasks: list[tuple] = []
    eval_index = 0
    run_name = args.run_name

    for budget_name in budget_names:
        config = BUDGETS[budget_name]
        for template in templates:
            tid = template["reset_template_id"]
            seed = args.seed + eval_index
            eval_index += 1
            log_path = run_dir / "logs" / f"{budget_name}__{tid}.log"
            tasks.append((template, config, budget_name, run_name, seed, str(log_path), args.max_speed_mps))

    # Run in parallel
    all_reports: list[dict] = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_info = {}
        for task_args in tasks:
            future = executor.submit(_run_one_task, task_args)
            budget_name = task_args[2]
            tid = task_args[0]["reset_template_id"]
            future_to_info[future] = (budget_name, tid)

        for future in as_completed(future_to_info):
            budget_name, tid = future_to_info[future]
            completed_count += 1

            try:
                res = future.result()
                all_reports.append(res["report"])
                status = res["status"]
                if status == "ok":
                    print(f"  [{completed_count}/{total_evals}] DONE  "
                          f"{budget_name} / {tid}")
                else:
                    print(f"  [{completed_count}/{total_evals}] ERROR "
                          f"{budget_name} / {tid}: {res.get('error','?')}")
            except Exception as e:
                print(f"  [{completed_count}/{total_evals}] CRASH "
                      f"{budget_name} / {tid}: {e}")
                all_reports.append({
                    "run_name": run_name,
                    "budget_name": budget_name,
                    "reset_template_id": tid,
                    "success": False,
                    "failure_reason": str(e),
                })

    # Write outputs
    write_summary_csv(all_reports, run_dir / "summary.csv")
    print(f"\nSaved: {run_dir / 'summary.csv'}")

    write_compact_summary(run_dir, BUDGETS, all_reports)

    manifest = {
        "run_dir": str(run_dir),
        "timestamp": ts,
        "templates_path": str(template_path),
        "budgets": {bn: BUDGETS[bn] for bn in budget_names},
        "num_templates": len(templates),
        "num_budgets": len(budget_names),
        "num_workers": num_workers,
        "total_evaluations": len(all_reports),
        "reports": [str(run_dir / "reports" / f"{r['budget_name']}__{r['reset_template_id']}.json")
                    for r in all_reports],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    file_list = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file())
    (run_dir / "file_list.txt").write_text("\n".join(file_list), encoding="utf-8")

    print(f"\n{'='*72}")
    print(f"Sweep complete. Directory: {run_dir}")
    print(f"Total evaluations: {len(all_reports)}")
    print(f"Workers used: {num_workers}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
