#!/usr/bin/env python3
"""
Render obstacle rollout videos in parallel via subprocess delegation.

Each video is rendered by a separate subprocess calling
scripts/render_closed_loop_rollout.py.  This avoids the contact_flags
attribute bug in the direct-env approach and isolates per-video crashes.

Usage (sixpack, 6 templates x 2 budgets = 12 videos):
  MUJOCO_GL=egl PYTHONPATH=. python scripts/render_obstacle_sixpack_videos_parallel.py \
    --templates data/sim/metadata/reset_templates_obstacle_sixpack_v0.json \
    --output-dir runs/obstacle_videos \
    --num-workers 12

Usage (difficulty, 6 splits x 10 templates x 2 budgets = 120 videos):
  MUJOCO_GL=egl PYTHONPATH=. python scripts/render_obstacle_sixpack_videos_parallel.py \
    --templates data/sim/metadata/reset_templates_obstacle_difficulty_v0.json \
    --output-dir runs/obstacle_videos \
    --num-workers 12
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_LAYOUTS = [
    "test_sim_layout_ood_blocking_easy",
    "test_sim_layout_ood_blocking_medium",
    "test_sim_layout_ood_blocking_hard",
    "test_sim_layout_ood_passage_direct_wide",
    "test_sim_layout_ood_passage_direct_medium",
    "test_sim_layout_ood_passage_direct_narrow",
]


BUDGETS = {
    "c23_strict600": {
        "horizon": 80,
        "execute_steps": 20,
        "max_mpc_steps": 30,
        "num_samples": 1024,
        "num_elites": 96,
        "num_iterations": 5,
        "stop_pos_threshold": 0.0015,
        "stop_theta_threshold_deg": 3.0,
    },
    "c23_strict800": {
        "horizon": 80,
        "execute_steps": 20,
        "max_mpc_steps": 40,
        "num_samples": 1024,
        "num_elites": 96,
        "num_iterations": 5,
        "stop_pos_threshold": 0.0015,
        "stop_theta_threshold_deg": 3.0,
    },
}


@dataclass(frozen=True)
class RenderJob:
    budget_name: str
    split: str
    template_index: int
    template_id: str
    out_video: str
    log_file: str


@dataclass
class RenderResult:
    budget_name: str
    split: str
    template_index: int
    template_id: str
    out_video: str
    log_file: str
    returncode: int
    ok: bool
    command: str


# ---------------------------------------------------------------------------
# Template validation
# ---------------------------------------------------------------------------

def load_and_validate_templates(
    template_path: Path,
    layouts: list[str],
    template_indices: list[int],
) -> dict[str, list[dict]]:
    """Load templates, group by split, validate indices.

    Returns dict mapping split_name -> list of template dicts (sorted by index).
    Exits with error if any split is missing or any index is out of range.
    """
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    with open(template_path, encoding="utf-8") as f:
        all_templates = json.load(f)

    by_split: dict[str, list[dict]] = {}
    for t in all_templates:
        split = t["split"]
        by_split.setdefault(split, []).append(t)

    # Sort each split by reset_template_id for stable ordering
    for split in by_split:
        by_split[split].sort(key=lambda t: t["reset_template_id"])

    # Validate splits exist
    missing_splits = [s for s in layouts if s not in by_split]
    if missing_splits:
        available = sorted(by_split.keys())
        print(f"ERROR: Splits not found in template file: {missing_splits}", file=sys.stderr)
        print(f"Available splits: {available}", file=sys.stderr)
        sys.exit(1)

    # Validate indices
    for split in layouts:
        n = len(by_split[split])
        bad = [i for i in template_indices if i >= n]
        if bad:
            print(
                f"ERROR: Split '{split}' has {n} templates (indices 0..{n-1}), "
                f"but requested indices include {bad}",
                file=sys.stderr,
            )
            sys.exit(1)

    return by_split


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

def build_command(job: RenderJob, args: argparse.Namespace, config: dict) -> list[str]:
    return [
        sys.executable,
        "scripts/render_closed_loop_rollout.py",
        "--templates", args.templates,
        "--split", job.split,
        "--template-index", str(job.template_index),
        "--camera", args.camera,
        "--horizon", str(config["horizon"]),
        "--execute-steps", str(config["execute_steps"]),
        "--max-mpc-steps", str(config["max_mpc_steps"]),
        "--num-samples", str(config["num_samples"]),
        "--num-elites", str(config["num_elites"]),
        "--num-iterations", str(config["num_iterations"]),
        "--strict-pose-stop",
        "--stop-pos-threshold", str(config["stop_pos_threshold"]),
        "--stop-theta-threshold-deg", str(config["stop_theta_threshold_deg"]),
        "--width", str(args.width),
        "--height", str(args.height),
        "--fps", str(args.fps),
        "--out-video", job.out_video,
    ]


# ---------------------------------------------------------------------------
# Single-job runner
# ---------------------------------------------------------------------------

def run_one(job: RenderJob, args: argparse.Namespace, config: dict) -> RenderResult:
    cmd = build_command(job, args, config)

    env = os.environ.copy()
    env["MUJOCO_GL"] = args.mujoco_gl
    env["PYTHONPATH"] = args.pythonpath
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    command_str = " ".join(cmd)

    if args.dry_run:
        Path(job.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(job.log_file).write_text("[DRY RUN]\n" + command_str + "\n", encoding="utf-8")
        return RenderResult(
            budget_name=job.budget_name,
            split=job.split,
            template_index=job.template_index,
            template_id=job.template_id,
            out_video=job.out_video,
            log_file=job.log_file,
            returncode=0,
            ok=True,
            command=command_str,
        )

    Path(job.log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(job.log_file, "w", encoding="utf-8") as f:
        f.write(f"COMMAND:\n{command_str}\n\n")
        f.flush()
        proc = subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            env=env, text=True, check=False,
        )

    return RenderResult(
        budget_name=job.budget_name,
        split=job.split,
        template_index=job.template_index,
        template_id=job.template_id,
        out_video=job.out_video,
        log_file=job.log_file,
        returncode=proc.returncode,
        ok=(proc.returncode == 0),
        command=command_str,
    )


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def make_jobs(
    args: argparse.Namespace,
    by_split: dict[str, list[dict]],
    layouts: list[str],
    template_indices: list[int],
    run_dir: Path,
) -> list[RenderJob]:
    videos_dir = run_dir / "videos"
    logs_dir = run_dir / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    budget_names = args.budgets if args.budgets else list(BUDGETS.keys())
    jobs: list[RenderJob] = []

    for budget_name in budget_names:
        for split in layouts:
            for idx in template_indices:
                template = by_split[split][idx]
                tid = template["reset_template_id"]
                short = split.replace("test_sim_layout_ood_", "")
                out_video = str(videos_dir / f"{budget_name}__{short}__idx{idx:02d}__{tid}.mp4")
                log_file = str(logs_dir / f"{budget_name}__{short}__idx{idx:02d}__{tid}.log")
                jobs.append(RenderJob(
                    budget_name=budget_name,
                    split=split,
                    template_index=idx,
                    template_id=tid,
                    out_video=out_video,
                    log_file=log_file,
                ))

    return jobs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_reports(
    run_dir: Path,
    args: argparse.Namespace,
    jobs: list[RenderJob],
    results: list[RenderResult],
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "templates": args.templates,
        "num_jobs": len(jobs),
        "num_workers": args.num_workers,
        "budgets": args.budgets if args.budgets else list(BUDGETS.keys()),
        "layouts": args.layouts or DEFAULT_LAYOUTS,
        "template_indices": args.template_indices_parsed,
        "results": [
            {
                "budget_name": r.budget_name,
                "split": r.split,
                "template_index": r.template_index,
                "template_id": r.template_id,
                "ok": r.ok,
                "returncode": r.returncode,
                "out_video": r.out_video,
                "log_file": r.log_file,
            }
            for r in results
        ],
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    lines = [
        f"run_dir: {run_dir}",
        f"num_jobs: {len(jobs)}",
        f"ok: {sum(r.ok for r in results)} / {len(results)}",
        "",
    ]
    for r in results:
        status = "OK" if r.ok else f"FAIL({r.returncode})"
        lines.append(f"  {status}  {r.budget_name}/{r.split} idx={r.template_index}  {r.template_id}")

    (run_dir / "compact_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render obstacle rollout videos in parallel via subprocess.",
    )
    parser.add_argument(
        "--templates",
        default="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json",
        help="Template JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/obstacle_videos",
        help="Output directory for videos and logs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=12,
        help="Parallel subprocess workers.",
    )
    parser.add_argument(
        "--layouts",
        nargs="*",
        default=None,
        help="Explicit split/layout list. Default: 6 obstacle layouts.",
    )
    parser.add_argument(
        "--template-indices",
        default="0,1",
        help="Comma-separated template indices per layout (e.g. '0,1' or '0').",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=None,
        help="Budget configs to run. Default: all (c23_strict600, c23_strict800).",
    )

    # Video settings
    parser.add_argument("--camera", default="topdown")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)

    # Env
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--pythonpath", default=".")

    parser.add_argument("--dry-run", action="store_true",
                        help="Write commands/logs but do not render.")
    return parser.parse_args()


def parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate budget names early
    budget_names = args.budgets if args.budgets else list(BUDGETS.keys())
    for bn in budget_names:
        if bn not in BUDGETS:
            print(f"ERROR: Unknown budget '{bn}'. Available: {list(BUDGETS.keys())}", file=sys.stderr)
            sys.exit(1)

    # Validate render script exists
    render_script = Path("scripts/render_closed_loop_rollout.py")
    if not render_script.exists():
        print(f"ERROR: Render script not found: {render_script}", file=sys.stderr)
        sys.exit(1)

    layouts = args.layouts or DEFAULT_LAYOUTS
    template_indices = parse_csv_ints(args.template_indices)
    args.template_indices_parsed = template_indices

    template_path = Path(args.templates)
    by_split = load_and_validate_templates(template_path, layouts, template_indices)

    # Print template info
    total_templates = sum(len(by_split[s]) for s in layouts)
    print(f"Templates: {template_path.name}")
    print(f"  Splits: {len(layouts)}, templates per split: {total_templates // len(layouts)}")
    print(f"  Indices: {template_indices}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"batch_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs(args, by_split, layouts, template_indices, run_dir)
    num_workers = min(args.num_workers, len(jobs))

    print(f"\n{'='*60}")
    print(f"Rendering {len(jobs)} videos with {num_workers} workers")
    print(f"Budgets: {budget_names}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    for j in jobs:
        print(f"  {j.budget_name}/{j.split} idx={j.template_index} -> {j.template_id}")

    if args.dry_run:
        print("\nDRY RUN: commands written to logs, videos will not be rendered.")

    # Run parallel subprocesses
    results: list[RenderResult] = []
    completed = 0

    with futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        future_to_job: dict[futures.Future, RenderJob] = {}
        for job in jobs:
            config = BUDGETS[job.budget_name]
            future = ex.submit(run_one, job, args, config)
            future_to_job[future] = job

        for fut in futures.as_completed(future_to_job):
            job = future_to_job[fut]
            completed += 1
            try:
                result = fut.result()
            except Exception as e:
                cmd = f"[exception before command built] {job.budget_name}/{job.split} idx={job.template_index}"
                result = RenderResult(
                    budget_name=job.budget_name,
                    split=job.split,
                    template_index=job.template_index,
                    template_id=job.template_id,
                    out_video=job.out_video,
                    log_file=job.log_file,
                    returncode=-999,
                    ok=False,
                    command=cmd,
                )
                Path(job.log_file).parent.mkdir(parents=True, exist_ok=True)
                Path(job.log_file).write_text(f"Exception:\n{repr(e)}\n", encoding="utf-8")

            results.append(result)
            status = "OK" if result.ok else f"FAIL({result.returncode})"
            print(f"  [{completed}/{len(jobs)}] {status}  {result.budget_name}/{result.split} idx={result.template_index}  {result.template_id}")

    results.sort(key=lambda r: (r.budget_name, r.split, r.template_index))
    write_reports(run_dir, args, jobs, results)

    ok_count = sum(r.ok for r in results)
    print(f"\n{'='*60}")
    print(f"Done: {ok_count}/{len(results)} succeeded")
    print(f"Summary: {run_dir / 'compact_summary.txt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
