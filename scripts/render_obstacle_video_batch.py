#!/usr/bin/env python3
"""
Render a fixed batch of obstacle rollout videos in parallel.

Default batch:
  - 6 single-obstacle templates:
      blocking_easy    index 0,1
      blocking_medium  index 0,1
      blocking_hard    index 0,1

  - 6 double-obstacle templates:
      passage_direct_wide       index 0,1
      passage_direct_medium     index 0,1
      passage_direct_narrow   index 0,1

This script only orchestrates calls to scripts/render_closed_loop_rollout.py.
It does not modify source files or templates.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
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


@dataclass(frozen=True)
class RenderJob:
    split: str
    template_index: int
    out_video: str
    log_file: str


@dataclass
class RenderResult:
    split: str
    template_index: int
    out_video: str
    log_file: str
    returncode: int
    ok: bool
    command: str


def parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def make_jobs(args: argparse.Namespace, run_dir: Path) -> list[RenderJob]:
    videos_dir = run_dir / "videos"
    logs_dir = run_dir / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    layouts = args.layouts or DEFAULT_LAYOUTS
    indices = parse_csv_ints(args.template_indices)

    jobs: list[RenderJob] = []
    for split in layouts:
        for idx in indices:
            short = split.replace("test_sim_layout_ood_", "")
            out_video = videos_dir / f"c23_strict800_{short}_idx{idx:02d}.mp4"
            log_file = logs_dir / f"c23_strict800_{short}_idx{idx:02d}.log"
            jobs.append(
                RenderJob(
                    split=split,
                    template_index=idx,
                    out_video=str(out_video),
                    log_file=str(log_file),
                )
            )
    return jobs


def build_command(job: RenderJob, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/render_closed_loop_rollout.py",
        "--templates",
        args.templates,
        "--split",
        job.split,
        "--template-index",
        str(job.template_index),
        "--camera",
        args.camera,
        "--horizon",
        str(args.horizon),
        "--execute-steps",
        str(args.execute_steps),
        "--max-mpc-steps",
        str(args.max_mpc_steps),
        "--num-samples",
        str(args.num_samples),
        "--num-elites",
        str(args.num_elites),
        "--num-iterations",
        str(args.num_iterations),
        "--strict-pose-stop",
        "--stop-pos-threshold",
        str(args.stop_pos_threshold),
        "--stop-theta-threshold-deg",
        str(args.stop_theta_threshold_deg),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--out-video",
        job.out_video,
    ]


def run_one(job: RenderJob, args: argparse.Namespace) -> RenderResult:
    cmd = build_command(job, args)

    env = os.environ.copy()
    env["MUJOCO_GL"] = args.mujoco_gl
    env["PYTHONPATH"] = args.pythonpath

    # Prevent 12 processes from each spawning many BLAS/OpenMP threads.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    command_str = " ".join(cmd)

    if args.dry_run:
        Path(job.log_file).write_text(
            "[DRY RUN]\n" + command_str + "\n",
            encoding="utf-8",
        )
        return RenderResult(
            split=job.split,
            template_index=job.template_index,
            out_video=job.out_video,
            log_file=job.log_file,
            returncode=0,
            ok=True,
            command=command_str,
        )

    with open(job.log_file, "w", encoding="utf-8") as f:
        f.write(f"COMMAND:\n{command_str}\n\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )

    return RenderResult(
        split=job.split,
        template_index=job.template_index,
        out_video=job.out_video,
        log_file=job.log_file,
        returncode=proc.returncode,
        ok=(proc.returncode == 0),
        command=command_str,
    )


def write_reports(run_dir: Path, args: argparse.Namespace, jobs: list[RenderJob], results: list[RenderResult]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "templates": args.templates,
        "jobs": args.jobs,
        "num_jobs": len(jobs),
        "config": {
            "horizon": args.horizon,
            "execute_steps": args.execute_steps,
            "max_mpc_steps": args.max_mpc_steps,
            "num_samples": args.num_samples,
            "num_elites": args.num_elites,
            "num_iterations": args.num_iterations,
            "strict_pose_stop": True,
            "stop_pos_threshold": args.stop_pos_threshold,
            "stop_theta_threshold_deg": args.stop_theta_threshold_deg,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "camera": args.camera,
        },
        "layouts": args.layouts or DEFAULT_LAYOUTS,
        "template_indices": parse_csv_ints(args.template_indices),
        "results": [asdict(r) for r in results],
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with open(run_dir / "render_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "template_index",
                "ok",
                "returncode",
                "out_video",
                "log_file",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "split": r.split,
                    "template_index": r.template_index,
                    "ok": r.ok,
                    "returncode": r.returncode,
                    "out_video": r.out_video,
                    "log_file": r.log_file,
                }
            )

    lines = []
    lines.append(f"run_dir: {run_dir}")
    lines.append(f"num_jobs: {len(jobs)}")
    lines.append(f"ok: {sum(r.ok for r in results)} / {len(results)}")
    lines.append("")
    for r in results:
        status = "OK" if r.ok else f"FAIL({r.returncode})"
        lines.append(f"{status}  {r.split} idx={r.template_index}  {r.out_video}")
    (run_dir / "compact_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 12 obstacle rollout videos in parallel.",
    )
    parser.add_argument(
        "--templates",
        default="data/sim/metadata/reset_templates_obstacle_difficulty_v0.json",
        help="Template JSON file.",
    )
    parser.add_argument(
        "--output-root",
        default="runs/obstacle_video_batches",
        help="Output root directory.",
    )
    parser.add_argument(
        "--run-name",
        default="c23_strict800_obstacle_12videos",
        help="Run name prefix.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=12,
        help="Parallel render jobs.",
    )
    parser.add_argument(
        "--layouts",
        nargs="*",
        default=None,
        help="Optional explicit split/layout list.",
    )
    parser.add_argument(
        "--template-indices",
        default="0,1",
        help="Comma-separated template indices per layout. Default: 0,1.",
    )

    # C23 strict800 reference config
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--execute-steps", type=int, default=20)
    parser.add_argument("--max-mpc-steps", type=int, default=40)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--num-elites", type=int, default=96)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--stop-pos-threshold", type=float, default=0.0015)
    parser.add_argument("--stop-theta-threshold-deg", type=float, default=3.0)

    # Video settings
    parser.add_argument("--camera", default="topdown")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)

    # Env
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--pythonpath", default=".")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write commands/logs but do not render.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = make_jobs(args, run_dir)

    print(f"Run dir: {run_dir}")
    print(f"Num render jobs: {len(jobs)}")
    print(f"Parallel workers: {args.jobs}")
    print(f"Templates: {args.templates}")
    print("")

    for j in jobs:
        print(f"  {j.split} idx={j.template_index} -> {j.out_video}")

    if args.dry_run:
        print("\nDRY RUN: commands will be written but videos will not be rendered.")

    results: list[RenderResult] = []
    with futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        future_to_job = {ex.submit(run_one, job, args): job for job in jobs}
        for fut in futures.as_completed(future_to_job):
            job = future_to_job[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = RenderResult(
                    split=job.split,
                    template_index=job.template_index,
                    out_video=job.out_video,
                    log_file=job.log_file,
                    returncode=-999,
                    ok=False,
                    command="",
                )
                Path(job.log_file).write_text(f"Exception:\n{repr(e)}\n", encoding="utf-8")

            results.append(result)
            status = "OK" if result.ok else f"FAIL({result.returncode})"
            print(f"[{status}] {result.split} idx={result.template_index}")

    results.sort(key=lambda r: (r.split, r.template_index))
    write_reports(run_dir, args, jobs, results)

    print("")
    print(f"Done. Summary: {run_dir / 'compact_summary.txt'}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"CSV: {run_dir / 'render_summary.csv'}")


if __name__ == "__main__":
    main()
