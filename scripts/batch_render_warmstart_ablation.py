#!/usr/bin/env python3
"""
Batch render all warmstart ablation episodes to videos.
Parallel with configurable workers (default 26 CPU cores).

Usage:
    MUJOCO_GL=egl python scripts/batch_render_warmstart_ablation.py --workers 26
"""
import subprocess, sys, os, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

EP_DIR = Path("runs/warmstart_ablation/episodes")
RENDER_SCRIPT = Path("scripts/render_episode_from_replay.py")


def render_one(ep_dir: Path, height: int, width: int) -> dict:
    """Render a single episode directory."""
    out_file = ep_dir / "top_rgb_224.mp4"
    if out_file.exists():
        return {"dir": ep_dir.name, "status": "skipped", "reason": "exists"}

    # Use lerobot env python which has mujoco
    py = "/home/brucewu/miniconda3/envs/lerobot/bin/python"
    cmd = [
        py, str(RENDER_SCRIPT),
        "--episode_dir", str(ep_dir),
        "--height", str(height),
        "--width", str(width),
    ]
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        elapsed = time.time() - t0
        if r.returncode == 0:
            size_mb = out_file.stat().st_size / 1e6 if out_file.exists() else 0
            return {"dir": ep_dir.name, "status": "ok", "elapsed": elapsed, "size_mb": size_mb}
        else:
            return {"dir": ep_dir.name, "status": "error", "returncode": r.returncode,
                    "stderr": r.stderr[-300:], "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        return {"dir": ep_dir.name, "status": "timeout", "elapsed": time.time() - t0}
    except Exception as e:
        return {"dir": ep_dir.name, "status": "error", "error": str(e), "elapsed": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=26)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()

    ep_dirs = sorted([d for d in EP_DIR.iterdir() if d.is_dir()])
    print(f"=== Batch Render Warmstart Ablation ===")
    print(f"  Episodes: {len(ep_dirs)}")
    print(f"  Workers:  {args.workers}")
    print(f"  Size:     {args.height}x{args.width}")
    print()

    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(render_one, d, args.height, args.width): d for d in ep_dirs}
        done = 0
        for f in as_completed(futs):
            done += 1
            r = f.result()
            status = r["status"]
            elapsed = r.get("elapsed", 0)
            size = r.get("size_mb", 0)
            icon = "✅" if status == "ok" else ("⏭️" if status == "skipped" else "❌")
            print(f"  [{done:2d}/{len(ep_dirs)}] {icon} {status:8s} {elapsed:5.1f}s {size:5.1f}MB  {r['dir']}", flush=True)
            results.append(r)

    total = time.time() - t0
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] not in ("ok", "skipped"))

    print(f"\n=== Done: {ok} rendered, {skipped} skipped, {failed} failed in {total:.0f}s ===")

    # List output files
    videos = sorted(EP_DIR.glob("*/top_rgb_224.mp4"))
    if videos:
        total_mb = sum(v.stat().st_size for v in videos) / 1e6
        print(f"\n{len(videos)} videos, total {total_mb:.1f} MB")
        print(f"Dir: {EP_DIR.absolute()}")


if __name__ == "__main__":
    main()
