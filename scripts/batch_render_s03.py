#!/usr/bin/env python3
"""Batch render all s03 warmstart ablation episodes."""
import subprocess, sys, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

EP_DIR = Path("runs/warmstart_s03/episodes")
RENDER_SCRIPT = Path("scripts/render_episode_from_replay.py")
PY = "/home/brucewu/miniconda3/envs/lerobot/bin/python"

def render_one(ep_dir):
    out_file = ep_dir / "top_rgb_224.mp4"
    if out_file.exists():
        return {"dir": ep_dir.name, "status": "skipped"}
    cmd = [PY, str(RENDER_SCRIPT), "--episode_dir", str(ep_dir), "--height", "224", "--width", "224"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           env={**os.environ, "MUJOCO_GL": "egl"})
        elapsed = time.time() - t0
        if r.returncode == 0:
            size_mb = out_file.stat().st_size / 1e6 if out_file.exists() else 0
            return {"dir": ep_dir.name, "status": "ok", "elapsed": elapsed, "size_mb": size_mb}
        return {"dir": ep_dir.name, "status": "error", "stderr": r.stderr[-200:], "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        return {"dir": ep_dir.name, "status": "timeout", "elapsed": time.time() - t0}
    except Exception as e:
        return {"dir": ep_dir.name, "status": "error", "error": str(e), "elapsed": time.time() - t0}

def main():
    ep_dirs = sorted([d for d in EP_DIR.iterdir() if d.is_dir()])
    print(f"=== Render s03 Ablation: {len(ep_dirs)} episodes ===\n")
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=26) as ex:
        futs = {ex.submit(render_one, d): d for d in ep_dirs}
        done = 0
        for f in as_completed(futs):
            done += 1
            r = f.result()
            icon = "✅" if r["status"] == "ok" else ("⏭️" if r["status"] == "skipped" else "❌")
            print(f"  [{done:2d}/{len(ep_dirs)}] {icon} {r['status']:8s} {r.get('elapsed',0):5.1f}s {r.get('size_mb',0):5.1f}MB  {r['dir']}", flush=True)
            results.append(r)
    total = time.time() - t0
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] not in ("ok", "skipped"))
    videos = sorted(EP_DIR.glob("*/top_rgb_224.mp4"))
    total_mb = sum(v.stat().st_size for v in videos) / 1e6
    print(f"\n=== Done: {ok} rendered, {skipped} skipped, {failed} failed in {total:.0f}s ===")
    print(f"{len(videos)} videos, total {total_mb:.1f} MB")

if __name__ == "__main__":
    main()
