#!/usr/bin/env python3
"""Render 1080p video from episode replay (qpos/qvel trace).

Usage:
    MUJOCO_GL=egl python scripts/render_1080p.py \
        --episode_dir runs/.../episodes/xxx

Renders both 1080p (1920x1080) and 224x224 videos.
Uses EGL for GPU rendering (minimal GPU memory, ~50MB per process).
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco
from src.envs.mujoco_push_env import MujocoPushEnv


def render_episode(ep_dir: Path, height: int, width: int, suffix: str):
    """Render a single episode to MP4."""
    out_name = f"top_rgb_{suffix}.mp4"
    out_path = ep_dir / out_name

    if out_path.exists():
        print(f"  SKIP {out_name} (exists)")
        return True

    # Load metadata
    with open(ep_dir / "metadata.json") as f:
        meta = json.load(f)

    # Load replay
    replay = np.load(ep_dir / "replay.npz")
    qpos_trace = replay["qpos"]
    qvel_trace = replay["qvel"]

    # Load episode for template info
    episode = np.load(ep_dir / "episode.npz", allow_pickle=True)
    goal_pose = episode["goal_pose"]
    obs_pos = episode["obstacle_positions"]
    obs_rad = episode["obstacle_radii"]

    # Setup env with obstacles
    env = MujocoPushEnv()

    # Rebuild model with obstacles if present
    if len(obs_pos) > 0 and len(obs_pos[0]) > 0:
        obstacles = []
        for i in range(len(obs_pos)):
            r = float(obs_rad[i])
            obstacles.append({
                "pose": {"x": float(obs_pos[i][0]), "y": float(obs_pos[i][1])},
                "size_x": r * 2, "size_y": r * 2,
            })
        template_stub = {
            "object_initial_pose": {"x": 0, "y": 0},
            "goal_pose": {"x": float(goal_pose[0]), "y": float(goal_pose[1])},
            "ee_initial_pose": {"x": 0.10, "y": 0.18},
            "obstacles": obstacles,
        }
        env.reset_from_template(template_stub)

    env.goal_pose = goal_pose.copy()
    env._sync_goal_visuals()

    # Increase offscreen buffer if needed
    vis_global = env.model.vis.global_
    if width > vis_global.offwidth:
        vis_global.offwidth = width
    if height > vis_global.offheight:
        vis_global.offheight = height

    # Setup renderer
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    try:
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown")
    except Exception:
        cam_id = -1

    # Render frames
    frames = []
    for i in range(len(qpos_trace)):
        env.data.qpos[:] = qpos_trace[i]
        env.data.qvel[:] = qvel_trace[i]
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, camera=cam_id)
        frames.append(renderer.render())

    renderer.close()

    # Save as mp4
    try:
        import imageio
        imageio.mimsave(str(out_path), frames, fps=10)
        print(f"  OK {out_name} ({height}x{width}, {len(frames)} frames)")
        return True
    except ImportError:
        np_path = ep_dir / f"top_rgb_{suffix}.npy"
        np.save(np_path, np.array(frames))
        print(f"  imageio not available, saved npy: {np_path}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, required=True)
    parser.add_argument("--render_1080p", action="store_true", default=True)
    parser.add_argument("--render_224", action="store_true", default=True)
    args = parser.parse_args()

    ep_dir = Path(args.episode_dir)
    ep_name = ep_dir.name

    if not (ep_dir / "replay.npz").exists():
        print(f"SKIP {ep_name}: no replay.npz")
        return 1

    if not (ep_dir / "metadata.json").exists():
        print(f"SKIP {ep_name}: no metadata.json")
        return 1

    ok = True
    if args.render_1080p:
        ok &= render_episode(ep_dir, 1080, 1920, "1080p")
    if args.render_224:
        ok &= render_episode(ep_dir, 224, 224, "224")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
