#!/usr/bin/env python3
"""Render video from episode replay (qpos/qvel trace).

Usage:
    MUJOCO_GL=egl python scripts/render_episode_from_replay.py \
        --episode_dir runs/.../episodes/xxx --height 224 --width 224
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco
from src.envs.mujoco_push_env import MujocoPushEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ep_dir = Path(args.episode_dir)

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

    # Setup env
    env = MujocoPushEnv()
    env.goal_pose = goal_pose.copy()
    env._sync_goal_visuals()

    # Setup renderer
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    try:
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown")
    except:
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
    out_path = Path(args.output) if args.output else ep_dir / "top_rgb_224.mp4"
    try:
        import imageio
        imageio.mimsave(str(out_path), frames, fps=10)
        print(f"Saved: {out_path}")
    except ImportError:
        # Save as npy fallback
        np_path = ep_dir / "top_rgb_224.npy"
        np.save(np_path, np.array(frames))
        print(f"imageio not available, saved npy: {np_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
