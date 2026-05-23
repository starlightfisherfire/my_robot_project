#!/usr/bin/env python3
"""
convert_mppi_to_state16.py — Convert mppi_stage2c data to canonical_state16 format.

Reads mppi_stage2c npz files (compact 5-dim state) and rebuilds
canonical_state16 [T, 6, 16] from raw fields.

Output: data/sim/mppi_stage2c_state16/ with same structure.

Usage:
    PYTHONPATH=. python scripts/convert_mppi_to_state16.py
    PYTHONPATH=. python scripts/convert_mppi_to_state16.py --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Token indices in canonical_state16 [T, N, D]
IDX_EE = 0
IDX_OBJ = 1
IDX_GOAL = 2
IDX_OBS_BASE = 3  # obstacles at 3, 4, 5

# Feature indices in D=16
FEAT_X = 0
FEAT_Y = 1
FEAT_SIN_THETA = 2
FEAT_COS_THETA = 3
FEAT_VX = 4
FEAT_VY = 5
FEAT_OMEGA = 6
FEAT_SIZE_X = 7
FEAT_SIZE_Y = 8
FEAT_SHAPE_T = 9
FEAT_SHAPE_L = 10
FEAT_SHAPE_OTHER = 11
FEAT_MASS = 12
FEAT_FRICTION = 13
FEAT_CONTACT = 14
FEAT_VALID = 15

# Default physical properties for T-shape object
OBJ_SIZE_X = 0.048
OBJ_SIZE_Y = 0.048
OBJ_MASS = 0.038
OBJ_FRICTION = 0.8

# Default EE properties (from env)
EE_SIZE_X = 0.015
EE_SIZE_Y = 0.015


def convert_episode(npz_path: str) -> dict:
    """Convert one episode from compact to canonical_state16.

    Returns dict with new npz data arrays, or None if conversion fails.
    """
    data = np.load(npz_path, allow_pickle=True)

    # Check required fields
    required = ["object_poses", "ee_positions", "goal_pose",
                 "actions_physical", "actions_norm"]
    for k in required:
        if k not in data:
            print(f"  SKIP: missing {k}")
            return None

    object_poses = data["object_poses"]      # [T, 3]
    ee_positions = data["ee_positions"]       # [T, 2]
    goal_pose = data["goal_pose"]             # [3]
    actions_norm = data["actions_norm"]       # [T, 2]
    actions_physical = data["actions_physical"]  # [T, 2]

    T = len(object_poses)

    # Optional fields
    if "actual_object_velocities" in data:
        obj_vel = data["actual_object_velocities"]  # [T, 2]
    else:
        obj_vel = np.zeros((T, 2), dtype=np.float32)

    if "actual_ee_velocities" in data:
        ee_vel = data["actual_ee_velocities"]  # [T, 2]
    else:
        ee_vel = np.zeros((T, 2), dtype=np.float32)

    contact_flags = data.get("contact_flags", np.zeros(T, dtype=bool))
    collision_flags = data.get("collision_flags", np.zeros(T, dtype=bool))

    # Obstacle features: [10] = obs0(x,y,θ,sx,sy,unused) + obs1(...) + obs2(,unused)
    obstacle_features = data.get("obstacle_features", np.zeros(10, dtype=np.float32))

    # Build canonical_state16 [T, 6, 16]
    states = np.zeros((T, 6, 16), dtype=np.float32)

    # --- Token 0: End-Effector ---
    states[:, IDX_EE, FEAT_X] = ee_positions[:, 0]
    states[:, IDX_EE, FEAT_Y] = ee_positions[:, 1]
    states[:, IDX_EE, FEAT_SIN_THETA] = 0.0  # EE has no orientation
    states[:, IDX_EE, FEAT_COS_THETA] = 1.0
    states[:, IDX_EE, FEAT_VX] = ee_vel[:, 0]
    states[:, IDX_EE, FEAT_VY] = ee_vel[:, 1]
    states[:, IDX_EE, FEAT_OMEGA] = 0.0
    states[:, IDX_EE, FEAT_SIZE_X] = EE_SIZE_X
    states[:, IDX_EE, FEAT_SIZE_Y] = EE_SIZE_Y
    states[:, IDX_EE, FEAT_CONTACT] = contact_flags.astype(np.float32)
    states[:, IDX_EE, FEAT_VALID] = 1.0

    # --- Token 1: Manipulated Object ---
    states[:, IDX_OBJ, FEAT_X] = object_poses[:, 0]
    states[:, IDX_OBJ, FEAT_Y] = object_poses[:, 1]
    states[:, IDX_OBJ, FEAT_SIN_THETA] = np.sin(object_poses[:, 2])
    states[:, IDX_OBJ, FEAT_COS_THETA] = np.cos(object_poses[:, 2])
    states[:, IDX_OBJ, FEAT_VX] = obj_vel[:, 0]
    states[:, IDX_OBJ, FEAT_VY] = obj_vel[:, 1]
    # omega from finite difference of theta (more accurate than 0)
    if T > 1:
        dtheta = np.diff(object_poses[:, 2])
        # Wrap to [-pi, pi]
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        omega = np.zeros(T, dtype=np.float32)
        omega[1:] = dtheta / 0.1  # control_dt = 0.1
        states[:, IDX_OBJ, FEAT_OMEGA] = omega
    states[:, IDX_OBJ, FEAT_SIZE_X] = OBJ_SIZE_X
    states[:, IDX_OBJ, FEAT_SIZE_Y] = OBJ_SIZE_Y
    states[:, IDX_OBJ, FEAT_SHAPE_T] = 1.0  # T-shape
    states[:, IDX_OBJ, FEAT_MASS] = OBJ_MASS
    states[:, IDX_OBJ, FEAT_FRICTION] = OBJ_FRICTION
    states[:, IDX_OBJ, FEAT_VALID] = 1.0

    # --- Token 2: Goal ---
    states[:, IDX_GOAL, FEAT_X] = goal_pose[0]
    states[:, IDX_GOAL, FEAT_Y] = goal_pose[1]
    states[:, IDX_GOAL, FEAT_SIN_THETA] = np.sin(goal_pose[2])
    states[:, IDX_GOAL, FEAT_COS_THETA] = np.cos(goal_pose[2])
    states[:, IDX_GOAL, FEAT_SIZE_X] = OBJ_SIZE_X
    states[:, IDX_GOAL, FEAT_SIZE_Y] = OBJ_SIZE_Y
    states[:, IDX_GOAL, FEAT_SHAPE_T] = 1.0
    states[:, IDX_GOAL, FEAT_MASS] = OBJ_MASS
    states[:, IDX_GOAL, FEAT_FRICTION] = OBJ_FRICTION
    states[:, IDX_GOAL, FEAT_VALID] = 1.0

    # --- Tokens 3-5: Obstacles ---
    for oi in range(3):
        base = oi * 3  # obstacle_features: [obs0_x, obs0_y, obs0_unused, obs0_sx, obs0_sy, ...]
        # Actually obstacle_features layout varies, let's check
        # From the data: [0.349, 0.269, 0.373, 0.06, 0.12, 0, 0, 0, 0, 0]
        # This looks like: obs0(x, y, θ_or_size, sx, sy), then zeros for unused obstacles
        idx = IDX_OBS_BASE + oi
        if oi == 0 and len(obstacle_features) >= 5:
            ox = obstacle_features[0]
            oy = obstacle_features[1]
            osx = obstacle_features[3]
            osy = obstacle_features[4]
            if osx > 0 and osy > 0:  # valid obstacle
                states[:, idx, FEAT_X] = ox
                states[:, idx, FEAT_Y] = oy
                states[:, idx, FEAT_COS_THETA] = 1.0
                states[:, idx, FEAT_SIZE_X] = osx
                states[:, idx, FEAT_SIZE_Y] = osy
                states[:, idx, FEAT_MASS] = 0.5
                states[:, idx, FEAT_FRICTION] = 0.8
                states[:, idx, FEAT_VALID] = 1.0

    # Build next_states (shifted by 1)
    next_states = np.zeros_like(states)
    if T > 1:
        next_states[:-1] = states[1:]
        next_states[-1] = states[-1]  # repeat last

    # Build next_object_poses
    next_object_poses = np.zeros_like(object_poses)
    if T > 1:
        next_object_poses[:-1] = object_poses[1:]
        next_object_poses[-1] = object_poses[-1]

    # Build next_ee_positions
    next_ee_positions = np.zeros_like(ee_positions)
    if T > 1:
        next_ee_positions[:-1] = ee_positions[1:]
        next_ee_positions[-1] = ee_positions[-1]

    return {
        "states": states,
        "next_states": next_states,
        "actions_norm": actions_norm,
        "actions_physical": actions_physical,
        "object_poses": object_poses,
        "next_object_poses": next_object_poses,
        "ee_positions": ee_positions,
        "next_ee_positions": next_ee_positions,
        "contact_flags": contact_flags,
        "collision_flags": collision_flags,
        "goal_pose": goal_pose,
        "obstacle_features": obstacle_features,
        # Extra fields from original
        "actual_object_velocities": obj_vel,
        "actual_ee_velocities": ee_vel,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", default="data/sim/mppi_stage2c",
                        help="Source mppi dataset")
    parser.add_argument("--dest", default="data/sim/mppi_stage2c_state16",
                        help="Destination state16 dataset")
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    src_meta = src / "metadata" / "episodes.jsonl"
    src_eps = src / "episodes"

    if not src_meta.exists():
        print(f"Source metadata not found: {src_meta}")
        sys.exit(1)

    with open(src_meta) as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    print(f"Source: {src}")
    print(f"  Episodes: {len(episodes)}")

    if args.dry_run:
        # Just check one episode
        ep = episodes[0]
        npz_path = str(src_eps / f"{ep['episode_id']}.npz")
        result = convert_episode(npz_path)
        if result:
            print(f"  Dry run OK: states shape = {result['states'].shape}")
            print(f"  Token 0 (EE): x={result['states'][0, 0, 0]:.4f}, y={result['states'][0, 0, 1]:.4f}")
            print(f"  Token 1 (Obj): x={result['states'][0, 1, 0]:.4f}, y={result['states'][0, 1, 1]:.4f}")
            print(f"  Token 2 (Goal): x={result['states'][0, 2, 0]:.4f}, y={result['states'][0, 2, 1]:.4f}")
        return

    # Create destination
    dst.mkdir(parents=True, exist_ok=True)
    dst_eps = dst / "episodes"
    dst_meta = dst / "metadata"
    dst_eps.mkdir(exist_ok=True)
    dst_meta.mkdir(exist_ok=True)

    converted = 0
    failed = 0
    new_episodes = []

    for i, ep in enumerate(episodes):
        ep_id = ep["episode_id"]
        src_npz = src_eps / f"{ep_id}.npz"

        if not src_npz.exists():
            print(f"  [{i+1}] {ep_id}: NPZ not found, skip")
            failed += 1
            continue

        result = convert_episode(str(src_npz))
        if result is None:
            print(f"  [{i+1}] {ep_id}: conversion failed")
            failed += 1
            continue

        # Save converted npz
        dst_npz = dst_eps / f"{ep_id}.npz"
        np.savez(dst_npz, **result)

        # Update metadata
        new_ep = ep.copy()
        new_ep["state_format"] = "canonical_state16"
        new_ep["state_shape"] = list(result["states"].shape)
        new_ep["source_dataset"] = "mppi_stage2c"
        new_episodes.append(new_ep)

        converted += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(episodes)}] converted {converted}, failed {failed}")

    # Write metadata
    with open(dst_meta / "episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    # Copy any other metadata files
    for f in (src / "metadata").iterdir():
        if f.name != "episodes.jsonl" and f.is_file():
            shutil.copy2(f, dst_meta / f.name)

    print(f"\nDone:")
    print(f"  Converted: {converted}")
    print(f"  Failed: {failed}")
    print(f"  Output: {dst}")


if __name__ == "__main__":
    main()
