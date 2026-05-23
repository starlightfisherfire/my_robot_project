#!/usr/bin/env python3
"""export_visual_state_v2.py — Export mppi_stage2c_state16 episodes to v2 format.

Usage:
    PYTHONPATH=. python scripts/export_visual_state_v2.py \
        --source data/sim/mppi_stage2c_state16 \
        --dest data/sim/visual_state_v2_smoke \
        --max-episodes 3 \
        --schema configs/state_schema/visual_structured_state_v2.yaml
"""

import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def export_episode(src_npz: str, v2_npz: str) -> bool:
    """Convert one state16 episode to v2 format."""
    data = np.load(src_npz, allow_pickle=True)
    states = data["states"]       # [T, 6, 16]
    obj_poses = data["object_poses"]
    ee_positions = data.get("ee_positions", np.zeros((len(states), 2), dtype=np.float32))
    goal_pose = data["goal_pose"]
    actions = data.get("actions_physical", np.zeros((len(states), 2), dtype=np.float32))
    next_obj = data.get("next_object_poses", obj_poses.copy())
    contact_flags = data.get("contact_flags", np.zeros(len(states), dtype=np.float32))
    obstacle_feats = data.get("obstacle_features", np.zeros(10, dtype=np.float32))

    T = len(states)

    # Build v2 payload with individual feature arrays
    payload = {
        # Object features
        "obj_x": states[:, 1, 0].copy(),
        "obj_y": states[:, 1, 1].copy(),
        "obj_sin_theta": states[:, 1, 2].copy(),
        "obj_cos_theta": states[:, 1, 3].copy(),
        "obj_vx": states[:, 1, 4].copy(),
        "obj_vy": states[:, 1, 5].copy(),
        "obj_omega": states[:, 1, 6].copy(),
        "obj_size_x": np.full(T, states[0, 1, 7] if states[0, 1, 7] != 0 else 0.048, dtype=np.float32),
        "obj_size_y": np.full(T, states[0, 1, 8] if states[0, 1, 8] != 0 else 0.048, dtype=np.float32),
        "obj_shape_T": np.ones(T, dtype=np.float32),
        "obj_shape_L": np.zeros(T, dtype=np.float32),
        
        # Proprio features
        "ee_x": states[:, 0, 0].copy(),
        "ee_y": states[:, 0, 1].copy(),
        "ee_vx": states[:, 0, 4].copy(),
        "ee_vy": states[:, 0, 5].copy(),
        "prev_action_dx": np.zeros(T, dtype=np.float32),
        "prev_action_dy": np.zeros(T, dtype=np.float32),
        
        # Goal features
        "goal_x": np.full(T, goal_pose[0], dtype=np.float32),
        "goal_y": np.full(T, goal_pose[1], dtype=np.float32),
        "goal_sin_theta": np.full(T, np.sin(goal_pose[2]), dtype=np.float32),
        "goal_cos_theta": np.full(T, np.cos(goal_pose[2]), dtype=np.float32),
        
        # Obstacle features
        "obs_x": np.full(T, obstacle_feats[0] if len(obstacle_feats) > 0 else 0.0, dtype=np.float32),
        "obs_y": np.full(T, obstacle_feats[1] if len(obstacle_feats) > 1 else 0.0, dtype=np.float32),
        "obs_size_x": np.full(T, obstacle_feats[3] if len(obstacle_feats) > 3 else 0.0, dtype=np.float32),
        "obs_size_y": np.full(T, obstacle_feats[4] if len(obstacle_feats) > 4 else 0.0, dtype=np.float32),
        
        # Relation features (derived)
        "ee_obj_dist": np.linalg.norm(states[:, 0, :2] - states[:, 1, :2], axis=1).astype(np.float32),
        "ee_obj_dx": (states[:, 0, 0] - states[:, 1, 0]).astype(np.float32),
        "ee_obj_dy": (states[:, 0, 1] - states[:, 1, 1]).astype(np.float32),
        "obj_goal_dist": np.linalg.norm(states[:, 1, :2] - goal_pose[:2], axis=1).astype(np.float32),
        "obj_goal_dx": (states[:, 1, 0] - goal_pose[0]).astype(np.float32),
        "obj_goal_dy": (states[:, 1, 1] - goal_pose[1]).astype(np.float32),
        "obj_obs1_dist": np.zeros(T, dtype=np.float32),
        "obj_obs1_dx": np.zeros(T, dtype=np.float32),
        "obj_obs1_dy": np.zeros(T, dtype=np.float32),
        "contact_proxy": contact_flags.astype(np.float32),
        "motion_alignment": np.zeros(T, dtype=np.float32),
        
        # Temporal features
        "action_h1_dx": np.zeros(T, dtype=np.float32),
        "action_h1_dy": np.zeros(T, dtype=np.float32),
        "obj_moved_h1": np.zeros(T, dtype=np.float32),
        "contact_h1": np.zeros(T, dtype=np.float32),
        "stuck_proxy": np.zeros(T, dtype=np.float32),
        
        # Nuisance placeholders
        "obj_color_r": np.zeros(T, dtype=np.float32),
        "obj_color_g": np.zeros(T, dtype=np.float32),
        "obj_color_b": np.zeros(T, dtype=np.float32),
        "light_pos_x": np.zeros(T, dtype=np.float32),
        "light_pos_y": np.zeros(T, dtype=np.float32),
        "ambient_light": np.ones(T, dtype=np.float32) * 0.3,
        "camera_fovy": np.ones(T, dtype=np.float32) * 50.0,
        
        # Privileged placeholders (default zero, not main input)
        "object_mass": np.ones(T, dtype=np.float32) * 0.038,
        "object_friction": np.ones(T, dtype=np.float32) * 0.8,
        "true_contact_flag": np.zeros(T, dtype=np.float32),
        "contact_force_x": np.zeros(T, dtype=np.float32),
        "contact_force_y": np.zeros(T, dtype=np.float32),
        "contact_mu": np.zeros(T, dtype=np.float32),
        
        # Core arrays
        "actions_physical": actions[:T],
        "object_poses": obj_poses[:T],
        "next_object_poses": next_obj[:T],
        "goal_pose": goal_pose.astype(np.float32),
        "schema_version": np.array(["0.1"], dtype=object),
    }

    np.savez(v2_npz, **payload)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--schema", default="configs/state_schema/visual_structured_state_v2.yaml")
    args = parser.parse_args()

    src_dir = Path(args.source)
    src_meta = src_dir / "metadata" / "episodes.jsonl"
    src_eps = src_dir / "episodes"
    
    with open(src_meta) as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    episodes = episodes[:args.max_episodes]

    dest_eps = Path(args.dest) / "episodes"
    dest_meta = Path(args.dest) / "metadata"
    dest_eps.mkdir(parents=True, exist_ok=True)
    dest_meta.mkdir(parents=True, exist_ok=True)

    exported = 0
    new_meta = []
    for ep in episodes:
        src_npz = src_eps / f"{ep['episode_id']}.npz"
        dest_npz = dest_eps / f"{ep['episode_id']}.npz"
        if export_episode(str(src_npz), str(dest_npz)):
            exported += 1
            new_meta.append(ep)
        print(f"  [{exported}] {ep['episode_id']}")

    with open(dest_meta / "episodes.jsonl", "w") as f:
        for ep in new_meta:
            f.write(json.dumps(ep) + "\n")

    print(f"Exported {exported} episodes to {args.dest}")

if __name__ == "__main__":
    main()
