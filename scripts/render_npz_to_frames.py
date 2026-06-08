#!/usr/bin/env python3
"""
Render a single npz episode to 224×224 PNG frames using MuJoCo.

Usage:
  PYTHONPATH=. python scripts/render_npz_to_frames.py \
    --npz data/sim/mppi_stage2c/episodes/d9c0322d-20b.npz \
    --templates data/sim/metadata/reset_templates_v0.json \
    --out runs/lewn_preview/d9c0322d-20b
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.envs.mujoco_push_env import MujocoPushEnv


def find_template(templates: list[dict], episode_meta: dict) -> dict | None:
    """Match an episode to its reset template.

    The episode metadata uses template_id like:
      test_sim_layout_ood_passage_direct_narrow__passage_direct_narrow__T_shape__000007
    but the reset templates JSON uses:
      test_sim_layout_ood_narrow_passage__narrow_passage__T_shape__000007

    Strategy: match by (split, template_index) and family.
    """
    ep_split = episode_meta.get("split", "")
    ep_idx = episode_meta.get("template_index", 0)
    ep_family = episode_meta.get("family", "")

    candidates = []
    for t in templates:
        t_split = t.get("split", "")
        t_idx = t.get("template_index", 0)

        # Normalize split names for matching
        if ep_split.replace("passage_direct_", "passage_") == t_split.replace(
            "passage_", "passage_"
        ):
            # Now check if they share the same family type
            if (
                "narrow" in ep_split.lower()
                and "narrow" in t_split.lower()
                and ep_idx == t_idx
            ):
                candidates.append(t)

    if candidates:
        return candidates[0]

    # Fallback: match any template with same family
    for t in templates:
        t_family = t.get("layout_family", "")
        if (
            ep_family.replace("passage_direct_", "passage_")
            == t_family.replace("passage_", "passage_")
        ):
            return t

    return None


def _npz_obstacles_to_dict(obstacle_features: np.ndarray) -> list[dict]:
    """Convert npz obstacle_features to obstacle dicts for MuJoCo env.

    obstacle_features: (10,) = [obs1_x, obs1_y, obs1_θ, obs1_w, obs1_h,
                                obs2_x, obs2_y, obs2_θ, obs2_w, obs2_h]
    """
    obstacles = []
    for slot in range(2):
        base = slot * 5
        x = float(obstacle_features[base])
        y = float(obstacle_features[base + 1])
        theta = float(obstacle_features[base + 2])
        size_x = float(obstacle_features[base + 3])
        size_y = float(obstacle_features[base + 4])
        # Skip obstacles with zero size (inactive)
        if size_x <= 0 or size_y <= 0:
            continue
        obstacles.append({
            "obstacle_id": f"obs_{slot}",
            "pose": {"x": x, "y": y, "theta": theta},
            "size_x": size_x,
            "size_y": size_y,
            "shape": "box",
            "valid": True,
        })
    return obstacles


def render_episode(
    npz_path: str,
    template: dict | None,
    out_dir: Path,
    width: int = 224,
    height: int = 224,
    camera: str = "topdown",
    camera_angles: str = "topdown",  # "topdown", "tilted", "both"
) -> list[Path]:
    """Render all frames of one episode from npz data."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load npz
    data = np.load(npz_path, allow_pickle=True)
    object_poses = data["object_poses"]  # (T, 3): [x, y, θ]
    ee_positions = data["ee_positions"]  # (T, 2): [x, y]
    actions = data["actions_norm"]  # (T, 2): [dx, dy]
    obstacle_features = data["obstacle_features"]  # (10,)
    T = len(object_poses)

    # Build obstacles directly from npz data (bypasses template matching)
    obstacles = _npz_obstacles_to_dict(obstacle_features)

    # Create env with correct obstacles
    env = MujocoPushEnv(
        max_speed_mps=0.3,
        pusher_mass=0.300,
    )
    env._rebuild_model_with_obstacles(obstacles)
    env._current_obstacle_signature = None  # avoid rebuild on reset
    env.reset()  # basic reset to init

    # Set correct goal pose from npz
    if "goal_pose" in data:
        goal_pose = data["goal_pose"]
        env.goal_pose = np.array([float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])], dtype=np.float64)
        env._sync_goal_visuals()  # update green marker position
    
    # Make goal marker fully opaque (default alpha=0.25 is too transparent)
    for i in range(env.model.ngeom):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and 'goal' in name.lower():
            rgba = env.model.geom_rgba[i].copy()
            rgba[3] = 0.8  # set alpha to 80%
            env.model.geom_rgba[i] = rgba

    # Create offscreen renderer
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    # Look up camera
    cam_id = -1
    for i in range(env.model.ncam):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name == camera:
            cam_id = i
            break

    # Build full frame sequence: include next_state of last step as final frame
    all_object_poses = list(object_poses)  # list of (3,) arrays
    all_ee_positions = list(ee_positions)
    if "next_object_poses" in data and "next_ee_positions" in data:
        all_object_poses.append(data["next_object_poses"][-1])
        all_ee_positions.append(data["next_ee_positions"][-1])

    frame_paths = []
    for i in range(len(all_object_poses)):
        obj_pose = all_object_poses[i]
        ee_pos = all_ee_positions[i]

        obj_x, obj_y, obj_theta = float(obj_pose[0]), float(obj_pose[1]), float(obj_pose[2])
        ee_x, ee_y = float(ee_pos[0]), float(ee_pos[1])

        # Set object pose: free joint qpos = [x, y, z, qw, qx, qy, qz]
        z = 0.006
        qw = float(np.cos(obj_theta / 2.0))
        qz = float(np.sin(obj_theta / 2.0))
        env.data.qpos[env.object_qpos_adr : env.object_qpos_adr + 7] = np.array(
            [obj_x, obj_y, z, qw, 0.0, 0.0, qz], dtype=np.float64
        )
        env.data.qvel[env.object_qvel_adr : env.object_qvel_adr + 6] = 0.0

        # Set pusher (ee) slide joints
        env.data.qpos[env.pusher_x_qpos_adr] = ee_x
        env.data.qpos[env.pusher_y_qpos_adr] = ee_y
        env.data.qvel[env.pusher_x_qvel_adr] = 0.0
        env.data.qvel[env.pusher_y_qvel_adr] = 0.0

        # Forward kinematics
        mujoco.mj_forward(env.model, env.data)

        # Render
        renderer.update_scene(env.data, camera=cam_id)
        pixels = renderer.render()  # [H, W, 3] uint8

        # Save
        frame_path = out_dir / f"frame_{i:04d}.png"
        img = Image.fromarray(pixels)
        img.save(frame_path)
        frame_paths.append(frame_path)

        if i == 0:
            print(f"  Step 0: obj=({obj_x:.4f},{obj_y:.4f},θ={obj_theta:.4f})  ee=({ee_x:.4f},{ee_y:.4f})")
        if i == T - 1:
            print(f"  Step {T-1}: obj=({obj_x:.4f},{obj_y:.4f},θ={obj_theta:.4f})  ee=({ee_x:.4f},{ee_y:.4f})")

    renderer.close()
    return frame_paths


def main():
    parser = argparse.ArgumentParser(description="Render npz episode to frames")
    parser.add_argument("--npz", required=True, help="Path to npz episode file")
    parser.add_argument(
        "--templates",
        default="data/sim/metadata/reset_templates_v0.json",
        help="Reset templates JSON",
    )
    parser.add_argument("--out", required=True, help="Output directory for frames")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--camera", default="topdown")
    parser.add_argument(
        "--metadata",
        default="data/sim/mppi_stage2c/metadata/episodes.jsonl",
        help="Episode metadata JSONL",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"ERROR: npz not found: {npz_path}")
        sys.exit(1)

    # Load templates
    with open(args.templates) as f:
        templates = json.load(f)

    # Load episode metadata
    episode_id = npz_path.stem
    episode_meta = None
    with open(args.metadata) as f:
        for line in f:
            ep = json.loads(line)
            if ep["episode_id"] == episode_id:
                episode_meta = ep
                break

    if episode_meta is None:
        print(f"WARNING: episode {episode_id} not found in metadata")
    else:
        print(f"Episode: {episode_id}")
        print(f"  Family: {episode_meta.get('family', '?')}")
        print(f"  Success: {episode_meta.get('success', '?')}")

    out_dir = Path(args.out)
    print(f"\nRendering {npz_path.name} → {out_dir}/")
    print(f"  Resolution: {args.width}×{args.height}")
    print(f"  Using obstacle data directly from npz (bypasses template matching)")

    frame_paths = render_episode(
        str(npz_path),
        None,  # obstacles built directly from npz data
        out_dir,
        width=args.width,
        height=args.height,
        camera=args.camera,
    )

    print(f"\nDone! {len(frame_paths)} frames rendered to {out_dir}/")

    # Also save a JSON with frame-action mapping
    data = np.load(npz_path, allow_pickle=True)
    actions = data["actions_norm"]
    obj_poses = data["object_poses"]
    ee_pos = data["ee_positions"]
    has_next = "next_object_poses" in data and "next_ee_positions" in data
    next_obj = data["next_object_poses"] if has_next else None
    next_ee = data["next_ee_positions"] if has_next else None
    mapping = []
    for i, fp in enumerate(frame_paths):
        # Last frame is the post-action state (if present)
        if i < len(obj_poses):
            obj = obj_poses[i].tolist()
            ee = ee_pos[i].tolist()
            act = actions[i].tolist() if i < len(actions) else [0.0, 0.0]
        else:
            obj = next_obj[-1].tolist()
            ee = next_ee[-1].tolist()
            act = [0.0, 0.0]  # no action after final state
        mapping.append({"frame": fp.name, "step": i, "object_pose": obj, "ee_position": ee, "action_norm": act})
    mapping_path = out_dir / "frame_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Frame-action mapping saved to {mapping_path}")


if __name__ == "__main__":
    main()
