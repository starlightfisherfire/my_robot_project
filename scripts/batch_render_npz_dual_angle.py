#!/usr/bin/env python3
"""
Batch render all npz episodes with dual camera angles (topdown + tilted 45°).

Uses multiprocessing for GPU-accelerated parallel rendering.
Each worker gets its own MuJoCo/EGL context.

Usage:
  MUJOCO_GL=egl PYTHONPATH=. python scripts/batch_render_npz_dual_angle.py \
    --npz-dir data/sim/mppi_stage2c/episodes \
    --out runs/lewn_dataset/dual_frames \
    --workers 10
"""
from __future__ import annotations

import argparse, os, sys, json, time, glob, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np

def render_one_episode(args):
    """Render one npz episode with both camera angles. Runs in worker process."""
    npz_path, out_root, width, height = args
    eid = Path(npz_path).stem
    
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import mujoco
        from PIL import Image
        from src.envs.mujoco_push_env import MujocoPushEnv
        
        data = np.load(npz_path, allow_pickle=True)
        goal = data["goal_pose"]
        obs = data["obstacle_features"]
        
        # Build obstacles
        obstacles = []
        for slot in range(2):
            base = slot * 5
            sz_x = float(obs[base+3])
            sz_y = float(obs[base+4])
            if sz_x <= 0 or sz_y <= 0:
                continue
            obstacles.append({
                "obstacle_id": f"obs_{slot}",
                "pose": {"x": float(obs[base]), "y": float(obs[base+1]), "theta": float(obs[base+2])},
                "size_x": sz_x, "size_y": sz_y, "shape": "box", "valid": True,
            })
        
        env = MujocoPushEnv(max_speed_mps=0.3, pusher_mass=0.300)
        env._rebuild_model_with_obstacles(obstacles if obstacles else None)
        env._current_obstacle_signature = None
        env.reset()
        env.goal_pose = np.array([goal[0], goal[1], goal[2]], dtype=np.float64)
        env._sync_goal_visuals()
        
        # Make goal marker visible
        for i in range(env.model.ngeom):
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and 'goal' in name.lower():
                rgba = env.model.geom_rgba[i].copy()
                rgba[3] = 0.8
                env.model.geom_rgba[i] = rgba
        
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        workspace_center = np.array([0.35, 0.25, 0.01])
        
        # Camera objects (must use MjvCamera FREE, model camera mods don't work)
        cam_top = mujoco.MjvCamera()
        cam_top.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam_top.lookat[:] = workspace_center
        cam_top.distance = 0.85
        cam_top.elevation = -90.0
        cam_top.azimuth = 0.0
        
        cam_tilt = mujoco.MjvCamera()
        cam_tilt.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam_tilt.lookat[:] = workspace_center
        cam_tilt.distance = 0.55
        cam_tilt.elevation = -45.0
        cam_tilt.azimuth = 90.0
        
        def get_camera(angle):
            return cam_top if angle == "topdown" else cam_tilt
        
        # Build frame list
        obj_poses = list(data["object_poses"]) + [data["next_object_poses"][-1]]
        ee_positions = list(data["ee_positions"]) + [data["next_ee_positions"][-1]]
        actions = data["actions_norm"]
        
        # Render both angles
        frame_counts = {}
        for angle in ["topdown", "tilted"]:
            cam = get_camera(angle)
            angle_dir = Path(out_root) / angle / eid
            angle_dir.mkdir(parents=True, exist_ok=True)
            
            mapping = []
            for fi, (obj, ee) in enumerate(zip(obj_poses, ee_positions)):
                ox, oy, ot = float(obj[0]), float(obj[1]), float(obj[2])
                ex, ey = float(ee[0]), float(ee[1])
                
                z = 0.006
                qw = float(np.cos(ot / 2.0))
                qz = float(np.sin(ot / 2.0))
                env.data.qpos[env.object_qpos_adr:env.object_qpos_adr+7] = np.array(
                    [ox, oy, z, qw, 0.0, 0.0, qz], dtype=np.float64)
                env.data.qpos[env.pusher_x_qpos_adr] = ex
                env.data.qpos[env.pusher_y_qpos_adr] = ey
                mujoco.mj_forward(env.model, env.data)
                renderer.update_scene(env.data, camera=cam)
                pixels = renderer.render()
                
                fname = f"frame_{fi:04d}.png"
                Image.fromarray(pixels).save(angle_dir / fname)
                
                act = actions[fi].tolist() if fi < len(actions) else [0.0, 0.0]
                mapping.append({"frame": fname, "step": fi, "object_pose": [ox, oy, ot], "ee_position": [ex, ey], "action_norm": act})
            
            with open(angle_dir / "frame_mapping.json", "w") as f:
                json.dump(mapping, f, indent=2)
            frame_counts[angle] = len(mapping)
        
        renderer.close()
        
        return {"status": "ok", "eid": eid, "frames_topdown": frame_counts["topdown"], "frames_tilted": frame_counts["tilted"]}
    
    except Exception as e:
        return {"status": "failed", "eid": eid, "reason": str(e)[:200]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-dir", default="data/sim/mppi_stage2c/episodes")
    parser.add_argument("--out", default="runs/lewn_dataset/dual_frames")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    args = parser.parse_args()
    
    npz_dir = Path(args.npz_dir)
    npz_files = sorted(npz_dir.glob("*.npz"))
    
    print(f"=== Dual-Angle Batch Render ===")
    print(f"  Source: {npz_dir}")
    print(f"  Episodes: {len(npz_files)}")
    print(f"  Output: {args.out}/topdown/ + {args.out}/tilted/")
    print(f"  Resolution: {args.width}×{args.height}")
    print(f"  Workers: {args.workers}")
    print(f"  Angles: topdown (俯视) + tilted (45°斜上)")
    print()
    
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    
    tasks = [(str(f), str(out_root), args.width, args.height) for f in npz_files]
    
    t0 = time.time()
    ok, failed = 0, 0
    total_frames = 0
    
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(render_one_episode, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            if r["status"] == "ok":
                ok += 1
                total_frames += r.get("frames_topdown", 0) + r.get("frames_tilted", 0)
            else:
                failed += 1
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i+1) * (len(tasks) - i - 1)
                print(f"  [{i+1}/{len(tasks)}] ok={ok} fail={failed} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"  Episodes: {ok} ok, {failed} failed")
    print(f"  Total frames: {total_frames} ({total_frames//2} per angle)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {out_root}/")


if __name__ == "__main__":
    main()
