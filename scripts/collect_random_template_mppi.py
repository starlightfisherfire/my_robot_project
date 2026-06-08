#!/usr/bin/env python3
"""
Collect random-template MPPI trajectories for LeWM pixel dataset.

Config:
  MPPI: T=0.2, N=2048, H=100, σ=0.5, speed=0.3, ex=10
  Families: open, blocking_hard/medium/easy, passage (4 types)
  Per family: 50 random templates
  Parallel: 9 workers
"""
from __future__ import annotations

import os, sys, time, json, random, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco

from src.envs.mujoco_push_env import MujocoPushEnv
from src.data.template_generator import generate_template, is_template_valid
from src.planners.mppi import MPPI
from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost
from src.planners.cost_functions import CostWeights
from src.data.episode_writer import EpisodeWriter

# ── Config ──
MPPI_CFG = {
    "temperature": 0.2, "num_samples": 2048, "horizon": 100,
    "init_std": 0.5, "smoothing": 0.2, "execute_steps": 10,
    "max_mpc_steps": 100, "max_speed_mps": 0.3,
}
FAMILIES = [
    "open",
    "blocking_hard", "blocking_medium", "blocking_easy",
    "passage_direct_narrow", "passage_bypass_narrow",
    "passage_bypass_wide", "passage_bypass_medium",
]
PER_FAMILY = 50
WORKERS = 4  # Reduced to avoid OOM/systemd kill
OUT_DIR = Path("runs/lewn_dataset/random_templates")
FRAMES_DIR = OUT_DIR / "frames"

def family_to_layout(family: str) -> str:
    if "open" in family: return "open"
    if "blocking" in family: return "blocking"
    if "passage" in family: return "passage"
    return "open"

def run_one_episode(args):
    """Run one MPPI episode with a random template. Save npz + render frames."""
    family, seed, episode_idx = args
    
    rng = random.Random(seed)
    
    # Generate valid template
    for attempt in range(100):
        t = generate_template(family_to_layout(family), rng)
        if is_template_valid(t):
            break
    else:
        return {"status": "failed", "reason": "no_valid_template", "family": family, "idx": episode_idx}
    
    # Setup
    env = MujocoPushEnv(max_speed_mps=MPPI_CFG["max_speed_mps"], pusher_mass=0.300)
    env.reset_from_template(t)
    goal = env.get_goal_pose()
    
    weights = CostWeights(w_pos=1.0, w_theta=0.5, w_collision=10.0, w_action=0.01)
    def cost_fn(action_seq):
        return mujoco_oracle_rollout_cost(env=env, action_sequence=action_seq, weights=weights, restore_state=True)
    
    planner = MPPI(
        horizon=MPPI_CFG["horizon"], num_samples=MPPI_CFG["num_samples"],
        temperature=MPPI_CFG["temperature"], init_std=MPPI_CFG["init_std"],
        smoothing=MPPI_CFG["smoothing"],
    )
    
    # Data buffers
    ep = {
        "states": [], "actions_norm": [], "actions_physical": [],
        "next_states": [], "object_poses": [], "next_object_poses": [],
        "ee_positions": [], "next_ee_positions": [],
        "contact_flags": [], "collision_flags": [],
        "actual_ee_velocities": [], "actual_object_velocities": [],
        "goal_pose": goal,
        "obstacle_features": np.zeros(10, dtype=np.float32),
    }
    
    # Fill obstacle features
    for slot, obs in enumerate(t.get("obstacles", [])[:2]):
        base = slot * 5
        ep["obstacle_features"][base] = obs["pose"]["x"]
        ep["obstacle_features"][base+1] = obs["pose"]["y"]
        ep["obstacle_features"][base+2] = obs["pose"].get("theta", 0.0)
        ep["obstacle_features"][base+3] = obs.get("size_x", 0.08)
        ep["obstacle_features"][base+4] = obs.get("size_y", 0.06)
    
    best_dist = float('inf')
    success = False
    prev_mean = None
    mpc_steps = 0
    total_env_steps = 0
    
    try:
        for mpc_i in range(MPPI_CFG["max_mpc_steps"]):
            mpc_steps += 1
            result = planner.optimize(cost_fn, init_mean=prev_mean)
            
            for exec_i in range(MPPI_CFG["execute_steps"]):
                act_idx = exec_i
                if act_idx >= len(result.action_sequence):
                    break
                action = result.action_sequence[act_idx]
                
                prev_obj = env.get_object_pose()
                prev_ee = env.get_ee_pos()
                
                env.step(action)
                total_env_steps += 1
                
                new_obj = env.get_object_pose()
                new_ee = env.get_ee_pos()
                
                ep["states"].append(np.array([prev_obj[0], prev_obj[1], prev_obj[2], prev_ee[0], prev_ee[1]], dtype=np.float32))
                ep["actions_norm"].append(np.array(action, dtype=np.float32))
                ep["actions_physical"].append(np.array(action, dtype=np.float32) * MPPI_CFG["max_speed_mps"])
                ep["next_states"].append(np.array([new_obj[0], new_obj[1], new_obj[2], new_ee[0], new_ee[1]], dtype=np.float32))
                ep["object_poses"].append(np.array(prev_obj, dtype=np.float32))
                ep["next_object_poses"].append(np.array(new_obj, dtype=np.float32))
                ep["ee_positions"].append(np.array([prev_ee[0], prev_ee[1]], dtype=np.float32))
                ep["next_ee_positions"].append(np.array([new_ee[0], new_ee[1]], dtype=np.float32))
                ep["contact_flags"].append(bool(env.get_contact_flag()))
                ep["collision_flags"].append(bool(env.get_collision_flag()))
                ep["actual_ee_velocities"].append(np.zeros(2, dtype=np.float32))
                ep["actual_object_velocities"].append(np.zeros(2, dtype=np.float32))
                
                # Check early stop
                cur_dist = np.sqrt((new_obj[0]-goal[0])**2 + (new_obj[1]-goal[1])**2)
                cur_theta_err = abs((new_obj[2] - goal[2] + np.pi) % (2*np.pi) - np.pi) * 180/np.pi
                if cur_dist < best_dist:
                    best_dist = cur_dist
                
                if cur_dist < 0.002 and cur_theta_err < 10.0:
                    success = True
                    break
            
            if success:
                break
            
            # Warm-start
            prev_mean = np.zeros_like(result.mean)
            shift = MPPI_CFG["execute_steps"]
            if shift < len(result.mean):
                prev_mean[:-shift] = result.mean[shift:]
        
        # Save npz
        eid = f"{family}_{seed}"
        npz_path = OUT_DIR / "npz" / f"{eid}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            npz_path,
            states=np.array(ep["states"]),
            actions_norm=np.array(ep["actions_norm"]),
            actions_physical=np.array(ep["actions_physical"]),
            next_states=np.array(ep["next_states"]),
            object_poses=np.array(ep["object_poses"]),
            next_object_poses=np.array(ep["next_object_poses"]),
            ee_positions=np.array(ep["ee_positions"]),
            next_ee_positions=np.array(ep["next_ee_positions"]),
            contact_flags=np.array(ep["contact_flags"]),
            collision_flags=np.array(ep["collision_flags"]),
            actual_ee_velocities=np.array(ep["actual_ee_velocities"]),
            actual_object_velocities=np.array(ep["actual_object_velocities"]),
            goal_pose=ep["goal_pose"],
            obstacle_features=ep["obstacle_features"],
        )
        
        # Render frames
        frame_dir = FRAMES_DIR / eid
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Rebuild env with obstacles
        obs_list = []
        for obs in t.get("obstacles", []):
            obs_list.append({
                "obstacle_id": f"obs_{len(obs_list)}",
                "pose": obs["pose"],
                "size_x": obs.get("size_x", 0.08),
                "size_y": obs.get("size_y", 0.06),
                "shape": "box", "valid": True,
            })
        env._rebuild_model_with_obstacles(obs_list if obs_list else None)
        env._current_obstacle_signature = None
        env.reset()
        env.goal_pose = np.array([goal[0], goal[1], goal[2]], dtype=np.float64)
        env._sync_goal_visuals()
        
        renderer = mujoco.Renderer(env.model, height=224, width=224)
        try:
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown")
        except:
            cam_id = -1
        
        # Render all frames (object_poses + final next_object_poses)
        from PIL import Image as PILImage
        mapping = []
        all_obj = list(ep["object_poses"]) + [ep["next_object_poses"][-1]]
        all_ee = list(ep["ee_positions"]) + [ep["next_ee_positions"][-1]]
        
        for fi, (obj, ee) in enumerate(zip(all_obj, all_ee)):
            ox, oy, ot = float(obj[0]), float(obj[1]), float(obj[2])
            ex, ey = float(ee[0]), float(ee[1])
            z = 0.006
            qw = float(np.cos(ot / 2.0))
            qz = float(np.sin(ot / 2.0))
            env.data.qpos[env.object_qpos_adr:env.object_qpos_adr+7] = np.array([ox, oy, z, qw, 0.0, 0.0, qz], dtype=np.float64)
            env.data.qpos[env.pusher_x_qpos_adr] = ex
            env.data.qpos[env.pusher_y_qpos_adr] = ey
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=cam_id)
            pixels = renderer.render()
            PILImage.fromarray(pixels).save(frame_dir / f"frame_{fi:04d}.png")
            
            act = ep["actions_norm"][fi].tolist() if fi < len(ep["actions_norm"]) else [0.0, 0.0]
            mapping.append({"frame": f"frame_{fi:04d}.png", "step": fi, "object_pose": [ox, oy, ot], "ee_position": [ex, ey], "action_norm": act})
        
        with open(frame_dir / "frame_mapping.json", "w") as f:
            json.dump(mapping, f, indent=2)
        
        renderer.close()
        
        return {
            "status": "ok", "family": family, "idx": episode_idx,
            "success": success, "steps": total_env_steps, "best_dist": best_dist,
            "eid": eid,
        }
    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "reason": str(e), "family": family, "idx": episode_idx}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all tasks
    tasks = []
    seed_base = 10000
    for family in FAMILIES:
        for i in range(PER_FAMILY):
            seed = seed_base + FAMILIES.index(family) * 1000 + i
            tasks.append((family, seed, i))
    
    print(f"=== Random Template MPPI Sweep ===")
    print(f"  Families: {len(FAMILIES)}")
    print(f"  Per family: {PER_FAMILY}")
    print(f"  Total: {len(tasks)} episodes")
    print(f"  Workers: {WORKERS}")
    print(f"  MPPI: T={MPPI_CFG['temperature']} N={MPPI_CFG['num_samples']} H={MPPI_CFG['horizon']} σ={MPPI_CFG['init_std']} sp={MPPI_CFG['max_speed_mps']}")
    print(f"  Output: {OUT_DIR}")
    print()
    
    t0 = time.time()
    results = {"ok": 0, "failed": 0, "success": 0}
    
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(run_one_episode, t): t for t in tasks}
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            if r["status"] == "ok":
                results["ok"] += 1
                if r["success"]:
                    results["success"] += 1
            else:
                results["failed"] += 1
            
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i+1) * (len(tasks) - i - 1)
                print(f"  [{i+1}/{len(tasks)}] ok={results['ok']} succ={results['success']} fail={results['failed']} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"  Total: {len(tasks)} episodes")
    print(f"  OK: {results['ok']}, Failed: {results['failed']}")
    print(f"  Success: {results['success']} ({100*results['success']/max(results['ok'],1):.1f}%)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {OUT_DIR}")

if __name__ == "__main__":
    main()
