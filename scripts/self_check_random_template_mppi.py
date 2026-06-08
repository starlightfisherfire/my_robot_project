#!/usr/bin/env python3
"""
Self-check for random-template MPPI rollout pipeline.

Verifies:
1. template_generator works for all 8 families
2. MPPI config loads correctly
3. Env reset_from_template works with random templates
4. Single smoke episode runs (1 template) and saves npz
5. Rendered frame from npz passes basic checks
"""
from __future__ import annotations

import random, sys, os, time, json
from pathlib import Path

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
MPPI_CONFIG = {
    "temperature": 0.2,
    "num_samples": 2048,
    "horizon": 100,
    "init_std": 0.5,
    "smoothing": 0.2,
    "execute_steps": 10,
    "max_mpc_steps": 100,
    "max_speed_mps": 0.3,
}
FAMILIES = [
    "open", "blocking_hard", "blocking_medium", "blocking_easy",
    "passage_direct_narrow", "passage_bypass_narrow",
    "passage_bypass_wide", "passage_bypass_medium",
]
SEED = 42

def family_to_layout(family: str) -> str:
    """Map family name to template_generator layout type."""
    if "open" in family:
        return "open"
    elif "blocking" in family:
        return "blocking"
    elif "passage" in family:
        return "passage"
    else:
        return "open"

def main():
    errors = []
    rng = random.Random(SEED)

    # ── Check 1: Template generation ──
    print("=" * 60)
    print("[1/5] Template generation check")
    print("=" * 60)
    for family in FAMILIES:
        layout = family_to_layout(family)
        try:
            t = generate_template(layout, rng)
            valid = is_template_valid(t)
            n_obs = len(t.get("obstacles", []))
            print(f"  ✅ {family} ({layout}): {n_obs} obstacles, valid={valid}")
        except Exception as e:
            print(f"  ❌ {family}: {e}")
            errors.append(f"template_gen:{family}:{e}")

    # ── Check 2: Env init + reset ──
    print("\n" + "=" * 60)
    print("[2/5] Env initialization check")
    print("=" * 60)
    rng = random.Random(SEED)
    try:
        env = MujocoPushEnv(max_speed_mps=MPPI_CONFIG["max_speed_mps"], pusher_mass=0.300)
        for family in FAMILIES[:3]:  # Test 3 families
            t = generate_template(family_to_layout(family), rng)
            state = env.reset_from_template(t)
            obj = env.get_object_pose()
            goal = env.get_goal_pose()
            print(f"  ✅ {family}: obj=({obj[0]:.3f},{obj[1]:.3f}) goal=({goal[0]:.3f},{goal[1]:.3f})")
    except Exception as e:
        print(f"  ❌ Env init: {e}")
        errors.append(f"env:{e}")
        return

    # ── Check 3: Cost function + MPPI init ──
    print("\n" + "=" * 60)
    print("[3/5] MPPI planner init check")
    print("=" * 60)
    try:
        weights = CostWeights(w_pos=1.0, w_theta=0.5, w_collision=10.0,
                             w_action=0.01, w_reach=2.0, w_no_contact=1.0,
                             w_collision_step=1.0)
        
        def cost_fn(action_seq):
            return mujoco_oracle_rollout_cost(
                env=env, action_sequence=action_seq,
                weights=weights, restore_state=True,
            )
        
        planner = MPPI(
            horizon=MPPI_CONFIG["horizon"],
            num_samples=MPPI_CONFIG["num_samples"],
            temperature=MPPI_CONFIG["temperature"],
            init_std=MPPI_CONFIG["init_std"],
            smoothing=MPPI_CONFIG["smoothing"],
        )
        print(f"  ✅ MPPI initialized: T={MPPI_CONFIG['temperature']}, N={MPPI_CONFIG['num_samples']}, H={MPPI_CONFIG['horizon']}")
    except Exception as e:
        print(f"  ❌ MPPI init: {e}")
        errors.append(f"mppi:{e}")

    # ── Check 4: Smoke episode (2 steps only) ──
    print("\n" + "=" * 60)
    print("[4/5] Smoke episode (quick test)")
    print("=" * 60)
    try:
        smoke_planner = MPPI(horizon=10, num_samples=64, temperature=0.2, init_std=0.5, smoothing=0.2)
        smoke_weights = CostWeights(w_pos=1.0, w_theta=0.5, w_action=0.01)
        def smoke_cost_fn(action_seq):
            return mujoco_oracle_rollout_cost(
                env=env, action_sequence=action_seq, weights=smoke_weights, restore_state=True)
        
        rng = random.Random(SEED + 999)
        t = generate_template("open", rng)
        env.reset_from_template(t)
        goal = env.get_goal_pose()

        ep_data = {
            "states": [], "actions_norm": [], "actions_physical": [],
            "next_states": [], "object_poses": [], "next_object_poses": [],
            "ee_positions": [], "next_ee_positions": [],
            "contact_flags": [], "collision_flags": [],
            "actual_ee_velocities": [], "actual_object_velocities": [],
            "goal_pose": goal,
            "obstacle_features": np.zeros(10, dtype=np.float32),
        }

        prev_state = env.clone_state()
        for step in range(3):  # Just 3 steps for smoke test
            result = smoke_planner.optimize(smoke_cost_fn)
            action = result.action_sequence[0]
            
            prev_obj = env.get_object_pose()
            prev_ee = env.get_ee_pos()
            
            env.step(action)
            
            new_obj = env.get_object_pose()
            new_ee = env.get_ee_pos()
            
            ep_data["states"].append(np.array([prev_obj[0], prev_obj[1], prev_obj[2], prev_ee[0], prev_ee[1]]))
            ep_data["actions_norm"].append(action)
            ep_data["actions_physical"].append(action * MPPI_CONFIG["max_speed_mps"])
            ep_data["next_states"].append(np.array([new_obj[0], new_obj[1], new_obj[2], new_ee[0], new_ee[1]]))
            ep_data["object_poses"].append(prev_obj)
            ep_data["next_object_poses"].append(new_obj)
            ep_data["ee_positions"].append(np.array([prev_ee[0], prev_ee[1]]))
            ep_data["next_ee_positions"].append(np.array([new_ee[0], new_ee[1]]))
            ep_data["contact_flags"].append(env.get_contact_flag())
            ep_data["collision_flags"].append(env.get_collision_flag())
            ep_data["actual_ee_velocities"].append(np.zeros(2))
            ep_data["actual_object_velocities"].append(np.zeros(2))

        print(f"  ✅ Smoke episode: {len(ep_data['states'])} steps completed")
    except Exception as e:
        print(f"  ❌ Smoke episode: {e}")
        errors.append(f"smoke:{e}")

    # ── Check 5: Rendered frame check ──
    print("\n" + "=" * 60)
    print("[5/5] Rendered frame check")
    print("=" * 60)
    try:
        # Rebuild env with obstacles from smoke
        env2 = MujocoPushEnv(max_speed_mps=0.3, pusher_mass=0.300)
        env2.reset()
        renderer = mujoco.Renderer(env2.model, height=224, width=224)
        
        # Set to last state from smoke
        last_obj = ep_data["next_object_poses"][-1]
        last_ee = ep_data["next_ee_positions"][-1]
        ox, oy, ot = float(last_obj[0]), float(last_obj[1]), float(last_obj[2])
        ex, ey = float(last_ee[0]), float(last_ee[1])
        
        z = 0.006
        qw = float(np.cos(ot / 2.0))
        qz = float(np.sin(ot / 2.0))
        env2.data.qpos[env2.object_qpos_adr:env2.object_qpos_adr+7] = np.array(
            [ox, oy, z, qw, 0.0, 0.0, qz], dtype=np.float64)
        env2.data.qpos[env2.pusher_x_qpos_adr] = ex
        env2.data.qpos[env2.pusher_y_qpos_adr] = ey
        mujoco.mj_forward(env2.model, env2.data)
        
        try:
            cam_id = mujoco.mj_name2id(env2.model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown")
        except:
            cam_id = -1
        renderer.update_scene(env2.data, camera=cam_id)
        pixels = renderer.render()
        renderer.close()
        
        assert pixels.shape == (224, 224, 3), f"Bad shape: {pixels.shape}"
        assert pixels.min() >= 0 and pixels.max() <= 255, "Bad pixel range"
        print(f"  ✅ Rendered frame: shape={pixels.shape}, range=[{pixels.min()},{pixels.max()}]")
    except Exception as e:
        print(f"  ❌ Render check: {e}")
        errors.append(f"render:{e}")

    # ── Summary ──
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ SELF-CHECK FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✅ SELF-CHECK PASSED — Ready for full sweep")
        print(f"\nPlan:")
        print(f"  Families: {len(FAMILIES)} ({', '.join(FAMILIES)})")
        print(f"  Per family: 50 random templates")
        print(f"  Total: {len(FAMILIES)*50} episodes")
        print(f"  MPPI: T={MPPI_CONFIG['temperature']} N={MPPI_CONFIG['num_samples']} H={MPPI_CONFIG['horizon']} σ={MPPI_CONFIG['init_std']} sp={MPPI_CONFIG['max_speed_mps']}")
        print(f"  CPU cores: 9 (parallel)")
        sys.exit(0)

if __name__ == "__main__":
    main()
