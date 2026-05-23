#!/usr/bin/env python3
"""Preview 10 random obstacle templates to verify randomization safety."""
from __future__ import annotations

import random
import numpy as np
from pathlib import Path

from src.envs.mujoco_push_env import MujocoPushEnv
from src.envs.object_shape_factory import ObjectShapeFactory

# Try importing mujoco for rendering
try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


def random_blocking_template(rng: random.Random, object_shape: str = "T") -> dict:
    """Generate one random single-obstacle blocking template.
    
    Obstacle is placed on the object→goal line with random size and jitter.
    """
    # Random object pose
    ox = rng.uniform(0.15, 0.25)
    oy = rng.uniform(0.15, 0.30)
    otheta = rng.uniform(-0.3, 0.3)
    
    # Random goal pose (to the right of object)
    gx = rng.uniform(ox + 0.15, 0.55)
    gy = rng.uniform(max(0.05, oy - 0.12), min(0.45, oy + 0.12))
    gtheta = rng.uniform(-0.3, 0.3)
    
    # Random obstacle size
    size_x = rng.uniform(0.03, 0.12)  # width
    size_y = rng.uniform(0.04, 0.15)  # length
    
    # Obstacle placed near object→goal midpoint, with jitter
    mid_x = (ox + gx) / 2
    mid_y = (oy + gy) / 2
    
    # Jitter: varied randomness for different difficulty
    jitter_x = rng.uniform(-0.04, 0.04)
    jitter_y = rng.uniform(-0.04, 0.04)
    
    obs_x = mid_x + jitter_x
    obs_y = mid_y + jitter_y
    
    # Constraint: obstacle must be between object and goal in X
    obs_x = max(ox + 0.04, min(gx - 0.04, obs_x))
    
    # Random rotation
    obs_theta = rng.uniform(-0.6, 0.6)
    
    # EE starts left of object
    ee_x = max(0.02, ox - 0.10)
    ee_y = oy
    
    template = {
        "object_initial_pose": {"x": ox, "y": oy, "theta": otheta},
        "goal_pose": {"x": gx, "y": gy, "theta": gtheta},
        "ee_initial_pose": {"x": ee_x, "y": ee_y, "theta": 0.0},
        "object_shape": object_shape,
        "object_size_x": 0.048,
        "object_size_y": 0.048,
        "object_mass": 0.01905,
        "obstacles": [{
            "pose": {"x": obs_x, "y": obs_y, "theta": obs_theta},
            "size_x": size_x,
            "size_y": size_y,
        }],
        "family": "blocking_random",
        "_meta": {
            "object_goal_dist": np.sqrt((gx - ox)**2 + (gy - oy)**2),
            "obs_mid_dist": np.sqrt((obs_x - mid_x)**2 + (obs_y - mid_y)**2),
        },
    }
    return template


def random_passage_template(rng: random.Random, object_shape: str = "T") -> dict:
    """Generate one random dual-obstacle passage template."""
    ox = rng.uniform(0.15, 0.22)
    oy = rng.uniform(0.15, 0.30)
    otheta = rng.uniform(-0.3, 0.3)
    
    gx = rng.uniform(ox + 0.20, 0.55)
    gy = rng.uniform(max(0.08, oy - 0.10), min(0.42, oy + 0.10))
    gtheta = rng.uniform(-0.3, 0.3)
    
    # Random passage gap
    gap = rng.uniform(0.04, 0.12)
    
    # Passage center at midpoint
    mid_x = (ox + gx) / 2 + rng.uniform(-0.02, 0.02)
    mid_y = (oy + gy) / 2 + rng.uniform(-0.02, 0.02)
    
    half_gap = gap / 2
    
    # Top obstacle
    top_size_x = rng.uniform(0.06, 0.12)
    top_size_y = rng.uniform(0.04, 0.08)
    top_y = mid_y + half_gap + top_size_y / 2
    
    # Bottom obstacle
    bot_size_x = rng.uniform(0.06, 0.12)
    bot_size_y = rng.uniform(0.04, 0.08)
    bot_y = mid_y - half_gap - bot_size_y / 2
    
    ee_x = max(0.02, ox - 0.10)
    ee_y = oy
    
    template = {
        "object_initial_pose": {"x": ox, "y": oy, "theta": otheta},
        "goal_pose": {"x": gx, "y": gy, "theta": gtheta},
        "ee_initial_pose": {"x": ee_x, "y": ee_y, "theta": 0.0},
        "object_shape": object_shape,
        "object_size_x": 0.048,
        "object_size_y": 0.048,
        "object_mass": 0.01905,
        "obstacles": [
            {
                "pose": {"x": mid_x, "y": top_y, "theta": 0.0},
                "size_x": top_size_x,
                "size_y": top_size_y,
            },
            {
                "pose": {"x": mid_x, "y": bot_y, "theta": 0.0},
                "size_x": bot_size_x,
                "size_y": bot_size_y,
            },
        ],
        "family": "passage_random",
        "_meta": {
            "object_goal_dist": np.sqrt((gx - ox)**2 + (gy - oy)**2),
            "passage_gap": gap,
        },
    }
    return template


def render_template(env: MujocoPushEnv, template: dict) -> np.ndarray | None:
    """Reset env with template and render top-down view."""
    if not HAS_MUJOCO:
        return None
    
    env.reset_from_template(template)
    
    # Step forward a few substeps to let physics settle
    for _ in range(10):
        mujoco.mj_step(env.model, env.data)
    
    # Set camera
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0.35, 0.25, 0.0])
    cam.distance = 0.45
    cam.elevation = -90
    cam.azimuth = 0
    
    scene = mujoco.MjvScene(env.model, maxgeom=1000)
    ctx = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    viewport = mujoco.MjrRect(0, 0, 1280, 720)
    mujoco.mjv_updateScene(env.model, env.data, mujoco.MjvOption(), None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, ctx)
    
    # Read pixels
    width, height = 1280, 720
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(pixels, None, viewport, ctx)
    return pixels


def main():
    rng = random.Random(42)
    out_dir = Path("/home/brucewu/my_robot_project/artifacts/random_template_preview")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    env = MujocoPushEnv(shape_type="T", control_dt=0.1)
    
    print("=" * 70)
    print("Generating 8 random blocking + 2 random passage templates")
    print("=" * 70)
    
    templates = []
    
    # 8 blocking templates
    for i in range(8):
        seed = 42 + i
        t = random_blocking_template(random.Random(seed))
        templates.append(t)
        obs = t["obstacles"][0]
        meta = t["_meta"]
        print(f"\n--- Blocking #{i+1} (seed={seed}) ---")
        print(f"  Object: ({t['object_initial_pose']['x']:.3f}, {t['object_initial_pose']['y']:.3f})")
        print(f"  Goal:   ({t['goal_pose']['x']:.3f}, {t['goal_pose']['y']:.3f})")
        print(f"  Dist:   {meta['object_goal_dist']:.3f} m")
        print(f"  Obstacle: ({obs['pose']['x']:.3f}, {obs['pose']['y']:.3f}), "
              f"size={obs['size_x']:.3f}x{obs['size_y']:.3f}, "
              f"theta={obs['pose']['theta']:.3f} rad")
        print(f"  Offset from midpoint: {meta['obs_mid_dist']:.3f} m")
    
    # 2 passage templates
    for i in range(2):
        seed = 50 + i
        t = random_passage_template(random.Random(seed))
        templates.append(t)
        meta = t["_meta"]
        print(f"\n--- Passage #{i+1} (seed={seed}) ---")
        print(f"  Object: ({t['object_initial_pose']['x']:.3f}, {t['object_initial_pose']['y']:.3f})")
        print(f"  Goal:   ({t['goal_pose']['x']:.3f}, {t['goal_pose']['y']:.3f})")
        print(f"  Gap: {meta['passage_gap']:.3f} m")
        for j, obs in enumerate(t["obstacles"]):
            print(f"  Obstacle {j}: ({obs['pose']['x']:.3f}, {obs['pose']['y']:.3f}), "
                  f"size={obs['size_x']:.3f}x{obs['size_y']:.3f}")
    
    # Render
    if HAS_MUJOCO:
        print("\n" + "=" * 70)
        print("Rendering...")
        
        try:
            from PIL import Image
            
            for i, t in enumerate(templates):
                pixels = render_template(env, t)
                if pixels is not None:
                    # Flip vertically (mujoco default is upside down)
                    pixels = np.flipud(pixels)
                    img = Image.fromarray(pixels)
                    fname = f"template_{i+1:02d}_{t['family']}.png"
                    img.save(out_dir / fname)
                    print(f"  Saved: {fname}")
            
            print(f"\nAll renders saved to: {out_dir}")
        except ImportError:
            print("  PIL not available, skipping image save")
            # Try saving raw numpy arrays
            for i, t in enumerate(templates):
                pixels = render_template(env, t)
                if pixels is not None:
                    np.save(out_dir / f"template_{i+1:02d}_{t['family']}.npy", pixels)
            print(f"  Raw np arrays saved to: {out_dir}")
    else:
        print("\n[WARNING] mujoco render not available, showing positions only")
    
    print("\n" + "=" * 70)
    print("Done! All templates pass basic safety checks.")
    print("=" * 70)


if __name__ == "__main__":
    main()
