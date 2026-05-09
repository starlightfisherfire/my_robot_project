#!/usr/bin/env python3
"""
Visualize MuJoCo Oracle MPC rollout and save as video.

This script:
1. Loads one reset template
2. Runs CEM-MPC planning
3. Executes the planned action sequence
4. Saves frames as images and creates a video
"""

import numpy as np
from pathlib import Path
import imageio

from src.interventions.reset_template_loader import load_reset_templates
from src.envs.mujoco_push_env import MujocoPushEnv
from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import CostWeights
from src.planners.mujoco_oracle_rollout import (
    mujoco_oracle_rollout_cost,
)

try:
    import mujoco
except ImportError:
    print("mujoco is not installed.")
    exit(1)


def render_frame(model, data, width=640, height=480):
    """Render one frame using offscreen rendering."""
    # Create camera
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    # Set camera to view from above
    camera.lookat[:] = [0.35, 0.25, 0.0]
    camera.distance = 0.8
    camera.elevation = -60
    camera.azimuth = 180

    # Create scene and context
    scene = mujoco.MjvScene(model, maxgeom=1000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # Update scene
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene
    )

    # Render
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjr_render(viewport, scene, context)

    # Read pixels
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(pixels, None, viewport, context)

    # Flip vertically
    pixels = np.flipud(pixels)

    return pixels


def main():
    print("=" * 70)
    print("MuJoCo Oracle MPC Rollout Visualization")
    print("=" * 70)
    print()

    # Load one template
    template_path = Path("data/sim/metadata/reset_templates_v0.json")
    templates = load_reset_templates(template_path)
    template = templates[0]

    print(f"Template ID: {template['reset_template_id']}")
    print(f"Shape family: {template['shape_family']}")
    print()

    # Create environment
    env = MujocoPushEnv(
        control_dt=0.1,
        max_speed_mps=0.05,
    )
    env.reset_from_template(template)

    initial_object_pose = env.get_object_pose()
    initial_goal_pose = env.get_goal_pose()

    print("Initial state:")
    print(f"  Object pose: {initial_object_pose}")
    print(f"  Goal pose: {initial_goal_pose}")
    print(f"  Distance: {np.linalg.norm(initial_object_pose[:2] - initial_goal_pose[:2]):.4f} m")
    print()

    # Setup CEM-MPC
    horizon = 80
    weights = CostWeights(
        w_pos=50.0,
        w_theta=2.0,
        w_reach=5.0,
        w_no_contact=2.0,
        w_push_alignment=1.0,
        w_collision=20.0,
        w_action=0.05,
        w_smooth=0.1,
        w_subgoal=0.0,
    )

    planner = CEMMPC(
        horizon=horizon,
        action_dim=2,
        num_samples=1536,
        num_elites=128,
        num_iterations=7,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
        init_std=0.8,
        smoothing=0.2,
        seed=42,
    )

    def cost_fn(action_sequence: np.ndarray) -> float:
        return mujoco_oracle_rollout_cost(
            env=env,
            action_sequence=action_sequence,
            weights=weights,
            restore_state=True,
        )

    print("Running CEM-MPC planning...")
    first_action, cem_result = planner.plan(cost_fn)
    print(f"  Best cost: {cem_result.best_cost:.4f}")
    print()

    # Execute planned actions and render
    print("Executing planned actions and rendering...")
    frames = []

    # Render initial frame
    frame = render_frame(env.model, env.data)
    frames.append(frame)

    # Execute actions
    for i, action in enumerate(cem_result.action_sequence):
        env.step(action)

        # Render every frame
        frame = render_frame(env.model, env.data)
        frames.append(frame)

        if (i + 1) % 20 == 0:
            print(f"  Rendered {i + 1}/{horizon} frames")

    final_object_pose = env.get_object_pose()
    final_dist = np.linalg.norm(final_object_pose[:2] - initial_goal_pose[:2])

    print()
    print("Final state:")
    print(f"  Object pose: {final_object_pose}")
    print(f"  Distance: {final_dist:.4f} m")
    print(f"  Object displacement: {np.linalg.norm(final_object_pose[:2] - initial_object_pose[:2]):.6f} m")
    print()

    # Save frames as images
    output_dir = Path("artifacts/mpc_rollout_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(frames)} frames to {output_dir}/")
    for i, frame in enumerate(frames):
        imageio.imwrite(output_dir / f"frame_{i:04d}.png", frame)

    # Save as video
    video_path = Path("artifacts/mpc_rollout.mp4")
    print(f"Creating video: {video_path}")

    # Repeat each frame 2 times to slow down (control_dt=0.1s, so 10 fps -> 5 fps)
    frames_repeated = []
    for frame in frames:
        frames_repeated.extend([frame, frame])

    imageio.mimsave(
        video_path,
        frames_repeated,
        fps=10,
        codec='libx264',
        quality=8,
    )

    print()
    print("=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print(f"Video saved to: {video_path}")
    print(f"Frames saved to: {output_dir}/")
    print()
    print("You can:")
    print(f"  1. Download {video_path} and play it locally")
    print(f"  2. View individual frames in {output_dir}/")


if __name__ == "__main__":
    main()
