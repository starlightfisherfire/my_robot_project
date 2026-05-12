#!/usr/bin/env python3
"""
Render MuJoCo oracle-MPC closed-loop execution to video.

This script visualizes the actual execution process of closed-loop MPC,
allowing visual inspection of:
- EE effector movement patterns
- Contact timing with object
- Object displacement over time
- MPC replanning behavior

IMPORTANT: Set MUJOCO_GL environment variable BEFORE running this script:

    MUJOCO_GL=egl PYTHONPATH=. python scripts/render_closed_loop_rollout.py [args]

If EGL fails, try OSMesa:

    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_closed_loop_rollout.py [args]

Do NOT rely on the script's internal fallback logic. Set the backend explicitly
in your command line for reliable rendering.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ImportError as exc:
    print("ERROR: mujoco is not installed.", file=sys.stderr)
    print("Install MuJoCo before running this script.", file=sys.stderr)
    sys.exit(1)

try:
    import imageio
except ImportError as exc:
    print("ERROR: imageio is not installed.", file=sys.stderr)
    print("Install imageio: pip install imageio imageio-ffmpeg", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    print("ERROR: PIL is not installed.", file=sys.stderr)
    print("Install Pillow: pip install pillow", file=sys.stderr)
    sys.exit(1)

from src.envs.mujoco_push_env import MujocoPushEnv
from src.interventions.reset_template_loader import load_reset_templates
from src.metrics.mujoco_oracle_capacity import make_default_mujoco_cost_weights
from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import wrap_angle
from src.planners.mujoco_oracle_rollout import (
    mujoco_oracle_rollout_cost,
    rollout_action_sequence_mujoco,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render closed-loop MuJoCo oracle-MPC execution to video"
    )

    parser.add_argument(
        "--templates",
        type=str,
        default="data/sim/metadata/reset_templates_v0.json",
        help="Path to reset template JSON file.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train_sim_id",
        help="Split to select templates from.",
    )

    parser.add_argument(
        "--template-index",
        type=int,
        default=0,
        help="Index of template within the split.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=80,
        help="CEM planning horizon.",
    )

    parser.add_argument(
        "--execute-steps",
        type=int,
        default=5,
        help="Number of steps to execute per MPC iteration.",
    )

    parser.add_argument(
        "--max-mpc-steps",
        type=int,
        default=40,
        help="Maximum number of MPC replanning steps.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1536,
        help="CEM number of samples.",
    )

    parser.add_argument(
        "--num-elites",
        type=int,
        default=128,
        help="CEM number of elites.",
    )

    parser.add_argument(
        "--num-iterations",
        type=int,
        default=7,
        help="CEM number of iterations.",
    )

    parser.add_argument(
        "--out-video",
        type=str,
        default=None,
        help="Output video path. If None, auto-generated in artifacts/videos/.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Video width in pixels.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Video height in pixels.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frames per second.",
    )

    parser.add_argument(
        "--success-dist-threshold",
        type=float,
        default=0.05,
        help="Success distance threshold in meters (legacy, used for display only).",
    )

    parser.add_argument(
        "--success-pos-threshold",
        type=float,
        default=0.05,
        help="Strict pose early stop: position threshold in meters.",
    )

    parser.add_argument(
        "--stop-pos-threshold",
        type=float,
        default=None,
        dest="success_pos_threshold",
        help="Alias for --success-pos-threshold.",
    )

    parser.add_argument(
        "--success-theta-threshold-deg",
        type=float,
        default=180.0,
        help="Strict pose early stop: theta threshold in degrees. Set <180 to enable joint pose stop.",
    )

    parser.add_argument(
        "--stop-theta-threshold-deg",
        type=float,
        default=None,
        dest="success_theta_threshold_deg",
        help="Alias for --success-theta-threshold-deg.",
    )

    parser.add_argument(
        "--strict-pose-stop",
        action="store_true",
        default=False,
        help="Enable strict pose early stop with --stop-pos-threshold and --stop-theta-threshold-deg.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--pusher-radius",
        type=float,
        default=None,
        help="Pusher radius in meters (default: 0.010).",
    )

    parser.add_argument(
        "--pusher-halfheight",
        type=float,
        default=None,
        help="Pusher halfheight in meters (default: 0.014).",
    )

    parser.add_argument(
        "--pusher-z",
        type=float,
        default=None,
        help="Pusher z position in meters (default: 0.016).",
    )

    return parser.parse_args()


def setup_renderer(env: MujocoPushEnv, width: int, height: int):
    """Initialize MuJoCo renderer with fixed top-down camera."""
    gl_backend = os.environ.get("MUJOCO_GL", "not set")
    print(f"MUJOCO_GL backend: {gl_backend}")

    if gl_backend == "not set":
        print("WARNING: MUJOCO_GL not set. Renderer may fail.", file=sys.stderr)
        print("Recommended: MUJOCO_GL=egl or MUJOCO_GL=osmesa", file=sys.stderr)

    try:
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        print(f"✓ Renderer initialized: {width}x{height}")

        # Setup fixed top-down camera for planar pushing visualization
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [0.35, 0.25, 0.0]
        camera.distance = 0.8
        camera.elevation = -90
        camera.azimuth = 90

        return renderer, camera
    except Exception as e:
        print(f"✗ Failed to initialize renderer: {e}", file=sys.stderr)
        print(f"Try setting MUJOCO_GL explicitly:", file=sys.stderr)
        print(f"  MUJOCO_GL=egl python scripts/render_closed_loop_rollout.py ...", file=sys.stderr)
        print(f"  MUJOCO_GL=osmesa python scripts/render_closed_loop_rollout.py ...", file=sys.stderr)
        sys.exit(1)


def add_text_to_frame(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """
    Add text overlay to frame using PIL.

    Args:
        frame: RGB image array [H, W, 3]
        lines: List of text lines to render

    Returns:
        Frame with text overlay
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 16)
        except:
            font = ImageFont.load_default()

    text_height = 20 * len(lines)
    draw.rectangle([(10, 10), (650, 10 + text_height)], fill=(0, 0, 0))

    y_offset = 15
    for line in lines:
        draw.text((15, y_offset), line, fill=(255, 255, 255), font=font)
        y_offset += 20

    return np.array(img)


def render_frame_with_overlay(
    renderer,
    camera,
    env: MujocoPushEnv,
    template_id: str,
    mpc_step: int,
    exec_step_in_chunk: int,
    total_env_step: int,
    current_dist: float,
    current_theta_error_deg: float,
    contact_flag: float,
    object_displacement: float,
    planned_contact_first_step: int,
    execute_steps: int,
    horizon: int,
    strict_stop_active: bool = False,
) -> np.ndarray:
    """Render one frame with text overlay using fixed top-down camera."""
    renderer.update_scene(env.data, camera=camera)
    pixels = renderer.render()

    stop_str = "ON" if strict_stop_active else "off"
    text_lines = [
        f"Template: {template_id}",
        f"MPC Step: {mpc_step}",
        f"Exec Step: {exec_step_in_chunk}/{execute_steps}",
        f"Total Env Step: {total_env_step}",
        f"Dist: {current_dist*1000:.2f}mm  Theta: {current_theta_error_deg:.1f}deg",
        f"Contact: {contact_flag:.0f}",
        f"Object Disp: {object_displacement:.4f}m",
        f"Planned Contact@: {planned_contact_first_step}",
        f"Horizon: {horizon}  StrictStop:{stop_str}",
    ]

    frame_with_text = add_text_to_frame(pixels, text_lines)

    return frame_with_text


def find_first_contact_step(rollout_result) -> int:
    """Find the first step where contact occurs in a rollout."""
    for i, contact in enumerate(rollout_result.contact_flags):
        if contact > 0.5:
            return i
    return -1


def run_closed_loop_with_rendering(
    env: MujocoPushEnv,
    renderer,
    camera,
    template: dict[str, Any],
    planning_horizon: int,
    num_samples: int,
    num_elites: int,
    num_iterations: int,
    execute_steps: int,
    max_mpc_steps: int,
    seed: int,
    success_dist_threshold: float,
    success_pos_threshold: float = 0.05,
    success_theta_threshold_deg: float = 180.0,
) -> list[np.ndarray]:
    """
    Run closed-loop MPC and render each execution step.
    
    Returns:
        List of rendered frames (RGB images)
    """
    frames = []

    initial_object_pose = env.get_object_pose()
    initial_goal_pose = env.get_goal_pose()
    initial_dist = float(
        np.linalg.norm(initial_object_pose[:2] - initial_goal_pose[:2])
    )

    strict_stop_active = success_theta_threshold_deg < 180.0 or success_pos_threshold < success_dist_threshold
    print(f"\nTemplate: {template['reset_template_id']}")
    print(f"Initial distance: {initial_dist:.4f}m")
    print(f"Planning horizon: {planning_horizon}, Execute steps: {execute_steps}")
    print(f"Strict pose stop: pos<={success_pos_threshold*1000:.1f}mm AND theta<={success_theta_threshold_deg:.1f}deg  (active={strict_stop_active})")

    weights = make_default_mujoco_cost_weights()

    total_env_step = 0
    success = False
    best_dist = initial_dist

    initial_theta_error_deg = float(np.rad2deg(abs(wrap_angle(
        initial_object_pose[2] - initial_goal_pose[2]
    ))))

    frame = render_frame_with_overlay(
        renderer=renderer,
        camera=camera,
        env=env,
        template_id=template["reset_template_id"],
        mpc_step=0,
        exec_step_in_chunk=0,
        total_env_step=0,
        current_dist=initial_dist,
        current_theta_error_deg=initial_theta_error_deg,
        contact_flag=0.0,
        object_displacement=0.0,
        planned_contact_first_step=-1,
        execute_steps=execute_steps,
        horizon=planning_horizon,
        strict_stop_active=strict_stop_active,
    )
    frames.append(frame)

    for mpc_step in range(max_mpc_steps):
        current_object_pose = env.get_object_pose()
        current_dist = float(
            np.linalg.norm(current_object_pose[:2] - initial_goal_pose[:2])
        )

        print(f"\nMPC Step {mpc_step + 1}/{max_mpc_steps}, dist={current_dist:.4f}m")

        planner = CEMMPC(
            horizon=planning_horizon,
            action_dim=2,
            num_samples=num_samples,
            num_elites=num_elites,
            num_iterations=num_iterations,
            action_low=[-1.0, -1.0],
            action_high=[1.0, 1.0],
            init_std=0.8,
            smoothing=0.2,
            seed=seed + mpc_step,
        )

        def cost_fn(action_sequence: np.ndarray) -> float:
            return mujoco_oracle_rollout_cost(
                env=env,
                action_sequence=action_sequence,
                weights=weights,
                restore_state=True,
            )

        first_action, cem_result = planner.plan(cost_fn)

        planned_rollout = rollout_action_sequence_mujoco(
            env=env,
            action_sequence=cem_result.action_sequence,
            restore_state=True,
        )
        planned_contact_first_step = find_first_contact_step(planned_rollout)

        print(f"  CEM best_cost: {cem_result.best_cost:.4f}")
        print(f"  Planned contact@step: {planned_contact_first_step}")

        actions_to_execute = cem_result.action_sequence[:execute_steps]

        for exec_step_idx, action in enumerate(actions_to_execute):
            env.step(action)
            total_env_step += 1

            current_object_pose = env.get_object_pose()
            current_dist = float(
                np.linalg.norm(current_object_pose[:2] - initial_goal_pose[:2])
            )
            current_theta_error_deg = float(np.rad2deg(abs(wrap_angle(
                current_object_pose[2] - initial_goal_pose[2]
            ))))
            contact_flag = env.get_contact_flag()
            object_displacement = float(
                np.linalg.norm(current_object_pose[:2] - initial_object_pose[:2])
            )

            if current_dist < best_dist:
                best_dist = current_dist

            frame = render_frame_with_overlay(
                renderer=renderer,
                camera=camera,
                env=env,
                template_id=template["reset_template_id"],
                mpc_step=mpc_step + 1,
                exec_step_in_chunk=exec_step_idx + 1,
                total_env_step=total_env_step,
                current_dist=current_dist,
                current_theta_error_deg=current_theta_error_deg,
                contact_flag=contact_flag,
                object_displacement=object_displacement,
                planned_contact_first_step=planned_contact_first_step,
                execute_steps=execute_steps,
                horizon=planning_horizon,
                strict_stop_active=strict_stop_active,
            )
            frames.append(frame)

            # Strict pose early stop: both pos AND theta must be satisfied
            _pos_ok = current_dist < success_pos_threshold
            _theta_ok = current_theta_error_deg < success_theta_threshold_deg
            if _pos_ok and _theta_ok:
                success = True
                print(f"\n✓ STRICT POSE STOP at step {total_env_step}! dist={current_dist*1000:.2f}mm theta={current_theta_error_deg:.1f}deg")
                break
            # Legacy distance-only stop (only if strict stop not active)
            if not strict_stop_active and current_dist < success_dist_threshold:
                success = True
                print(f"\n✓ SUCCESS at step {total_env_step}!")
                break

        if success:
            break

    print(f"\nRendered {len(frames)} frames")
    print(f"Success: {success}, Best dist: {best_dist:.4f}m")

    return frames


def main() -> None:
    args = parse_args()

    template_path = Path(args.templates)
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    templates = load_reset_templates(template_path)

    templates = [t for t in templates if t["split"] == args.split]
    if not templates:
        print(f"ERROR: No templates found for split={args.split}", file=sys.stderr)
        sys.exit(1)

    if args.template_index < 0 or args.template_index >= len(templates):
        print(
            f"ERROR: template-index {args.template_index} out of range [0, {len(templates)-1}]",
            file=sys.stderr,
        )
        sys.exit(1)

    template = templates[args.template_index]

    print(f"Selected template: {template['reset_template_id']}")
    print(f"  split: {template['split']}")
    print(f"  layout_family: {template['layout_family']}")
    print(f"  shape_family: {template['shape_family']}")

    # Pusher geometry defaults
    pusher_radius = args.pusher_radius if args.pusher_radius is not None else 0.010
    pusher_halfheight = args.pusher_halfheight if args.pusher_halfheight is not None else 0.014
    pusher_z = args.pusher_z if args.pusher_z is not None else 0.016

    print(f"\nPusher geometry:")
    print(f"  radius: {pusher_radius:.4f}m")
    if pusher_halfheight is not None:
        print(f"  halfheight: {pusher_halfheight:.4f}m")
    print(f"  z: {pusher_z:.4f}m")

    env = MujocoPushEnv(
        control_dt=0.1,
        max_speed_mps=0.05,
        pusher_radius=pusher_radius,
        pusher_halfheight=pusher_halfheight,
        pusher_z=pusher_z,
    )
    env.reset_from_template(template)

    renderer, camera = setup_renderer(env, args.width, args.height)

    frames = run_closed_loop_with_rendering(
        env=env,
        renderer=renderer,
        camera=camera,
        template=template,
        planning_horizon=args.horizon,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_iterations=args.num_iterations,
        execute_steps=args.execute_steps,
        max_mpc_steps=args.max_mpc_steps,
        seed=args.seed,
        success_dist_threshold=args.success_dist_threshold,
        success_pos_threshold=args.success_pos_threshold,
        success_theta_threshold_deg=args.success_theta_threshold_deg,
    )

    if args.out_video is None:
        out_dir = Path("artifacts/videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / f"closed_loop_{template['reset_template_id']}_exec{args.execute_steps}_mpc{args.max_mpc_steps}.mp4"
    else:
        out_video = Path(args.out_video)
        out_video.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving video to: {out_video}")
    imageio.mimsave(
        out_video,
        frames,
        fps=args.fps,
        codec="libx264",
        quality=8,
    )

    print(f"✓ Video saved: {out_video}")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {len(frames) / args.fps:.2f}s")
    print(f"  Resolution: {args.width}x{args.height}")


if __name__ == "__main__":
    main()
