#!/usr/bin/env python3
"""
Render pusher capacity diagnostic video with heuristic policy.

Tests pusher physical capacity without MPC/CEM using direct commanded actions.

IMPORTANT: Set MUJOCO_GL environment variable BEFORE running this script:

    MUJOCO_GL=egl PYTHONPATH=. python scripts/render_pusher_capacity.py [args]

If EGL fails, try OSMesa:

    MUJOCO_GL=osmesa PYTHONPATH=. python scripts/render_pusher_capacity.py [args]

Does NOT modify:
- MujocoPushEnv default XML
- Cost functions
- CEM/MPC planners
- Reset templates

Only tests raw pusher control capacity with direct commanded actions.
"""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render pusher capacity diagnostic video"
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
        "--max-speed-mps",
        type=float,
        default=0.05,
        help="Maximum pusher speed in m/s.",
    )

    parser.add_argument(
        "--phase-a-steps",
        type=int,
        default=120,
        help="Maximum steps for phase A (approach).",
    )

    parser.add_argument(
        "--phase-b-steps",
        type=int,
        default=300,
        help="Maximum steps for phase B (push).",
    )

    parser.add_argument(
        "--pre-contact-offset",
        type=float,
        default=0.04,
        help="Distance behind object to start pushing (m).",
    )

    parser.add_argument(
        "--pre-contact-threshold",
        type=float,
        default=0.005,
        help="Distance threshold to consider pre-contact position reached (m).",
    )

    parser.add_argument(
        "--approach-timeout-switch",
        type=bool,
        default=True,
        help="Force switch to push phase if approach times out.",
    )

    print(f"MUJOCO_GL backend: {gl_backend}")

    if gl_backend == "not set":
        print("WARNING: MUJOCO_GL not set. Renderer may fail.", file=sys.stderr)
        print("Recommended: MUJOCO_GL=egl or MUJOCO_GL=osmesa", file=sys.stderr)

    try:
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        print(f"✓ Renderer initialized: {width}x{height}")

        # Setup fixed top-down camera (same as render_closed_loop_rollout.py)
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
        print(f"  MUJOCO_GL=egl python scripts/render_pusher_capacity.py ...", file=sys.stderr)
        print(f"  MUJOCO_GL=osmesa python scripts/render_pusher_capacity.py ...", file=sys.stderr)
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
    draw.rectangle([(10, 10), (750, 10 + text_height)], fill=(0, 0, 0))

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
    phase: str,
    total_step: int,
    current_dist: float,
    contact_flag: float,
    first_contact_step: int,
    object_displacement: float,
    object_displacement_after_contact: float,
    ee_object_dist: float,
    ee_to_pre_contact_dist: float,
    min_ee_object_dist: float,
    approach_timeout: bool,
    pusher_actual_speed: float,
    commanded_speed: float,
    max_speed_mps: float,
    commanded_action: np.ndarray,
    actual_ee_pos: np.ndarray,
    object_pose: np.ndarray,
) -> np.ndarray:
    """Render one frame with text overlay using fixed top-down camera."""
    renderer.update_scene(env.data, camera=camera)
    pixels = renderer.render()

    text_lines = [
        f"Template: {template_id}",
        f"Phase: {phase}",
        f"Total Step: {total_step}",
        f"Current Dist: {current_dist:.4f}m",
        f"Contact: {contact_flag:.0f}",
        f"First Contact@: {first_contact_step}",
        f"Object Disp: {object_displacement:.4f}m",
        f"Obj Disp After Contact: {object_displacement_after_contact:.4f}m",
        f"EE-Obj Dist: {ee_object_dist:.4f}m",
        f"EE-PreContact Dist: {ee_to_pre_contact_dist:.4f}m",
        f"Min EE-Obj Dist: {min_ee_object_dist:.4f}m",
        f"Approach Timeout: {approach_timeout}",
        f"Max Speed: {max_speed_mps:.2f}m/s",
        f"Pusher Actual Speed: {pusher_actual_speed*100:.2f}cm/s",
        f"Commanded Speed: {commanded_speed*100:.2f}cm/s",
        f"Cmd Action: [{commanded_action[0]:.2f}, {commanded_action[1]:.2f}]",
        f"EE Pos: [{actual_ee_pos[0]:.3f}, {actual_ee_pos[1]:.3f}]",
        f"Obj Pose: [{object_pose[0]:.3f}, {object_pose[1]:.3f}, {object_pose[2]:.2f}]",
    ]

    frame_with_text = add_text_to_frame(pixels, text_lines)

    return frame_with_text


def run_heuristic_policy_with_rendering(
    env: MujocoPushEnv,
    renderer,
    camera,
    template: dict[str, Any],
    max_speed_mps: float,
    phase_a_steps: int,
    phase_b_steps: int,
    pre_contact_offset: float,
    pre_contact_threshold: float,
    approach_timeout_switch: bool,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Run heuristic policy and render each step.

    Phase A: Move pusher to pre-contact position
    Phase B: Push object toward goal

    Returns:
        frames: List of rendered frames (RGB images)
        metrics: Dictionary with diagnostic metrics
    """
    frames = []

    # Get initial poses
    initial_object_pose = env.get_object_pose()
    initial_ee_pos = env.get_ee_pos()
    goal_pose = env.get_goal_pose()

    initial_dist = float(np.linalg.norm(initial_object_pose[:2] - goal_pose[:2]))

    print(f"\nTemplate: {template['reset_template_id']}")
    print(f"Initial distance: {initial_dist:.4f}m")
    print(f"Max speed: {max_speed_mps:.2f}m/s")

    # Calculate push direction (from object to goal)
    push_dir = normalize_vector(goal_pose[:2] - initial_object_pose[:2])

    # Calculate pre-contact position (behind object)
    pre_contact_pos = initial_object_pose[:2] - push_dir * pre_contact_offset

    print(f"Push direction: [{push_dir[0]:.3f}, {push_dir[1]:.3f}]")
    print(f"Pre-contact pos: [{pre_contact_pos[0]:.3f}, {pre_contact_pos[1]:.3f}]")

    # Tracking variables
    first_contact_step_global = -1
    first_contact_step_in_push_phase = -1
    time_to_first_contact = -1.0
    contact_steps = set()
    object_displacements = []
    pusher_speeds_phase_a = []
    commanded_speeds_phase_a = []
    pusher_speeds_phase_b = []
    commanded_speeds_phase_b = []
    ee_object_distances = []
    ee_to_pre_contact_distances = []
    min_ee_object_dist = float('inf')
    approach_timeout = False
    phase_a_actual_steps = 0
    phase_b_actual_steps = 0

    current_phase = "approach"
    total_step = 0
    prev_ee_pos = initial_ee_pos.copy()

    # Render initial frame
    frame = render_frame_with_overlay(
        renderer=renderer,
        camera=camera,
        env=env,
        template_id=template["reset_template_id"],
        phase=current_phase,
        total_step=0,
        current_dist=initial_dist,
        contact_flag=0.0,
        first_contact_step=-1,
        object_displacement=0.0,
        object_displacement_after_contact=0.0,
        ee_object_dist=np.linalg.norm(initial_ee_pos - initial_object_pose[:2]),
        ee_to_pre_contact_dist=np.linalg.norm(initial_ee_pos - pre_contact_pos),
        min_ee_object_dist=np.linalg.norm(initial_ee_pos - initial_object_pose[:2]),
        approach_timeout=False,
        pusher_actual_speed=0.0,
        commanded_speed=0.0,
        max_speed_mps=max_speed_mps,
        commanded_action=np.array([0.0, 0.0]),
        actual_ee_pos=initial_ee_pos,
        object_pose=initial_object_pose,
    )
    frames.append(frame)

    # Phase A: Approach
    print(f"\nPhase A: Approach (max {phase_a_steps} steps)")
    for step in range(phase_a_steps):
        current_ee_pos = env.get_ee_pos()
        current_object_pose = env.get_object_pose()
        contact_flag = env.get_contact_flag()

        # Record contact (for metrics only, not phase transition)
        if contact_flag:
            contact_steps.add(total_step)
            if first_contact_step_global < 0:
                first_contact_step_global = total_step
                time_to_first_contact = total_step * env.control_dt

        # Calculate distances
        ee_to_pre_contact_dist = float(np.linalg.norm(current_ee_pos - pre_contact_pos))
        ee_to_object_dist = float(np.linalg.norm(current_ee_pos - current_object_pose[:2]))
        ee_to_pre_contact_distances.append(ee_to_pre_contact_dist)
        ee_object_distances.append(ee_to_object_dist)
        min_ee_object_dist = min(min_ee_object_dist, ee_to_object_dist)

        # Phase transition: switch to push phase when close to pre-contact position
        if ee_to_pre_contact_dist < pre_contact_threshold:
            print(f"  Reached pre-contact position at step {total_step}, switching to push phase")
            current_phase = "push"
            phase_a_actual_steps = step
            break

        # Compute action: proportional velocity control toward pre-contact position
        delta = pre_contact_pos - current_ee_pos
        action = delta / (max_speed_mps * env.control_dt)
        action = np.clip(action, -1.0, 1.0)

        # Calculate commanded speed
        commanded_speed = np.linalg.norm(action) * max_speed_mps
        commanded_speeds_phase_a.append(commanded_speed)

        # Execute action
        env.step(action)
        total_step += 1

        # Record state after step
        new_ee_pos = env.get_ee_pos()
        new_object_pose = env.get_object_pose()
        new_contact_flag = env.get_contact_flag()

        if new_contact_flag:
            contact_steps.add(total_step)
            if first_contact_step_global < 0:
                first_contact_step_global = total_step
                time_to_first_contact = total_step * env.control_dt

        # Calculate actual pusher speed
        pusher_displacement = np.linalg.norm(new_ee_pos - prev_ee_pos)
        pusher_speed = pusher_displacement / env.control_dt
        pusher_speeds_phase_a.append(pusher_speed)
        prev_ee_pos = new_ee_pos.copy()

        # Calculate metrics
        current_dist = float(np.linalg.norm(new_object_pose[:2] - goal_pose[:2]))
        object_displacement = float(
            np.linalg.norm(new_object_pose[:2] - initial_object_pose[:2])
        )
        object_displacements.append(object_displacement)
        ee_object_dist = float(np.linalg.norm(new_ee_pos - new_object_pose[:2]))
        ee_to_pre_contact_dist = float(np.linalg.norm(new_ee_pos - pre_contact_pos))

        # Calculate object displacement after contact
        if first_contact_step_global >= 0:
            object_displacement_after_contact = object_displacement
        else:
            object_displacement_after_contact = 0.0

        # Render frame
        frame = render_frame_with_overlay(
            renderer=renderer,
            camera=camera,
            env=env,
            template_id=template["reset_template_id"],
            phase=current_phase,
            total_step=total_step,
            current_dist=current_dist,
            contact_flag=new_contact_flag,
            first_contact_step=first_contact_step_global,
            object_displacement=object_displacement,
            object_displacement_after_contact=object_displacement_after_contact,
            ee_object_dist=ee_object_dist,
            ee_to_pre_contact_dist=ee_to_pre_contact_dist,
            min_ee_object_dist=min_ee_object_dist,
            approach_timeout=approach_timeout,
            pusher_actual_speed=pusher_speed,
            commanded_speed=commanded_speed,
            max_speed_mps=max_speed_mps,
            commanded_action=action,
            actual_ee_pos=new_ee_pos,
            object_pose=new_object_pose,
        )
        frames.append(frame)

        if current_phase == "push":
            break
    else:
        # Phase A timeout
        if approach_timeout_switch:
            print(f"  Phase A timeout at step {total_step}, forcing switch to push phase")
            current_phase = "push"
            phase_a_actual_steps = phase_a_steps
            approach_timeout = True
        else:
            print(f"  Phase A timeout at step {total_step}, ending without push phase")
            phase_a_actual_steps = phase_a_steps

    # Phase B: Push
    if current_phase == "push":
        print(f"\nPhase B: Push (max {phase_b_steps} steps)")
        for step in range(phase_b_steps):
            current_ee_pos = env.get_ee_pos()
            current_object_pose = env.get_object_pose()
            contact_flag = env.get_contact_flag()

            # Record contact
            if contact_flag:
                contact_steps.add(total_step)
                if first_contact_step_global < 0:
                    first_contact_step_global = total_step
                    time_to_first_contact = total_step * env.control_dt
                if first_contact_step_in_push_phase < 0:
                    first_contact_step_in_push_phase = step

            # Calculate distances
            ee_to_object_dist = float(np.linalg.norm(current_ee_pos - current_object_pose[:2]))
            ee_object_distances.append(ee_to_object_dist)
            min_ee_object_dist = min(min_ee_object_dist, ee_to_object_dist)

            # Compute action: push along push_dir
            action = push_dir  # Full speed in push direction

            # Calculate commanded speed
            commanded_speed = np.linalg.norm(action) * max_speed_mps
            commanded_speeds_phase_b.append(commanded_speed)

            # Execute action
            env.step(action)
            total_step += 1
            phase_b_actual_steps += 1

            # Record state after step
            new_ee_pos = env.get_ee_pos()
            new_object_pose = env.get_object_pose()
            new_contact_flag = env.get_contact_flag()

            if new_contact_flag:
                contact_steps.add(total_step)
                if first_contact_step_global < 0:
                    first_contact_step_global = total_step
                    time_to_first_contact = total_step * env.control_dt
                if first_contact_step_in_push_phase < 0:
                    first_contact_step_in_push_phase = step

            # Calculate actual pusher speed
            pusher_displacement = np.linalg.norm(new_ee_pos - prev_ee_pos)
            pusher_speed = pusher_displacement / env.control_dt
            pusher_speeds_phase_b.append(pusher_speed)
            prev_ee_pos = new_ee_pos.copy()

            # Calculate metrics
            current_dist = float(np.linalg.norm(new_object_pose[:2] - goal_pose[:2]))
            object_displacement = float(
                np.linalg.norm(new_object_pose[:2] - initial_object_pose[:2])
            )
            object_displacements.append(object_displacement)
            ee_object_dist = float(np.linalg.norm(new_ee_pos - new_object_pose[:2]))
            ee_to_pre_contact_dist = float(np.linalg.norm(new_ee_pos - pre_contact_pos))

            # Calculate object displacement after contact
            if first_contact_step_global >= 0:
                object_displacement_after_contact = object_displacement
            else:
                object_displacement_after_contact = 0.0

            # Render frame
            frame = render_frame_with_overlay(
                renderer=renderer,
                camera=camera,
                env=env,
                template_id=template["reset_template_id"],
                phase=current_phase,
                total_step=total_step,
                current_dist=current_dist,
                contact_flag=new_contact_flag,
                first_contact_step=first_contact_step_global,
                object_displacement=object_displacement,
                object_displacement_after_contact=object_displacement_after_contact,
                ee_object_dist=ee_object_dist,
                ee_to_pre_contact_dist=ee_to_pre_contact_dist,
                min_ee_object_dist=min_ee_object_dist,
                approach_timeout=approach_timeout,
                pusher_actual_speed=pusher_speed,
                commanded_speed=commanded_speed,
                max_speed_mps=max_speed_mps,
                commanded_action=action,
                actual_ee_pos=new_ee_pos,
                object_pose=new_object_pose,
            )
            frames.append(frame)

    # Final metrics
    final_object_pose = env.get_object_pose()
    final_ee_pos = env.get_ee_pos()
    final_dist = float(np.linalg.norm(final_object_pose[:2] - goal_pose[:2]))
    dist_delta = initial_dist - final_dist

    total_object_displacement = float(
        np.linalg.norm(final_object_pose[:2] - initial_object_pose[:2])
    )

    # Phase A metrics
    mean_pusher_speed_phase_a = (
        float(np.mean(pusher_speeds_phase_a)) if pusher_speeds_phase_a else 0.0
    )
    mean_commanded_speed_phase_a = (
        float(np.mean(commanded_speeds_phase_a)) if commanded_speeds_phase_a else 0.0
    )
    pusher_tracking_ratio_phase_a = (
        mean_pusher_speed_phase_a / mean_commanded_speed_phase_a
        if mean_commanded_speed_phase_a > 0
        else 0.0
    )

    # Phase B metrics
    mean_pusher_speed_phase_b = (
        float(np.mean(pusher_speeds_phase_b)) if pusher_speeds_phase_b else 0.0
    )
    mean_commanded_speed_phase_b = (
        float(np.mean(commanded_speeds_phase_b)) if commanded_speeds_phase_b else 0.0
    )
    pusher_tracking_ratio_phase_b = (
        mean_pusher_speed_phase_b / mean_commanded_speed_phase_b
        if mean_commanded_speed_phase_b > 0
        else 0.0
    )

    # Distance metrics
    final_ee_to_pre_contact_dist = float(
        ee_to_pre_contact_distances[-1] if ee_to_pre_contact_distances else 0.0
    )

    # Contact metrics
    contact_rate = (
        len(contact_steps) / total_step if total_step > 0 else 0.0
    )

    print(f"\nRendered {len(frames)} frames")
    print(f"Phase A steps: {phase_a_actual_steps}")
    print(f"Phase B steps: {phase_b_actual_steps}")
    print(f"Approach timeout: {approach_timeout}")
    print(f"First contact (global): step {first_contact_step_global} ({time_to_first_contact:.2f}s)")
    print(f"First contact (push phase): step {first_contact_step_in_push_phase}")
    print(f"Final dist: {final_dist:.4f}m")
    print(f"Dist delta: {dist_delta:.4f}m")
    print(f"Object displacement: {total_object_displacement:.4f}m")
    print(f"Min EE-object dist: {min_ee_object_dist:.4f}m")
    print(f"Final EE-pre_contact dist: {final_ee_to_pre_contact_dist:.4f}m")
    print(f"Contact rate: {contact_rate*100:.1f}%")
    print(f"Pusher tracking Phase A: {pusher_tracking_ratio_phase_a*100:.1f}%")
    print(f"Pusher tracking Phase B: {pusher_tracking_ratio_phase_b*100:.1f}%")

    metrics = {
        "max_speed_mps": max_speed_mps,
        "phase_a_steps": phase_a_actual_steps,
        "phase_b_steps": phase_b_actual_steps,
        "total_steps": total_step,
        "approach_timeout": approach_timeout,
        "final_ee_to_pre_contact_dist": final_ee_to_pre_contact_dist,
        "min_ee_object_dist": min_ee_object_dist,
        "first_contact_step_global": first_contact_step_global,
        "first_contact_step_in_push_phase": first_contact_step_in_push_phase,
        "time_to_first_contact": time_to_first_contact,
        "contact_rate": contact_rate,
        "contact_steps_count": len(contact_steps),
        "initial_dist": initial_dist,
        "final_dist": final_dist,
        "dist_delta": dist_delta,
        "object_displacement": total_object_displacement,
        "mean_pusher_speed_phase_a": mean_pusher_speed_phase_a,
        "mean_commanded_speed_phase_a": mean_commanded_speed_phase_a,
        "pusher_tracking_ratio_phase_a": pusher_tracking_ratio_phase_a,
        "mean_pusher_speed_phase_b": mean_pusher_speed_phase_b,
        "mean_commanded_speed_phase_b": mean_commanded_speed_phase_b,
        "pusher_tracking_ratio_phase_b": pusher_tracking_ratio_phase_b,
        "final_object_pose": final_object_pose.tolist(),
        "final_ee_pos": final_ee_pos.tolist(),
    }

    return frames, metrics


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

    # Set pusher geometry defaults
    pusher_radius = args.pusher_radius if args.pusher_radius is not None else 0.010
    pusher_halfheight = args.pusher_halfheight if args.pusher_halfheight is not None else 0.014
    pusher_z = args.pusher_z if args.pusher_z is not None else 0.016

    print(f"\nPusher geometry:")
    print(f"  radius: {pusher_radius:.4f}m")
    if pusher_halfheight is not None:
        print(f"  halfheight: {pusher_halfheight:.4f}m")
    print(f"  z: {pusher_z:.4f}m")

    # Create environment with specified max_speed and pusher geometry
    env = MujocoPushEnv(
        control_dt=0.1,
        max_speed_mps=args.max_speed_mps,
        pusher_radius=pusher_radius,
        pusher_halfheight=pusher_halfheight,
        pusher_z=pusher_z,
    )
    env.reset_from_template(template)

    renderer, camera = setup_renderer(env, args.width, args.height)

    frames, metrics = run_heuristic_policy_with_rendering(
        env=env,
        renderer=renderer,
        camera=camera,
        template=template,
        max_speed_mps=args.max_speed_mps,
        phase_a_steps=args.phase_a_steps,
        phase_b_steps=args.phase_b_steps,
        pre_contact_offset=args.pre_contact_offset,
        pre_contact_threshold=args.pre_contact_threshold,
        approach_timeout_switch=args.approach_timeout_switch,
    )

    # Determine output video path
    if args.out_video is None:
        out_dir = Path("artifacts/videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        speed_str = f"{int(args.max_speed_mps * 100):03d}"
        out_video = out_dir / f"pusher_capacity_speed{speed_str}.mp4"
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

    # Save JSON summary
    json_dir = Path("runs/debug")
    json_dir.mkdir(parents=True, exist_ok=True)
    speed_str = f"{int(args.max_speed_mps * 100):03d}"
    json_path = json_dir / f"pusher_capacity_video_cylinder_speed{speed_str}.json"

    metrics["video_path"] = str(out_video)
    metrics["template"] = {
        "reset_template_id": template["reset_template_id"],
        "split": template["split"],
        "layout_family": template["layout_family"],
        "shape_family": template["shape_family"],
    }

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ JSON summary saved: {json_path}")


if __name__ == "__main__":
    main()
