#!/usr/bin/env python3
"""Render learned CEM closed-loop trials to video using GPU (EGL).

Usage:
  MUJOCO_GL=egl PYTHONPATH=. python scripts/render_learned_rollout.py \
    --checkpoint runs/pilot_state16_mppi_stage2c/train_flat/flat/checkpoints/best.pt \
    --normalizer runs/pilot_state16_mppi_stage2c/train_flat/normalizer_state16.json \
    --templates data/sim/metadata/reset_templates_obstacle_10family_v0.json \
    --max-videos 10 --max-steps 100 --max-speed 0.75 \
    --out-dir runs/fair_runner_repair_20260526_222900/videos
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Must set before mujoco import
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import imageio
from PIL import Image, ImageDraw

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.batched_rollout_cost import BatchedLearnedRolloutCostFn
from src.planners.cem_mpc import CEMMPC
from src.planners.cost_functions import CostWeights
from src.envs.mujoco_push_env import MujocoPushEnv
from scripts.run_learned_fair_closed_loop import (
    extract_state16_from_mujoco, compute_velocity
)


def render_frame(env, renderer, text_lines=None):
    """Render one MuJoCo frame and return PIL Image."""
    renderer.update_scene(env.data, camera="topdown")
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    if text_lines:
        draw = ImageDraw.Draw(img)
        y = 10
        for line in text_lines:
            draw.text((10, y), line, fill=(255, 255, 0))
            y += 20
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--templates", default="data/sim/metadata/reset_templates_obstacle_10family_v0.json")
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-speed", type=float, default=0.75)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-elites", type=int, default=64)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    # Load model with action_embed detection
    print(f"Loading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    # Auto-detect: old checkpoints don't have action_embed layer
    has_action_embed = any("action_embed" in k for k in ckpt["model_state_dict"].keys())
    use_action_embed = has_action_embed
    print(f"  Detected action_embed: {use_action_embed}")
    
    model = RIGWorldModel(model_type="flat", action_dim=2, gru_hidden=256, d_model=128,
                          head_hidden_dim=256, use_action_embed=use_action_embed)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    normalizer = StateNormalizer.load(args.normalizer)

    # Load templates
    with open(args.templates) as f:
        all_templates = json.load(f)

    # Select diverse templates: 3 open, 3 blocking_easy, 3 passage_direct_narrow, 1 extra
    families_wanted = {"open": 3, "blocking_easy": 3, "passage_direct_narrow": 3, "blocking_medium": 1}
    selected = []
    for fam, count in families_wanted.items():
        fam_tmpls = [t for t in all_templates if t.get("family", "") == fam]
        selected.extend(fam_tmpls[:count])
    selected = selected[: args.max_videos]

    print(f"Rendering {len(selected)} videos...")

    # MuJoCo renderer (GPU via EGL)
    env = MujocoPushEnv(shape_type="T", control_dt=0.1, max_speed_mps=args.max_speed)
    renderer = mujoco.Renderer(env.model, 480, 360)

    for vid_idx, template in enumerate(selected):
        template_id = template.get("reset_template_id", f"tpl_{vid_idx}")
        family = template.get("family", "unknown")
        short_id = template_id.split("__")[-1] if "__" in template_id else str(vid_idx)
        video_path = out_dir / f"learned_cem_{family}_{short_id}.mp4"

        print(f"\n[{vid_idx+1}/{len(selected)}] {family}: {short_id}")
        env.reset_from_template(template)

        # Initialize
        first_state = extract_state16_from_mujoco(env)
        history = np.tile(first_state[np.newaxis], (6, 1, 1))
        goal_pose = env.get_goal_pose()
        initial_dist = float(np.linalg.norm(first_state[1, :2] - goal_pose[:2]))

        # CEM planner
        cem = CEMMPC(horizon=args.horizon, action_dim=2,
                     num_samples=args.num_samples, num_elites=args.num_elites,
                     num_iterations=args.num_iterations,
                     action_low=-1.0, action_high=1.0, init_std=0.3, smoothing=0.2, seed=42)

        frames = []
        prev_state = first_state.copy()
        best_dist = float("inf")

        t0 = time.time()
        for step in range(args.max_steps):
            # Extract state
            curr_state = extract_state16_from_mujoco(env)
            if step > 0:
                curr_state = compute_velocity(prev_state, curr_state, env.control_dt)
            history[:-1] = history[1:]
            history[-1] = curr_state

            # Plan
            cost_fn = BatchedLearnedRolloutCostFn(
                model, normalizer, history.copy(), goal_pose, device="cpu", weights=CostWeights())
            result = cem.optimize(cost_fn)
            first_action = result.action_sequence[0]

            # Execute
            env.step(first_action)

            # Metrics
            obj_pose = env.get_object_pose()
            dist = float(np.linalg.norm(obj_pose[:2] - goal_pose[:2]))
            if dist < best_dist:
                best_dist = dist
            theta_err = np.degrees(abs(np.arctan2(
                np.sin(obj_pose[2] - goal_pose[2]), np.cos(obj_pose[2] - goal_pose[2]))))

            # Render frame
            text = [
                f"Step {step+1}/{args.max_steps}",
                f"Dist: {dist:.4f}m  Best: {best_dist:.4f}m  Init: {initial_dist:.4f}m",
                f"Theta: {theta_err:.0f}deg  Contact: {env.get_contact_flag()}",
                f"Cost: {result.best_cost:.3f}  Family: {family}",
            ]
            frame = render_frame(env, renderer, text)
            frames.append(frame)

            prev_state = curr_state.copy()

            # Early stop
            if dist < 0.002 and theta_err < 10:
                # Add a few more frames at success
                for _ in range(10):
                    frames.append(frame)
                break

            if (step + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  step {step+1}: dist={dist:.4f}m best={best_dist:.4f}m ({elapsed:.0f}s)")

        # Compile video
        elapsed = time.time() - t0
        print(f"  Compiled {len(frames)} frames in {elapsed:.0f}s, saving...")
        writer = imageio.get_writer(str(video_path), fps=args.fps, codec="libx264")
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()

        print(f"  ✅ {video_path} ({len(frames)} frames, {elapsed:.0f}s)")

        # Clear frames to save memory
        frames.clear()

    renderer.close()
    print(f"\nDone! {len(selected)} videos in {out_dir}")


if __name__ == "__main__":
    main()
