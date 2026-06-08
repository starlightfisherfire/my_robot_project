#!/usr/bin/env python3
"""Render videos for overnight dual-planner trials.

Re-runs one trial per (model, template) combination and renders to video.
Uses EGL rendering (CPU-side, no GPU memory impact).

Usage:
    MUJOCO_GL=egl PYTHONPATH=. python scripts/render_overnight_top_trials.py
"""

import sys, json, time, os
from pathlib import Path
import numpy as np
import torch

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

sys.path.insert(0, '/home/brucewu/my_robot_project')

import mujoco
from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.learned_planner_adapter import CEMLearnedPlanner
from src.planners.action_conventions import PAPER1_CONVENTION
from src.envs.mujoco_push_env import MujocoPushEnv

# Constants (same as overnight run)
EE_SIZE, OBJ_SIZE, OBJ_MASS, OBJ_FRICTION = 0.015, 0.048, 0.038, 0.8
RUN_DIR = Path('runs/overnight_dual_planner_closed_loop_20260525_033000')
OUT_DIR = Path('runs/rendered_videos')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ['flat', 'object_centric', 'causality_aware']


def extract_state16(env):
    ee = env.get_ee_pos(); obj = env.get_object_pose(); goal = env.get_goal_pose()
    t = np.zeros((6, 16), dtype=np.float32)
    t[0, :2] = ee; t[0, 3] = 1.0; t[0, 7:9] = EE_SIZE; t[0, 14] = float(env.get_contact_flag()); t[0, 15] = 1.0
    t[1, :2] = obj[:2]; t[1, 2] = np.sin(obj[2]); t[1, 3] = np.cos(obj[2])
    t[1, 7:9] = OBJ_SIZE; t[1, 9] = 1.0; t[1, 12:14] = [OBJ_MASS, OBJ_FRICTION]; t[1, 15] = 1.0
    t[2, :2] = goal[:2]; t[2, 2] = np.sin(goal[2]); t[2, 3] = np.cos(goal[2])
    t[2, 7:9] = OBJ_SIZE; t[2, 9] = 1.0; t[2, 12:14] = [OBJ_MASS, OBJ_FRICTION]; t[2, 15] = 1.0
    return t


def load_model(model_type):
    model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256, d_model=128, head_hidden_dim=256)
    ckpt = torch.load(f'runs/pilot_state16_mppi_stage2c/train_{model_type}/{model_type}/checkpoints/best.pt',
                       map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    norm = StateNormalizer.load(f'runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json')
    return model, norm


def setup_renderer(env, width=640, height=480):
    """Initialize MuJoCo EGL renderer."""
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    # Try named camera first
    try:
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjCamera, "topdown")
        return renderer, cam_id
    except Exception:
        return renderer, None


def run_and_render(env, planner, template, max_steps, exec_steps, renderer, cam_id):
    """Run one trial, collecting frames. Returns (result_dict, frames_list)."""
    env.reset_from_template(template)
    state = extract_state16(env)
    hist = np.tile(state[np.newaxis], (6, 1, 1))
    goal = env.get_goal_pose()
    init_d = float(np.linalg.norm(env.get_object_pose()[:2] - goal[:2]))
    best_d = init_d; final_d = init_d; contacts = 0; collisions = 0; total_steps = 0
    frames = []

    # Capture initial frame
    mujoco.mj_forward(env.model, env.data)
    if cam_id is not None:
        renderer.update_scene(env.data, camera=cam_id)
    else:
        renderer.update_scene(env.data)
    frames.append(renderer.render())

    while total_steps < max_steps:
        curr = extract_state16(env)
        hist[:-1] = hist[1:]
        hist[-1] = curr
        result = planner.plan(hist.copy(), goal)

        steps_to_exec = min(exec_steps, max_steps - total_steps)
        for s in range(steps_to_exec):
            action = result.action_sequence_norm[s] if s < len(result.action_sequence_norm) else result.first_action_norm
            env.step(action)
            total_steps += 1
            contacts += int(env.get_contact_flag())
            collisions += int(env.get_collision_flag())
            obj = env.get_object_pose()
            d = float(np.linalg.norm(obj[:2] - goal[:2]))
            best_d = min(best_d, d); final_d = d

            # Capture frame
            mujoco.mj_forward(env.model, env.data)
            if cam_id is not None:
                renderer.update_scene(env.data, camera=cam_id)
            else:
                renderer.update_scene(env.data)
            frames.append(renderer.render())

            te = abs(np.arctan2(np.sin(obj[2] - goal[2]), np.cos(obj[2] - goal[2])))
            if d < 0.02 and np.degrees(te) < 10:
                break
        if final_d < 0.02 and np.degrees(te) < 10:
            break

    summary = {
        'initial_dist': init_d, 'final_dist': final_d, 'best_dist': best_d,
        'distance_improvement': init_d - best_d,
        'improved': best_d < init_d - 0.001,
        'contact_rate': contacts / max(total_steps, 1),
        'collision_rate': collisions / max(total_steps, 1),
        'total_steps': total_steps,
        'success': bool(final_d < 0.02 and np.degrees(te) < 10),
    }
    return summary, frames


def save_video(frames, path, fps=20):
    """Save frames as MP4 using imageio."""
    import imageio
    writer = imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def main():
    # Load templates
    with open('data/sim/metadata/reset_templates_v0.json') as f:
        all_templates = json.load(f)
    tmpl_map = {t['reset_template_id']: t for t in all_templates}

    # Get unique templates from overnight run
    trial_files = sorted((RUN_DIR / 'trial_results').glob('*.json'))
    seen_templates = []
    for f in trial_files:
        with open(f) as fh:
            d = json.load(fh)
        tid = d.get('template_id', '')
        if tid not in [x[0] for x in seen_templates]:
            seen_templates.append((tid, d.get('family', 'unknown')))

    print(f"Templates to render: {len(seen_templates)}")
    print(f"Models: {MODELS}")
    print(f"Total videos: {len(seen_templates) * len(MODELS)}")
    print()

    conv = PAPER1_CONVENTION
    conv.max_speed_mps = 0.75

    results_log = []
    video_count = 0

    for model_type in MODELS:
        print(f"\n{'='*60}")
        print(f"Loading model: {model_type}")
        model, norm = load_model(model_type)
        planner = CEMLearnedPlanner(
            model=model, normalizer=norm, convention=conv,
            horizon=10, num_samples=32, num_elites=8,
            num_iterations=3, init_std=0.3, device='cpu'
        )

        for tid, family in seen_templates:
            template = tmpl_map[tid]
            video_name = f"{model_type}__{family}__{tid.split('__')[-1]}"
            video_path = OUT_DIR / f"{video_name}.mp4"

            print(f"  [{video_count+1}] {video_name} ...", end=' ', flush=True)
            t0 = time.time()

            env = MujocoPushEnv(shape_type='T', control_dt=conv.control_dt, max_speed_mps=conv.max_speed_mps)
            renderer, cam_id = setup_renderer(env)

            try:
                summary, frames = run_and_render(
                    env, planner, template,
                    max_steps=30, exec_steps=1, renderer=renderer, cam_id=cam_id
                )
                save_video(frames, video_path, fps=20)
                elapsed = time.time() - t0
                print(f"OK  best={summary['best_dist']:.4f}m  improved={summary['improved']}  "
                      f"steps={summary['total_steps']}  frames={len(frames)}  {elapsed:.1f}s")

                results_log.append({
                    'model': model_type, 'template_id': tid, 'family': family,
                    'video': str(video_path), **summary, 'render_time': elapsed,
                })
            except Exception as e:
                print(f"FAIL: {e}")
                results_log.append({
                    'model': model_type, 'template_id': tid, 'family': family,
                    'video': str(video_path), 'error': str(e)[:200],
                })
            finally:
                renderer.close()
                del env

            video_count += 1

        del model, norm, planner
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save summary
    with open(OUT_DIR / 'render_summary.json', 'w') as f:
        json.dump(results_log, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Done. {video_count} videos rendered to {OUT_DIR}")
    print(f"Summary: {OUT_DIR / 'render_summary.json'}")


if __name__ == '__main__':
    main()
