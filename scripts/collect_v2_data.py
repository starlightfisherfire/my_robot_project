#!/usr/bin/env python3
"""collect_v2_data.py — Collect visual_structured_state_v2 episodes from MuJoCo.

Uses MPPI best config to collect episodes with rich structured features.

Usage:
    PYTHONPATH=. python scripts/collect_v2_data.py \
        --output-dir data/sim/visual_state_v2_pilot \
        --num-episodes 200 \
        --max-steps 250 \
        --speed 0.3 \
        --temperature 0.1 \
        --horizon 100 \
        --num-samples 2048 \
        --pusher-mass 0.300
"""

import argparse, json, sys, time, uuid
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.envs.mujoco_push_env import MujocoPushEnv
from src.planners.mppi import MPPI
from src.planners.cost_functions import CostWeights, rollout_cost
from src.planners.mujoco_oracle_rollout import mujoco_oracle_rollout_cost


# === Feature extraction ===

def extract_v2_features(env, prev_obj_pose=None, prev_ee_pos=None, prev_action=None, dt=0.1):
    """Extract all v2 feature groups from MuJoCo env state."""
    ee_pos = env.get_ee_pos()
    obj_pose = env.get_object_pose()
    goal_pose = env.get_goal_pose()
    contact = env.get_contact_flag()
    collision = env.get_collision_flag()

    # --- Object tokens ---
    obj_x, obj_y = obj_pose[0], obj_pose[1]
    obj_sin, obj_cos = np.sin(obj_pose[2]), np.cos(obj_pose[2])

    # Velocity from finite diff (or MuJoCo if available)
    if prev_obj_pose is not None:
        obj_vx = (obj_pose[0] - prev_obj_pose[0]) / dt
        obj_vy = (obj_pose[1] - prev_obj_pose[1]) / dt
        dtheta = np.arctan2(np.sin(obj_pose[2] - prev_obj_pose[2]),
                            np.cos(obj_pose[2] - prev_obj_pose[2]))
        obj_omega = dtheta / dt
    else:
        obj_vx, obj_vy, obj_omega = 0.0, 0.0, 0.0

    object_tokens = {
        'obj_x': obj_x, 'obj_y': obj_y,
        'obj_sin_theta': obj_sin, 'obj_cos_theta': obj_cos,
        'obj_vx': obj_vx, 'obj_vy': obj_vy, 'obj_omega': obj_omega,
        'obj_size_x': 0.048, 'obj_size_y': 0.048,
        'obj_shape_T': 1.0, 'obj_shape_L': 0.0,
    }

    # --- Proprio tokens ---
    if prev_ee_pos is not None:
        ee_vx = (ee_pos[0] - prev_ee_pos[0]) / dt
        ee_vy = (ee_pos[1] - prev_ee_pos[1]) / dt
    else:
        ee_vx, ee_vy = 0.0, 0.0

    proprio_tokens = {
        'ee_x': ee_pos[0], 'ee_y': ee_pos[1],
        'ee_vx': ee_vx, 'ee_vy': ee_vy,
    }
    if prev_action is not None:
        proprio_tokens['prev_action_dx'] = prev_action[0]
        proprio_tokens['prev_action_dy'] = prev_action[1]
    else:
        proprio_tokens['prev_action_dx'] = 0.0
        proprio_tokens['prev_action_dy'] = 0.0

    # --- Goal tokens ---
    goal_tokens = {
        'goal_x': goal_pose[0], 'goal_y': goal_pose[1],
        'goal_sin_theta': np.sin(goal_pose[2]),
        'goal_cos_theta': np.cos(goal_pose[2]),
    }

    # --- Relation tokens ---
    ee_obj_dx = ee_pos[0] - obj_pose[0]
    ee_obj_dy = ee_pos[1] - obj_pose[1]
    ee_obj_dist = np.sqrt(ee_obj_dx**2 + ee_obj_dy**2)

    obj_goal_dx = goal_pose[0] - obj_pose[0]
    obj_goal_dy = goal_pose[1] - obj_pose[1]
    obj_goal_dist = np.sqrt(obj_goal_dx**2 + obj_goal_dy**2)

    # Motion alignment: how well current motion aligns with goal direction
    if obj_goal_dist > 0.001:
        goal_dir = np.array([obj_goal_dx, obj_goal_dy]) / obj_goal_dist
        motion_dir = np.array([obj_vx, obj_vy])
        motion_norm = np.linalg.norm(motion_dir)
        if motion_norm > 0.001:
            motion_alignment = np.dot(motion_dir / motion_norm, goal_dir)
        else:
            motion_alignment = 0.0
    else:
        motion_alignment = 0.0

    relation_tokens = {
        'ee_obj_dist': ee_obj_dist,
        'ee_obj_dx': ee_obj_dx, 'ee_obj_dy': ee_obj_dy,
        'obj_goal_dist': obj_goal_dist,
        'obj_goal_dx': obj_goal_dx, 'obj_goal_dy': obj_goal_dy,
        'obj_obs1_dist': 0.0, 'obj_obs1_dx': 0.0, 'obj_obs1_dy': 0.0,
        'contact_proxy': float(contact),
        'motion_alignment': motion_alignment,
    }

    # --- Temporal tokens ---
    temporal_tokens = {
        'action_h1_dx': prev_action[0] if prev_action is not None else 0.0,
        'action_h1_dy': prev_action[1] if prev_action is not None else 0.0,
        'obj_moved_h1': np.linalg.norm([obj_vx * dt, obj_vy * dt]),
        'contact_h1': float(contact),
        'stuck_proxy': 0.0,  # will be computed externally
    }

    # --- Nuisance placeholders ---
    nuisance = {
        'obj_color_r': 0.0, 'obj_color_g': 0.0, 'obj_color_b': 0.0,
        'light_pos_x': 0.0, 'light_pos_y': 0.0,
        'ambient_light': 0.3, 'camera_fovy': 50.0,
    }

    # --- Privileged physics ---
    privileged = {
        'object_mass': 0.038,
        'object_friction': 0.8,
        'true_contact_flag': float(contact),
        'contact_force_x': 0.0, 'contact_force_y': 0.0,
        'contact_mu': 0.0,
    }

    # --- Obstacle tokens ---
    obstacle_tokens = {
        'obs_x': 0.0, 'obs_y': 0.0,
        'obs_size_x': 0.0, 'obs_size_y': 0.0,
    }

    return {
        **object_tokens, **proprio_tokens, **goal_tokens,
        **relation_tokens, **temporal_tokens,
        **nuisance, **privileged, **obstacle_tokens,
    }


def collect_one_episode(env, mppi_cfg, cost_w, max_steps, template=None):
    """Collect one episode using MPPI."""
    if template is not None:
        env.reset_from_template(template)
    else:
        env.reset()

    mppi = MPPI(
        horizon=mppi_cfg['horizon'], action_dim=2,
        num_samples=mppi_cfg['num_samples'],
        num_iterations=1,
        init_std=mppi_cfg['init_std'],
        temperature=mppi_cfg['temperature'],
        seed=42,
    )

    steps_data = []
    prev_obj = None
    prev_ee = None
    prev_action = None
    stuck_count = 0

    for step in range(max_steps):
        features = extract_v2_features(env, prev_obj, prev_ee, prev_action)

        # Compute stuck proxy (last 5 steps with <1mm movement)
        if len(steps_data) >= 5:
            recent_moves = [s['obj_moved_h1'] for s in steps_data[-5:]]
            features['stuck_proxy'] = float(sum(1 for m in recent_moves if m < 0.001) / 5)

        # MPPI plan
        goal_pose = env.get_goal_pose()

        def cost_fn(action_seq):
            return mujoco_oracle_rollout_cost(
                env=env, action_sequence=action_seq,
                goal_pose=goal_pose, weights=cost_w, restore_state=True,
            )

        try:
            action, _ = mppi.plan(cost_fn)
        except Exception:
            action = np.zeros(2)

        # Clip and execute
        action = np.clip(action, -1.0, 1.0)
        action_phys = action * env.max_speed_mps

        prev_obj = env.get_object_pose().copy()
        prev_ee = env.get_ee_pos().copy()
        prev_action = action.copy()

        env.step(action)

        # Record
        features['actions_physical'] = action_phys.tolist()
        features['actions_norm'] = action.tolist()
        features['object_poses'] = prev_obj.tolist()
        features['next_object_poses'] = env.get_object_pose().tolist()
        features['ee_positions'] = prev_ee.tolist()
        features['next_ee_positions'] = env.get_ee_pos().tolist()
        features['goal_pose'] = goal_pose.tolist()
        features['contact_flag'] = float(env.get_contact_flag())
        features['collision_flag'] = float(env.get_collision_flag())

        steps_data.append(features)

        # Check success
        obj = env.get_object_pose()
        dist = np.linalg.norm(obj[:2] - goal_pose[:2])
        theta_err = abs(np.arctan2(np.sin(obj[2] - goal_pose[2]),
                                    np.cos(obj[2] - goal_pose[2])))
        if dist < 0.02 and np.degrees(theta_err) < 10.0:
            break

    return steps_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-episodes', type=int, default=200)
    parser.add_argument('--max-steps', type=int, default=250)
    parser.add_argument('--speed', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=2048)
    parser.add_argument('--init-std', type=float, default=0.5)
    parser.add_argument('--pusher-mass', type=float, default=0.300)
    parser.add_argument('--control-dt', type=float, default=0.1)
    parser.add_argument('--template-file', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Create output dirs
    out_dir = Path(args.output_dir)
    ep_dir = out_dir / 'episodes'
    meta_dir = out_dir / 'metadata'
    ep_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Load templates
    if args.template_file:
        tf = args.template_file
    else:
        tf = str(REPO / 'data/sim/metadata/reset_templates_obstacle_10family_v0.json')
        if not Path(tf).exists():
            tf = str(REPO / 'data/sim/metadata/reset_templates_v0.json')

    with open(tf) as f:
        all_templates = json.load(f)

    # Group by family
    by_family = {}
    for t in all_templates:
        fam = t.get('family', 'unknown')
        by_family.setdefault(fam, []).append(t)

    print(f'Templates: {len(all_templates)} from {len(by_family)} families')

    # MPPI config
    mppi_cfg = {
        'horizon': args.horizon,
        'num_samples': args.num_samples,
        'init_std': args.init_std,
        'temperature': args.temperature,
    }

    cost_w = CostWeights()

    # Create env
    env = MujocoPushEnv(shape_type='T', control_dt=args.control_dt,
                         max_speed_mps=args.speed)

    # Collect episodes
    rng = np.random.RandomState(args.seed)
    metadata = []
    total_transitions = 0

    for ep_idx in range(args.num_episodes):
        # Pick template round-robin from families
        fam_keys = list(by_family.keys())
        fam = fam_keys[ep_idx % len(fam_keys)]
        templates = by_family[fam]
        template = templates[ep_idx % len(templates)]

        t0 = time.time()
        try:
            steps = collect_one_episode(env, mppi_cfg, cost_w, args.max_steps, template)
            elapsed = time.time() - t0

            ep_id = str(uuid.uuid4())[:12]
            ep_path = ep_dir / f'{ep_id}.npz'

            # Convert to arrays
            T = len(steps)
            arrays = {}
            for key in steps[0]:
                vals = [s[key] for s in steps]
                if isinstance(vals[0], (list, np.ndarray)):
                    arrays[key] = np.array(vals, dtype=np.float32)
                elif isinstance(vals[0], dict):
                    continue
                else:
                    arrays[key] = np.array(vals, dtype=np.float32)

            # Add scalar metadata arrays
            arrays['schema_version'] = np.array(['0.1'], dtype=object)

            np.savez(ep_path, **arrays)

            final_dist = np.linalg.norm(
                np.array(steps[-1]['next_object_poses'][:2]) -
                np.array(steps[-1]['goal_pose'][:2])
            )
            success = final_dist < 0.02

            meta = {
                'episode_id': ep_id,
                'family': fam,
                'num_transitions': T,
                'success': bool(success),
                'final_dist': float(final_dist),
                'template_id': template.get('template_id', ep_idx),
                'source': 'v2_collector',
                'schema_version': '0.1',
                'speed_mps': args.speed,
                'temperature': args.temperature,
            }
            metadata.append(meta)

            # Append to jsonl
            with open(meta_dir / 'episodes.jsonl', 'a') as f:
                f.write(json.dumps(meta) + '\n')

            total_transitions += T
            status = '✅' if success else '❌'
            print(f'  [{ep_idx+1}/{args.num_episodes}] {status} {fam} T={T} dist={final_dist:.4f}m time={elapsed:.1f}s')

        except Exception as e:
            print(f'  [{ep_idx+1}/{args.num_episodes}] ❌ ERROR: {e}')

    print(f'\nDone: {len(metadata)} episodes, {total_transitions} transitions')
    print(f'Output: {out_dir}')


if __name__ == '__main__':
    main()
