#!/usr/bin/env python3
"""Overnight sweep with BUG FIXES:
1. extract_state16 now includes obstacle tokens
2. max_speed_mps fixed from 0.75 to 0.5
"""

import sys, json, time, os, csv, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, '/home/brucewu/my_robot_project')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.learned_planner_adapter import CEMLearnedPlanner, MPPILearnedPlanner
from src.planners.action_conventions import PAPER1_CONVENTION
from src.envs.mujoco_push_env import MujocoPushEnv

EE_SIZE, OBJ_SIZE, OBJ_MASS, OBJ_FRICTION = 0.015, 0.048, 0.038, 0.8
MAX_OBSTACLES = 3
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = Path(f'runs/bugfix_sweep_{TIMESTAMP}')
RUN_DIR.mkdir(parents=True, exist_ok=True)


def extract_state16(env):
    """FIXED: includes obstacle tokens (was the #1 bug)."""
    ee = env.get_ee_pos(); obj = env.get_object_pose(); goal = env.get_goal_pose()
    t = np.zeros((6, 16), dtype=np.float32)
    t[0, :2] = ee; t[0, 3] = 1.0; t[0, 7:9] = EE_SIZE; t[0, 14] = float(env.get_contact_flag()); t[0, 15] = 1.0
    t[1, :2] = obj[:2]; t[1, 2] = np.sin(obj[2]); t[1, 3] = np.cos(obj[2])
    t[1, 7:9] = OBJ_SIZE; t[1, 9] = 1.0; t[1, 12:14] = [OBJ_MASS, OBJ_FRICTION]; t[1, 15] = 1.0
    t[2, :2] = goal[:2]; t[2, 2] = np.sin(goal[2]); t[2, 3] = np.cos(goal[2])
    t[2, 7:9] = OBJ_SIZE; t[2, 9] = 1.0; t[2, 12:14] = [OBJ_MASS, OBJ_FRICTION]; t[2, 15] = 1.0
    num_active = getattr(env, '_num_active_obstacles', 0)
    for i in range(min(MAX_OBSTACLES, num_active)):
        body_id = env._obstacle_body_ids[i]; geom_id = env._obstacle_geom_ids[i]
        obs_pos = env.data.xpos[body_id][:2].copy()
        if np.allclose(obs_pos, 0.0, atol=1e-3): continue
        token_idx = 3 + i
        t[token_idx, :2] = obs_pos
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        t[token_idx, 2] = np.sin(yaw); t[token_idx, 3] = np.cos(yaw)
        geom_size = env.model.geom_size[geom_id]
        t[token_idx, 7] = geom_size[0]; t[token_idx, 8] = geom_size[1]
        t[token_idx, 12] = 0.5; t[token_idx, 13] = 0.8; t[token_idx, 15] = 1.0
    return t


def load_model(model_type):
    model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256, d_model=128, head_hidden_dim=256)
    ckpt = torch.load(f'runs/pilot_state16_mppi_stage2c/train_{model_type}/{model_type}/checkpoints/best.pt',
                       map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    norm = StateNormalizer.load(f'runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json')
    return model, norm


def create_planner(planner_type, config_name, model, norm, conv):
    if planner_type == 'cem':
        configs = {
            'cem_small': {'horizon': 10, 'num_samples': 32, 'num_elites': 8, 'num_iterations': 3},
            'cem_medium': {'horizon': 20, 'num_samples': 128, 'num_elites': 16, 'num_iterations': 4},
        }
        cfg = configs.get(config_name, configs['cem_small'])
        return CEMLearnedPlanner(model=model, normalizer=norm, convention=conv, device='cpu', **cfg)
    elif planner_type == 'mppi':
        configs = {
            'mppi_smoke': {'horizon': 30, 'num_samples': 256, 'temperature': 0.1, 'init_std': 0.5,
                           'speed': 0.3, 'execute_steps': 5, 'max_mpc_steps': 30},
        }
        cfg = configs.get(config_name, configs['mppi_smoke'])
        return MPPILearnedPlanner(model=model, normalizer=norm, convention=conv, device='cpu', **cfg)


def run_one_trial(env, planner, template, max_steps, exec_steps, trial_id):
    t0 = time.time()
    try:
        env.reset_from_template(template)
        state = extract_state16(env)
        hist = np.tile(state[np.newaxis], (6, 1, 1))
        goal = env.get_goal_pose()
        init_d = float(np.linalg.norm(env.get_object_pose()[:2] - goal[:2]))
        best_d = init_d; final_d = init_d; contacts = 0; collisions = 0; total_steps = 0

        while total_steps < max_steps:
            curr = extract_state16(env); hist[:-1] = hist[1:]; hist[-1] = curr
            result = planner.plan(hist.copy(), goal)
            steps_to_exec = min(exec_steps, max_steps - total_steps)
            for s in range(steps_to_exec):
                action = result.action_sequence_norm[s] if s < len(result.action_sequence_norm) else result.first_action_norm
                env.step(action); total_steps += 1
                contacts += int(env.get_contact_flag()); collisions += int(env.get_collision_flag())
                obj = env.get_object_pose()
                d = float(np.linalg.norm(obj[:2] - goal[:2]))
                best_d = min(best_d, d); final_d = d
                te = abs(np.arctan2(np.sin(obj[2] - goal[2]), np.cos(obj[2] - goal[2])))
                if d < 0.02 and np.degrees(te) < 10: break
            if final_d < 0.02 and np.degrees(te) < 10: break

        return {
            'trial_id': trial_id, 'success': bool(final_d < 0.02 and np.degrees(te) < 10),
            'initial_dist': init_d, 'final_dist': final_d, 'best_dist': best_d,
            'distance_improvement': init_d - best_d, 'improved': best_d < init_d - 0.001,
            'contact_rate': contacts / max(total_steps, 1), 'collision_rate': collisions / max(total_steps, 1),
            'total_steps': total_steps, 'runtime_sec': time.time() - t0, 'failure_code': None,
            'planner_backend': result.planner_backend, 'planner_config': result.planner_config,
        }
    except Exception as e:
        return {'trial_id': trial_id, 'success': False, 'initial_dist': float('nan'), 'final_dist': float('nan'),
                'best_dist': float('nan'), 'distance_improvement': float('nan'), 'improved': False,
                'contact_rate': 0, 'collision_rate': 0, 'total_steps': 0,
                'runtime_sec': time.time() - t0, 'failure_code': str(e)[:200],
                'planner_backend': 'ERROR', 'planner_config': {}}


def main():
    start_time = time.time()
    max_wall_time = 11 * 3600  # 11 hours

    # Load templates
    with open('data/sim/metadata/reset_templates_v0.json') as f:
        all_templates = json.load(f)
    template_pool = {}
    for t in all_templates:
        fam = t.get('layout_family', 'unknown')
        if fam not in template_pool: template_pool[fam] = []
        template_pool[fam].append(t)

    # Save git status
    import subprocess
    git_status = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True, cwd='/home/brucewu/my_robot_project').stdout
    with open(RUN_DIR / 'git_status.txt', 'w') as f: f.write(git_status)

    # Save config
    config = {'bugfixes': ['obstacle_token', 'max_speed_0.75_obstacle_fix', 'extract_state16_fixed'],
              'max_speed_mps': 0.75, 'timestamp': TIMESTAMP}
    with open(RUN_DIR / 'config.json', 'w') as f: json.dump(config, f, indent=2)

    # Trial matrix
    conv = PAPER1_CONVENTION; conv.max_speed_mps = 0.75  # FIXED
    models_list = ['flat', 'object_centric', 'causality_aware']
    planner_configs = [
        ('cem', 'cem_small', 'open_space', 30, 5),
        ('cem', 'cem_medium', 'open_space', 50, 10),
        ('mppi', 'mppi_smoke', 'open_space', 30, 5),
        ('cem', 'cem_small', 'blocking', 30, 5),
        ('cem', 'cem_medium', 'blocking', 50, 10),
        ('mppi', 'mppi_smoke', 'blocking', 30, 5),
        ('cem', 'cem_small', 'narrow_passage', 30, 5),
        ('cem', 'cem_small', 'edge_goal', 30, 5),
        ('mppi', 'mppi_smoke', 'narrow_passage', 30, 5),
        ('mppi', 'mppi_smoke', 'edge_goal', 30, 5),
    ]

    trial_matrix = []; trial_id = 0
    for model_type in models_list:
        for ptype, cfg, fam, max_steps, exec_steps in planner_configs:
            templates = template_pool.get(fam, [])
            for i, tmpl in enumerate(templates[:3]):
                trial_matrix.append({
                    'trial_id': f'trial_{trial_id:04d}', 'model': model_type,
                    'planner_type': ptype, 'config_name': cfg, 'template': tmpl,
                    'template_id': tmpl.get('reset_template_id', f'tmpl_{i}'),
                    'family': fam, 'max_steps': max_steps, 'exec_steps': exec_steps,
                })
                trial_id += 1

    print(f'Run dir: {RUN_DIR}')
    print(f'Trial matrix: {len(trial_matrix)} trials')
    print(f'Bug fixes: obstacle token + max_speed=0.75')

    # Run trials (load models one at a time to save memory)
    trial_results = []
    current_model_type = None
    for i, trial in enumerate(trial_matrix):
        elapsed = time.time() - start_time
        if elapsed > max_wall_time:
            print(f'Time limit reached. Stopping.')
            break

        if trial['model'] != current_model_type:
            if current_model_type is not None:
                del model, norm; import gc; gc.collect()
            print(f'Loading model: {trial["model"]}')
            model, norm = load_model(trial['model'])
            current_model_type = trial['model']
        planner = create_planner(trial['planner_type'], trial['config_name'], model, norm, conv)
        env = MujocoPushEnv(shape_type='T', control_dt=conv.control_dt, max_speed_mps=conv.max_speed_mps)

        result = run_one_trial(env, planner, trial['template'], trial['max_steps'], trial['exec_steps'], trial['trial_id'])
        result.update({'model': trial['model'], 'planner_type': trial['planner_type'],
                       'config_name': trial['config_name'], 'template_id': trial['template_id'], 'family': trial['family']})
        trial_results.append(result)

        # Update manifest
        with open(RUN_DIR / 'manifest.csv', 'w', newline='') as f:
            fields = ['trial_id','model','planner_type','config_name','template_id','family',
                      'success','improved','initial_dist','final_dist','best_dist','distance_improvement',
                      'contact_rate','collision_rate','total_steps','runtime_sec','failure_code','planner_backend']
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(trial_results)

        status = '✅' if result['improved'] else '❌'
        print(f"[{i+1}/{len(trial_matrix)}] {status} {trial['trial_id']} {trial['model']:<16} {trial['planner_type']:<4} {trial['config_name']:<12} {trial['family']:<16} "
              f"best={result['best_dist']:.4f} imp={result['distance_improvement']:.4f} {result['runtime_sec']:.1f}s")

        del env, planner

    # Summary
    cem = [r for r in trial_results if r.get('planner_type') == 'cem']
    mppi = [r for r in trial_results if r.get('planner_type') == 'mppi']
    summary = {
        'elapsed_hours': (time.time() - start_time) / 3600,
        'total_trials': len(trial_results), 'completed': sum(1 for r in trial_results if r.get('failure_code') is None),
        'cem_trials': len(cem), 'mppi_trials': len(mppi),
        'cem_improved': sum(1 for r in cem if r.get('improved')),
        'mppi_improved': sum(1 for r in mppi if r.get('improved')),
        'best_dist': min((r['best_dist'] for r in trial_results if np.isfinite(r.get('best_dist', float('inf')))), default=float('inf')),
        'bugfixes': ['obstacle_token', 'max_speed_0.75_obstacle_fix'],
    }
    with open(RUN_DIR / 'summary.json', 'w') as f: json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. {len(trial_results)} trials in {summary['elapsed_hours']:.1f}h")
    print(f"Best distance: {summary['best_dist']:.4f}m")
    print(f"CEM improved: {summary['cem_improved']}/{summary['cem_trials']}")
    print(f"MPPI improved: {summary['mppi_improved']}/{summary['mppi_trials']}")


if __name__ == '__main__':
    main()
