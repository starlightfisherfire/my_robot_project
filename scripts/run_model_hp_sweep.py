#!/usr/bin/env python3
"""Hyperparameter sweep for all 3 models × CEM + MPPI.

Goal: Find the best achievable performance for each model by sweeping
planner hyperparameters (horizon, samples, iterations, etc.)

Usage:
    nohup python scripts/run_model_hp_sweep.py > runs/model_hp_sweep.log 2>&1 &
"""

import sys, json, time, os, csv
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
RUN_DIR = Path(f'runs/model_hp_sweep_{TIMESTAMP}')
RUN_DIR.mkdir(parents=True, exist_ok=True)


def extract_state16(env):
    """FIXED: includes obstacle tokens."""
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


def create_planner(planner_type, cfg, model, norm, conv):
    if planner_type == 'cem':
        return CEMLearnedPlanner(model=model, normalizer=norm, convention=conv, device='cpu', **cfg)
    elif planner_type == 'mppi':
        return MPPILearnedPlanner(model=model, normalizer=norm, convention=conv, device='cpu', **cfg)


def run_trial(env, planner, template, max_steps, exec_steps):
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
        'initial_dist': init_d, 'final_dist': final_d, 'best_dist': best_d,
        'distance_improvement': init_d - best_d, 'improved': best_d < init_d - 0.001,
        'contact_rate': contacts / max(total_steps, 1), 'collision_rate': collisions / max(total_steps, 1),
        'total_steps': total_steps, 'success': bool(final_d < 0.02 and np.degrees(te) < 10),
    }


def main():
    start_time = time.time()
    max_wall_time = 11 * 3600

    conv = PAPER1_CONVENTION; conv.max_speed_mps = 0.75

    # Load templates (use 1 per family for speed, focus on planner HP)
    with open('data/sim/metadata/reset_templates_v0.json') as f:
        all_templates = json.load(f)
    # Pick representative templates: 1 open_space, 1 blocking, 1 narrow_passage, 1 edge_goal
    selected = []
    for fam in ['open_space', 'blocking', 'narrow_passage', 'edge_goal']:
        for t in all_templates:
            if t.get('layout_family') == fam:
                selected.append(t)
                break
    tids = [t.get('reset_template_id','?')[-8:] for t in selected]
    print(f'Selected {len(selected)} templates: {tids}')

    # Sweep configs
    cem_configs = {
        'cem_xs':     {'horizon': 5,  'num_samples': 16,  'num_elites': 4,  'num_iterations': 2, 'init_std': 0.3},
        'cem_small':  {'horizon': 10, 'num_samples': 32,  'num_elites': 8,  'num_iterations': 3, 'init_std': 0.3},
        'cem_medium': {'horizon': 20, 'num_samples': 128, 'num_elites': 16, 'num_iterations': 4, 'init_std': 0.3},
        'cem_large':  {'horizon': 30, 'num_samples': 256, 'num_elites': 32, 'num_iterations': 5, 'init_std': 0.3},
        'cem_xlarge': {'horizon': 50, 'num_samples': 512, 'num_elites': 64, 'num_iterations': 6, 'init_std': 0.3},
        'cem_deep':   {'horizon': 20, 'num_samples': 64,  'num_elites': 16, 'num_iterations': 8, 'init_std': 0.3},
        'cem_wide':   {'horizon': 20, 'num_samples': 256, 'num_elites': 32, 'num_iterations': 3, 'init_std': 0.3},
        'cem_tight':  {'horizon': 15, 'num_samples': 64,  'num_elites': 16, 'num_iterations': 5, 'init_std': 0.15},
    }

    mppi_configs = {
        'mppi_xs':     {'horizon': 15, 'num_samples': 64,  'temperature': 0.1, 'init_std': 0.5, 'speed': 0.3, 'execute_steps': 3, 'max_mpc_steps': 20},
        'mppi_small':  {'horizon': 30, 'num_samples': 256, 'temperature': 0.1, 'init_std': 0.5, 'speed': 0.3, 'execute_steps': 5, 'max_mpc_steps': 30},
        'mppi_medium': {'horizon': 60, 'num_samples': 1024,'temperature': 0.1, 'init_std': 0.5, 'speed': 0.3, 'execute_steps': 10,'max_mpc_steps': 80},
        'mppi_large':  {'horizon': 100,'num_samples': 2048,'temperature': 0.1, 'init_std': 0.5, 'speed': 0.3, 'execute_steps': 10,'max_mpc_steps': 100},
        'mppi_cold':   {'horizon': 60, 'num_samples': 1024,'temperature': 0.02,'init_std': 0.5, 'speed': 0.3, 'execute_steps': 10,'max_mpc_steps': 80},
        'mppi_hot':    {'horizon': 60, 'num_samples': 1024,'temperature': 0.5, 'init_std': 0.5, 'speed': 0.3, 'execute_steps': 10,'max_mpc_steps': 80},
        'mppi_tight':  {'horizon': 60, 'num_samples': 1024,'temperature': 0.1, 'init_std': 0.2, 'speed': 0.3, 'execute_steps': 10,'max_mpc_steps': 80},
        'mppi_fast':   {'horizon': 40, 'num_samples': 512, 'temperature': 0.1, 'init_std': 0.5, 'speed': 0.5, 'execute_steps': 5, 'max_mpc_steps': 50},
    }

    models_list = ['flat', 'object_centric', 'causality_aware']

    # Build trial matrix
    trial_matrix = []
    tid = 0
    for model_type in models_list:
        for cfg_name, cfg in cem_configs.items():
            for tmpl in selected:
                trial_matrix.append({
                    'trial_id': f't{tid:04d}', 'model': model_type,
                    'planner_type': 'cem', 'cfg_name': cfg_name, 'cfg': cfg,
                    'template': tmpl, 'template_id': tmpl['reset_template_id'],
                    'family': tmpl['layout_family'], 'max_steps': 50, 'exec_steps': 10,
                })
                tid += 1
        for cfg_name, cfg in mppi_configs.items():
            for tmpl in selected:
                trial_matrix.append({
                    'trial_id': f't{tid:04d}', 'model': model_type,
                    'planner_type': 'mppi', 'cfg_name': cfg_name, 'cfg': cfg,
                    'template': tmpl, 'template_id': tmpl['reset_template_id'],
                    'family': tmpl['layout_family'], 'max_steps': 80, 'exec_steps': cfg['execute_steps'],
                })
                tid += 1

    print(f'Trial matrix: {len(trial_matrix)} trials')
    print(f'  Models: {len(models_list)} × CEM configs: {len(cem_configs)} × Templates: {len(selected)} = {len(models_list)*len(cem_configs)*len(selected)}')
    print(f'  Models: {len(models_list)} × MPPI configs: {len(mppi_configs)} × Templates: {len(selected)} = {len(models_list)*len(mppi_configs)*len(selected)}')

    # Save config
    with open(RUN_DIR / 'sweep_config.json', 'w') as f:
        json.dump({'cem_configs': cem_configs, 'mppi_configs': mppi_configs,
                   'models': models_list, 'max_speed_mps': 0.75, 'obstacle_fix': True}, f, indent=2, default=str)

    # Run
    fields = ['trial_id','model','planner_type','cfg_name','template_id','family',
              'success','improved','initial_dist','final_dist','best_dist','distance_improvement',
              'contact_rate','collision_rate','total_steps','runtime_sec','planner_backend']
    trial_results = []
    current_model_type = None

    for i, trial in enumerate(trial_matrix):
        elapsed = time.time() - start_time
        if elapsed > max_wall_time:
            print(f'Time limit reached.')
            break

        # Load model on demand
        if trial['model'] != current_model_type:
            if current_model_type is not None:
                del model, norm; import gc; gc.collect()
            print(f'Loading model: {trial["model"]}')
            model, norm = load_model(trial['model'])
            current_model_type = trial['model']

        planner = create_planner(trial['planner_type'], trial['cfg'], model, norm, conv)
        env = MujocoPushEnv(shape_type='T', control_dt=conv.control_dt, max_speed_mps=conv.max_speed_mps)

        t0 = time.time()
        try:
            result = run_trial(env, planner, trial['template'], trial['max_steps'], trial['exec_steps'])
            result['runtime_sec'] = time.time() - t0
            result['planner_backend'] = 'CEM_BEST_LEARNED_ROLLOUT' if trial['planner_type'] == 'cem' else 'MPPI_LEARNED_ROLLOUT'
        except Exception as e:
            result = {'success': False, 'initial_dist': float('nan'), 'final_dist': float('nan'),
                      'best_dist': float('nan'), 'distance_improvement': float('nan'), 'improved': False,
                      'contact_rate': 0, 'collision_rate': 0, 'total_steps': 0,
                      'runtime_sec': time.time() - t0, 'planner_backend': 'ERROR'}
        result.update({'trial_id': trial['trial_id'], 'model': trial['model'],
                       'planner_type': trial['planner_type'], 'cfg_name': trial['cfg_name'],
                       'template_id': trial['template_id'], 'family': trial['family']})
        trial_results.append(result)

        # Write manifest
        with open(RUN_DIR / 'manifest.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(trial_results)

        status = '✅' if result.get('improved') else '❌'
        print(f"[{i+1}/{len(trial_matrix)}] {status} {trial['model']:<16} {trial['planner_type']:<4} {trial['cfg_name']:<12} {trial['family']:<16} "
              f"best={result.get('best_dist', float('nan')):.4f} {result.get('runtime_sec', 0):.1f}s")

        del env, planner

    # Summary
    with open(RUN_DIR / 'summary.json', 'w') as f:
        json.dump({'elapsed_hours': (time.time()-start_time)/3600, 'total_trials': len(trial_results),
                   'obstacle_fix': True, 'max_speed_mps': 0.75}, f, indent=2)

    print(f"\nDone. {len(trial_results)} trials in {(time.time()-start_time)/3600:.1f}h")


if __name__ == '__main__':
    main()
