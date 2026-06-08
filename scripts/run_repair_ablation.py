#!/usr/bin/env python3
"""A/B verification: simplified vs full cost, with early stop.

Max 36 trials. Tests whether full cost + early stop improves final_dist/drift.
"""
import sys, json, time, os, csv
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/home/brucewu/my_robot_project')
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.learned_planner_adapter import CEMLearnedPlanner, MPPILearnedPlanner
from src.planners.action_conventions import PAPER1_CONVENTION
from src.planners.cost_functions import CostWeights
from src.envs.mujoco_push_env import MujocoPushEnv

EE_SIZE, OBJ_SIZE, OBJ_MASS, OBJ_FRICTION = 0.015, 0.048, 0.038, 0.8
MAX_OBSTACLES = 3
RUN_DIR = Path('runs/repair_ablation_' + time.strftime('%Y%m%d_%H%M%S'))
RUN_DIR.mkdir(parents=True, exist_ok=True)


def extract_state16(env):
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
        token_idx = 3 + i; t[token_idx, :2] = obs_pos
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        t[token_idx, 2] = np.sin(yaw); t[token_idx, 3] = np.cos(yaw)
        geom_size = env.model.geom_size[geom_id]
        t[token_idx, 7] = geom_size[0]; t[token_idx, 8] = geom_size[1]
        t[token_idx, 12] = 0.5; t[token_idx, 13] = 0.8; t[token_idx, 15] = 1.0
    return t


def get_obstacle_info(env):
    """Extract obstacle positions and radii for full cost."""
    positions = []; radii = []
    num_active = getattr(env, '_num_active_obstacles', 0)
    for i in range(min(MAX_OBSTACLES, num_active)):
        body_id = env._obstacle_body_ids[i]
        obs_pos = env.data.xpos[body_id][:2].copy()
        if np.allclose(obs_pos, 0.0, atol=1e-3): continue
        geom_id = env._obstacle_geom_ids[i]
        geom_size = env.model.geom_size[geom_id]
        positions.append(obs_pos)
        radii.append(max(geom_size[0], geom_size[1]))
    if positions:
        return np.array(positions), np.array(radii)
    return None, None


def load_model(model_type):
    model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256, d_model=128, head_hidden_dim=256)
    ckpt = torch.load(f'runs/pilot_state16_mppi_stage2c/train_{model_type}/{model_type}/checkpoints/best.pt',
                       map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    norm = StateNormalizer.load(f'runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json')
    return model, norm


def run_trial(env, planner, template, max_steps, exec_steps=10, stop_pos=0.002, stop_theta=10.0):
    """Run trial with early stop."""
    env.reset_from_template(template)
    state = extract_state16(env)
    hist = np.tile(state[np.newaxis], (6, 1, 1))
    goal = env.get_goal_pose()
    init_d = float(np.linalg.norm(env.get_object_pose()[:2] - goal[:2]))
    best_d = init_d; best_step = 0; final_d = init_d
    contacts = 0; collisions = 0; total_steps = 0
    ever_success_pos = False; ever_success_pose = False
    early_stopped = False; early_stop_reason = ''

    while total_steps < max_steps:
        curr = extract_state16(env); hist[:-1] = hist[1:]; hist[-1] = curr
        result = planner.plan(hist.copy(), goal)
        for s in range(min(exec_steps, max_steps - total_steps)):
            action = result.action_sequence_norm[s] if s < len(result.action_sequence_norm) else result.first_action_norm
            env.step(action); total_steps += 1
            contacts += int(env.get_contact_flag()); collisions += int(env.get_collision_flag())
            obj = env.get_object_pose()
            d = float(np.linalg.norm(obj[:2] - goal[:2]))
            te = abs(np.arctan2(np.sin(obj[2] - goal[2]), np.cos(obj[2] - goal[2])))
            if d < best_d: best_d = d; best_step = total_steps
            if d < 0.002: ever_success_pos = True
            if d < 0.002 and np.degrees(te) < 10: ever_success_pose = True
            # Early stop
            if d < stop_pos and np.degrees(te) < stop_theta:
                early_stopped = True; early_stop_reason = f'pos={d:.4f}m_theta={np.degrees(te):.1f}deg'
                break
        if early_stopped: break

    final_obj = env.get_object_pose()
    final_d = float(np.linalg.norm(final_obj[:2] - goal[:2]))
    final_te = abs(np.arctan2(np.sin(final_obj[2] - goal[2]), np.cos(final_obj[2] - goal[2])))
    success = bool(final_d < 0.002 and np.degrees(final_te) < 10.0)

    return {
        'success': success, 'ever_success_pos_only': ever_success_pos,
        'ever_success_pose': ever_success_pose,
        'initial_dist': init_d, 'best_dist': best_d, 'final_dist': final_d,
        'drift_after_best': final_d - best_d, 'best_step': best_step,
        'final_theta_error_deg': float(np.degrees(final_te)),
        'contact_rate': contacts / max(total_steps, 1),
        'collision_rate': collisions / max(total_steps, 1),
        'total_steps': total_steps, 'early_stop_triggered': early_stopped,
        'early_stop_reason': early_stop_reason,
    }


def main():
    conv = PAPER1_CONVENTION; conv.max_speed_mps = 0.3  # Fair speed
    weights = CostWeights(w_pos=10.0, w_theta=2.0, w_reach=5.0, w_no_contact=2.0,
                          w_push_alignment=1.0, w_collision=20.0, w_collision_step=1.0,
                          w_proximity=5.0, w_action=0.05, w_smooth=0.1)

    with open('data/sim/metadata/reset_templates_v0.json') as f:
        all_templates = json.load(f)
    selected = []
    for fam in ['open_space', 'blocking', 'narrow_passage', 'edge_goal']:
        fam_t = [t for t in all_templates if t.get('layout_family') == fam]
        selected.extend(fam_t[:2])  # 2 per family = 8 templates

    models_list = ['flat', 'object_centric', 'causality_aware']
    configs = [
        ('cem', 'simplified', {'horizon': 50, 'num_samples': 128, 'num_elites': 16, 'num_iterations': 4, 'init_std': 0.3}),
        ('cem', 'full', {'horizon': 50, 'num_samples': 128, 'num_elites': 16, 'num_iterations': 4, 'init_std': 0.3}),
        ('mppi', 'simplified', {'horizon': 30, 'num_samples': 256, 'temperature': 0.1, 'init_std': 0.5, 'speed': 0.3, 'execute_steps': 10, 'max_mpc_steps': 10}),
        ('mppi', 'full', {'horizon': 30, 'num_samples': 256, 'temperature': 0.1, 'init_std': 0.5, 'speed': 0.3, 'execute_steps': 10, 'max_mpc_steps': 10}),
    ]

    trial_matrix = []; tid = 0
    for model_type in models_list:
        for planner_type, cost_mode, cfg in configs:
            for tmpl in selected:
                trial_matrix.append({'tid': f't{tid:03d}', 'model': model_type,
                    'planner': planner_type, 'cost_mode': cost_mode, 'cfg': cfg,
                    'template': tmpl, 'family': tmpl.get('layout_family', '?')})
                tid += 1

    print(f'A/B verification: {len(trial_matrix)} trials')
    print(f'  Models: {models_list}')
    print(f'  Cost modes: simplified vs full')
    fams = [t.get('layout_family','?')[-4:] for t in selected]
    print(f'  Templates: {len(selected)} ({fams})')
    print(f'  max_speed: {conv.max_speed_mps}')
    print(f'  early_stop: 2mm + 10°')
    print(f'  Run dir: {RUN_DIR}')

    fields = ['tid','model','planner','cost_mode','family','template_id',
              'success','ever_success_pos_only','ever_success_pose',
              'initial_dist','best_dist','final_dist','drift_after_best','best_step',
              'final_theta_error_deg','contact_rate','collision_rate',
              'total_steps','early_stop_triggered','early_stop_reason','runtime_sec']
    results = []; current_model = None

    for i, trial in enumerate(trial_matrix):
        if trial['model'] != current_model:
            if current_model: del model, norm; import gc; gc.collect()
            model, norm = load_model(trial['model'])
            current_model = trial['model']
            print(f'Loaded model: {current_model}')

        obs_pos, obs_rad = None, None
        env = MujocoPushEnv(shape_type='T', control_dt=conv.control_dt, max_speed_mps=conv.max_speed_mps)

        if trial['cost_mode'] == 'full':
            env.reset_from_template(trial['template'])
            obs_pos, obs_rad = get_obstacle_info(env)
            planner_args = {'model': model, 'normalizer': norm, 'convention': conv, 'device': 'cpu',
                           'cost_mode': 'full', 'cost_weights': weights,
                           'obstacle_positions': obs_pos, 'obstacle_radii': obs_rad}
        else:
            planner_args = {'model': model, 'normalizer': norm, 'convention': conv, 'device': 'cpu',
                           'cost_mode': 'simplified'}

        if trial['planner'] == 'cem':
            planner = CEMLearnedPlanner(**planner_args, **trial['cfg'])
        else:
            planner = MPPILearnedPlanner(**planner_args, **trial['cfg'])

        t0 = time.time()
        exec_s = trial['cfg'].get('execute_steps', 10)
        result = run_trial(env, planner, trial['template'], max_steps=100, exec_steps=exec_s, stop_pos=0.002, stop_theta=10.0)
        result['runtime_sec'] = time.time() - t0
        result.update({'tid': trial['tid'], 'model': trial['model'], 'planner': trial['planner'],
                       'cost_mode': trial['cost_mode'], 'family': trial['family'],
                       'template_id': trial['template'].get('reset_template_id', '?')})
        results.append(result)

        with open(RUN_DIR / 'manifest.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(results)

        s = '✅' if result['success'] else ('📈' if result['ever_success_pose'] else '❌')
        print(f"[{i+1}/{len(trial_matrix)}] {s} {trial['model']:<16} {trial['planner']:<4} {trial['cost_mode']:<12} {trial['family']:<12} "
              f"best={result['best_dist']:.4f} final={result['final_dist']:.4f} drift={result['drift_after_best']:.4f} stop={result['early_stop_triggered']}")
        del env, planner

    # Summary
    simp = [r for r in results if r['cost_mode'] == 'simplified']
    full = [r for r in results if r['cost_mode'] == 'full']
    summary = {
        'simplified': {'trials': len(simp), 'success': sum(r['success'] for r in simp),
                       'ever_success': sum(r['ever_success_pose'] for r in simp),
                       'mean_best': np.mean([r['best_dist'] for r in simp]),
                       'mean_final': np.mean([r['final_dist'] for r in simp]),
                       'mean_drift': np.mean([r['drift_after_best'] for r in simp])},
        'full': {'trials': len(full), 'success': sum(r['success'] for r in full),
                 'ever_success': sum(r['ever_success_pose'] for r in full),
                 'mean_best': np.mean([r['best_dist'] for r in full]),
                 'mean_final': np.mean([r['final_dist'] for r in full]),
                 'mean_drift': np.mean([r['drift_after_best'] for r in full])},
    }
    with open(RUN_DIR / 'summary.json', 'w') as f: json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Simplified: best={summary['simplified']['mean_best']:.4f} final={summary['simplified']['mean_final']:.4f} drift={summary['simplified']['mean_drift']:.4f} success={summary['simplified']['success']}/{summary['simplified']['trials']}")
    print(f"Full:       best={summary['full']['mean_best']:.4f} final={summary['full']['mean_final']:.4f} drift={summary['full']['mean_drift']:.4f} success={summary['full']['success']}/{summary['full']['trials']}")
    print(f"\nRun dir: {RUN_DIR}")

if __name__ == '__main__':
    main()
