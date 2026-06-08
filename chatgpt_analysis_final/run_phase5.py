#!/usr/bin/env python3
"""Phase 5: Open-space closed-loop with corrected action convention.

CEM config: horizon=10, samples=32, elites=4, iterations=2, speed=0.75
"""

import sys, json, time, os
sys.path.insert(0, '/home/brucewu/my_robot_project')
import numpy as np
import torch
import yaml

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.learned_planner_adapter import CEMLearnedPlanner
from src.planners.action_conventions import PAPER1_CONVENTION
from src.envs.mujoco_push_env import MujocoPushEnv
from src.metrics.topology_geometry import compute_all_metrics

EE_SIZE, OBJ_SIZE, OBJ_MASS, OBJ_FRICTION = 0.015, 0.048, 0.038, 0.8

def extract_state16(env):
    ee = env.get_ee_pos(); obj = env.get_object_pose(); goal = env.get_goal_pose()
    t = np.zeros((6, 16), dtype=np.float32)
    t[0, :2] = ee; t[0, 3] = 1.0; t[0, 7:9] = EE_SIZE; t[0, 14] = float(env.get_contact_flag()); t[0, 15] = 1.0
    t[1, :2] = obj[:2]; t[1, 2] = np.sin(obj[2]); t[1, 3] = np.cos(obj[2])
    t[1, 7:9] = OBJ_SIZE; t[1, 9] = 1.0; t[1, 12:14] = [OBJ_MASS, OBJ_FRICTION]; t[1, 15] = 1.0
    t[2, :2] = goal[:2]; t[2, 2] = np.sin(goal[2]); t[2, 3] = np.cos(goal[2])
    t[2, 7:9] = OBJ_SIZE; t[2, 9] = 1.0; t[2, 12:14] = [OBJ_MASS, OBJ_FRICTION]; t[2, 15] = 1.0
    return t

def run_one(env, planner, template, max_steps, exec_steps=10):
    env.reset_from_template(template)
    state = extract_state16(env)
    hist = np.tile(state[np.newaxis], (6, 1, 1))
    goal = env.get_goal_pose()
    init_d = float(np.linalg.norm(env.get_object_pose()[:2] - goal[:2]))
    best_d = init_d; final_d = init_d; contacts = 0; total_steps = 0
    first_action_norm = None; planned_costs = []; zero_costs = []
    
    while total_steps < max_steps:
        curr = extract_state16(env); hist[:-1] = hist[1:]; hist[-1] = curr
        result = planner.plan(hist.copy(), goal)
        if first_action_norm is None:
            first_action_norm = result.first_action_norm.copy()
        planned_costs.append(result.planned_cost)
        zero_costs.append(result.zero_cost)
        
        steps_to_exec = min(exec_steps, max_steps - total_steps)
        for s in range(steps_to_exec):
            action = result.action_sequence_norm[s] if s < len(result.action_sequence_norm) else result.first_action_norm
            env.step(action)
            total_steps += 1
            contacts += int(env.get_contact_flag())
            obj = env.get_object_pose()
            d = float(np.linalg.norm(obj[:2] - goal[:2]))
            best_d = min(best_d, d); final_d = d
            te = abs(np.arctan2(np.sin(obj[2] - goal[2]), np.cos(obj[2] - goal[2])))
            if d < 0.02 and np.degrees(te) < 10:
                break
        
        if final_d < 0.02 and np.degrees(te) < 10:
            break
    
    ee = env.get_ee_pos(); obj = env.get_object_pose()
    return {
        'success': bool(final_d < 0.02 and np.degrees(te) < 10),
        'initial_dist': init_d, 'final_dist': final_d, 'best_dist': best_d,
        'improved': best_d < init_d - 0.001,
        'total_steps': total_steps, 'contact_rate': contacts / max(total_steps, 1),
        'ee_obj_dist': float(np.linalg.norm(ee - obj[:2])),
        'first_action_norm': first_action_norm.tolist() if first_action_norm is not None else None,
        'mean_planned_cost': float(np.mean(planned_costs)) if planned_costs else None,
        'mean_zero_cost': float(np.mean(zero_costs)) if zero_costs else None,
    }

# Config
CEM_CONFIG = {'horizon': 10, 'num_samples': 32, 'num_elites': 4, 'num_iterations': 2, 'init_std': 0.3}
MAX_SPEED = 0.75; MAX_STEPS = 30; EXEC_STEPS = 10

conv = PAPER1_CONVENTION; conv.max_speed_mps = MAX_SPEED

# Load templates
with open('configs/eval/closed_loop_smoke_templates.yaml') as f:
    matrix = yaml.safe_load(f)
open_space_templates = matrix['open_space']

# Find template objects
with open('data/sim/metadata/reset_templates_v0.json') as f:
    all_t = json.load(f)
template_map = {t['reset_template_id']: t for t in all_t}
templates = [template_map[t['template_id']] for t in open_space_templates]

print(f'Config: CEM={CEM_CONFIG}, speed={MAX_SPEED}, max_steps={MAX_STEPS}, exec_steps={EXEC_STEPS}', flush=True)
print(f'Templates: {len(templates)} open_space', flush=True)

base = '/home/brucewu/my_robot_project/runs/closed_loop_action_planner_fix/open_space/cem'
os.makedirs(base, exist_ok=True)

all_results = {}
for model_type in ['flat', 'object_centric', 'causality_aware']:
    print(f'\n=== {model_type} ===', flush=True)
    model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256, d_model=128, head_hidden_dim=256)
    ckpt = torch.load(f'runs/pilot_state16_mppi_stage2c/train_{model_type}/{model_type}/checkpoints/best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    norm = StateNormalizer.load(f'runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json')
    
    planner = CEMLearnedPlanner(model=model, normalizer=norm, convention=conv,
        horizon=CEM_CONFIG['horizon'], num_samples=CEM_CONFIG['num_samples'],
        num_elites=CEM_CONFIG['num_elites'], num_iterations=CEM_CONFIG['num_iterations'],
        init_std=CEM_CONFIG['init_std'], device='cpu')
    env = MujocoPushEnv(shape_type='T', control_dt=conv.control_dt, max_speed_mps=conv.max_speed_mps)
    
    model_results = []
    for i, tmpl in enumerate(templates):
        # Compute topology metrics
        topo_metrics = compute_all_metrics(tmpl)
        
        t0 = time.time()
        r = run_one(env, planner, tmpl, MAX_STEPS, EXEC_STEPS)
        r['runtime'] = time.time() - t0
        r['template_idx'] = i; r['template_id'] = tmpl['reset_template_id']; r['model'] = model_type
        r['planner_backend'] = 'CEM_BEST_LEARNED_ROLLOUT'; r['planner_config'] = CEM_CONFIG
        r['action_convention'] = conv.describe()
        
        # Add topology metrics to result
        r['topology'] = {
            'family': tmpl.get('layout_family', 'unknown'),
            'difficulty_level': topo_metrics['classification']['difficulty_level'],
            'difficulty_score': topo_metrics['classification']['difficulty_score'],
            'topology_family_pred': topo_metrics['classification']['topology_family_pred'],
            'object_goal_distance': topo_metrics['basic_metrics']['object_goal_distance'],
            'goal_edge_distance': topo_metrics['basic_metrics']['goal_edge_distance'],
            'obstacle_count': topo_metrics['basic_metrics']['obstacle_count'],
            'direct_path_blocked': topo_metrics['path_metrics']['direct_path_blocked'],
            'blocking_score': topo_metrics['path_metrics']['blocking_score'],
            'passage_width_estimate': topo_metrics['path_metrics']['passage_width_estimate'],
            'edge_goal_score': topo_metrics['path_metrics']['edge_goal_score'],
            'reachable_contact_sides': topo_metrics['contact_metrics']['reachable_contact_sides'],
            'approach_feasibility_score': topo_metrics['contact_metrics']['approach_feasibility_score'],
            'dominant_geometric_challenge': topo_metrics['classification']['dominant_geometric_challenge'],
            'geometry_valid': topo_metrics['validation']['is_valid'],
            'geometry_warnings': topo_metrics['classification']['warnings'],
        }
        
        model_results.append(r)
        s = '✅' if r['success'] else ('📈' if r['improved'] else '📉')
        print(f'  [{i+1}] {s} init={r["initial_dist"]:.3f} best={r["best_dist"]:.3f} final={r["final_dist"]:.3f} contact={r["contact_rate"]:.2f} ({r["runtime"]:.0f}s)', flush=True)
    
    all_results[model_type] = model_results
    with open(f'{base}/{model_type}.json', 'w') as f:
        json.dump({'config': CEM_CONFIG, 'convention': conv.describe(), 'results': model_results}, f, indent=2)

# Summary
print(f'\n{"="*50}\nSUMMARY\n{"="*50}', flush=True)
for m in ['flat', 'object_centric', 'causality_aware']:
    rs = all_results[m]
    sc = sum(r['success'] for r in rs); imp = sum(r['improved'] for r in rs)
    mb = np.mean([r['best_dist'] for r in rs]); mc = np.mean([r['contact_rate'] for r in rs])
    print(f'{m}: success={sc}/{len(rs)}, improved={imp}/{len(rs)}, mean_best={mb:.4f}, contact={mc:.2f}', flush=True)

summary = {'config': CEM_CONFIG, 'convention': conv.describe(), 'models': {}}
for m in ['flat', 'object_centric', 'causality_aware']:
    rs = all_results[m]
    summary['models'][m] = {
        'success': sum(r['success'] for r in rs), 'improved': sum(r['improved'] for r in rs), 'total': len(rs),
        'mean_best': float(np.mean([r['best_dist'] for r in rs])),
        'mean_contact': float(np.mean([r['contact_rate'] for r in rs])),
    }
with open(f'{base}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Done!', flush=True)
