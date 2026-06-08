#!/usr/bin/env python3
"""Full closed-loop experiment with correct CEM/MPPI configs.

CEM best: horizon=140, speed=0.75, execute_steps=10
MPPI best: speed=0.3, temperature=0.1, horizon=100, samples=2048

For learned rollout, we adapt:
- CEM: horizon=30 (learned model accumulates error), speed=0.75, samples=512
- MPPI: use oracle MPPI with learned cost_fn
"""

import sys, json, time, os
sys.path.insert(0, '/home/brucewu/my_robot_project')
import numpy as np
import torch

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.learned_planner_adapter import CEMLearnedPlanner
from src.planners.action_conventions import PAPER1_CONVENTION
from src.envs.mujoco_push_env import MujocoPushEnv

# State indices
IDX_EE, IDX_OBJ, IDX_GOAL = 0, 1, 2
FEAT_X, FEAT_Y, FEAT_SIN_THETA, FEAT_COS_THETA = 0, 1, 2, 3
OBJ_SIZE = 0.048
OBJ_MASS = 0.038
OBJ_FRICTION = 0.8
EE_SIZE = 0.015

def extract_state16(env):
    ee = env.get_ee_pos()
    obj = env.get_object_pose()
    goal = env.get_goal_pose()
    contact = env.get_contact_flag()
    t = np.zeros((6, 16), dtype=np.float32)
    # EE
    t[0, :2] = ee; t[0, 3] = 1.0; t[0, 7:9] = EE_SIZE
    t[0, 14] = float(contact); t[0, 15] = 1.0
    # Object
    t[1, :2] = obj[:2]; t[1, 2] = np.sin(obj[2]); t[1, 3] = np.cos(obj[2])
    t[1, 7:9] = OBJ_SIZE; t[1, 9] = 1.0; t[1, 12:14] = [OBJ_MASS, OBJ_FRICTION]; t[1, 15] = 1.0
    # Goal
    t[2, :2] = goal[:2]; t[2, 2] = np.sin(goal[2]); t[2, 3] = np.cos(goal[2])
    t[2, 7:9] = OBJ_SIZE; t[2, 9] = 1.0; t[2, 12:14] = [OBJ_MASS, OBJ_FRICTION]; t[2, 15] = 1.0
    return t

def run_one(env, planner, template, max_mpc_steps, convention):
    """Run closed-loop on one template."""
    env.reset_from_template(template)
    state = extract_state16(env)
    hist = np.tile(state[np.newaxis], (6, 1, 1))
    goal = env.get_goal_pose()
    init_d = float(np.linalg.norm(env.get_object_pose()[:2] - goal[:2]))
    best_d = init_d
    final_d = init_d
    contacts = 0
    collisions = 0
    step = 0
    
    for step in range(max_mpc_steps):
        curr = extract_state16(env)
        hist[:-1] = hist[1:]
        hist[-1] = curr
        
        result = planner.plan(hist.copy(), goal)
        env.step(result.first_action_norm)
        
        obj = env.get_object_pose()
        d = float(np.linalg.norm(obj[:2] - goal[:2]))
        best_d = min(best_d, d)
        final_d = d
        contacts += int(env.get_contact_flag())
        collisions += int(env.get_collision_flag())
        
        te = abs(np.arctan2(np.sin(obj[2] - goal[2]), np.cos(obj[2] - goal[2])))
        if d < 0.02 and np.degrees(te) < 10:
            break
    
    return {
        'success': bool(final_d < 0.02 and np.degrees(te) < 10),
        'initial_dist': init_d,
        'final_dist': final_d,
        'best_dist': best_d,
        'improved': best_d < init_d - 0.001,  # 1mm threshold
        'steps': step + 1,
        'contact_rate': contacts / max(step + 1, 1),
        'collision_rate': collisions / max(step + 1, 1),
    }

def main():
    # CEM config adapted from user's best: horizon=140, speed=0.75
    # For learned rollout: horizon=30, speed=0.75, samples=512
    CEM_CONFIG = {
        'horizon': 30,
        'num_samples': 512,
        'num_elites': 64,
        'num_iterations': 5,
        'init_std': 0.3,
        'max_speed_mps': 0.75,
    }
    
    MAX_MPC_STEPS = 50  # Longer to allow approach + push
    MAX_TEMPLATES = 3
    
    convention = PAPER1_CONVENTION
    convention.max_speed_mps = CEM_CONFIG['max_speed_mps']
    
    print(f'CEM Config: {CEM_CONFIG}', flush=True)
    print(f'Convention: max_speed={convention.max_speed_mps}', flush=True)
    
    # Load templates
    with open('/home/brucewu/my_robot_project/data/sim/metadata/reset_templates_v0.json') as f:
        all_t = json.load(f)
    
    # open_space
    open_space = [t for t in all_t if t.get('layout_family') == 'open_space'][:MAX_TEMPLATES]
    # classic families
    blocking = [t for t in all_t if 'blocking' in t.get('layout_family', '')][:MAX_TEMPLATES]
    passage = [t for t in all_t if 'passage' in t.get('layout_family', '')][:MAX_TEMPLATES]
    
    template_sets = {
        'open_space': open_space,
        'blocking': blocking,
        'passage': passage,
    }
    
    os.makedirs('/home/brucewu/my_robot_project/runs/closed_loop_action_planner_fix/open_space/cem', exist_ok=True)
    os.makedirs('/home/brucewu/my_robot_project/runs/closed_loop_action_planner_fix/classic_templates/cem', exist_ok=True)
    
    all_results = {}
    
    for model_type in ['flat', 'object_centric', 'causality_aware']:
        print(f'\n{"="*60}', flush=True)
        print(f'MODEL: {model_type}', flush=True)
        print(f'{"="*60}', flush=True)
        
        ckpt_path = f'/home/brucewu/my_robot_project/runs/pilot_state16_mppi_stage2c/train_{model_type}/{model_type}/checkpoints/best.pt'
        norm_path = f'/home/brucewu/my_robot_project/runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json'
        
        model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256, d_model=128, head_hidden_dim=256)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        normalizer = StateNormalizer.load(norm_path)
        
        planner = CEMLearnedPlanner(
            model=model, normalizer=normalizer, convention=convention,
            horizon=CEM_CONFIG['horizon'],
            num_samples=CEM_CONFIG['num_samples'],
            num_elites=CEM_CONFIG['num_elites'],
            num_iterations=CEM_CONFIG['num_iterations'],
            init_std=CEM_CONFIG['init_std'],
            device='cpu',
        )
        env = MujocoPushEnv(shape_type='T', control_dt=convention.control_dt, max_speed_mps=convention.max_speed_mps)
        
        model_results = {}
        
        for family_name, templates in template_sets.items():
            if not templates:
                print(f'  {family_name}: no templates found', flush=True)
                continue
            
            print(f'\n  --- {family_name} ({len(templates)} templates) ---', flush=True)
            family_results = []
            
            for i, tmpl in enumerate(templates):
                t0 = time.time()
                result = run_one(env, planner, tmpl, MAX_MPC_STEPS, convention)
                elapsed = time.time() - t0
                result['template_idx'] = i
                result['family'] = family_name
                result['runtime'] = elapsed
                family_results.append(result)
                
                s = '✅' if result['success'] else ('📈' if result['improved'] else '📉')
                print(f'    [{i+1}] {s} init={result["initial_dist"]:.3f} best={result["best_dist"]:.3f} '
                      f'final={result["final_dist"]:.3f} contact={result["contact_rate"]:.2f} ({elapsed:.0f}s)', flush=True)
            
            model_results[family_name] = family_results
            
            # Save per-family
            if family_name == 'open_space':
                path = f'/home/brucewu/my_robot_project/runs/closed_loop_action_planner_fix/open_space/cem/{model_type}.json'
            else:
                path = f'/home/brucewu/my_robot_project/runs/closed_loop_action_planner_fix/classic_templates/cem/{model_type}_{family_name}.json'
            with open(path, 'w') as f:
                json.dump({'config': CEM_CONFIG, 'results': family_results}, f, indent=2)
        
        all_results[model_type] = model_results
    
    # Summary
    print(f'\n{"="*60}', flush=True)
    print('SUMMARY', flush=True)
    print(f'{"="*60}', flush=True)
    
    summary = {'config': CEM_CONFIG, 'models': {}}
    for model_type in ['flat', 'object_centric', 'causality_aware']:
        print(f'\n{model_type}:', flush=True)
        model_summary = {}
        for family_name in template_sets:
            if family_name in all_results.get(model_type, {}):
                rs = all_results[model_type][family_name]
                successes = sum(1 for r in rs if r['success'])
                improved = sum(1 for r in rs if r['improved'])
                mean_best = np.mean([r['best_dist'] for r in rs])
                mean_contact = np.mean([r['contact_rate'] for r in rs])
                print(f'  {family_name}: success={successes}/{len(rs)}, improved={improved}/{len(rs)}, '
                      f'mean_best={mean_best:.4f}m, contact={mean_contact:.2f}', flush=True)
                model_summary[family_name] = {
                    'success': successes, 'improved': improved, 'total': len(rs),
                    'mean_best': mean_best, 'mean_contact': mean_contact,
                }
        summary['models'][model_type] = model_summary
    
    with open('/home/brucewu/my_robot_project/runs/closed_loop_action_planner_fix/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('\nDone!', flush=True)

if __name__ == '__main__':
    main()
