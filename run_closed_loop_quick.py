#!/usr/bin/env python3
"""Quick closed-loop test with small CEM config."""

import sys, json, time
sys.path.insert(0, '/home/brucewu/my_robot_project')
import numpy as np
import torch

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.learned_planner_adapter import CEMLearnedPlanner
from src.planners.action_conventions import PAPER1_CONVENTION
from src.envs.mujoco_push_env import MujocoPushEnv

IDX_EE, IDX_OBJ, IDX_GOAL = 0, 1, 2
FEAT_X, FEAT_Y, FEAT_SIN_THETA, FEAT_COS_THETA = 0, 1, 2, 3
OBJ_SIZE = 0.048
OBJ_MASS = 0.038
OBJ_FRICTION = 0.8
EE_SIZE = 0.015

def extract_state16(env):
    ee_pos = env.get_ee_pos()
    obj_pose = env.get_object_pose()
    goal_pose = env.get_goal_pose()
    contact = env.get_contact_flag()
    tokens = np.zeros((6, 16), dtype=np.float32)
    tokens[0, 0] = ee_pos[0]; tokens[0, 1] = ee_pos[1]; tokens[0, 3] = 1.0
    tokens[0, 7] = EE_SIZE; tokens[0, 8] = EE_SIZE; tokens[0, 14] = float(contact); tokens[0, 15] = 1.0
    tokens[1, 0] = obj_pose[0]; tokens[1, 1] = obj_pose[1]
    tokens[1, 2] = np.sin(obj_pose[2]); tokens[1, 3] = np.cos(obj_pose[2])
    tokens[1, 7] = OBJ_SIZE; tokens[1, 8] = OBJ_SIZE; tokens[1, 9] = 1.0
    tokens[1, 12] = OBJ_MASS; tokens[1, 13] = OBJ_FRICTION; tokens[1, 15] = 1.0
    tokens[2, 0] = goal_pose[0]; tokens[2, 1] = goal_pose[1]
    tokens[2, 2] = np.sin(goal_pose[2]); tokens[2, 3] = np.cos(goal_pose[2])
    tokens[2, 7] = OBJ_SIZE; tokens[2, 8] = OBJ_SIZE; tokens[2, 9] = 1.0
    tokens[2, 12] = OBJ_MASS; tokens[2, 13] = OBJ_FRICTION; tokens[2, 15] = 1.0
    return tokens

HORIZON = 10
SAMPLES = 64
ELITES = 8
ITERS = 3
MAX_STEPS = 15
MAX_TEMPLATES = 3

convention = PAPER1_CONVENTION
print(f'Config: H={HORIZON} S={SAMPLES} E={ELITES} I={ITERS} steps={MAX_STEPS}', flush=True)

with open('/home/brucewu/my_robot_project/data/sim/metadata/reset_templates_v0.json') as f:
    all_t = json.load(f)
open_space = [t for t in all_t if t.get('layout_family') == 'open_space']
templates = open_space[:MAX_TEMPLATES]

all_results = {}
for model_type in ['flat', 'object_centric', 'causality_aware']:
    print(f'\n=== {model_type} ===', flush=True)
    ckpt_path = f'/home/brucewu/my_robot_project/runs/pilot_state16_mppi_stage2c/train_{model_type}/{model_type}/checkpoints/best.pt'
    norm_path = f'/home/brucewu/my_robot_project/runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json'
    
    model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256, d_model=128, head_hidden_dim=256)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    normalizer = StateNormalizer.load(norm_path)
    
    planner = CEMLearnedPlanner(model=model, normalizer=normalizer, convention=convention,
        horizon=HORIZON, num_samples=SAMPLES, num_elites=ELITES,
        num_iterations=ITERS, init_std=0.3, device='cpu')
    env = MujocoPushEnv(shape_type='T', control_dt=0.1, max_speed_mps=0.5)
    
    model_results = []
    for i, tmpl in enumerate(templates):
        t0 = time.time()
        env.reset_from_template(tmpl)
        state = extract_state16(env)
        hist = np.tile(state[np.newaxis], (6, 1, 1))
        goal = env.get_goal_pose()
        init_d = float(np.linalg.norm(env.get_object_pose()[:2] - goal[:2]))
        best_d = init_d
        final_d = init_d
        contacts = 0
        
        for step in range(MAX_STEPS):
            curr = extract_state16(env)
            hist[:-1] = hist[1:]
            hist[-1] = curr
            res = planner.plan(hist.copy(), goal)
            env.step(res.first_action_norm)
            obj = env.get_object_pose()
            d = float(np.linalg.norm(obj[:2] - goal[:2]))
            best_d = min(best_d, d)
            final_d = d
            contacts += int(env.get_contact_flag())
            te = abs(np.arctan2(np.sin(obj[2]-goal[2]), np.cos(obj[2]-goal[2])))
            if d < 0.02 and np.degrees(te) < 10:
                break
        
        elapsed = time.time() - t0
        model_results.append({'init': init_d, 'final': final_d, 'best': best_d, 'improved': best_d < init_d, 'steps': step+1, 'time': elapsed})
        s = '📈' if best_d < init_d else '📉'
        print(f'  [{i+1}] {s} {init_d:.3f}→{best_d:.3f} ({elapsed:.1f}s)', flush=True)
    
    all_results[model_type] = model_results

print('\n=== SUMMARY ===', flush=True)
for m in ['flat', 'object_centric', 'causality_aware']:
    rs = all_results[m]
    imp = sum(1 for r in rs if r['improved'])
    mb = np.mean([r['best'] for r in rs])
    print(f'{m}: {imp}/{len(rs)} improved, mean_best={mb:.4f}m', flush=True)

with open('/home/brucewu/my_robot_project/runs/closed_loop_action_planner_fix/open_space/cem/all_models.json', 'w') as f:
    json.dump({'results': all_results}, f, indent=2)
print('\nDone!', flush=True)
