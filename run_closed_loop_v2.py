#!/usr/bin/env python3
"""Closed-loop with reasonable CEM config for learned rollout.

Key insight: learned rollout is slower than oracle MuJoCo,
so we use smaller CEM but more MPC steps with re-planning.

Config: horizon=15, samples=128, iterations=3, speed=0.75
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

EE_SIZE = 0.015
OBJ_SIZE = 0.048
OBJ_MASS = 0.038
OBJ_FRICTION = 0.8

def extract_state16(env):
    ee = env.get_ee_pos()
    obj = env.get_object_pose()
    goal = env.get_goal_pose()
    contact = env.get_contact_flag()
    t = np.zeros((6, 16), dtype=np.float32)
    t[0, :2] = ee; t[0, 3] = 1.0; t[0, 7:9] = EE_SIZE; t[0, 14] = float(contact); t[0, 15] = 1.0
    t[1, :2] = obj[:2]; t[1, 2] = np.sin(obj[2]); t[1, 3] = np.cos(obj[2])
    t[1, 7:9] = OBJ_SIZE; t[1, 9] = 1.0; t[1, 12:14] = [OBJ_MASS, OBJ_FRICTION]; t[1, 15] = 1.0
    t[2, :2] = goal[:2]; t[2, 2] = np.sin(goal[2]); t[2, 3] = np.cos(goal[2])
    t[2, 7:9] = OBJ_SIZE; t[2, 9] = 1.0; t[2, 12:14] = [OBJ_MASS, OBJ_FRICTION]; t[2, 15] = 1.0
    return t

def run_one(env, planner, template, max_steps, conv):
    env.reset_from_template(template)
    state = extract_state16(env)
    hist = np.tile(state[np.newaxis], (6, 1, 1))
    goal = env.get_goal_pose()
    init_d = float(np.linalg.norm(env.get_object_pose()[:2] - goal[:2]))
    best_d = init_d
    final_d = init_d
    contacts = 0
    
    for step in range(max_steps):
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
        te = abs(np.arctan2(np.sin(obj[2] - goal[2]), np.cos(obj[2] - goal[2])))
        if d < 0.02 and np.degrees(te) < 10:
            break
    
    return {
        'success': bool(final_d < 0.02 and np.degrees(te) < 10),
        'initial_dist': init_d, 'final_dist': final_d, 'best_dist': best_d,
        'improved': best_d < init_d - 0.001,
        'steps': step + 1, 'contact_rate': contacts / max(step + 1, 1),
    }

# Config
CEM_HORIZON = 15
CEM_SAMPLES = 128
CEM_ELITES = 16
CEM_ITERS = 3
MAX_SPEED = 0.75
MAX_STEPS = 30
MAX_TEMPLATES = 3

conv = PAPER1_CONVENTION
conv.max_speed_mps = MAX_SPEED

print(f'CEM: H={CEM_HORIZON} S={CEM_SAMPLES} E={CEM_ELITES} I={CEM_ITERS} speed={MAX_SPEED}', flush=True)

with open('/home/brucewu/my_robot_project/data/sim/metadata/reset_templates_v0.json') as f:
    all_t = json.load(f)

families = {
    'open_space': [t for t in all_t if t.get('layout_family') == 'open_space'],
    'blocking': [t for t in all_t if 'blocking' in t.get('layout_family', '')],
    'passage': [t for t in all_t if 'passage' in t.get('layout_family', '')],
}

base = '/home/brucewu/my_robot_project/runs/closed_loop_action_planner_fix'
for d in ['open_space/cem', 'classic_templates/cem']:
    os.makedirs(f'{base}/{d}', exist_ok=True)

all_results = {}

for model_type in ['flat', 'object_centric', 'causality_aware']:
    print(f'\n{"="*50}\n{model_type}\n{"="*50}', flush=True)
    
    model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256, d_model=128, head_hidden_dim=256)
    ckpt = torch.load(f'/home/brucewu/my_robot_project/runs/pilot_state16_mppi_stage2c/train_{model_type}/{model_type}/checkpoints/best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    norm = StateNormalizer.load(f'/home/brucewu/my_robot_project/runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json')
    
    planner = CEMLearnedPlanner(model=model, normalizer=norm, convention=conv,
        horizon=CEM_HORIZON, num_samples=CEM_SAMPLES, num_elites=CEM_ELITES,
        num_iterations=CEM_ITERS, init_std=0.3, device='cpu')
    env = MujocoPushEnv(shape_type='T', control_dt=conv.control_dt, max_speed_mps=conv.max_speed_mps)
    
    model_results = {}
    for fam, templates in families.items():
        ts = templates[:MAX_TEMPLATES]
        if not ts:
            continue
        print(f'\n  {fam} ({len(ts)}):', flush=True)
        fam_results = []
        for i, tmpl in enumerate(ts):
            t0 = time.time()
            r = run_one(env, planner, tmpl, MAX_STEPS, conv)
            r['runtime'] = time.time() - t0
            r['template_idx'] = i
            r['family'] = fam
            fam_results.append(r)
            s = '✅' if r['success'] else ('📈' if r['improved'] else '📉')
            print(f'    [{i+1}] {s} init={r["initial_dist"]:.3f} best={r["best_dist"]:.3f} '
                  f'final={r["final_dist"]:.3f} contact={r["contact_rate"]:.2f} ({r["runtime"]:.0f}s)', flush=True)
        
        model_results[fam] = fam_results
        path = f'{base}/{"open_space" if fam == "open_space" else "classic_templates"}/cem/{model_type}{"" if fam == "open_space" else "_" + fam}.json'
        with open(path, 'w') as f:
            json.dump({'results': fam_results}, f, indent=2)
    
    all_results[model_type] = model_results

# Summary
print(f'\n{"="*50}\nSUMMARY\n{"="*50}', flush=True)
for m in ['flat', 'object_centric', 'causality_aware']:
    print(f'\n{m}:', flush=True)
    for fam in families:
        if fam in all_results.get(m, {}):
            rs = all_results[m][fam]
            print(f'  {fam}: success={sum(r["success"] for r in rs)}/{len(rs)}, '
                  f'improved={sum(r["improved"] for r in rs)}/{len(rs)}, '
                  f'mean_best={np.mean([r["best_dist"] for r in rs]):.4f}', flush=True)

with open(f'{base}/summary.json', 'w') as f:
    json.dump({'results': all_results}, f, indent=2)
print('\nDone!', flush=True)
