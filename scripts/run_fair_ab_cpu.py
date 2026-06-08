#!/usr/bin/env python3
"""Fair A/B: old 10ep vs new 50ep checkpoints - CPU-friendly version.
Reduced: horizon=30, budget=150, CEM samples=200/iter=3, MPPI samples=400.
"""
import sys, json, time, os, csv
from pathlib import Path
import numpy as np
import torch
from multiprocessing import Pool

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

# Reduced parameters for CPU
HORIZON = 30
MAX_BUDGET = 150
CEM_SAMPLES = 200
CEM_ELITES = 32
CEM_ITERS = 3
MPPI_SAMPLES = 400
EXEC_STEPS = 8
MAX_SPEED = 0.3

TERMINAL_WEIGHTS = CostWeights(
    w_pos=10.0, w_theta=2.0, w_reach=5.0, w_no_contact=2.0,
    w_push_alignment=1.0, w_collision=20.0, w_collision_step=1.0,
    w_proximity=5.0, w_action=0.05, w_smooth=0.1,
)


def extract_state16(env):
    ee = env.get_ee_pos(); obj = env.get_object_pose(); goal = env.get_goal_pose()
    t = np.zeros((6, 16), dtype=np.float32)
    t[0, :2] = ee; t[0, 3] = 1.0; t[0, 7:9] = EE_SIZE; t[0, 14] = float(env.get_contact_flag()); t[0, 15] = 1.0
    t[1, :2] = obj[:2]; t[1, 2] = np.sin(obj[2]); t[1, 3] = np.cos(obj[2])
    t[1, 7:9] = OBJ_SIZE; t[1, 9] = 1.0; t[1, 15] = 1.0
    t[2, :2] = goal[:2]; t[2, 2] = np.sin(goal[2]); t[2, 3] = np.cos(goal[2])
    t[2, 7:9] = OBJ_SIZE; t[2, 9] = 1.0; t[2, 15] = 1.0
    num_active = getattr(env, '_num_active_obstacles', 0)
    for i in range(min(MAX_OBSTACLES, num_active)):
        body_id = env._obstacle_body_ids[i]; geom_id = env._obstacle_geom_ids[i]
        obs_pos = env.data.xpos[body_id][:2].copy()
        if np.allclose(obs_pos, 0.0, atol=1e-3): continue
        token_idx = 3 + i; t[token_idx, :2] = obs_pos
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        t[token_idx, 2] = np.sin(yaw); t[token_idx, 3] = np.cos(yaw)
        gs = env.model.geom_size[geom_id]
        t[token_idx, 7] = gs[0]; t[token_idx, 8] = gs[1]
        t[token_idx, 15] = 1.0
    return t


def get_obstacle_info(env):
    positions = []; radii = []
    num_active = getattr(env, '_num_active_obstacles', 0)
    for i in range(min(MAX_OBSTACLES, num_active)):
        body_id = env._obstacle_body_ids[i]
        obs_pos = env.data.xpos[body_id][:2].copy()
        if np.allclose(obs_pos, 0.0, atol=1e-3): continue
        gs = env.model.geom_size[env._obstacle_geom_ids[i]]
        positions.append(obs_pos); radii.append(max(gs[0], gs[1]))
    if positions: return np.array(positions), np.array(radii)
    return None, None


def run_trial(args):
    tid, model_type, ckpt_type, planner_type, tmpl = args
    conv = PAPER1_CONVENTION; conv.max_speed_mps = MAX_SPEED

    if ckpt_type == 'new_50ep':
        ckpt_path = f'runs/retrain_nomass_50ep/{model_type}/checkpoints/best.pt'
        norm_path = f'runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json'
    else:
        ckpt_path = f'runs/pilot_state16_mppi_stage2c/train_{model_type}/{model_type}/checkpoints/best.pt'
        norm_path = f'runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json'

    model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256, d_model=128, head_hidden_dim=256)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    norm = StateNormalizer.load(norm_path)

    env = MujocoPushEnv(shape_type='T', control_dt=conv.control_dt, max_speed_mps=conv.max_speed_mps)
    env.reset_from_template(tmpl)
    obs_pos, obs_rad = get_obstacle_info(env)

    planner_kwargs = {
        'model': model, 'normalizer': norm, 'convention': conv, 'device': 'cpu',
        'cost_mode': 'full', 'cost_weights': TERMINAL_WEIGHTS,
        'obstacle_positions': obs_pos, 'obstacle_radii': obs_rad,
    }

    if planner_type == 'cem':
        planner = CEMLearnedPlanner(
            horizon=HORIZON, num_samples=CEM_SAMPLES, num_elites=CEM_ELITES,
            num_iterations=CEM_ITERS, init_std=0.3, **planner_kwargs)
        mpc_steps = (MAX_BUDGET + EXEC_STEPS - 1) // EXEC_STEPS
    else:
        planner = MPPILearnedPlanner(
            horizon=HORIZON, num_samples=MPPI_SAMPLES, temperature=0.1, init_std=0.5,
            speed=0.3, execute_steps=EXEC_STEPS,
            max_mpc_steps=(MAX_BUDGET + EXEC_STEPS - 1) // EXEC_STEPS,
            **planner_kwargs)

    env.reset_from_template(tmpl)
    state = extract_state16(env); hist = np.tile(state[np.newaxis], (6, 1, 1))
    goal = env.get_goal_pose()
    init_d = float(np.linalg.norm(env.get_object_pose()[:2] - goal[:2]))
    best_d = init_d; best_step = 0; final_d = init_d; total_steps = 0
    contacts = 0; collisions = 0
    ever_pos = False; ever_pose = False; early_stopped = False; es_reason = ''

    t0 = time.time()
    while total_steps < MAX_BUDGET:
        curr = extract_state16(env); hist[:-1] = hist[1:]; hist[-1] = curr
        result = planner.plan(hist.copy(), goal)
        steps_todo = min(EXEC_STEPS, MAX_BUDGET - total_steps)
        for s in range(steps_todo):
            action = result.action_sequence_norm[s] if s < len(result.action_sequence_norm) else result.first_action_norm
            env.step(action); total_steps += 1
            contacts += int(env.get_contact_flag()); collisions += int(env.get_collision_flag())
            obj = env.get_object_pose()
            d = float(np.linalg.norm(obj[:2] - goal[:2]))
            te = abs(np.arctan2(np.sin(obj[2] - goal[2]), np.cos(obj[2] - goal[2])))
            if d < best_d: best_d = d; best_step = total_steps
            if d < 0.005: ever_pos = True
            if d < 0.005 and np.degrees(te) < 10: ever_pose = True
            if d < 0.002 and np.degrees(te) < 10:
                early_stopped = True; es_reason = f'{d:.4f}m_{np.degrees(te):.1f}deg'; break
        if early_stopped: break

    final_obj = env.get_object_pose()
    final_d = float(np.linalg.norm(final_obj[:2] - goal[:2]))
    final_te = abs(np.arctan2(np.sin(final_obj[2] - goal[2]), np.cos(final_obj[2] - goal[2])))
    rt = time.time() - t0
    del env, model, norm, planner

    return {
        'model': model_type, 'checkpoint': ckpt_type, 'planner': planner_type,
        'family': tmpl.get('layout_family', '?'), 'template_id': tmpl.get('reset_template_id', '?'),
        'horizon': HORIZON, 'execute_steps': EXEC_STEPS, 'total_budget': MAX_BUDGET,
        'early_stop_enabled': True, 'early_stop_triggered': early_stopped, 'early_stop_step': total_steps if early_stopped else -1,
        'cost_mode': 'full_terminal', 'max_speed_mps': MAX_SPEED,
        'success': bool(final_d < 0.002 and np.degrees(final_te) < 10),
        'ever_success_pos_only': ever_pos, 'ever_success_pose': ever_pose,
        'initial_dist': init_d, 'best_dist': float(best_d), 'final_dist': float(final_d),
        'drift_after_best': float(final_d - best_d), 'best_step': best_step,
        'final_theta_error_deg': float(np.degrees(final_te)),
        'contact_rate': contacts / max(total_steps, 1),
        'collision_rate': collisions / max(total_steps, 1), 'total_steps': total_steps,
        'runtime_sec': rt,
    }


def main():
    RUN_DIR = Path('runs/fair_ab_cpu_' + time.strftime('%Y%m%d_%H%M%S'))
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    with open('data/sim/metadata/reset_templates_v0.json') as f:
        all_t = json.load(f)
    selected = []
    for fam in ['open_space', 'blocking', 'narrow_passage', 'edge_goal']:
        fam_t = [t for t in all_t if t.get('layout_family') == fam]
        selected.extend(fam_t[:2])

    # Both old and new checkpoints for flat model
    trial_args = []
    tid = 0
    for ckpt_type in ['old_10ep', 'new_50ep']:
        for model_type in ['flat']:
            for planner_type in ['cem', 'mppi']:
                for tmpl in selected:
                    trial_args.append((f't{tid:03d}', model_type, ckpt_type, planner_type, tmpl))
                    tid += 1

    print(f'=== Fair A/B Test (CPU-optimized) ===')
    print(f'Config: horizon={HORIZON}, budget={MAX_BUDGET}, early_stop=2mm+10°, cost=full_terminal')
    print(f'CEM: samples={CEM_SAMPLES}, elites={CEM_ELITES}, iters={CEM_ITERS}')
    print(f'MPPI: samples={MPPI_SAMPLES}, exec_steps={EXEC_STEPS}')
    print(f'Checkpoints: flat old(10ep) vs new(50ep nomass)')
    print(f'Templates: 8 (open×2, blocking×2, narrow×2, edge×2)')
    print(f'Trials: {len(trial_args)}, Workers: 12')
    print(f'Run dir: {RUN_DIR}')
    sys.stdout.flush()

    n_workers = 12
    t0 = time.time()
    results = []

    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_trial, trial_args)):
            results.append(result)
            s = '✅' if result['success'] else ('📈' if result['ever_success_pose'] else '❌')
            print(f"[{i+1}/{len(trial_args)}] {s} {result['model']:<6} {result['checkpoint']:<8} "
                  f"{result['planner']:<4} {result['family']:<12} best={result['best_dist']:.4f} "
                  f"final={result['final_dist']:.4f} drift={result['drift_after_best']:.4f} "
                  f"es={result['early_stop_triggered']} {result['runtime_sec']:.0f}s")
            sys.stdout.flush()

            sorted_r = sorted(results, key=lambda x: x['family'])
            fields = ['model','checkpoint','planner','family','template_id','horizon','execute_steps',
                      'total_budget','early_stop_enabled','early_stop_triggered','early_stop_step',
                      'cost_mode','max_speed_mps','success','ever_success_pos_only','ever_success_pose',
                      'initial_dist','best_dist','final_dist','drift_after_best','best_step',
                      'final_theta_error_deg','contact_rate','collision_rate','total_steps','runtime_sec']
            with open(RUN_DIR / 'summary.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                for r in sorted_r: writer.writerow(r)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed/60:.1f}min')

    old = [r for r in results if r['checkpoint'] == 'old_10ep']
    new = [r for r in results if r['checkpoint'] == 'new_50ep']
    o_best = np.mean([r["best_dist"] for r in old]); o_final = np.mean([r["final_dist"] for r in old])
    n_best = np.mean([r["best_dist"] for r in new]); n_final = np.mean([r["final_dist"] for r in new])
    o_succ = sum(r["success"] for r in old); n_succ = sum(r["success"] for r in new)
    o_ever = sum(r["ever_success_pose"] for r in old); n_ever = sum(r["ever_success_pose"] for r in new)

    print(f'\nOld 10ep: best={o_best:.4f} final={o_final:.4f} success={o_succ} ever={o_ever}')
    print(f'New 50ep: best={n_best:.4f} final={n_final:.4f} success={n_succ} ever={n_ever}')

    with open(RUN_DIR / 'summary.json', 'w') as f:
        json.dump({
            'config': {
                'horizon': HORIZON, 'budget': MAX_BUDGET, 'early_stop': '2mm+10deg',
                'cem_samples': CEM_SAMPLES, 'cem_iters': CEM_ITERS,
                'mppi_samples': MPPI_SAMPLES, 'exec_steps': EXEC_STEPS,
            },
            'old_10ep': {
                'mean_best': float(o_best), 'mean_final': float(o_final),
                'success': o_succ, 'ever_success': o_ever,
            },
            'new_50ep': {
                'mean_best': float(n_best), 'mean_final': float(n_final),
                'success': n_succ, 'ever_success': n_ever,
            }
        }, f, indent=2)


if __name__ == '__main__':
    main()
