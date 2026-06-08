#!/usr/bin/env python3
"""Strict Fair A/B: flat + object_centric, old 10ep vs new 50ep.
FAIR protocol: horizon=100, budget=1000, CEM(512×5), MPPI(1024), full_terminal, early_stop.
28 CPU workers.
"""
import sys, json, time, os, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from multiprocessing import Pool

sys.path.insert(0, '/home/brucewu/my_robot_project')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.learned_planner_adapter import CEMLearnedPlanner, MPPILearnedPlanner
from src.planners.action_conventions import PAPER1_CONVENTION
from src.planners.cost_functions import CostWeights
from src.envs.mujoco_push_env import MujocoPushEnv

# ============================================================
# FAIR PROTOCOL PARAMETERS (LOCKED)
# ============================================================
HORIZON = 100
TOTAL_BUDGET = 1000
EXEC_STEPS = 10
MAX_SPEED = 0.3
EARLY_STOP_POS = 0.002
EARLY_STOP_THETA_DEG = 10

CEM_SAMPLES = 512
CEM_ELITES = 64
CEM_ITERS = 5
CEM_INIT_STD = 0.3

MPPI_SAMPLES = 1024
MPPI_TEMPERATURE = 0.1
MPPI_INIT_STD = 0.5

N_WORKERS = 28

EE_SIZE, OBJ_SIZE = 0.015, 0.048
MAX_OBSTACLES = 3

TERMINAL_WEIGHTS = CostWeights(
    w_pos=10.0, w_theta=2.0, w_reach=5.0, w_no_contact=2.0,
    w_push_alignment=1.0, w_collision=20.0, w_collision_step=1.0,
    w_proximity=5.0, w_action=0.05, w_smooth=0.1,
)

# ============================================================
# Helpers
# ============================================================
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

def get_checkpoint_paths(model_type, ckpt_type):
    if ckpt_type == 'new_50ep':
        ckpt = f'runs/retrain_nomass_50ep/{model_type}/checkpoints/best.pt'
        norm = f'runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json'
    else:
        ckpt = f'runs/pilot_state16_mppi_stage2c/train_{model_type}/{model_type}/checkpoints/best.pt'
        norm = f'runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json'
    return ckpt, norm

# ============================================================
# Trial runner
# ============================================================
def run_trial(args):
    tid, model_type, ckpt_type, planner_type, tmpl = args
    conv = PAPER1_CONVENTION; conv.max_speed_mps = MAX_SPEED
    ckpt_path, norm_path = get_checkpoint_paths(model_type, ckpt_type)
    failure_code = ''

    try:
        model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256, d_model=128, head_hidden_dim=256)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict']); model.eval()
        norm = StateNormalizer.load(norm_path)
    except Exception as e:
        failure_code = f'MODEL_LOAD_FAILED:{str(e)[:80]}'
        return _empty_result(model_type, ckpt_type, planner_type, tmpl, failure_code)

    try:
        env = MujocoPushEnv(shape_type='T', control_dt=conv.control_dt, max_speed_mps=MAX_SPEED)
        env.reset_from_template(tmpl)
        obs_pos, obs_rad = get_obstacle_info(env)
    except Exception as e:
        failure_code = f'ENV_INIT_FAILED:{str(e)[:80]}'
        return _empty_result(model_type, ckpt_type, planner_type, tmpl, failure_code)

    planner_kwargs = {
        'model': model, 'normalizer': norm, 'convention': conv, 'device': 'cpu',
        'cost_mode': 'full', 'cost_weights': TERMINAL_WEIGHTS,
        'obstacle_positions': obs_pos, 'obstacle_radii': obs_rad,
    }

    if planner_type == 'cem':
        planner = CEMLearnedPlanner(
            horizon=HORIZON, num_samples=CEM_SAMPLES, num_elites=CEM_ELITES,
            num_iterations=CEM_ITERS, init_std=CEM_INIT_STD, **planner_kwargs)
    else:
        planner = MPPILearnedPlanner(
            horizon=HORIZON, num_samples=MPPI_SAMPLES, temperature=MPPI_TEMPERATURE,
            init_std=MPPI_INIT_STD, speed=MAX_SPEED, execute_steps=EXEC_STEPS,
            max_mpc_steps=(TOTAL_BUDGET + EXEC_STEPS - 1) // EXEC_STEPS,
            **planner_kwargs)

    env.reset_from_template(tmpl)
    state = extract_state16(env); hist = np.tile(state[np.newaxis], (6, 1, 1))
    goal = env.get_goal_pose()
    init_d = float(np.linalg.norm(env.get_object_pose()[:2] - goal[:2]))
    best_d = init_d; best_step = 0; best_theta = float('inf')
    total_steps = 0; contacts = 0; collisions = 0
    ever_pos = False; ever_pose = False; early_stopped = False
    cost_breakdown_summary = ''

    t0 = time.time()
    try:
        while total_steps < TOTAL_BUDGET:
            curr = extract_state16(env); hist[:-1] = hist[1:]; hist[-1] = curr
            result = planner.plan(hist.copy(), goal)
            steps_todo = min(EXEC_STEPS, TOTAL_BUDGET - total_steps)
            for s in range(steps_todo):
                action = result.action_sequence_norm[s] if s < len(result.action_sequence_norm) else result.first_action_norm
                env.step(action); total_steps += 1
                contacts += int(env.get_contact_flag()); collisions += int(env.get_collision_flag())
                obj = env.get_object_pose()
                d = float(np.linalg.norm(obj[:2] - goal[:2]))
                te = abs(np.arctan2(np.sin(obj[2] - goal[2]), np.cos(obj[2] - goal[2])))
                if d < best_d:
                    best_d = d; best_step = total_steps
                    best_theta = float(np.degrees(te))
                if d < 0.005: ever_pos = True
                if d < 0.005 and np.degrees(te) < 10: ever_pose = True
                if d < EARLY_STOP_POS and np.degrees(te) < EARLY_STOP_THETA_DEG:
                    early_stopped = True; break
            if early_stopped: break
    except Exception as e:
        failure_code = f'TRIAL_ERROR:{str(e)[:80]}'

    final_obj = env.get_object_pose()
    final_d = float(np.linalg.norm(final_obj[:2] - goal[:2]))
    final_te = abs(np.arctan2(np.sin(final_obj[2] - goal[2]), np.cos(final_obj[2] - goal[2])))
    rt = time.time() - t0

    # Cost breakdown (last plan)
    try:
        cb = getattr(planner, '_last_cost_breakdown', None) or getattr(
            getattr(planner, 'cost_fn', None), 'cost_breakdown', None) or {}
        cost_breakdown_summary = json.dumps({k: round(float(v), 4) for k, v in cb.items()})
    except:
        pass

    del env, model, norm, planner

    return {
        'model': model_type, 'checkpoint': ckpt_type, 'planner': planner_type,
        'family': tmpl.get('layout_family', '?'), 'template_id': tmpl.get('reset_template_id', '?'),
        'horizon': HORIZON, 'execute_steps': EXEC_STEPS, 'max_mpc_steps': (TOTAL_BUDGET + EXEC_STEPS - 1) // EXEC_STEPS,
        'total_budget': TOTAL_BUDGET,
        'early_stop_enabled': True, 'early_stop_triggered': early_stopped,
        'early_stop_step': total_steps if early_stopped else -1,
        'cost_mode': 'full_terminal', 'max_speed_mps': MAX_SPEED,
        'final_success_pose': bool(final_d < EARLY_STOP_POS and np.degrees(final_te) < EARLY_STOP_THETA_DEG),
        'ever_success_pose': ever_pose,
        'ever_success_pos_only': ever_pos,
        'initial_dist': init_d, 'best_dist': float(best_d), 'final_dist': float(final_d),
        'drift_after_best': float(final_d - best_d), 'best_step': best_step,
        'best_theta_error': float(best_theta), 'final_theta_error': float(np.degrees(final_te)),
        'contact_rate': contacts / max(total_steps, 1),
        'collision_rate': collisions / max(total_steps, 1), 'total_steps': total_steps,
        'runtime': rt, 'failure_code': failure_code,
        'cost_breakdown_summary': cost_breakdown_summary,
    }

def _empty_result(model_type, ckpt_type, planner_type, tmpl, failure_code):
    return {
        'model': model_type, 'checkpoint': ckpt_type, 'planner': planner_type,
        'family': tmpl.get('layout_family', '?'), 'template_id': tmpl.get('reset_template_id', '?'),
        'horizon': HORIZON, 'execute_steps': EXEC_STEPS, 'max_mpc_steps': (TOTAL_BUDGET + EXEC_STEPS - 1) // EXEC_STEPS,
        'total_budget': TOTAL_BUDGET,
        'early_stop_enabled': True, 'early_stop_triggered': False, 'early_stop_step': -1,
        'cost_mode': 'full_terminal', 'max_speed_mps': MAX_SPEED,
        'final_success_pose': False, 'ever_success_pose': False, 'ever_success_pos_only': False,
        'initial_dist': -1, 'best_dist': -1, 'final_dist': -1,
        'drift_after_best': -1, 'best_step': -1,
        'best_theta_error': -1, 'final_theta_error': -1,
        'contact_rate': -1, 'collision_rate': -1, 'total_steps': 0,
        'runtime': 0, 'failure_code': failure_code,
        'cost_breakdown_summary': '',
    }


# ============================================================
# Save helpers
# ============================================================
FIELDS = ['model','checkpoint','planner','family','template_id','horizon','execute_steps',
          'max_mpc_steps','total_budget','early_stop_enabled','early_stop_triggered','early_stop_step',
          'cost_mode','max_speed_mps','final_success_pose','ever_success_pose','ever_success_pos_only',
          'initial_dist','best_dist','final_dist','drift_after_best','best_step',
          'best_theta_error','final_theta_error','contact_rate','collision_rate','total_steps',
          'runtime','failure_code','cost_breakdown_summary']

def save_all(results, run_dir):
    sorted_r = sorted(results, key=lambda x: (x['family'], x['checkpoint'], x['planner']))
    with open(run_dir / 'fair_ab_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore'); w.writeheader()
        for r in sorted_r: w.writerow(r)

    # by_checkpoint.csv
    groups = defaultdict(list)
    for r in results: groups[(r['checkpoint'], r['model'], r['planner'])].append(r)
    with open(run_dir / 'by_checkpoint.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['checkpoint','model','planner','trials','success','ever_success','mean_best','mean_final','mean_drift','mean_runtime'])
        for (ck, m, p), items in sorted(groups.items()):
            w.writerow([ck, m, p, len(items),
                sum(x['final_success_pose'] for x in items),
                sum(x['ever_success_pose'] for x in items),
                round(np.mean([x['best_dist'] for x in items if x['best_dist']>=0]), 4),
                round(np.mean([x['final_dist'] for x in items if x['final_dist']>=0]), 4),
                round(np.mean([x['drift_after_best'] for x in items if x['drift_after_best']>=0]), 4),
                round(np.mean([x['runtime'] for x in items]), 0)])

    # by_family.csv
    fgroups = defaultdict(list)
    for r in results: fgroups[r['family']].append(r)
    with open(run_dir / 'by_family.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['family','trials','success','ever_success','mean_best','mean_final','mean_drift'])
        for fam, items in sorted(fgroups.items()):
            w.writerow([fam, len(items),
                sum(x['final_success_pose'] for x in items),
                sum(x['ever_success_pose'] for x in items),
                round(np.mean([x['best_dist'] for x in items if x['best_dist']>=0]), 4),
                round(np.mean([x['final_dist'] for x in items if x['final_dist']>=0]), 4),
                round(np.mean([x['drift_after_best'] for x in items if x['drift_after_best']>=0]), 4)])

    # best_cases.csv
    valid = [r for r in results if r['best_dist'] >= 0]
    valid.sort(key=lambda x: x['best_dist'])
    with open(run_dir / 'best_cases.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore'); w.writeheader()
        for r in valid[:10]: w.writerow(r)

    # drift_analysis.csv
    with open(run_dir / 'drift_analysis.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['checkpoint','planner','family','template_id','best_dist','final_dist','drift','best_step','total_steps'])
        for r in sorted_r:
            if r['best_dist'] >= 0:
                w.writerow([r['checkpoint'], r['planner'], r['family'], r['template_id'],
                    r['best_dist'], r['final_dist'], r['drift_after_best'], r['best_step'], r['total_steps']])

    # summary.json
    old = [r for r in results if r['checkpoint'] == 'old_10ep' and r['best_dist'] >= 0]
    new = [r for r in results if r['checkpoint'] == 'new_50ep' and r['best_dist'] >= 0]
    json.dump({
        'protocol': {
            'horizon': HORIZON, 'total_budget': TOTAL_BUDGET, 'execute_steps': EXEC_STEPS,
            'early_stop': f'{EARLY_STOP_POS}m+{EARLY_STOP_THETA_DEG}deg',
            'cost_mode': 'full_terminal', 'max_speed_mps': MAX_SPEED,
            'cem': f'samples={CEM_SAMPLES},elites={CEM_ELITES},iters={CEM_ITERS}',
            'mppi': f'samples={MPPI_SAMPLES},temperature={MPPI_TEMPERATURE}',
            'workers': N_WORKERS,
        },
        'old_10ep': _summary(old),
        'new_50ep': _summary(new),
        'cem': _summary([r for r in results if r['planner']=='cem' and r['best_dist']>=0]),
        'mppi': _summary([r for r in results if r['planner']=='mppi' and r['best_dist']>=0]),
    }, open(run_dir / 'fair_ab_summary.json', 'w'), indent=2)

def _summary(items):
    if not items: return {}
    return {
        'trials': len(items),
        'success': sum(x['final_success_pose'] for x in items),
        'ever_success': sum(x['ever_success_pose'] for x in items),
        'mean_best': round(np.mean([x['best_dist'] for x in items]), 6),
        'mean_final': round(np.mean([x['final_dist'] for x in items]), 6),
        'mean_drift': round(np.mean([x['drift_after_best'] for x in items]), 6),
        'mean_runtime': round(np.mean([x['runtime'] for x in items]), 0),
    }

# ============================================================
# Main
# ============================================================
def main():
    RUN_DIR = Path('runs/fair_ab_test_' + time.strftime('%Y%m%d_%H%M%S'))
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Load templates: 2 per family
    with open('data/sim/metadata/reset_templates_v0.json') as f:
        all_t = json.load(f)
    selected = []
    for fam in ['open_space', 'blocking', 'narrow_passage', 'edge_goal']:
        fam_t = [t for t in all_t if t.get('layout_family') == fam]
        selected.extend(fam_t[:2])

    # Build trial matrix: flat + object_centric, old + new, CEM + MPPI
    trial_args = []
    tid = 0
    for model_type in ['flat', 'object_centric']:
        for ckpt_type in ['old_10ep', 'new_50ep']:
            for planner_type in ['cem', 'mppi']:
                for tmpl in selected:
                    trial_args.append((f't{tid:03d}', model_type, ckpt_type, planner_type, tmpl))
                    tid += 1

    n_trials = len(trial_args)

    print(f'=== STRICT FAIR A/B TEST ===')
    print(f'Protocol: horizon={HORIZON}, budget={TOTAL_BUDGET}, early_stop={EARLY_STOP_POS}m+{EARLY_STOP_THETA_DEG}°, cost=full_terminal')
    print(f'CEM: samples={CEM_SAMPLES}, elites={CEM_ELITES}, iters={CEM_ITERS}')
    print(f'MPPI: samples={MPPI_SAMPLES}, T={MPPI_TEMPERATURE}, std={MPPI_INIT_STD}')
    print(f'Models: flat, object_centric | Checkpoints: old_10ep, new_50ep')
    print(f'Families: open×2, blocking×2, narrow×2, edge×2')
    print(f'Total trials: {n_trials} | Workers: {N_WORKERS} CPU cores')
    print(f'Run dir: {RUN_DIR}')
    sys.stdout.flush()

    t0 = time.time()
    results = []

    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(run_trial, trial_args)):
            results.append(result)
            s = '✅' if result['final_success_pose'] else ('📈' if result['ever_success_pose'] else '❌')
            fail = f' [FAIL:{result["failure_code"]}]' if result['failure_code'] else ''
            print(f"[{i+1}/{n_trials}] {s} {result['model']:<6} {result['checkpoint']:<8} "
                  f"{result['planner']:<4} {result['family']:<12} best={result['best_dist']:.4f} "
                  f"final={result['final_dist']:.4f} drift={result['drift_after_best']:.4f} "
                  f"es={result['early_stop_triggered']} {result['runtime']:.0f}s{fail}")
            sys.stdout.flush()
            save_all(results, RUN_DIR)

    elapsed = time.time() - t0
    print(f'\n=== DONE in {elapsed/60:.1f}min ===')
    save_all(results, RUN_DIR)

    old = [r for r in results if r['checkpoint'] == 'old_10ep' and r['best_dist'] >= 0]
    new = [r for r in results if r['checkpoint'] == 'new_50ep' and r['best_dist'] >= 0]
    print(f'Old 10ep: best={np.mean([r["best_dist"] for r in old]):.4f} final={np.mean([r["final_dist"] for r in old]):.4f} success={sum(r["final_success_pose"] for r in old)} ever={sum(r["ever_success_pose"] for r in old)}')
    print(f'New 50ep: best={np.mean([r["best_dist"] for r in new]):.4f} final={np.mean([r["final_dist"] for r in new]):.4f} success={sum(r["final_success_pose"] for r in new)} ever={sum(r["ever_success_pose"] for r in new)}')

    if old:
        print(f'\nCEM: best={np.mean([r["best_dist"] for r in results if r["planner"]=="cem" and r["best_dist"]>=0]):.4f} success={sum(r["final_success_pose"] for r in results if r["planner"]=="cem")}')
        print(f'MPPI: best={np.mean([r["best_dist"] for r in results if r["planner"]=="mppi" and r["best_dist"]>=0]):.4f} success={sum(r["final_success_pose"] for r in results if r["planner"]=="mppi")}')

if __name__ == '__main__':
    main()
