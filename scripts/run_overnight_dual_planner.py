#!/usr/bin/env python3
"""Overnight Dual-Planner Closed-Loop Runner.

Runs CEM and MPPI with learned rollout on open_space and classic templates.
Writes per-trial JSON and generates summaries.
"""

import sys, json, time, os, csv, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import yaml

sys.path.insert(0, '/home/brucewu/my_robot_project')

from src.models.rig_world import RIGWorldModel
from src.data.state_normalizer import StateNormalizer
from src.planners.learned_planner_adapter import CEMLearnedPlanner, MPPILearnedPlanner
from src.planners.action_conventions import PAPER1_CONVENTION
from src.envs.mujoco_push_env import MujocoPushEnv

# Thread control for parallel execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Constants
EE_SIZE, OBJ_SIZE, OBJ_MASS, OBJ_FRICTION = 0.015, 0.048, 0.038, 0.8
MAX_OBSTACLES = 3
RUN_DIR = Path('runs/overnight_dual_planner_closed_loop_20260525_033000')


def extract_state16(env):
    """Extract canonical_state16 from env, INCLUDING obstacle tokens.

    Token layout (matching training data format):
        0: EE       [x, y, 0, 1, vx, vy, 0, size_x, size_y, 0, 0, 0, 0, 0, contact, valid]
        1: OBJ      [x, y, sinθ, cosθ, vx, vy, ω, size_x, size_y, 1, 0, 0, mass, fric, 0, valid]
        2: GOAL     [x, y, sinθ, cosθ, 0, 0, 0, size_x, size_y, 1, 0, 0, mass, fric, 0, valid]
        3-5: OBSTACLE tokens (valid=1 if obstacle exists, 0 for padding)
    """
    ee = env.get_ee_pos()
    obj = env.get_object_pose()
    goal = env.get_goal_pose()
    t = np.zeros((6, 16), dtype=np.float32)

    # Token 0: EE
    t[0, :2] = ee
    t[0, 3] = 1.0
    t[0, 7:9] = EE_SIZE
    t[0, 14] = float(env.get_contact_flag())
    t[0, 15] = 1.0

    # Token 1: OBJ
    t[1, :2] = obj[:2]
    t[1, 2] = np.sin(obj[2])
    t[1, 3] = np.cos(obj[2])
    t[1, 7:9] = OBJ_SIZE
    t[1, 9] = 1.0
    t[1, 12:14] = [OBJ_MASS, OBJ_FRICTION]
    t[1, 15] = 1.0

    # Token 2: GOAL
    t[2, :2] = goal[:2]
    t[2, 2] = np.sin(goal[2])
    t[2, 3] = np.cos(goal[2])
    t[2, 7:9] = OBJ_SIZE
    t[2, 9] = 1.0
    t[2, 12:14] = [OBJ_MASS, OBJ_FRICTION]
    t[2, 15] = 1.0

    # Tokens 3-5: OBSTACLES (critical fix: training data has obstacles!)
    num_active = getattr(env, '_num_active_obstacles', 0)
    for i in range(min(MAX_OBSTACLES, num_active)):
        body_id = env._obstacle_body_ids[i]
        geom_id = env._obstacle_geom_ids[i]
        # Skip inactive obstacles (body at origin = unused slot)
        obs_pos = env.data.xpos[body_id][:2].copy()
        if np.allclose(obs_pos, 0.0, atol=1e-3):
            continue
        token_idx = 3 + i
        t[token_idx, :2] = obs_pos
        # Obstacle orientation from rotation matrix (column 0 for 2D)
        rot_mat = env.data.xmat[body_id].reshape(3, 3)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        t[token_idx, 2] = np.sin(yaw)
        t[token_idx, 3] = np.cos(yaw)
        # Obstacle size (half-extents, matching training data convention)
        geom_size = env.model.geom_size[geom_id]
        t[token_idx, 7] = geom_size[0]  # half-extent x
        t[token_idx, 8] = geom_size[1]  # half-extent y
        # Mass and friction (from template defaults)
        t[token_idx, 12] = 0.5  # obstacle mass (matches training data)
        t[token_idx, 13] = 0.8  # obstacle friction (matches training data)
        t[token_idx, 15] = 1.0  # valid flag

    return t


def run_one_trial(env, planner, template, max_steps, exec_steps, trial_id):
    """Run one closed-loop trial. Returns result dict."""
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
                env.step(action)
                total_steps += 1
                contacts += int(env.get_contact_flag())
                collisions += int(env.get_collision_flag())
                obj = env.get_object_pose()
                d = float(np.linalg.norm(obj[:2] - goal[:2]))
                best_d = min(best_d, d); final_d = d
                te = abs(np.arctan2(np.sin(obj[2] - goal[2]), np.cos(obj[2] - goal[2])))
                if d < 0.02 and np.degrees(te) < 10:
                    break
            if final_d < 0.02 and np.degrees(te) < 10:
                break
        
        elapsed = time.time() - t0
        return {
            'trial_id': trial_id,
            'success': bool(final_d < 0.02 and np.degrees(te) < 10),
            'initial_dist': init_d, 'final_dist': final_d, 'best_dist': best_d,
            'distance_improvement': init_d - best_d,
            'improved': best_d < init_d - 0.001,
            'contact_rate': contacts / max(total_steps, 1),
            'collision_rate': collisions / max(total_steps, 1),
            'total_steps': total_steps,
            'runtime_sec': elapsed,
            'failure_code': None,
            'planner_backend': result.planner_backend,
            'planner_config': result.planner_config,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            'trial_id': trial_id,
            'success': False, 'initial_dist': float('nan'), 'final_dist': float('nan'),
            'best_dist': float('nan'), 'distance_improvement': float('nan'), 'improved': False,
            'contact_rate': 0, 'collision_rate': 0, 'total_steps': 0,
            'runtime_sec': elapsed, 'failure_code': str(e)[:200],
            'planner_backend': 'ERROR', 'planner_config': {},
        }


def load_model(model_type):
    """Load model and normalizer."""
    model = RIGWorldModel(model_type=model_type, action_dim=2, gru_hidden=256, d_model=128, head_hidden_dim=256)
    ckpt = torch.load(f'runs/pilot_state16_mppi_stage2c/train_{model_type}/{model_type}/checkpoints/best.pt', 
                       map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    norm = StateNormalizer.load(f'runs/pilot_state16_mppi_stage2c/train_{model_type}/normalizer_state16.json')
    return model, norm


def create_planner(planner_type, config_name, model, norm, conv):
    """Create planner instance."""
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
            'mppi_medium': {'horizon': 60, 'num_samples': 1024, 'temperature': 0.1, 'init_std': 0.5,
                            'speed': 0.3, 'execute_steps': 10, 'max_mpc_steps': 80},
        }
        cfg = configs.get(config_name, configs['mppi_smoke'])
        return MPPILearnedPlanner(model=model, normalizer=norm, convention=conv, device='cpu', **cfg)
    else:
        raise ValueError(f'Unknown planner type: {planner_type}')


def load_templates():
    """Load template pool."""
    with open('data/sim/metadata/reset_templates_v0.json') as f:
        all_t = json.load(f)
    
    families = {}
    for t in all_t:
        fam = t.get('layout_family', 'unknown')
        if fam not in families:
            families[fam] = []
        families[fam].append(t)
    
    # Build template pool
    pool = {
        'open_space': families.get('open_space', [])[:10] + families.get('mild_offset', [])[:5],
        'blocking': families.get('blocking', [])[:5],
        'narrow_passage': families.get('narrow_passage', [])[:5],
        'edge_goal': families.get('edge_goal', [])[:5],
    }
    
    return pool


def write_trial_result(trial_result, trial_dir):
    """Write individual trial result JSON."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    path = trial_dir / f"{trial_result['trial_id']}.json"
    with open(path, 'w') as f:
        json.dump(trial_result, f, indent=2, default=str, allow_nan=False)


def update_manifest(trial_results, run_dir):
    """Update manifest.csv from trial results."""
    manifest_path = run_dir / 'manifest.csv'
    fields = ['trial_id', 'model', 'planner_type', 'config_name', 'template_id', 'family',
              'success', 'improved', 'initial_dist', 'final_dist', 'best_dist',
              'distance_improvement', 'contact_rate', 'collision_rate', 'total_steps',
              'runtime_sec', 'failure_code', 'planner_backend']
    
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in trial_results:
            writer.writerow(r)


def update_progress(trial_results, run_dir, start_time):
    """Update progress summary."""
    elapsed = time.time() - start_time
    total = len(trial_results)
    completed = sum(1 for r in trial_results if r.get('failure_code') is None)
    failed = total - completed
    
    cem_results = [r for r in trial_results if r.get('planner_type') == 'cem']
    mppi_results = [r for r in trial_results if r.get('planner_type') == 'mppi']
    
    summary = {
        'elapsed_hours': elapsed / 3600,
        'total_trials': total,
        'completed': completed,
        'failed': failed,
        'cem_trials': len(cem_results),
        'mppi_trials': len(mppi_results),
        'cem_improved': sum(1 for r in cem_results if r.get('improved')),
        'mppi_improved': sum(1 for r in mppi_results if r.get('improved')),
        'best_dist': min((r['best_dist'] for r in trial_results if np.isfinite(r.get('best_dist', float('nan')))), default=float('nan')),
        'success_count': sum(1 for r in trial_results if r.get('success')),
    }
    
    with open(run_dir / 'progress_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    with open(run_dir / 'progress_summary.md', 'w') as f:
        f.write(f"# Progress Summary\n\n")
        f.write(f"**Elapsed:** {elapsed/3600:.1f} hours\n")
        f.write(f"**Total:** {total} trials\n")
        f.write(f"**Completed:** {completed}, **Failed:** {failed}\n")
        f.write(f"**CEM:** {len(cem_results)} trials, {summary['cem_improved']} improved\n")
        f.write(f"**MPPI:** {len(mppi_results)} trials, {summary['mppi_improved']} improved\n")
        f.write(f"**Best dist:** {summary['best_dist']:.4f}m\n")
        f.write(f"**Success:** {summary['success_count']}\n")


def main():
    start_time = time.time()
    max_wall_time = 11 * 3600  # 11 hours
    
    print(f'=== Overnight Dual-Planner Closed-Loop ===')
    print(f'Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Run dir: {RUN_DIR}')
    
    # Load templates
    template_pool = load_templates()
    print(f'Templates: {sum(len(v) for v in template_pool.values())} total')
    
    # Action convention
    conv = PAPER1_CONVENTION
    conv.max_speed_mps = 0.5  # Fixed: was 0.75, but training data max is 0.5
    
    # Trial matrix
    trial_matrix = []
    trial_id = 0
    
    models = ['flat', 'object_centric', 'causality_aware']
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
    
    for model_type in models:
        for planner_type, config_name, family, max_steps, exec_steps in planner_configs:
            templates = template_pool.get(family, [])
            for i, tmpl in enumerate(templates[:3]):  # Max 3 per family initially
                trial_matrix.append({
                    'trial_id': f'trial_{trial_id:04d}',
                    'model': model_type,
                    'planner_type': planner_type,
                    'config_name': config_name,
                    'template': tmpl,
                    'template_id': tmpl.get('reset_template_id', f'tmpl_{i}'),
                    'family': family,
                    'max_steps': max_steps,
                    'exec_steps': exec_steps,
                })
                trial_id += 1
    
    print(f'Trial matrix: {len(trial_matrix)} trials')
    
    # Run trials
    trial_results = []
    consecutive_failures = 0
    
    # Load models once
    models_loaded = {}
    for model_type in models:
        print(f'Loading model: {model_type}')
        models_loaded[model_type] = load_model(model_type)
    
    # Create planners once
    planners_cache = {}
    
    for i, trial in enumerate(trial_matrix):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > max_wall_time:
            print(f'\n⏰ Time limit reached ({elapsed/3600:.1f}h). Stopping.')
            break
        
        model_type = trial['model']
        planner_type = trial['planner_type']
        config_name = trial['config_name']
        cache_key = (model_type, planner_type, config_name)
        
        # Create planner if not cached
        if cache_key not in planners_cache:
            model, norm = models_loaded[model_type]
            planners_cache[cache_key] = create_planner(planner_type, config_name, model, norm, conv)
        
        planner = planners_cache[cache_key]
        env = MujocoPushEnv(shape_type='T', control_dt=conv.control_dt, max_speed_mps=conv.max_speed_mps)
        
        # Run trial
        print(f'[{i+1}/{len(trial_matrix)}] {trial["trial_id"]}: {model_type} {planner_type}/{config_name} {trial["family"]}...', 
              end=' ', flush=True)
        
        result = run_one_trial(env, planner, trial['template'], trial['max_steps'], trial['exec_steps'], trial['trial_id'])
        result.update({
            'model': model_type, 'planner_type': planner_type, 'config_name': config_name,
            'template_id': trial['template_id'], 'family': trial['family'],
        })
        
        trial_results.append(result)
        
        # Print result
        if result.get('failure_code'):
            print(f'❌ FAIL: {result["failure_code"][:50]}')
            consecutive_failures += 1
        else:
            s = '✅' if result['success'] else ('📈' if result['improved'] else '📉')
            print(f'{s} best={result["best_dist"]:.3f} contact={result["contact_rate"]:.2f} ({result["runtime_sec"]:.0f}s)')
            consecutive_failures = 0
        
        # Check consecutive failures
        if consecutive_failures >= 10:
            print(f'\n❌ 10 consecutive failures. Stopping.')
            break
        
        # Write trial result
        write_trial_result(result, RUN_DIR / 'trial_results')
        
        # Update manifest and progress periodically
        if (i + 1) % 5 == 0:
            update_manifest(trial_results, RUN_DIR)
            update_progress(trial_results, RUN_DIR, start_time)
    
    # Final updates
    update_manifest(trial_results, RUN_DIR)
    update_progress(trial_results, RUN_DIR, start_time)
    
    # Generate summary
    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    
    cem_results = [r for r in trial_results if r.get('planner_type') == 'cem']
    mppi_results = [r for r in trial_results if r.get('planner_type') == 'mppi']
    
    print(f'Total: {len(trial_results)} trials')
    print(f'CEM: {len(cem_results)} trials, {sum(1 for r in cem_results if r.get("improved"))} improved')
    print(f'MPPI: {len(mppi_results)} trials, {sum(1 for r in mppi_results if r.get("improved"))} improved')
    print(f'Success: {sum(1 for r in trial_results if r.get("success"))}')
    
    best = min(trial_results, key=lambda r: r.get('best_dist', float('inf')))
    print(f'Best: {best["best_dist"]:.4f}m ({best["model"]} {best["planner_type"]})')
    
    # Save final summary
    summary = {
        'total_trials': len(trial_results),
        'cem_trials': len(cem_results), 'mppi_trials': len(mppi_results),
        'cem_improved': sum(1 for r in cem_results if r.get('improved')),
        'mppi_improved': sum(1 for r in mppi_results if r.get('improved')),
        'success_count': sum(1 for r in trial_results if r.get('success')),
        'best_dist': best['best_dist'], 'best_case': best['trial_id'],
        'wall_time_hours': (time.time() - start_time) / 3600,
    }
    with open(RUN_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f'\nDone! Results in {RUN_DIR}')


if __name__ == '__main__':
    main()
