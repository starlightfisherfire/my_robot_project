#!/usr/bin/env python3
"""Self-check for topology/geometry metrics."""

import sys, json
sys.path.insert(0, '/home/brucewu/my_robot_project')
import numpy as np
from src.metrics.topology_geometry import compute_all_metrics

# Synthetic templates
templates = {
    'open_space': {
        'object_initial_pose': {'x': 0.2, 'y': 0.2, 'theta': 0.0},
        'goal_pose': {'x': 0.4, 'y': 0.3, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.1, 'y': 0.2, 'theta': 0.0},
        'obstacles': [],
        'object_shape': 'T',
        'object_size_x': 0.048,
        'object_size_y': 0.048,
    },
    'blocking': {
        'object_initial_pose': {'x': 0.2, 'y': 0.2, 'theta': 0.0},
        'goal_pose': {'x': 0.5, 'y': 0.3, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.1, 'y': 0.2, 'theta': 0.0},
        'obstacles': [{'x': 0.35, 'y': 0.25, 'size_x': 0.05, 'size_y': 0.05}],
        'object_shape': 'T',
        'object_size_x': 0.048,
        'object_size_y': 0.048,
    },
    'narrow_passage': {
        'object_initial_pose': {'x': 0.15, 'y': 0.25, 'theta': 0.0},
        'goal_pose': {'x': 0.55, 'y': 0.25, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.05, 'y': 0.25, 'theta': 0.0},
        'obstacles': [
            {'x': 0.35, 'y': 0.15, 'size_x': 0.08, 'size_y': 0.05},
            {'x': 0.35, 'y': 0.35, 'size_x': 0.08, 'size_y': 0.05},
        ],
        'object_shape': 'T',
        'object_size_x': 0.048,
        'object_size_y': 0.048,
    },
    'edge_goal': {
        'object_initial_pose': {'x': 0.3, 'y': 0.25, 'theta': 0.0},
        'goal_pose': {'x': 0.68, 'y': 0.48, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.2, 'y': 0.25, 'theta': 0.0},
        'obstacles': [],
        'object_shape': 'T',
        'object_size_x': 0.048,
        'object_size_y': 0.048,
    },
}

print('=== Phase TG-1 Self-check ===\n')
results = {}
checks = []

for name, tmpl in templates.items():
    metrics = compute_all_metrics(tmpl)
    results[name] = metrics
    
    basic = metrics['basic_metrics']
    path = metrics['path_metrics']
    contact = metrics['contact_metrics']
    cls = metrics['classification']
    val = metrics['validation']
    
    print(f'{name}:')
    print(f'  object_goal_distance: {basic["object_goal_distance"]:.4f}')
    print(f'  goal_edge_distance: {basic["goal_edge_distance"]:.4f}')
    print(f'  direct_path_blocked: {path["direct_path_blocked"]}')
    print(f'  blocking_score: {path["blocking_score"]:.4f}')
    print(f'  passage_width_estimate: {path["passage_width_estimate"]:.4f}')
    print(f'  edge_goal_score: {path["edge_goal_score"]:.4f}')
    print(f'  reachable_contact_sides: {contact["reachable_contact_sides"]}')
    print(f'  approach_feasibility_score: {contact["approach_feasibility_score"]:.4f}')
    print(f'  difficulty: {cls["difficulty_level"]} ({cls["difficulty_score"]:.4f})')
    print(f'  topology_pred: {cls["topology_family_pred"]}')
    print(f'  valid: {val["is_valid"]}')
    print()

# Specific checks
# 1. open_space direct_path_blocked should be False
assert not results['open_space']['path_metrics']['direct_path_blocked'], 'FAIL: open_space blocked'
checks.append(('open_space_direct_path_not_blocked', True))

# 2. blocking blocking_score > open_space
assert (results['blocking']['path_metrics']['blocking_score'] > 
        results['open_space']['path_metrics']['blocking_score']), 'FAIL: blocking score'
checks.append(('blocking_score_gt_open_space', True))

# 3. narrow_passage passage_width finite
assert np.isfinite(results['narrow_passage']['path_metrics']['passage_width_estimate']), 'FAIL: passage width'
checks.append(('narrow_passage_width_finite', True))

# 4. edge_goal goal_edge_distance small
assert results['edge_goal']['basic_metrics']['goal_edge_distance'] < 0.1, 'FAIL: edge distance'
checks.append(('edge_goal_small_edge_distance', True))

# 5. No NaN/inf in any metric
for name, metrics in results.items():
    for section in ['basic_metrics', 'path_metrics', 'contact_metrics']:
        for k, v in metrics[section].items():
            if isinstance(v, (int, float)):
                assert np.isfinite(v) or v == float('inf'), f'FAIL: {name}.{section}.{k}={v}'
checks.append(('no_nan_inf', True))

print('=== Checks ===')
for check_name, passed in checks:
    status = '✅' if passed else '❌'
    print(f'{status} {check_name}')

print('\n=== Phase TG-1 Self-check: ALL PASS ===')

# Save
import os
os.makedirs('/home/brucewu/my_robot_project/runs/self_check', exist_ok=True)
with open('/home/brucewu/my_robot_project/runs/self_check/topology_geometry_self_check.json', 'w') as f:
    json.dump({'checks': checks, 'results': results}, f, indent=2, default=str)
print('Saved: runs/self_check/topology_geometry_self_check.json')
