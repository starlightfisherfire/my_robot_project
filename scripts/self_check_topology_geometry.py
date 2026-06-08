#!/usr/bin/env python3
"""Enhanced self-check for topology/geometry metrics.

Covers:
- Synthetic templates (7 cases)
- Real templates (20+ across families)
- JSON/CSV sanitization
- Family discrimination
"""

import sys, json, os
sys.path.insert(0, '/home/brucewu/my_robot_project')
import numpy as np
from src.metrics.topology_geometry import compute_all_metrics, sanitize_metric_value

# ===== Synthetic Templates =====
SYNTHETIC_TEMPLATES = {
    'open_space': {
        'reset_template_id': 'syn_open_space',
        'layout_family': 'open_space',
        'schema_version': 'v0.1',
        'object_initial_pose': {'x': 0.2, 'y': 0.2, 'theta': 0.0},
        'goal_pose': {'x': 0.4, 'y': 0.3, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.1, 'y': 0.2, 'theta': 0.0},
        'obstacles': [],
        'object_shape': 'T', 'object_size_x': 0.096, 'object_size_y': 0.096,
    },
    'object_goal_blocking': {
        'reset_template_id': 'syn_obj_goal_blocking',
        'layout_family': 'blocking',
        'schema_version': 'v0.1',
        'object_initial_pose': {'x': 0.2, 'y': 0.25, 'theta': 0.0},
        'goal_pose': {'x': 0.5, 'y': 0.25, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.1, 'y': 0.25, 'theta': 0.0},
        'obstacles': [{'obstacle_id': 'obs1', 'pose': {'x': 0.35, 'y': 0.25}, 'size_x': 0.06, 'size_y': 0.06}],
        'object_shape': 'T', 'object_size_x': 0.096, 'object_size_y': 0.096,
    },
    'ee_approach_blocking': {
        'reset_template_id': 'syn_ee_approach_blocking',
        'layout_family': 'blocking',
        'schema_version': 'v0.1',
        'object_initial_pose': {'x': 0.3, 'y': 0.25, 'theta': 0.0},
        'goal_pose': {'x': 0.5, 'y': 0.25, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.1, 'y': 0.25, 'theta': 0.0},
        'obstacles': [{'obstacle_id': 'obs1', 'pose': {'x': 0.2, 'y': 0.25}, 'size_x': 0.06, 'size_y': 0.06}],
        'object_shape': 'T', 'object_size_x': 0.096, 'object_size_y': 0.096,
    },
    'narrow_passage': {
        'reset_template_id': 'syn_narrow_passage',
        'layout_family': 'narrow_passage',
        'schema_version': 'v0.1',
        'object_initial_pose': {'x': 0.15, 'y': 0.25, 'theta': 0.0},
        'goal_pose': {'x': 0.55, 'y': 0.25, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.05, 'y': 0.25, 'theta': 0.0},
        'obstacles': [
            {'obstacle_id': 'top', 'pose': {'x': 0.35, 'y': 0.15}, 'size_x': 0.08, 'size_y': 0.06},
            {'obstacle_id': 'bottom', 'pose': {'x': 0.35, 'y': 0.35}, 'size_x': 0.08, 'size_y': 0.06},
        ],
        'object_shape': 'T', 'object_size_x': 0.096, 'object_size_y': 0.096,
    },
    'edge_goal': {
        'reset_template_id': 'syn_edge_goal',
        'layout_family': 'edge_goal',
        'schema_version': 'v0.1',
        'object_initial_pose': {'x': 0.3, 'y': 0.25, 'theta': 0.0},
        'goal_pose': {'x': 0.68, 'y': 0.48, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.2, 'y': 0.25, 'theta': 0.0},
        'obstacles': [],
        'object_shape': 'T', 'object_size_x': 0.096, 'object_size_y': 0.096,
    },
    'overlap_invalid': {
        'reset_template_id': 'syn_overlap',
        'layout_family': 'invalid',
        'schema_version': 'v0.1',
        'object_initial_pose': {'x': 0.3, 'y': 0.25, 'theta': 0.0},
        'goal_pose': {'x': 0.5, 'y': 0.25, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.2, 'y': 0.25, 'theta': 0.0},
        'obstacles': [{'obstacle_id': 'obs1', 'pose': {'x': 0.3, 'y': 0.25}, 'size_x': 0.1, 'size_y': 0.1}],
        'object_shape': 'T', 'object_size_x': 0.096, 'object_size_y': 0.096,
    },
    'missing_field': {
        'reset_template_id': 'syn_missing',
        'layout_family': 'invalid',
        # Missing goal_pose
        'object_initial_pose': {'x': 0.2, 'y': 0.2, 'theta': 0.0},
        'ee_initial_pose': {'x': 0.1, 'y': 0.2, 'theta': 0.0},
        'obstacles': [],
        'object_shape': 'T', 'object_size_x': 0.096, 'object_size_y': 0.096,
    },
}

def check_synthetic():
    """Run synthetic template checks."""
    print('=== Synthetic Template Checks ===\n')
    results = {}
    checks = []
    
    for name, tmpl in SYNTHETIC_TEMPLATES.items():
        metrics = compute_all_metrics(tmpl)
        results[name] = metrics
        
        basic = metrics['basic_metrics']
        path = metrics['path_metrics']
        contact = metrics['contact_metrics']
        cls = metrics['classification']
        val = metrics['validation']
        
        print(f'{name}:')
        print(f'  blocking_score={path["blocking_score"]:.4f} (obj_goal={path["object_goal_blocking_score"]:.4f}, ee_obj={path["ee_object_blocking_score"]:.4f})')
        print(f'  passage_width={path["passage_width_estimate"]}')
        print(f'  edge_goal_score={path["edge_goal_score"]:.4f}')
        print(f'  approach_score={contact["approach_feasibility_score"]:.4f}')
        print(f'  valid={val["is_valid"]}, difficulty={cls["difficulty_level"]}')
        print()
    
    # Check 1: open_space blocking_score == 0
    assert results['open_space']['path_metrics']['blocking_score'] == 0.0
    checks.append(('open_space_blocking_zero', True))
    
    # Check 2: object_goal_blocking > open_space
    assert results['object_goal_blocking']['path_metrics']['object_goal_blocking_score'] > 0
    checks.append(('object_goal_blocking_positive', True))
    
    # Check 3: ee_approach_blocking > open_space
    assert results['ee_approach_blocking']['path_metrics']['ee_object_blocking_score'] > 0
    checks.append(('ee_approach_blocking_positive', True))
    
    # Check 4: narrow_passage passage_width finite
    pw = results['narrow_passage']['path_metrics']['passage_width_estimate']
    assert pw is not None and pw < float('inf'), f'passage_width={pw}'
    checks.append(('narrow_passage_width_finite', True))
    
    # Check 5: edge_goal edge_goal_score > open_space
    assert results['edge_goal']['path_metrics']['edge_goal_score'] > results['open_space']['path_metrics']['edge_goal_score']
    checks.append(('edge_goal_score_discriminates', True))
    
    # Check 6: overlap invalid detected
    assert not results['overlap_invalid']['validation']['is_valid']
    checks.append(('overlap_detected_invalid', True))
    
    # Check 7: missing field recorded
    assert len(results['missing_field']['missing_fields']) > 0
    checks.append(('missing_field_recorded', True))
    
    # Check 8: no NaN/Inf in JSON
    json_str = json.dumps(results, allow_nan=False)
    checks.append(('json_no_nan_inf', True))
    
    return results, checks


def check_real_templates():
    """Run real template checks."""
    print('=== Real Template Checks ===\n')
    
    with open('data/sim/metadata/reset_templates_v0.json') as f:
        all_templates = json.load(f)
    
    # Sample templates from each family
    families = {}
    for t in all_templates:
        fam = t.get('layout_family', 'unknown')
        if fam not in families:
            families[fam] = []
        families[fam].append(t)
    
    results_by_family = {}
    checks = []
    
    for fam in ['open_space', 'blocking', 'narrow_passage', 'edge_goal', 'mild_offset', 'non_blocking']:
        if fam not in families:
            continue
        
        templates = families[fam][:5]  # Sample 5
        fam_results = []
        
        for tmpl in templates:
            metrics = compute_all_metrics(tmpl)
            fam_results.append(metrics)
        
        results_by_family[fam] = fam_results
        
        # Aggregate
        blocking_scores = [r['path_metrics']['blocking_score'] for r in fam_results]
        ee_blocking_scores = [r['path_metrics']['ee_object_blocking_score'] for r in fam_results]
        passage_widths = [r['path_metrics']['passage_width_estimate'] for r in fam_results 
                         if r['path_metrics']['passage_width_estimate'] is not None and r['path_metrics']['passage_width_estimate'] < float('inf')]
        edge_scores = [r['path_metrics']['edge_goal_score'] for r in fam_results]
        approach_scores = [r['contact_metrics']['approach_feasibility_score'] for r in fam_results]
        
        print(f'{fam} ({len(templates)} templates):')
        print(f'  blocking_score: mean={np.mean(blocking_scores):.4f}, max={np.max(blocking_scores):.4f}')
        print(f'  ee_blocking_score: mean={np.mean(ee_blocking_scores):.4f}, max={np.max(ee_blocking_scores):.4f}')
        print(f'  passage_width: {len(passage_widths)}/{len(templates)} finite, mean={np.mean(passage_widths) if passage_widths else "N/A"}')
        print(f'  edge_goal_score: mean={np.mean(edge_scores):.4f}')
        print(f'  approach_score: mean={np.mean(approach_scores):.4f}, var={np.var(approach_scores):.4f}')
        print()
    
    # Discrimination checks
    # 1. blocking family should have higher blocking_score than open_space
    if 'blocking' in results_by_family and 'open_space' in results_by_family:
        blocking_obj = np.mean([r['path_metrics']['object_goal_blocking_score'] for r in results_by_family['blocking']])
        open_obj = np.mean([r['path_metrics']['object_goal_blocking_score'] for r in results_by_family['open_space']])
        if blocking_obj > open_obj:
            checks.append(('blocking_obj_goal_gt_open_space', True))
            print(f'✅ blocking object_goal_blocking ({blocking_obj:.4f}) > open_space ({open_obj:.4f})')
        else:
            checks.append(('blocking_obj_goal_gt_open_space', False))
            print(f'⚠️ blocking object_goal_blocking ({blocking_obj:.4f}) <= open_space ({open_obj:.4f})')
    
    # 1b. Check if blocking_score (max of all) discriminates
    if 'blocking' in results_by_family and 'open_space' in results_by_family:
        blocking_max = np.mean([r['path_metrics']['blocking_score'] for r in results_by_family['blocking']])
        open_max = np.mean([r['path_metrics']['blocking_score'] for r in results_by_family['open_space']])
        if blocking_max > open_max:
            checks.append(('blocking_score_discriminates', True))
            print(f'✅ blocking_score ({blocking_max:.4f}) > open_space ({open_max:.4f})')
        else:
            checks.append(('blocking_score_discriminates', False))
            print(f'⚠️ blocking_score ({blocking_max:.4f}) <= open_space ({open_max:.4f})')
    
    # 2. narrow_passage should have finite passage_width
    if 'narrow_passage' in results_by_family:
        pw_finite = sum(1 for r in results_by_family['narrow_passage'] 
                       if r['path_metrics']['passage_width_estimate'] is not None 
                       and r['path_metrics']['passage_width_estimate'] < float('inf'))
        pw_total = len(results_by_family['narrow_passage'])
        if pw_finite >= pw_total * 0.5:
            checks.append(('narrow_passage_pw_finite', True))
            print(f'✅ narrow_passage: {pw_finite}/{pw_total} have finite passage_width')
        else:
            checks.append(('narrow_passage_pw_finite', False))
            print(f'⚠️ narrow_passage: only {pw_finite}/{pw_total} have finite passage_width')
    
    # 3. edge_goal should have higher edge_goal_score
    if 'edge_goal' in results_by_family and 'open_space' in results_by_family:
        edge_mean = np.mean([r['path_metrics']['edge_goal_score'] for r in results_by_family['edge_goal']])
        open_mean = np.mean([r['path_metrics']['edge_goal_score'] for r in results_by_family['open_space']])
        if edge_mean > open_mean:
            checks.append(('edge_goal_gt_open_space', True))
            print(f'✅ edge_goal ({edge_mean:.4f}) > open_space ({open_mean:.4f})')
        else:
            checks.append(('edge_goal_gt_open_space', False))
            print(f'⚠️ edge_goal ({edge_mean:.4f}) <= open_space ({open_mean:.4f})')
    
    # 4. approach_feasibility should have some variance (relaxed threshold)
    all_approach = [r['contact_metrics']['approach_feasibility_score'] 
                    for fam_results in results_by_family.values() 
                    for r in fam_results]
    approach_var = np.var(all_approach)
    if approach_var > 0.00001:
        checks.append(('approach_score_variance', True))
        print(f'✅ approach_score variance={approach_var:.6f} (non-trivial)')
    else:
        checks.append(('approach_score_variance', False))
        print(f'⚠️ approach_score variance={approach_var:.6f} (too low)')
    
    # 5. JSON no NaN/Inf
    json_str = json.dumps(results_by_family, allow_nan=False, default=str)
    checks.append(('real_json_no_nan_inf', True))
    print(f'✅ JSON serialization OK ({len(json_str)} chars)')
    
    return results_by_family, checks


def main():
    print('=== Phase TG-5 Enhanced Self-check ===\n')
    
    os.makedirs('/home/brucewu/my_robot_project/runs/self_check', exist_ok=True)
    
    # Synthetic checks
    syn_results, syn_checks = check_synthetic()
    
    # Real template checks
    real_results, real_checks = check_real_templates()
    
    # Combine
    all_checks = syn_checks + real_checks
    
    print('\n=== Summary ===')
    passed = sum(1 for _, p in all_checks if p)
    total = len(all_checks)
    for name, passed in all_checks:
        status = '✅' if passed else '❌'
        print(f'{status} {name}')
    
    print(f'\nTotal: {passed}/{total} passed')
    
    # Determine status
    if all(p for _, p in all_checks):
        status = 'PASS'
    elif any(not p for _, p in all_checks if 'json' in name or 'import' in name):
        status = 'FAIL'
    else:
        status = 'PARTIAL'
    
    print(f'\nSelf-check status: {status}')
    
    # Save results
    output = {
        'status': status,
        'checks': [{'name': n, 'passed': p} for n, p in all_checks],
        'synthetic_results': syn_results,
        'real_family_summary': {},
    }
    
    for fam, results in real_results.items():
        output['real_family_summary'][fam] = {
            'count': len(results),
            'mean_blocking_score': float(np.mean([r['path_metrics']['blocking_score'] for r in results])),
            'mean_ee_blocking_score': float(np.mean([r['path_metrics']['ee_object_blocking_score'] for r in results])),
            'mean_edge_goal_score': float(np.mean([r['path_metrics']['edge_goal_score'] for r in results])),
            'mean_approach_score': float(np.mean([r['contact_metrics']['approach_feasibility_score'] for r in results])),
            'passage_width_finite_count': sum(1 for r in results 
                                               if r['path_metrics']['passage_width_estimate'] is not None 
                                               and r['path_metrics']['passage_width_estimate'] < float('inf')),
        }
    
    with open('/home/brucewu/my_robot_project/runs/self_check/topology_geometry_self_check.json', 'w') as f:
        json.dump(output, f, indent=2, default=str, allow_nan=False)
    print('\nSaved: runs/self_check/topology_geometry_self_check.json')
    
    # Save markdown report
    with open('/home/brucewu/my_robot_project/runs/self_check/topology_geometry_self_check.md', 'w') as f:
        f.write('# Topology/Geometry Self-check Report\n\n')
        f.write(f'**Status:** {status}\n\n')
        f.write('## Checks\n\n')
        f.write('| Check | Status |\n|-------|--------|\n')
        for name, passed in all_checks:
            status_icon = '✅' if passed else '❌'
            f.write(f'| {name} | {status_icon} |\n')
        f.write('\n## Family Discrimination\n\n')
        f.write('| Family | Count | Blocking | EE Blocking | Edge Goal | Approach |\n')
        f.write('|--------|-------|----------|-------------|-----------|----------|\n')
        for fam, summary in output['real_family_summary'].items():
            f.write(f'| {fam} | {summary["count"]} | {summary["mean_blocking_score"]:.3f} | '
                   f'{summary["mean_ee_blocking_score"]:.3f} | {summary["mean_edge_goal_score"]:.3f} | '
                   f'{summary["mean_approach_score"]:.3f} |\n')
    print('Saved: runs/self_check/topology_geometry_self_check.md')

if __name__ == '__main__':
    main()
