#!/usr/bin/env python3
"""Phase TG-2: Audit all templates for topology/geometry metrics."""

import sys, json, csv, os
sys.path.insert(0, '/home/brucewu/my_robot_project')
from pathlib import Path
import yaml
from src.metrics.topology_geometry import compute_all_metrics

def main():
    # Load templates
    template_path = Path('data/sim/metadata/reset_templates_v0.json')
    with open(template_path) as f:
        all_templates = json.load(f)
    
    print(f'Loaded {len(all_templates)} templates')
    
    # Audit each template
    all_metrics = []
    invalid_templates = []
    
    for tmpl in all_templates:
        template_id = tmpl.get('reset_template_id', 'unknown')
        family = tmpl.get('layout_family', 'unknown')
        
        try:
            metrics = compute_all_metrics(tmpl)
            metrics['template_id'] = template_id
            metrics['family'] = family
            all_metrics.append(metrics)
            
            if not metrics['validation']['is_valid']:
                invalid_templates.append({
                    'template_id': template_id,
                    'family': family,
                    'reasons': metrics['validation']['invalid_reasons'],
                })
        except Exception as e:
            print(f'  ERROR: {template_id}: {e}')
            invalid_templates.append({
                'template_id': template_id,
                'family': family,
                'reasons': [str(e)],
            })
    
    print(f'Processed: {len(all_metrics)}, Invalid: {len(invalid_templates)}')
    
    # Output directory
    out_dir = Path('runs/topology_geometry_audit')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Per-template CSV
    csv_path = out_dir / 'template_topology_geometry.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'template_id', 'family',
            'object_goal_distance', 'goal_edge_distance', 'object_edge_distance',
            'min_object_obstacle_distance', 'min_goal_obstacle_distance',
            'ee_to_object_distance', 'obstacle_count',
            'direct_path_blocked', 'obstacle_between_object_goal',
            'blocking_score', 'passage_width_estimate', 'edge_goal_score',
            'reachable_contact_sides', 'approach_feasibility_score',
            'ee_object_goal_alignment',
            'difficulty_score', 'difficulty_level', 'dominant_geometric_challenge',
            'topology_family_pred', 'is_valid', 'warnings',
        ])
        for m in all_metrics:
            b = m['basic_metrics']
            p = m['path_metrics']
            c = m['contact_metrics']
            cl = m['classification']
            v = m['validation']
            writer.writerow([
                m['template_id'], m['family'],
                f'{b["object_goal_distance"]:.4f}', f'{b["goal_edge_distance"]:.4f}',
                f'{b["object_edge_distance"]:.4f}',
                f'{b["min_object_obstacle_distance"]:.4f}' if b['min_object_obstacle_distance'] < float('inf') else 'inf',
                f'{b["min_goal_obstacle_distance"]:.4f}' if b['min_goal_obstacle_distance'] < float('inf') else 'inf',
                f'{b["ee_to_object_distance"]:.4f}', b['obstacle_count'],
                p['direct_path_blocked'], p['obstacle_between_object_goal'],
                f'{p["blocking_score"]:.4f}',
                f'{p["passage_width_estimate"]:.4f}' if p['passage_width_estimate'] < float('inf') else 'inf',
                f'{p["edge_goal_score"]:.4f}',
                c['reachable_contact_sides'], f'{c["approach_feasibility_score"]:.4f}',
                f'{c["ee_object_goal_alignment"]:.4f}',
                f'{cl["difficulty_score"]:.4f}', cl['difficulty_level'],
                cl['dominant_geometric_challenge'], cl['topology_family_pred'],
                v['is_valid'], '; '.join(cl['warnings']),
            ])
    print(f'Saved: {csv_path}')
    
    # Per-template JSON
    json_path = out_dir / 'template_topology_geometry.json'
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f'Saved: {json_path}')
    
    # By-family summary
    families = {}
    for m in all_metrics:
        fam = m['family']
        if fam not in families:
            families[fam] = []
        families[fam].append(m)
    
    summary_path = out_dir / 'by_family_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'family', 'count', 'mean_object_goal_distance', 'mean_goal_edge_distance',
            'mean_blocking_score', 'mean_passage_width', 'mean_edge_goal_score',
            'mean_approach_feasibility', 'mean_difficulty', 'invalid_count',
        ])
        for fam, metrics_list in sorted(families.items()):
            n = len(metrics_list)
            mog = sum(m['basic_metrics']['object_goal_distance'] for m in metrics_list) / n
            ged = sum(m['basic_metrics']['goal_edge_distance'] for m in metrics_list) / n
            bs = sum(m['path_metrics']['blocking_score'] for m in metrics_list) / n
            pw_vals = [m['path_metrics']['passage_width_estimate'] for m in metrics_list 
                       if m['path_metrics']['passage_width_estimate'] < float('inf')]
            pw = sum(pw_vals) / len(pw_vals) if pw_vals else float('inf')
            eg = sum(m['path_metrics']['edge_goal_score'] for m in metrics_list) / n
            af = sum(m['contact_metrics']['approach_feasibility_score'] for m in metrics_list) / n
            diff = sum(m['classification']['difficulty_score'] for m in metrics_list) / n
            inv = sum(1 for m in metrics_list if not m['validation']['is_valid'])
            writer.writerow([
                fam, n, f'{mog:.4f}', f'{ged:.4f}', f'{bs:.4f}',
                f'{pw:.4f}' if pw < float('inf') else 'inf',
                f'{eg:.4f}', f'{af:.4f}', f'{diff:.4f}', inv,
            ])
    print(f'Saved: {summary_path}')
    
    # Invalid templates
    if invalid_templates:
        inv_path = out_dir / 'invalid_templates.csv'
        with open(inv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['template_id', 'family', 'reasons'])
            for inv in invalid_templates:
                writer.writerow([inv['template_id'], inv['family'], '; '.join(inv['reasons'])])
        print(f'Saved: {inv_path}')
    
    # Recommended template pool
    pool = {'open_space': [], 'blocking': [], 'narrow_passage': [], 'edge_goal': [], 'other': []}
    for m in all_metrics:
        if not m['validation']['is_valid']:
            continue
        fam = m['classification']['topology_family_pred']
        if fam not in pool:
            fam = 'other'
        pool[fam].append({
            'template_id': m['template_id'],
            'actual_family': m['family'],
            'difficulty': m['classification']['difficulty_level'],
            'difficulty_score': m['classification']['difficulty_score'],
        })
    
    # Sort by difficulty within each family
    for fam in pool:
        pool[fam].sort(key=lambda x: x['difficulty_score'])
    
    pool_path = out_dir / 'recommended_closed_loop_template_pool.yaml'
    with open(pool_path, 'w') as f:
        yaml.dump(pool, f, default_flow_style=False, sort_keys=False)
    print(f'Saved: {pool_path}')
    
    # Print summary
    print(f'\n=== Summary by Family ===')
    for fam, metrics_list in sorted(families.items()):
        n = len(metrics_list)
        bs = sum(m['path_metrics']['blocking_score'] for m in metrics_list) / n
        diff = sum(m['classification']['difficulty_score'] for m in metrics_list) / n
        print(f'  {fam}: {n} templates, mean_blocking={bs:.3f}, mean_difficulty={diff:.3f}')
    
    print(f'\nDone!')

if __name__ == '__main__':
    main()
