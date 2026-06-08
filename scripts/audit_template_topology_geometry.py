#!/usr/bin/env python3
"""Phase TG-6: Audit all templates with fixed topology/geometry metrics."""

import sys, json, csv, os, argparse
sys.path.insert(0, '/home/brucewu/my_robot_project')
from pathlib import Path
import yaml
import numpy as np
from src.metrics.topology_geometry import compute_all_metrics, sanitize_metric_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template-source', default='data/sim/metadata/reset_templates_v0.json')
    parser.add_argument('--out-dir', default='runs/topology_geometry_audit_fixed')
    parser.add_argument('--max-templates', type=int, default=None)
    parser.add_argument('--families', default=None, help='Comma-separated families to include')
    parser.add_argument('--strict', action='store_true', help='Strict mode: fail on warnings')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load templates
    with open(args.template_source) as f:
        all_templates = json.load(f)
    
    if args.max_templates:
        all_templates = all_templates[:args.max_templates]
    
    if args.families:
        families_filter = set(args.families.split(','))
        all_templates = [t for t in all_templates if t.get('layout_family', 'unknown') in families_filter]
    
    print(f'Processing {len(all_templates)} templates...')
    
    # Audit each template
    all_metrics = []
    invalid_templates = []
    schema_warnings_all = []
    
    for tmpl in all_templates:
        template_id = tmpl.get('reset_template_id', 'unknown')
        family = tmpl.get('layout_family', 'unknown')
        
        try:
            metrics = compute_all_metrics(tmpl)
            all_metrics.append(metrics)
            
            # Check validity
            if not metrics['validation']['is_valid']:
                invalid_templates.append({
                    'template_id': template_id,
                    'family': family,
                    'reasons': metrics['validation']['invalid_reasons'],
                })
            
            # Collect schema warnings
            for w in metrics.get('schema_warnings', []):
                schema_warnings_all.append({'template_id': template_id, 'family': family, 'warning': w})
            
        except Exception as e:
            print(f'  ERROR: {template_id}: {e}')
            invalid_templates.append({
                'template_id': template_id,
                'family': family,
                'reasons': [str(e)],
            })
    
    print(f'Processed: {len(all_metrics)}, Invalid: {len(invalid_templates)}, Schema warnings: {len(schema_warnings_all)}')
    
    # 1. Per-template CSV
    csv_path = out_dir / 'template_topology_geometry.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'template_id', 'family', 'topology_pred', 'difficulty_level', 'difficulty_score',
            'object_goal_distance', 'goal_edge_distance_adjusted',
            'min_object_obstacle_distance_raw', 'min_goal_obstacle_distance_raw',
            'obstacle_count',
            'object_goal_path_blocked', 'ee_object_path_blocked',
            'blocking_score', 'object_goal_blocking_score', 'ee_object_blocking_score',
            'passage_width_estimate', 'edge_goal_score',
            'reachable_contact_sides', 'approach_feasibility_score',
            'ee_object_goal_alignment',
            'is_valid', 'dominant_geometric_challenge', 'warnings',
        ])
        for m in all_metrics:
            b = m['basic_metrics']
            p = m['path_metrics']
            c = m['contact_metrics']
            cl = m['classification']
            v = m['validation']
            writer.writerow([
                m['template_id'], m['source_family'],
                cl['topology_family_pred'], cl['difficulty_level'], f'{cl["difficulty_score"]:.4f}',
                f'{b["object_goal_distance"]:.4f}', f'{b["goal_edge_distance_adjusted"]:.4f}',
                sanitize_metric_value(b['min_object_obstacle_distance_raw']),
                sanitize_metric_value(b['min_goal_obstacle_distance_raw']),
                b['obstacle_count'],
                p['object_goal_path_blocked'], p['ee_object_path_blocked'],
                f'{p["blocking_score"]:.4f}', f'{p["object_goal_blocking_score"]:.4f}', f'{p["ee_object_blocking_score"]:.4f}',
                sanitize_metric_value(p['passage_width_estimate']),
                f'{p["edge_goal_score"]:.4f}',
                c['reachable_contact_sides'], f'{c["approach_feasibility_score"]:.4f}',
                f'{c["ee_object_goal_alignment"]:.4f}',
                v['is_valid'], cl['dominant_geometric_challenge'],
                '; '.join(cl.get('warnings', [])),
            ])
    print(f'Saved: {csv_path}')
    
    # 2. Per-template JSON
    json_path = out_dir / 'template_topology_geometry.json'
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str, allow_nan=False)
    print(f'Saved: {json_path}')
    
    # 3. By-family summary
    families = {}
    for m in all_metrics:
        fam = m['source_family']
        if fam not in families:
            families[fam] = []
        families[fam].append(m)
    
    summary_path = out_dir / 'by_family_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'family', 'count', 'mean_object_goal_distance', 'mean_goal_edge_distance',
            'mean_blocking_score', 'mean_obj_goal_blocking', 'mean_ee_blocking',
            'mean_passage_width', 'passage_finite_pct', 'mean_edge_goal_score',
            'mean_approach_score', 'mean_difficulty', 'invalid_count',
        ])
        for fam, metrics_list in sorted(families.items()):
            n = len(metrics_list)
            mog = np.mean([m['basic_metrics']['object_goal_distance'] for m in metrics_list])
            ged = np.mean([m['basic_metrics']['goal_edge_distance_adjusted'] for m in metrics_list])
            bs = np.mean([m['path_metrics']['blocking_score'] for m in metrics_list])
            obs = np.mean([m['path_metrics']['object_goal_blocking_score'] for m in metrics_list])
            ebs = np.mean([m['path_metrics']['ee_object_blocking_score'] for m in metrics_list])
            pw_vals = [m['path_metrics']['passage_width_estimate'] for m in metrics_list 
                       if m['path_metrics']['passage_width_estimate'] is not None]
            pw = np.mean(pw_vals) if pw_vals else None
            pw_pct = len(pw_vals) / n * 100
            eg = np.mean([m['path_metrics']['edge_goal_score'] for m in metrics_list])
            af = np.mean([m['contact_metrics']['approach_feasibility_score'] for m in metrics_list])
            diff = np.mean([m['classification']['difficulty_score'] for m in metrics_list])
            inv = sum(1 for m in metrics_list if not m['validation']['is_valid'])
            writer.writerow([
                fam, n, f'{mog:.4f}', f'{ged:.4f}', f'{bs:.4f}', f'{obs:.4f}', f'{ebs:.4f}',
                f'{pw:.4f}' if pw is not None else 'N/A',
                f'{pw_pct:.0f}%', f'{eg:.4f}', f'{af:.4f}', f'{diff:.4f}', inv,
            ])
    print(f'Saved: {summary_path}')
    
    # 4. Invalid templates
    if invalid_templates:
        inv_path = out_dir / 'invalid_templates.csv'
        with open(inv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['template_id', 'family', 'reasons'])
            for inv in invalid_templates:
                writer.writerow([inv['template_id'], inv['family'], '; '.join(inv['reasons'])])
        print(f'Saved: {inv_path}')
    
    # 5. Schema warnings
    if schema_warnings_all:
        warn_path = out_dir / 'schema_warnings.csv'
        with open(warn_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['template_id', 'family', 'warning'])
            for w in schema_warnings_all:
                writer.writerow([w['template_id'], w['family'], w['warning']])
        print(f'Saved: {warn_path}')
    
    # 6. Recommended template pool
    pool = {'open_space_easy': [], 'open_space_medium': [], 
            'blocking_high': [], 'narrow_passage_low_width': [], 'edge_goal_high': []}
    
    for m in all_metrics:
        if not m['validation']['is_valid']:
            continue
        
        cl = m['classification']
        p = m['path_metrics']
        entry = {
            'template_id': m['template_id'],
            'actual_family': m['source_family'],
            'difficulty': cl['difficulty_level'],
            'difficulty_score': cl['difficulty_score'],
            'topology_pred': cl['topology_family_pred'],
        }
        
        # Categorize
        pred = cl['topology_family_pred']
        diff = cl['difficulty_level']
        
        if pred == 'open_space' and diff == 'easy':
            pool['open_space_easy'].append(entry)
        elif pred == 'open_space' and diff == 'medium':
            pool['open_space_medium'].append(entry)
        elif pred == 'blocking' and p['blocking_score'] > 0.5:
            pool['blocking_high'].append(entry)
        elif pred == 'narrow_passage' and p['passage_width_estimate'] is not None:
            pool['narrow_passage_low_width'].append({**entry, 'passage_width': p['passage_width_estimate']})
        elif pred == 'edge_goal' and p['edge_goal_score'] > 0.5:
            pool['edge_goal_high'].append(entry)
    
    # Sort each category
    for cat in pool:
        pool[cat].sort(key=lambda x: x['difficulty_score'])
    
    pool_path = out_dir / 'recommended_closed_loop_template_pool.yaml'
    with open(pool_path, 'w') as f:
        yaml.dump(pool, f, default_flow_style=False, sort_keys=False)
    print(f'Saved: {pool_path}')
    
    # Print summary
    print(f'\n=== Summary by Family ===')
    for fam, metrics_list in sorted(families.items()):
        n = len(metrics_list)
        bs = np.mean([m['path_metrics']['blocking_score'] for m in metrics_list])
        eg = np.mean([m['path_metrics']['edge_goal_score'] for m in metrics_list])
        pw_vals = [m['path_metrics']['passage_width_estimate'] for m in metrics_list 
                   if m['path_metrics']['passage_width_estimate'] is not None]
        pw = np.mean(pw_vals) if pw_vals else float('inf')
        pw_str = f'{pw:.3f}' if pw < float('inf') else 'N/A'
        print(f'  {fam}: {n} templates, blocking={bs:.3f}, edge={eg:.3f}, passage_width={pw_str}')
    
    print(f'\n=== Recommended Pool ===')
    for cat, entries in pool.items():
        print(f'  {cat}: {len(entries)} templates')
    
    print(f'\nDone!')

if __name__ == '__main__':
    main()
