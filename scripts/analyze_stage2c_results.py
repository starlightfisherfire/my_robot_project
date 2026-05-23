#!/usr/bin/env python3
"""MPPI Stage 2C analysis script with self-validation.
Generates visualizations, statistical analysis, and insights.
Supports self-test with synthetic data, or real data from sweep.
"""
import csv, sys, argparse, math, json, os
from pathlib import Path
from collections import defaultdict
import numpy as np

# ── Try importing visualization libs ──────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping visualizations")

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available, skipping statistical tests")


# ── Data loading ──────────────────────────────────────────────────────────

def load_manifest(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            status = r.get('status','')
            if status in ('completed','True','true'):
                rows.append(r)
            elif status == '' and r.get('error','') == '' and r.get('ee_path_length_m','') != '':
                rows.append(r)
    return rows

def sf(r, key, default=0.0):
    try:
        v = r.get(key, default)
        if v == '' or v is None:
            return default
        return float(v)
    except (ValueError, TypeError):
        return default

def sb(r, key):
    v = str(r.get(key,'false')).lower()
    return v in ('true','1','yes','True')


# ── Synthetic data generation for self-test ───────────────────────────────

def generate_synthetic_manifest(n_rows=100, output_path=None):
    """Generate synthetic manifest for self-testing."""
    np.random.seed(42)
    
    top_configs = ['A', 'B', 'C']
    horizons = [100, 120, 140]
    num_samples_list = [1024, 2048]
    init_stds = [0.5, 0.7, 1.0]
    families = ['passage_direct_narrow', 'passage_bypass_wide', 'passage_bypass_medium',
                'passage_bypass_narrow', 'blocking_hard']
    
    rows = []
    for i in range(n_rows):
        tc = np.random.choice(top_configs)
        h = np.random.choice(horizons)
        n = np.random.choice(num_samples_list)
        std = np.random.choice(init_stds)
        fam = np.random.choice(families)
        
        # Simulate realistic metrics
        base_success = 0.5 + 0.1 * (h - 100) / 40 + 0.05 * (n - 1024) / 1024
        success = np.random.random() < base_success
        
        ee_path = np.random.exponential(5.0) + 1.0
        obj_path = ee_path * np.random.uniform(0.3, 0.8)
        net_progress = np.random.uniform(0.01, 0.3) if success else np.random.uniform(-0.05, 0.1)
        
        wasted = ee_path / max(net_progress, 0.001)
        wasted_capped = min(wasted, 100.0)
        prog_eff = net_progress / max(ee_path, 0.001)
        
        # Segment metrics
        early_frac = np.random.uniform(0.2, 0.4)
        middle_frac = np.random.uniform(0.2, 0.4)
        late_frac = 1.0 - early_frac - middle_frac
        
        early_ee = ee_path * early_frac
        middle_ee = ee_path * middle_frac
        late_ee = ee_path * late_frac
        
        early_prog = net_progress * np.random.uniform(0.0, 0.3)
        middle_prog = net_progress * np.random.uniform(0.0, 0.3)
        late_prog = net_progress - early_prog - middle_prog
        
        row = {
            'stage': 'stage2c',
            'top_config_label': tc,
            'config': f'mppi_sp{tc}_h{h}_n{n}_std{std}_{fam}_idx0',
            'family': fam,
            'horizon': str(h),
            'num_samples': str(n),
            'init_std': str(std),
            'status': 'completed',
            'success': str(success),
            'success_pose_2mm_10deg': str(success),
            'success_pose_5mm_10deg': str(success and np.random.random() > 0.1),
            'success_pose_10mm_10deg': str(success and np.random.random() > 0.2),
            'reached_pose_10mm_10deg_once': str(success or np.random.random() > 0.3),
            'ee_path_length_m': str(round(ee_path, 6)),
            'object_path_length_m': str(round(obj_path, 6)),
            'net_progress_m': str(round(net_progress, 6)),
            'progress_efficiency_ee': str(round(prog_eff, 8)),
            'wasted_motion_ratio': str(round(wasted, 4)),
            'wasted_motion_ratio_capped': str(round(wasted_capped, 4)),
            'random_walk_flag': str(not success and ee_path > 1.5 and net_progress < 0.05),
            'inefficient_success_flag': str(success and wasted_capped > 20),
            'excessive_wander_flag': str(ee_path > 10.0 or wasted_capped > 50),
            'clean_success_flag': str(success and wasted_capped <= 20 and prog_eff >= 0.05),
            'meaningless_exploration_flag': str(np.random.random() < 0.2),
            'front_loaded_wander_flag': str(np.random.random() < 0.3),
            'late_breakthrough_flag': str(success and np.random.random() < 0.3),
            'early_ee_path_length_m': str(round(early_ee, 6)),
            'middle_ee_path_length_m': str(round(middle_ee, 6)),
            'late_ee_path_length_m': str(round(late_ee, 6)),
            'early_object_path_length_m': str(round(obj_path * early_frac, 6)),
            'middle_object_path_length_m': str(round(obj_path * middle_frac, 6)),
            'late_object_path_length_m': str(round(obj_path * late_frac, 6)),
            'early_progress_m': str(round(early_prog, 6)),
            'middle_progress_m': str(round(middle_prog, 6)),
            'late_progress_m': str(round(late_prog, 6)),
            'early_progress_efficiency_ee': str(round(early_prog / max(early_ee, 0.001), 8)),
            'middle_progress_efficiency_ee': str(round(middle_prog / max(middle_ee, 0.001), 8)),
            'late_progress_efficiency_ee': str(round(late_prog / max(late_ee, 0.001), 8)),
            'early_action_direction_change_count': str(np.random.randint(0, 10)),
            'middle_action_direction_change_count': str(np.random.randint(0, 10)),
            'late_action_direction_change_count': str(np.random.randint(0, 10)),
            'early_contact_count': str(np.random.randint(0, 5)),
            'middle_contact_count': str(np.random.randint(0, 5)),
            'late_contact_count': str(np.random.randint(0, 5)),
            'early_best_dist_improvement_m': str(round(np.random.uniform(0, 0.1), 6)),
            'middle_best_dist_improvement_m': str(round(np.random.uniform(0, 0.1), 6)),
            'late_best_dist_improvement_m': str(round(np.random.uniform(0, 0.1), 6)),
            'path_includes_initial_position': 'True',
            'ee_positions_count': str(np.random.randint(100, 200)),
            'object_positions_count': str(np.random.randint(100, 200)),
            'total_env_steps': str(np.random.randint(100, 200)),
            'runtime_sec': str(round(np.random.uniform(10, 30), 2)),
            'collision_count': str(np.random.randint(0, 5)),
            'contact_count': str(np.random.randint(0, 20)),
        }
        rows.append(row)
    
    if output_path:
        fieldnames = list(rows[0].keys())
        with open(output_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Generated {n_rows} synthetic rows to {output_path}")
    
    return rows


# ── Data validation ───────────────────────────────────────────────────────

def validate_manifest(rows):
    """Validate manifest data integrity. Returns (pass, errors, warnings)."""
    errors = []
    warnings = []
    
    if not rows:
        errors.append("Manifest is empty")
        return False, errors, warnings
    
    n = len(rows)
    print(f"Validating {n} rows...")
    
    # Check required fields
    required_fields = [
        'top_config_label', 'horizon', 'num_samples', 'init_std', 'family',
        'ee_path_length_m', 'wasted_motion_ratio_capped', 'progress_efficiency_ee',
        'early_ee_path_length_m', 'middle_ee_path_length_m', 'late_ee_path_length_m',
        'early_progress_m', 'middle_progress_m', 'late_progress_m',
        'path_includes_initial_position', 'ee_positions_count', 'object_positions_count'
    ]
    
    for field in required_fields:
        missing = sum(1 for r in rows if r.get(field, '') == '')
        if missing > 0:
            errors.append(f"Field '{field}' missing in {missing}/{n} rows")
    
    # Check numeric ranges
    for i, r in enumerate(rows):
        row_id = r.get('config', f'row_{i}')
        
        # ee_path_length_m >= 0
        ee = sf(r, 'ee_path_length_m', -1)
        if ee < 0:
            errors.append(f"[{row_id}] ee_path_length_m={ee} < 0")
        
        # wasted_motion_ratio_capped <= 100
        wcap = sf(r, 'wasted_motion_ratio_capped', -1)
        if wcap > 100:
            errors.append(f"[{row_id}] wasted_motion_ratio_capped={wcap} > 100")
        
        # Segment sums
        ee_total = sf(r, 'ee_path_length_m', 0)
        ee_early = sf(r, 'early_ee_path_length_m', 0)
        ee_middle = sf(r, 'middle_ee_path_length_m', 0)
        ee_late = sf(r, 'late_ee_path_length_m', 0)
        ee_sum = ee_early + ee_middle + ee_late
        if ee_total > 0 and abs(ee_sum - ee_total) > max(1e-5, 0.01 * ee_total):
            errors.append(f"[{row_id}] segment EE path sum={ee_sum:.4f} != total={ee_total:.4f}")
        
        prog_total = sf(r, 'net_progress_m', 0)
        prog_early = sf(r, 'early_progress_m', 0)
        prog_middle = sf(r, 'middle_progress_m', 0)
        prog_late = sf(r, 'late_progress_m', 0)
        prog_sum = prog_early + prog_middle + prog_late
        if abs(prog_total) > 1e-6 and abs(prog_sum - prog_total) > max(1e-5, 0.01 * abs(prog_total)):
            warnings.append(f"[{row_id}] segment progress sum={prog_sum:.4f} != total={prog_total:.4f}")
    
    # Check group sizes
    by_tc = defaultdict(int)
    by_h = defaultdict(int)
    by_n = defaultdict(int)
    by_std = defaultdict(int)
    for r in rows:
        by_tc[r.get('top_config_label', '?')] += 1
        by_h[str(r.get('horizon', '?'))] += 1
        by_n[str(r.get('num_samples', '?'))] += 1
        by_std[str(r.get('init_std', '?'))] += 1
    
    if len(by_tc) < 2:
        warnings.append(f"Only {len(by_tc)} top_config values found")
    if len(by_h) < 2:
        warnings.append(f"Only {len(by_h)} horizon values found")
    
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    
    return len(errors) == 0, errors, warnings


# ── Statistical analysis ──────────────────────────────────────────────────

def compute_stats(rows, group_by=None):
    """Compute summary statistics, optionally grouped."""
    def rate(cnt, total):
        return round(cnt/total*100, 1) if total > 0 else 0.0
    
    def avg(rows, key):
        vs = [sf(r, key) for r in rows if r.get(key, '') != '']
        return round(sum(vs)/len(vs), 4) if vs else 0.0
    
    def cnt(rows, key):
        return sum(1 for r in rows if sb(r, key))
    
    if group_by:
        groups = defaultdict(list)
        for r in rows:
            groups[r.get(group_by, '?')].append(r)
    else:
        groups = {'all': rows}
    
    results = {}
    for gname, grows in groups.items():
        t = len(grows)
        if t == 0:
            continue
        
        s2 = cnt(grows, 'success_pose_2mm_10deg')
        rw = cnt(grows, 'random_walk_flag')
        ine = cnt(grows, 'inefficient_success_flag')
        exc = cnt(grows, 'excessive_wander_flag')
        cln = cnt(grows, 'clean_success_flag')
        mne = cnt(grows, 'meaningless_exploration_flag')
        
        results[gname] = {
            'count': t,
            'success_rate': rate(s2, t),
            'clean_success_rate': rate(cln, t),
            'random_walk_rate': rate(rw, t),
            'inefficient_success_rate': rate(ine, t),
            'excessive_wander_rate': rate(exc, t),
            'meaningless_exploration_rate': rate(mne, t),
            'mean_ee_path': avg(grows, 'ee_path_length_m'),
            'mean_wasted_capped': avg(grows, 'wasted_motion_ratio_capped'),
            'mean_prog_eff': avg(grows, 'progress_efficiency_ee'),
            'mean_early_ee': avg(grows, 'early_ee_path_length_m'),
            'mean_middle_ee': avg(grows, 'middle_ee_path_length_m'),
            'mean_late_ee': avg(grows, 'late_ee_path_length_m'),
            'mean_early_prog': avg(grows, 'early_progress_m'),
            'mean_middle_prog': avg(grows, 'middle_progress_m'),
            'mean_late_prog': avg(grows, 'late_progress_m'),
        }
    
    return results


def statistical_tests(rows):
    """Perform statistical tests on key metrics."""
    results = {}
    
    if not HAS_SCIPY:
        return results
    
    # Group by horizon
    by_h = defaultdict(list)
    for r in rows:
        by_h[str(r.get('horizon', '?'))].append(r)
    
    horizons = sorted(by_h.keys(), key=float)
    if len(horizons) >= 2:
        h1, h2 = horizons[0], horizons[-1]
        eff1 = [sf(r, 'progress_efficiency_ee') for r in by_h[h1]]
        eff2 = [sf(r, 'progress_efficiency_ee') for r in by_h[h2]]
        
        if len(eff1) >= 2 and len(eff2) >= 2:
            t_stat, p_val = scipy_stats.ttest_ind(eff1, eff2)
            results[f'horizon_{h1}_vs_{h2}_prog_eff'] = {
                'test': 't-test',
                'statistic': round(t_stat, 4),
                'p_value': round(p_val, 6),
                'significant': p_val < 0.05
            }
    
    # Group by num_samples
    by_n = defaultdict(list)
    for r in rows:
        by_n[str(r.get('num_samples', '?'))].append(r)
    
    samples = sorted(by_n.keys(), key=float)
    if len(samples) >= 2:
        n1, n2 = samples[0], samples[-1]
        succ1 = [sb(r, 'success_pose_2mm_10deg') for r in by_n[n1]]
        succ2 = [sb(r, 'success_pose_2mm_10deg') for r in by_n[n2]]
        
        if len(succ1) >= 2 and len(succ2) >= 2:
            # Chi-square test for success rates
            table = [
                [sum(succ1), len(succ1) - sum(succ1)],
                [sum(succ2), len(succ2) - sum(succ2)]
            ]
            chi2, p_val, dof, expected = scipy_stats.chi2_contingency(table)
            results[f'samples_{n1}_vs_{n2}_success'] = {
                'test': 'chi-square',
                'statistic': round(chi2, 4),
                'p_value': round(p_val, 6),
                'significant': p_val < 0.05
            }
    
    return results


# ── Visualization ─────────────────────────────────────────────────────────

def create_visualizations(rows, output_dir):
    """Generate analysis plots."""
    if not HAS_MPL:
        print("Skipping visualizations (matplotlib not available)")
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    
    # Prepare data
    by_tc = defaultdict(list)
    by_h = defaultdict(list)
    by_n = defaultdict(list)
    by_std = defaultdict(list)
    by_family = defaultdict(list)
    
    for r in rows:
        by_tc[r.get('top_config_label', '?')].append(r)
        by_h[str(r.get('horizon', '?'))].append(r)
        by_n[str(r.get('num_samples', '?'))].append(r)
        by_std[str(r.get('init_std', '?'))].append(r)
        by_family[r.get('family', '?')].append(r)
    
    # 1. Success rate by top_config
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stage 2C Analysis: Success & Efficiency by Config', fontsize=14)
    
    # Success rate by top_config
    tc_labels = sorted(by_tc.keys())
    tc_success = [sum(1 for r in by_tc[tc] if sb(r, 'success_pose_2mm_10deg')) / len(by_tc[tc]) * 100 for tc in tc_labels]
    tc_clean = [sum(1 for r in by_tc[tc] if sb(r, 'clean_success_flag')) / len(by_tc[tc]) * 100 for tc in tc_labels]
    
    x = np.arange(len(tc_labels))
    width = 0.35
    axes[0, 0].bar(x - width/2, tc_success, width, label='Success', color='#2ecc71')
    axes[0, 0].bar(x + width/2, tc_clean, width, label='Clean Success', color='#27ae60')
    axes[0, 0].set_xlabel('Top Config')
    axes[0, 0].set_ylabel('Rate (%)')
    axes[0, 0].set_title('Success Rate by Top Config')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(tc_labels)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Waste ratio by horizon
    h_labels = sorted(by_h.keys(), key=float)
    h_waste = [np.mean([sf(r, 'wasted_motion_ratio_capped') for r in by_h[h]]) for h in h_labels]
    h_ee = [np.mean([sf(r, 'ee_path_length_m') for r in by_h[h]]) for h in h_labels]
    
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()
    ax1.bar(np.arange(len(h_labels)) - 0.2, h_waste, 0.4, label='Waste Capped', color='#e74c3c', alpha=0.7)
    ax2.bar(np.arange(len(h_labels)) + 0.2, h_ee, 0.4, label='EE Path', color='#3498db', alpha=0.7)
    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('Waste Capped', color='#e74c3c')
    ax2.set_ylabel('EE Path (m)', color='#3498db')
    ax1.set_title('Waste & EE Path by Horizon')
    ax1.set_xticks(np.arange(len(h_labels)))
    ax1.set_xticklabels(h_labels)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Progress efficiency by num_samples
    n_labels = sorted(by_n.keys(), key=float)
    n_eff = [np.mean([sf(r, 'progress_efficiency_ee') for r in by_n[n]]) for n in n_labels]
    n_eff_std = [np.std([sf(r, 'progress_efficiency_ee') for r in by_n[n]]) for n in n_labels]
    
    axes[1, 0].bar(np.arange(len(n_labels)), n_eff, yerr=n_eff_std, capsize=5, color='#9b59b6', alpha=0.8)
    axes[1, 0].set_xlabel('Num Samples')
    axes[1, 0].set_ylabel('Progress Efficiency')
    axes[1, 0].set_title('Progress Efficiency by Num Samples')
    axes[1, 0].set_xticks(np.arange(len(n_labels)))
    axes[1, 0].set_xticklabels(n_labels)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Meaningless exploration by family
    fam_labels = sorted(by_family.keys())
    fam_mne = [sum(1 for r in by_family[f] if sb(r, 'meaningless_exploration_flag')) / len(by_family[f]) * 100 for f in fam_labels]
    
    axes[1, 1].barh(np.arange(len(fam_labels)), fam_mne, color='#e67e22', alpha=0.8)
    axes[1, 1].set_ylabel('Family')
    axes[1, 1].set_xlabel('Meaningless Exploration Rate (%)')
    axes[1, 1].set_title('Meaningless Exploration by Family')
    axes[1, 1].set_yticks(np.arange(len(fam_labels)))
    axes[1, 1].set_yticklabels([f.replace('passage_', 'p_').replace('blocking_', 'b_') for f in fam_labels], fontsize=8)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'stage2c_analysis_overview.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plots.append(plot_path)
    print(f"  Saved: {plot_path}")
    
    # 2. Segment analysis heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Segment Analysis: Early / Middle / Late', fontsize=14)
    
    # EE path by config
    tc_labels_sorted = sorted(by_tc.keys())
    segments = ['early', 'middle', 'late']
    
    ee_data = []
    for tc in tc_labels_sorted:
        row = []
        for seg in segments:
            val = np.mean([sf(r, f'{seg}_ee_path_length_m') for r in by_tc[tc]])
            row.append(val)
        ee_data.append(row)
    
    im = axes[0].imshow(ee_data, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(segments)
    axes[0].set_yticks(range(len(tc_labels_sorted)))
    axes[0].set_yticklabels(tc_labels_sorted)
    axes[0].set_title('EE Path (m)')
    plt.colorbar(im, ax=axes[0])
    
    # Progress by config
    prog_data = []
    for tc in tc_labels_sorted:
        row = []
        for seg in segments:
            val = np.mean([sf(r, f'{seg}_progress_m') for r in by_tc[tc]])
            row.append(val)
        prog_data.append(row)
    
    im = axes[1].imshow(prog_data, cmap='YlGn', aspect='auto')
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(segments)
    axes[1].set_yticks(range(len(tc_labels_sorted)))
    axes[1].set_yticklabels(tc_labels_sorted)
    axes[1].set_title('Progress (m)')
    plt.colorbar(im, ax=axes[1])
    
    # Efficiency by config
    eff_data = []
    for tc in tc_labels_sorted:
        row = []
        for seg in segments:
            val = np.mean([sf(r, f'{seg}_progress_efficiency_ee') for r in by_tc[tc]])
            row.append(val)
        eff_data.append(row)
    
    im = axes[2].imshow(eff_data, cmap='YlGnBu', aspect='auto')
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(segments)
    axes[2].set_yticks(range(len(tc_labels_sorted)))
    axes[2].set_yticklabels(tc_labels_sorted)
    axes[2].set_title('Progress Efficiency')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plot_path = output_dir / 'stage2c_segment_heatmap.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plots.append(plot_path)
    print(f"  Saved: {plot_path}")
    
    # 3. Scatter: EE path vs Progress efficiency
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'A': '#2ecc71', 'B': '#3498db', 'C': '#e74c3c'}
    for tc in sorted(by_tc.keys()):
        ee = [sf(r, 'ee_path_length_m') for r in by_tc[tc]]
        eff = [sf(r, 'progress_efficiency_ee') for r in by_tc[tc]]
        succ = [sb(r, 'success_pose_2mm_10deg') for r in by_tc[tc]]
        
        for i in range(len(ee)):
            marker = 'o' if succ[i] else 'x'
            alpha = 0.8 if succ[i] else 0.4
            ax.scatter(ee[i], eff[i], c=colors.get(tc, 'gray'), marker=marker, alpha=alpha, s=50)
    
    # Add legend
    for tc in sorted(colors.keys()):
        ax.scatter([], [], c=colors[tc], marker='o', label=f'Config {tc} (success)', s=50)
        ax.scatter([], [], c=colors[tc], marker='x', label=f'Config {tc} (fail)', s=50)
    
    ax.set_xlabel('EE Path Length (m)')
    ax.set_ylabel('Progress Efficiency')
    ax.set_title('EE Path vs Progress Efficiency (○=success, ×=failure)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    
    plot_path = output_dir / 'stage2c_ee_vs_efficiency.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plots.append(plot_path)
    print(f"  Saved: {plot_path}")
    
    return plots


# ── Insight generation ────────────────────────────────────────────────────

def generate_insights(rows, stats, stat_tests):
    """Generate textual insights from analysis."""
    insights = []
    
    # Best config
    by_tc = stats.get('by_top_config', {})
    if by_tc:
        best_tc = max(by_tc.keys(), key=lambda k: by_tc[k].get('success_rate', 0))
        insights.append(f"🏆 Best top_config: {best_tc} (success={by_tc[best_tc]['success_rate']}%, clean={by_tc[best_tc]['clean_success_rate']}%)")
    
    # Horizon effect
    by_h = stats.get('by_horizon', {})
    if len(by_h) >= 2:
        horizons = sorted(by_h.keys(), key=lambda x: float(x) if x.replace('.','').isdigit() else 999)
        h_low, h_high = horizons[0], horizons[-1]
        waste_low = by_h[h_low].get('mean_wasted_capped', 0)
        waste_high = by_h[h_high].get('mean_wasted_capped', 0)
        if waste_high > waste_low * 1.2:
            insights.append(f"⚠️ Higher horizon ({h_high}) has {waste_high/waste_low:.1f}x more waste than {h_low}")
    
    # Family comparison
    by_fam = stats.get('by_family', {})
    if by_fam:
        worst_fam = max(by_fam.keys(), key=lambda k: by_fam[k].get('meaningless_exploration_rate', 0))
        insights.append(f"🔴 Worst family for meaningless exploration: {worst_fam} ({by_fam[worst_fam]['meaningless_exploration_rate']}%)")
        
        best_fam = max(by_fam.keys(), key=lambda k: by_fam[k].get('clean_success_rate', 0))
        insights.append(f"🟢 Best family for clean success: {best_fam} ({by_fam[best_fam]['clean_success_rate']}%)")
    
    # Statistical significance
    for test_name, result in stat_tests.items():
        if result.get('significant', False):
            insights.append(f"📊 Significant: {test_name} (p={result['p_value']:.4f})")
    
    # Segment patterns
    overall = stats.get('all', {})
    if overall:
        early_eff = overall.get('mean_early_prog', 0) / max(overall.get('mean_early_ee', 0.001), 0.001)
        late_eff = overall.get('mean_late_prog', 0) / max(overall.get('mean_late_ee', 0.001), 0.001)
        if late_eff > early_eff * 2:
            insights.append("📈 Late-stage efficiency is 2x+ higher than early-stage (front-loaded wandering detected)")
    
    return insights


# ── Main analysis ─────────────────────────────────────────────────────────

def run_analysis(manifest_path, output_dir, self_test=False):
    """Run full analysis pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MPPI Stage 2C Analysis")
    print("=" * 60)
    
    # Load or generate data
    if self_test:
        print("\n[SELF-TEST] Generating synthetic data...")
        synthetic_path = output_dir / 'synthetic_manifest.csv'
        rows = generate_synthetic_manifest(100, synthetic_path)
    else:
        print(f"\nLoading manifest: {manifest_path}")
        rows = load_manifest(manifest_path)
    
    if not rows:
        print("ERROR: No data to analyze")
        return False
    
    print(f"Loaded {len(rows)} rows")
    
    # Validate
    print("\n" + "=" * 60)
    print("Data Validation")
    print("=" * 60)
    valid, errors, warnings = validate_manifest(rows)
    
    if errors:
        print("\n❌ ERRORS:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more")
    
    if warnings:
        print("\n⚠️ WARNINGS:")
        for w in warnings[:5]:
            print(f"  - {w}")
    
    if not valid:
        print("\n❌ VALIDATION FAILED - Analysis aborted")
        return False
    
    print("\n✅ VALIDATION PASSED")
    
    # Compute statistics
    print("\n" + "=" * 60)
    print("Statistical Analysis")
    print("=" * 60)
    
    stats = {
        'all': compute_stats(rows),
        'by_top_config': compute_stats(rows, 'top_config_label'),
        'by_horizon': compute_stats(rows, 'horizon'),
        'by_num_samples': compute_stats(rows, 'num_samples'),
        'by_init_std': compute_stats(rows, 'init_std'),
        'by_family': compute_stats(rows, 'family'),
    }
    
    # Flatten 'all' stats
    if 'all' in stats and stats['all']:
        stats['all'] = list(stats['all'].values())[0]
    
    # Print key stats
    print(f"\nOverall: {len(rows)} runs")
    if stats['all']:
        print(f"  Success rate: {stats['all'].get('success_rate', '?')}%")
        print(f"  Clean success: {stats['all'].get('clean_success_rate', '?')}%")
        print(f"  Meaningless exploration: {stats['all'].get('meaningless_exploration_rate', '?')}%")
    
    print("\nBy top_config:")
    for tc, s in sorted(stats['by_top_config'].items()):
        print(f"  {tc}: succ={s['success_rate']}%, clean={s['clean_success_rate']}%, waste={s['mean_wasted_capped']:.1f}")
    
    # Statistical tests
    stat_tests = statistical_tests(rows)
    if stat_tests:
        print("\nStatistical tests:")
        for name, result in stat_tests.items():
            sig = "✅ significant" if result['significant'] else "❌ not significant"
            print(f"  {name}: p={result['p_value']:.4f} ({sig})")
    
    # Generate insights
    insights = generate_insights(rows, stats, stat_tests)
    print("\n" + "=" * 60)
    print("Key Insights")
    print("=" * 60)
    for insight in insights:
        print(f"  {insight}")
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    plots = create_visualizations(rows, output_dir)
    
    # Save analysis report
    report_path = output_dir / 'stage2c_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write("# MPPI Stage 2C Analysis Report\n\n")
        f.write(f"Generated from {len(rows)} runs\n\n")
        
        f.write("## Key Insights\n\n")
        for insight in insights:
            f.write(f"- {insight}\n")
        f.write("\n")
        
        f.write("## Statistics by Top Config\n\n")
        f.write("| Config | Success | Clean | Waste | Prog Eff |\n")
        f.write("|--------|---------|-------|-------|----------|\n")
        for tc, s in sorted(stats['by_top_config'].items()):
            f.write(f"| {tc} | {s['success_rate']}% | {s['clean_success_rate']}% | {s['mean_wasted_capped']:.1f} | {s['mean_prog_eff']:.4f} |\n")
        f.write("\n")
        
        f.write("## Statistics by Horizon\n\n")
        f.write("| Horizon | Success | Clean | Waste | EE Path |\n")
        f.write("|---------|---------|-------|-------|--------|\n")
        for h, s in sorted(stats['by_horizon'].items(), key=lambda x: float(x[0]) if x[0].replace('.','').isdigit() else 999):
            f.write(f"| {h} | {s['success_rate']}% | {s['clean_success_rate']}% | {s['mean_wasted_capped']:.1f} | {s['mean_ee_path']:.2f} |\n")
        f.write("\n")
        
        if stat_tests:
            f.write("## Statistical Tests\n\n")
            f.write("| Test | Statistic | p-value | Significant |\n")
            f.write("|------|-----------|---------|-------------|\n")
            for name, result in stat_tests.items():
                sig = "✅" if result['significant'] else "❌"
                f.write(f"| {name} | {result['statistic']:.4f} | {result['p_value']:.6f} | {sig} |\n")
            f.write("\n")
        
        f.write("## Visualizations\n\n")
        for plot in plots:
            f.write(f"![{plot.stem}]({plot.name})\n")
    
    print(f"\n  Report: {report_path}")
    print(f"  Plots: {output_dir}/")
    for p in plots:
        print(f"    {p.name}")
    
    # Save raw stats as JSON
    stats_path = output_dir / 'stage2c_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Stats JSON: {stats_path}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    
    return True


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MPPI Stage 2C Analysis')
    parser.add_argument('--manifest', help='Path to manifest.csv')
    parser.add_argument('--output-dir', default='runs/stage2c_analysis', help='Output directory')
    parser.add_argument('--self-test', action='store_true', help='Run self-test with synthetic data')
    args = parser.parse_args()
    
    if args.self_test:
        success = run_analysis(None, args.output_dir, self_test=True)
    elif args.manifest:
        success = run_analysis(args.manifest, args.output_dir, self_test=False)
    else:
        # Try to find latest Stage 2C run
        runs_dir = Path('runs')
        stage2c_runs = sorted(runs_dir.glob('mppi_stage2c_*'), reverse=True)
        if stage2c_runs:
            manifest = stage2c_runs[0] / 'manifest.csv'
            if manifest.exists():
                print(f"Found latest Stage 2C run: {stage2c_runs[0]}")
                success = run_analysis(manifest, args.output_dir, self_test=False)
            else:
                print("No manifest found in latest run, running self-test")
                success = run_analysis(None, args.output_dir, self_test=True)
        else:
            print("No Stage 2C runs found, running self-test")
            success = run_analysis(None, args.output_dir, self_test=True)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
