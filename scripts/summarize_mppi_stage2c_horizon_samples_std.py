#!/usr/bin/env python3
"""MPPI Stage 2C horizon × samples × init_std summarizer — 15 output files + full stats + scoring."""
import csv, sys, argparse, math
from pathlib import Path
from collections import defaultdict

def load_manifest(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            status = r.get('status','')
            # Accept completed runs, or runs with valid data (no error and has ee_path_length_m)
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

def rate(cnt, total):
    return round(cnt/total*100, 1) if total > 0 else 0.0

def avg(rows, key):
    vs = [sf(r,key) for r in rows if r.get(key,'') != '']
    return round(sum(vs)/len(vs), 4) if vs else 0.0

def med(rows, key):
    vs = sorted(sf(r,key) for r in rows if r.get(key,'') != '')
    return round(vs[len(vs)//2], 4) if vs else 0.0

def cnt(rows, key):
    return sum(1 for r in rows if sb(r, key))

def ft_cnt(rows, ft):
    return sum(1 for r in rows if r.get('failure_type','') == ft)

def summarize(rows, group_label='all'):
    t = len(rows)
    if t == 0:
        return {}
    s2 = cnt(rows, 'success_pose_2mm_10deg')
    s5 = cnt(rows, 'success_pose_5mm_10deg')
    s10 = cnt(rows, 'success_pose_10mm_10deg')
    rw = cnt(rows, 'random_walk_flag')
    ine = cnt(rows, 'inefficient_success_flag')
    exc = cnt(rows, 'excessive_wander_flag')
    cln = cnt(rows, 'clean_success_flag')
    mne = cnt(rows, 'meaningless_exploration_flag')
    flw = cnt(rows, 'front_loaded_wander_flag')
    ltb = cnt(rows, 'late_breakthrough_flag')
    reg = cnt(rows, 'regressed_after_near_success')
    cs = ft_cnt(rows, 'collision_stuck')
    return {
        'group': group_label, 'count': t,
        # Success
        'success_pose_2mm_10deg_rate': rate(s2, t),
        'success_pose_5mm_10deg_rate': rate(s5, t),
        'success_pose_10mm_10deg_rate': rate(s10, t),
        'reached_pose_10mm_10deg_once_rate': rate(cnt(rows, 'reached_pose_10mm_10deg_once'), t),
        # Failure types
        'regression_rate': rate(reg, t),
        'collision_stuck_rate': rate(cs, t),
        # Distance/angle
        'mean_final_pos_dist_m': avg(rows,'final_pos_dist_m'),
        'median_final_pos_dist_m': med(rows,'final_pos_dist_m'),
        'mean_best_pos_dist_m': avg(rows,'best_pos_dist_m'),
        'median_best_pos_dist_m': med(rows,'best_pos_dist_m'),
        'mean_final_theta_error_deg': avg(rows,'final_theta_error_deg'),
        'mean_best_pose_score': avg(rows,'best_pose_score'),
        # Path efficiency
        'mean_ee_path_length_m': avg(rows,'ee_path_length_m'),
        'mean_object_path_length_m': avg(rows,'object_path_length_m'),
        'mean_wasted_motion_ratio': avg(rows,'wasted_motion_ratio'),
        'mean_wasted_motion_ratio_capped': avg(rows,'wasted_motion_ratio_capped'),
        'mean_progress_efficiency_ee': avg(rows,'progress_efficiency_ee'),
        'mean_object_to_ee_motion_ratio': avg(rows,'object_to_ee_motion_ratio'),
        'random_walk_rate': rate(rw, t),
        'inefficient_success_rate': rate(ine, t),
        'excessive_wander_rate': rate(exc, t),
        'clean_success_rate': rate(cln, t),
        'meaningless_exploration_rate': rate(mne, t),
        'front_loaded_wander_rate': rate(flw, t),
        'late_breakthrough_rate': rate(ltb, t),
        # Segment metrics
        'mean_early_ee_path_length_m': avg(rows,'early_ee_path_length_m'),
        'mean_middle_ee_path_length_m': avg(rows,'middle_ee_path_length_m'),
        'mean_late_ee_path_length_m': avg(rows,'late_ee_path_length_m'),
        'mean_early_progress_m': avg(rows,'early_progress_m'),
        'mean_middle_progress_m': avg(rows,'middle_progress_m'),
        'mean_late_progress_m': avg(rows,'late_progress_m'),
        'mean_early_progress_efficiency_ee': avg(rows,'early_progress_efficiency_ee'),
        'mean_middle_progress_efficiency_ee': avg(rows,'middle_progress_efficiency_ee'),
        'mean_late_progress_efficiency_ee': avg(rows,'late_progress_efficiency_ee'),
        # Other
        'mean_collision_count': avg(rows,'collision_count'),
        'mean_contact_count': avg(rows,'contact_count'),
        'mean_total_env_steps': avg(rows,'total_env_steps'),
        'mean_runtime_sec': avg(rows,'runtime_sec'),
        'saved_episodes': t,
        'saved_transitions': int(sum(sf(r,'total_env_steps',0) for r in rows)),
    }

def calc_score(s):
    """Stage 2C score — penalizes meaningless exploration."""
    return (
        130 * float(s.get('success_pose_2mm_10deg_rate', 0)) / 100
        + 60 * float(s.get('success_pose_5mm_10deg_rate', 0)) / 100
        + 25 * float(s.get('reached_pose_10mm_10deg_once_rate', 0)) / 100
        + 20 * float(s.get('clean_success_rate', 0)) / 100
        - 25 * float(s.get('random_walk_rate', 0)) / 100
        - 20 * float(s.get('inefficient_success_rate', 0)) / 100
        - 20 * float(s.get('excessive_wander_rate', 0)) / 100
        - 25 * float(s.get('meaningless_exploration_rate', 0)) / 100
        - 15 * float(s.get('regression_rate', 0)) / 100
        - 10 * float(s.get('collision_stuck_rate', 0)) / 100
        - 10 * float(s.get('mean_final_pos_dist_m', 0))
        - 3 * float(s.get('mean_wasted_motion_ratio_capped', 0))
        + 10 * float(s.get('mean_progress_efficiency_ee', 0))
        - 0.005 * float(s.get('mean_runtime_sec', 0))
    )

def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, help='Path to run directory containing manifest.csv')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / 'manifest.csv'
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}")
        sys.exit(1)

    rows = load_manifest(manifest_path)
    print(f"Loaded {len(rows)} completed runs from {manifest_path}")

    out_dir = run_dir
    base_fields = list(summarize(rows[:1]).keys()) if rows else []

    # Group rows
    by_top_config = defaultdict(list)
    by_horizon = defaultdict(list)
    by_num_samples = defaultdict(list)
    by_init_std = defaultdict(list)
    by_tc_h = defaultdict(list)
    by_config_full = defaultdict(list)
    by_family = defaultdict(list)
    by_family_h = defaultdict(list)

    for r in rows:
        tc = r.get('top_config_label', '?')
        h = str(r.get('horizon', '?'))
        n = str(r.get('num_samples', '?'))
        std = str(r.get('init_std', '?'))
        fm = r.get('family', '?')
        cfg = r.get('config', f"{tc}_h{h}_n{n}_std{std}_{fm}")

        by_top_config[tc].append(r)
        by_horizon[h].append(r)
        by_num_samples[n].append(r)
        by_init_std[std].append(r)
        by_tc_h[(tc, h)].append(r)
        by_config_full[cfg].append(r)
        by_family[fm].append(r)
        by_family_h[(fm, h)].append(r)

    # 1. summary_by_top_config.csv
    stc = []
    for k in sorted(by_top_config):
        s = summarize(by_top_config[k], f"config={k}")
        if s: s['top_config_label'] = k; stc.append(s)
    write_csv(out_dir/"summary_by_top_config.csv", stc, ['top_config_label'] + base_fields)

    # 2. summary_by_horizon.csv
    sh = []
    for k in sorted(by_horizon, key=float):
        s = summarize(by_horizon[k], f"horizon={k}")
        if s: s['horizon'] = int(k); sh.append(s)
    write_csv(out_dir/"summary_by_horizon.csv", sh, ['horizon'] + base_fields)

    # 3. summary_by_num_samples.csv
    sns = []
    for k in sorted(by_num_samples, key=float):
        s = summarize(by_num_samples[k], f"n={k}")
        if s: s['num_samples'] = int(k); sns.append(s)
    write_csv(out_dir/"summary_by_num_samples.csv", sns, ['num_samples'] + base_fields)

    # 4. summary_by_init_std.csv
    sstd = []
    for k in sorted(by_init_std, key=float):
        s = summarize(by_init_std[k], f"std={k}")
        if s: s['init_std'] = float(k); sstd.append(s)
    write_csv(out_dir/"summary_by_init_std.csv", sstd, ['init_std'] + base_fields)

    # 5. summary_by_top_config_horizon.csv
    stch = []
    for (tc, h) in sorted(by_tc_h, key=lambda x: (x[0], float(x[1]) if x[1].replace('.','').isdigit() else 999)):
        s = summarize(by_tc_h[(tc, h)], f"{tc}_h{h}")
        if s: s['top_config_label'] = tc; s['horizon'] = int(h); stch.append(s)
    write_csv(out_dir/"summary_by_top_config_horizon.csv", stch, ['top_config_label','horizon'] + base_fields)

    # 6. summary_by_config_full.csv
    scf = []
    for k in sorted(by_config_full):
        s = summarize(by_config_full[k], k)
        if s and by_config_full[k]:
            r0 = by_config_full[k][0]
            s['top_config_label'] = r0.get('top_config_label','')
            s['horizon'] = int(r0.get('horizon',0))
            s['num_samples'] = int(r0.get('num_samples',0))
            s['init_std'] = float(r0.get('init_std',0))
            s['family'] = r0.get('family','')
            scf.append(s)
    fields_cfg = ['top_config_label','horizon','num_samples','init_std','family'] + base_fields
    write_csv(out_dir/"summary_by_config_full.csv", scf, fields_cfg)

    # 7. summary_by_family.csv
    sf_ = []
    for k in sorted(by_family):
        s = summarize(by_family[k], k)
        if s: s['family'] = k; sf_.append(s)
    write_csv(out_dir/"summary_by_family.csv", sf_, ['family'] + base_fields)

    # 8. summary_by_family_horizon.csv
    sfh = []
    for (fm, h) in sorted(by_family_h, key=lambda x: (x[0], float(x[1]) if x[1].replace('.','').isdigit() else 999)):
        s = summarize(by_family_h[(fm, h)], f"{fm}_h{h}")
        if s: s['family'] = fm; s['horizon'] = int(h); sfh.append(s)
    write_csv(out_dir/"summary_by_family_horizon.csv", sfh, ['family','horizon'] + base_fields)

    # 9. summary_by_exploration_efficiency.csv — group by clean_success_rate quartiles
    # Sort configs by clean_success_rate, split into quartiles
    if scf:
        scf_sorted = sorted(scf, key=lambda s: s.get('clean_success_rate', 0))
        q = len(scf_sorted) // 4 or 1
        quartiles = [
            ('Q1_lowest', scf_sorted[:q]),
            ('Q2', scf_sorted[q:2*q]),
            ('Q3', scf_sorted[2*q:3*q]),
            ('Q4_highest', scf_sorted[3*q:]),
        ]
        see = []
        for qlabel, qrows_cfg in quartiles:
            # Collect all manifest rows for these configs
            cfg_set = set(r.get('config','') for r in qrows_cfg)
            qrows = [r for r in rows if r.get('config','') in cfg_set]
            if qrows:
                s = summarize(qrows, qlabel)
                if s: see.append(s)
        write_csv(out_dir/"summary_by_exploration_efficiency.csv", see, base_fields)

    # 10. summary_by_segment_efficiency.csv — compare early/middle/late across groups
    sse_rows = []
    overall = summarize(rows)
    if overall:
        overall['group'] = 'all'
        sse_rows.append(overall)
    for k in sorted(by_top_config):
        s = summarize(by_top_config[k], f"config={k}")
        if s: sse_rows.append(s)
    for k in sorted(by_horizon, key=float):
        s = summarize(by_horizon[k], f"horizon={k}")
        if s: sse_rows.append(s)
    write_csv(out_dir/"summary_by_segment_efficiency.csv", sse_rows, base_fields)

    # 11. top_mppi_stage2c_configs.csv — score all full configs
    tops = sorted(scf, key=lambda s: calc_score(s), reverse=True)
    for r in tops:
        r['score'] = round(calc_score(r), 2)
    fields_top = ['score','top_config_label','horizon','num_samples','init_std','family'] + base_fields
    write_csv(out_dir/"top_mppi_stage2c_configs.csv", tops, fields_top)

    # 12. inefficient_cases.csv
    inefficient = [r for r in rows if sb(r, 'inefficient_success_flag') or sb(r, 'excessive_wander_flag')]
    ineff_fields = ['config','top_config_label','horizon','num_samples','init_std','family',
                    'success_pose_2mm_10deg','ee_path_length_m','net_progress_m',
                    'wasted_motion_ratio_capped','inefficient_success_flag','excessive_wander_flag',
                    'meaningless_exploration_flag','failure_type']
    write_csv(out_dir/"inefficient_cases.csv",
              [{k: r.get(k,'') for k in ineff_fields} for r in inefficient],
              ineff_fields)

    # 13. late_breakthrough_cases.csv
    late_bt = [r for r in rows if sb(r, 'late_breakthrough_flag')]
    ltb_fields = ['config','top_config_label','horizon','num_samples','init_std','family',
                  'success_pose_2mm_10deg','ee_path_length_m','early_progress_m','middle_progress_m',
                  'late_progress_m','total_progress_m','late_breakthrough_flag',
                  'front_loaded_wander_flag','meaningless_exploration_flag']
    write_csv(out_dir/"late_breakthrough_cases.csv",
              [{k: r.get(k,'') for k in ltb_fields} for r in late_bt],
              ltb_fields)

    # 14. training_data_summary.csv
    td = [
        ('total_episodes', len(rows)),
        ('total_transitions', int(sum(sf(r,'total_env_steps',0) for r in rows))),
        ('success_count', sum(1 for r in rows if sb(r,'success'))),
        ('failure_count', sum(1 for r in rows if not sb(r,'success'))),
        ('success_pose_2mm_10deg_count', cnt(rows, 'success_pose_2mm_10deg')),
        ('random_walk_count', cnt(rows, 'random_walk_flag')),
        ('inefficient_success_count', cnt(rows, 'inefficient_success_flag')),
        ('excessive_wander_count', cnt(rows, 'excessive_wander_flag')),
        ('clean_success_count', cnt(rows, 'clean_success_flag')),
        ('meaningless_exploration_count', cnt(rows, 'meaningless_exploration_flag')),
        ('total_contact', int(sum(sf(r,'contact_count',0) for r in rows))),
        ('total_collision', int(sum(sf(r,'collision_count',0) for r in rows))),
        ('horizon_count', len(set(r.get('horizon','') for r in rows))),
        ('num_samples_count', len(set(r.get('num_samples','') for r in rows))),
        ('init_std_count', len(set(r.get('init_std','') for r in rows))),
        ('family_count', len(set(r.get('family','') for r in rows))),
        ('template_count', len(set(r.get('template_id','') for r in rows))),
        ('mean_transitions_per_episode', round(sum(sf(r,'total_env_steps',0) for r in rows) / max(len(rows),1), 1)),
        ('state16_training_ready', 'false'),
    ]
    write_csv(out_dir/"training_data_summary.csv",
              [{'metric': str(m), 'value': str(v)} for m, v in td],
              ['metric','value'])

    # 15. stage2c_horizon_samples_std_summary.md
    md = []
    md.append("# MPPI Stage 2C Horizon × Samples × Init-Std Summary\n")

    md.append("## 1. Purpose\n")
    md.append("本轮围绕 Stage 2B top configs (speed=0.3/T=0.1, speed=0.5/T=0.2, speed=0.2/T=0.3)，")
    md.append("测试 horizon、num_samples 和 init_std，并重点量化无效探索。")
    md.append(f"扫描 3 top_configs × 3 horizons × 2 num_samples × 3 init_std × 8 templates = {len(rows)} completed runs。\n")

    md.append("## 2. Best overall config\n")
    if tops:
        best = tops[0]
        md.append(f"- **best top_config_label**: {best.get('top_config_label','?')}")
        md.append(f"- **best horizon**: {best.get('horizon','?')}")
        md.append(f"- **best num_samples**: {best.get('num_samples','?')}")
        md.append(f"- **best init_std**: {best.get('init_std','?')}")
        md.append(f"- **best full config**: {best.get('group','?')}")
        md.append(f"- **success rate**: {best.get('success_pose_2mm_10deg_rate','?')}%")
        md.append(f"- **clean_success_rate**: {best.get('clean_success_rate','?')}%")
        md.append(f"- **meaningless_exploration_rate**: {best.get('meaningless_exploration_rate','?')}%")
        md.append(f"- **wasted_motion_ratio_capped**: {best.get('mean_wasted_motion_ratio_capped','?')}")
    md.append("")

    md.append("## 3. Horizon analysis\n")
    md.append("| Horizon | Runs | 2mm10° | 5mm10° | Clean% | MNE% | EE_Path | WasteCap | ProgEff | Runtime |")
    md.append("|---------|------|--------|--------|--------|------|---------|----------|---------|---------|")
    for s in sh:
        md.append(f"| {s.get('horizon','?')} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['success_pose_5mm_10deg_rate']}% | {s['clean_success_rate']}% | {s['meaningless_exploration_rate']}% | {s['mean_ee_path_length_m']:.2f} | {s['mean_wasted_motion_ratio_capped']:.1f} | {s['mean_progress_efficiency_ee']:.4f} | {s['mean_runtime_sec']:.1f}s |")
    md.append("")
    if len(sh) >= 2:
        h100 = next((s for s in sh if s.get('horizon') == 100), None)
        h140 = next((s for s in sh if s.get('horizon') == 140), None)
        if h100 and h140:
            md.append(f"- horizon=100 vs 140: succ={h100['success_pose_2mm_10deg_rate']}% vs {h140['success_pose_2mm_10deg_rate']}%")
            md.append(f"  - EE path: {h100['mean_ee_path_length_m']:.2f} vs {h140['mean_ee_path_length_m']:.2f}")
            md.append(f"  - Waste cap: {h100['mean_wasted_motion_ratio_capped']:.1f} vs {h140['mean_wasted_motion_ratio_capped']:.1f}")
    md.append("")

    md.append("## 4. Num samples analysis\n")
    md.append("| Samples | Runs | 2mm10° | Clean% | MNE% | EE_Path | WasteCap | Runtime |")
    md.append("|---------|------|--------|--------|------|---------|----------|---------|")
    for s in sns:
        md.append(f"| {s.get('num_samples','?')} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['clean_success_rate']}% | {s['meaningless_exploration_rate']}% | {s['mean_ee_path_length_m']:.2f} | {s['mean_wasted_motion_ratio_capped']:.1f} | {s['mean_runtime_sec']:.1f}s |")
    md.append("")

    md.append("## 5. Init std analysis\n")
    md.append("| Std | Runs | 2mm10° | Clean% | MNE% | EE_Path | WasteCap | ProgEff |")
    md.append("|-----|------|--------|--------|------|---------|----------|---------|")
    for s in sstd:
        md.append(f"| {s.get('init_std','?')} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['clean_success_rate']}% | {s['meaningless_exploration_rate']}% | {s['mean_ee_path_length_m']:.2f} | {s['mean_wasted_motion_ratio_capped']:.1f} | {s['mean_progress_efficiency_ee']:.4f} |")
    md.append("")

    md.append("## 6. Exploration efficiency analysis\n")
    md.append("### Configs with high success but high meaningless exploration:\n")
    high_succ_high_mne = [s for s in scf if s.get('success_pose_2mm_10deg_rate',0) > 50 and s.get('meaningless_exploration_rate',0) > 20]
    if high_succ_high_mne:
        for s in sorted(high_succ_high_mne, key=lambda x: x.get('meaningless_exploration_rate',0), reverse=True)[:5]:
            md.append(f"- {s['group']}: succ={s['success_pose_2mm_10deg_rate']}%, mne={s['meaningless_exploration_rate']}%")
    else:
        md.append("- None found")
    md.append("")

    md.append("### Configs with high clean_success_rate:\n")
    high_clean = sorted(scf, key=lambda s: s.get('clean_success_rate',0), reverse=True)[:5]
    for s in high_clean:
        md.append(f"- {s['group']}: clean={s['clean_success_rate']}%, succ={s['success_pose_2mm_10deg_rate']}%")
    md.append("")

    md.append("## 7. Segment analysis\n")
    md.append("### Overall segment comparison\n")
    if overall:
        md.append(f"- Early: EE_path={overall['mean_early_ee_path_length_m']:.3f}m, progress={overall['mean_early_progress_m']:.4f}m, eff={overall['mean_early_progress_efficiency_ee']:.4f}")
        md.append(f"- Middle: EE_path={overall['mean_middle_ee_path_length_m']:.3f}m, progress={overall['mean_middle_progress_m']:.4f}m, eff={overall['mean_middle_progress_efficiency_ee']:.4f}")
        md.append(f"- Late: EE_path={overall['mean_late_ee_path_length_m']:.3f}m, progress={overall['mean_late_progress_m']:.4f}m, eff={overall['mean_late_progress_efficiency_ee']:.4f}")
    md.append("")
    md.append(f"- **late_breakthrough_rate**: {overall.get('late_breakthrough_rate',0)}%")
    md.append(f"- **front_loaded_wander_rate**: {overall.get('front_loaded_wander_rate',0)}%")
    md.append("")

    md.append("## 8. Top config comparison\n")
    md.append("| Config | Runs | 2mm10° | Clean% | MNE% | EE_Path | WasteCap | ProgEff | Score |")
    md.append("|--------|------|--------|--------|------|---------|----------|---------|-------|")
    for s in stc:
        sc = calc_score(s)
        md.append(f"| {s.get('top_config_label','?')} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['clean_success_rate']}% | {s['meaningless_exploration_rate']}% | {s['mean_ee_path_length_m']:.2f} | {s['mean_wasted_motion_ratio_capped']:.1f} | {s['mean_progress_efficiency_ee']:.4f} | {sc:.1f} |")
    md.append("")

    md.append("## 9. Family-wise behavior\n")
    md.append("| Family | Runs | 2mm10° | Clean% | MNE% | EE_Path | WasteCap | ProgEff |")
    md.append("|--------|------|--------|--------|------|---------|----------|---------|")
    for s in sf_:
        md.append(f"| {s['family']} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['clean_success_rate']}% | {s['meaningless_exploration_rate']}% | {s['mean_ee_path_length_m']:.2f} | {s['mean_wasted_motion_ratio_capped']:.1f} | {s['mean_progress_efficiency_ee']:.4f} |")
    md.append("")

    # passage_direct_narrow detailed analysis
    pdn = [s for s in sf_ if 'passage_direct_narrow' in s.get('family','')]
    if pdn:
        md.append("### passage_direct_narrow: horizon breakdown\n")
        md.append("| Horizon | Runs | 2mm10° | Clean% | MNE% | EE_Path | WasteCap |")
        md.append("|---------|------|--------|--------|------|---------|----------|")
        pdn_h = [s for s in sfh if 'passage_direct_narrow' in s.get('family','')]
        for s in pdn_h:
            md.append(f"| {s.get('horizon','?')} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['clean_success_rate']}% | {s['meaningless_exploration_rate']}% | {s['mean_ee_path_length_m']:.2f} | {s['mean_wasted_motion_ratio_capped']:.1f} |")
        md.append("")

    md.append("## 10. Training data quality\n")
    best_train = max(scf, key=lambda s: s.get('mean_progress_efficiency_ee',0) - s.get('random_walk_rate',0)*0.5, default=None)
    if best_train:
        md.append(f"- **Cleanest trajectories**: {best_train['group']} (prog_eff={best_train['mean_progress_efficiency_ee']:.4f}, RW={best_train['random_walk_rate']}%)")
    md.append(f"- **state16_training_ready**: false (compact rollout state)")
    md.append("")

    md.append("## 11. Recommendation after Stage 2C\n")
    if tops:
        best = tops[0]
        md.append(f"### Final Oracle-MPPI recommended config:")
        md.append(f"- top_config: {best.get('top_config_label','?')}")
        md.append(f"- horizon: {best.get('horizon','?')}")
        md.append(f"- num_samples: {best.get('num_samples','?')}")
        md.append(f"- init_std: {best.get('init_std','?')}")
        md.append(f"- success: {best.get('success_pose_2mm_10deg_rate','?')}%")
        md.append(f"- clean_success: {best.get('clean_success_rate','?')}%")
        md.append(f"- meaningless_exploration: {best.get('meaningless_exploration_rate','?')}%")
    md.append("")
    md.append("### 是否需要 Stage 2D cost-efficiency ablation:")
    md.append("- 待用户决定")
    md.append("")
    md.append("### 是否应该停止 planner sweep、转向 learned model training:")
    md.append("- 待用户决定")
    md.append("")

    # Appendix: Top 10 configs
    md.append("## Appendix: Top 10 configs by score\n")
    md.append("| Rank | Config | H | N | Std | Family | 2mm10° | Clean% | MNE% | WasteCap | ProgEff | Score |")
    md.append("|------|--------|---|---|-----|--------|--------|--------|------|----------|---------|-------|")
    for i, s in enumerate(tops[:10], 1):
        sc = s.get('score', calc_score(s))
        md.append(f"| {i} | {s.get('top_config_label','?')} | {s.get('horizon','?')} | {s.get('num_samples','?')} | {s.get('init_std','?')} | {s.get('family','?')} | {s.get('success_pose_2mm_10deg_rate',0)}% | {s.get('clean_success_rate',0)}% | {s.get('meaningless_exploration_rate',0)}% | {s.get('mean_wasted_motion_ratio_capped',0):.1f} | {s.get('mean_progress_efficiency_ee',0):.4f} | {sc:.1f} |")
    md.append("")

    md.append(f"\n*Raw data: {manifest_path}*")
    md.append(f"\n*Generated: MPPI Stage 2C Horizon × Samples × Init-Std Sweep*")

    report_path = out_dir / "stage2c_horizon_samples_std_summary.md"
    report_path.write_text('\n'.join(md))
    print(f"\n  Report: {report_path}")
    print(f"  Outputs: {out_dir}/")
    for f in sorted(out_dir.glob("summary_*.csv")):
        print(f"    {f.name}")
    for f in sorted(out_dir.glob("top_*.csv")):
        print(f"    {f.name}")
    for f in sorted(out_dir.glob("*inefficient*.csv")):
        print(f"    {f.name}")
    for f in sorted(out_dir.glob("*late_breakthrough*.csv")):
        print(f"    {f.name}")
    for f in sorted(out_dir.glob("training_*.csv")):
        print(f"    {f.name}")

if __name__ == "__main__":
    main()
