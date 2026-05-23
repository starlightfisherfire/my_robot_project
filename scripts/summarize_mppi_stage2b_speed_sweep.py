#!/usr/bin/env python3
"""MPPI Stage 2B speed sweep summarizer — 9 output files + full stats + scoring."""
import csv, sys, argparse, math
from pathlib import Path
from collections import defaultdict

def load_manifest(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get('status','') in ('completed','True','true'):
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
    return {
        'group': group_label, 'count': t,
        # Success
        'success_pose_2mm_10deg_rate': rate(s2, t),
        'success_pose_5mm_10deg_rate': rate(s5, t),
        'success_pose_10mm_10deg_rate': rate(s10, t),
        'success_pos_1mm_rate': rate(cnt(rows,'success_pos_1mm'), t),
        'success_pos_2mm_rate': rate(cnt(rows,'success_pos_2mm'), t),
        'success_pos_5mm_rate': rate(cnt(rows,'success_pos_5mm'), t),
        'success_pos_10mm_rate': rate(cnt(rows,'success_pos_10mm'), t),
        'success_pos_50mm_rate': rate(cnt(rows,'success_pos_50mm'), t),
        'success_pose_5mm_5deg_rate': rate(cnt(rows,'success_pose_5mm_5deg'), t),
        'success_pose_10mm_5deg_rate': rate(cnt(rows,'success_pose_10mm_5deg'), t),
        'reached_pos_5mm_once_rate': rate(cnt(rows,'reached_pos_5mm_once'), t),
        'reached_pos_10mm_once_rate': rate(cnt(rows,'reached_pos_10mm_once'), t),
        'reached_pose_5mm_10deg_once_rate': rate(cnt(rows,'reached_pose_5mm_10deg_once'), t),
        'reached_pose_10mm_10deg_once_rate': rate(cnt(rows,'reached_pose_10mm_10deg_once'), t),
        'regression_rate': rate(cnt(rows,'regressed_after_near_success'), t),
        # Failure types
        'no_contact_rate': rate(ft_cnt(rows,'no_contact'), t),
        'collision_stuck_rate': rate(ft_cnt(rows,'collision_stuck'), t),
        'low_progress_rate': rate(ft_cnt(rows,'low_progress'), t),
        'theta_failed_rate': rate(ft_cnt(rows,'position_reached_theta_failed'), t),
        'regressed_type_rate': rate(ft_cnt(rows,'regressed_after_near_success'), t),
        # Distance/angle
        'mean_final_pos_dist_m': avg(rows,'final_pos_dist_m'),
        'median_final_pos_dist_m': med(rows,'final_pos_dist_m'),
        'mean_best_pos_dist_m': avg(rows,'best_pos_dist_m'),
        'median_best_pos_dist_m': med(rows,'best_pos_dist_m'),
        'mean_final_theta_error_deg': avg(rows,'final_theta_error_deg'),
        'mean_best_pose_score': avg(rows,'best_pose_score'),
        'mean_initial_pos_dist_m': avg(rows,'initial_pos_dist_m'),
        'mean_total_progress_m': avg(rows,'total_progress_m'),
        # Path efficiency
        'mean_ee_path_length_m': avg(rows,'ee_path_length_m'),
        'mean_object_path_length_m': avg(rows,'object_path_length_m'),
        'mean_net_progress_m': avg(rows,'net_progress_m'),
        'mean_progress_efficiency_ee': avg(rows,'progress_efficiency_ee'),
        'mean_wasted_motion_ratio': avg(rows,'wasted_motion_ratio'),
        'mean_wasted_motion_ratio_capped': avg(rows,'wasted_motion_ratio_capped'),
        'random_walk_rate': rate(rw, t),
        'inefficient_success_rate': rate(ine, t),
        'mean_action_smoothness': avg(rows,'action_smoothness_mean'),
        'mean_action_direction_change_count': avg(rows,'action_direction_change_count'),
        'mean_contact_efficiency': avg(rows,'contact_efficiency'),
        # Other
        'mean_collision_count': avg(rows,'collision_count'),
        'mean_contact_count': avg(rows,'contact_count'),
        'mean_total_env_steps': avg(rows,'total_env_steps'),
        'mean_runtime_sec': avg(rows,'runtime_sec'),
        'saved_episodes': t,
        'saved_transitions': int(sum(sf(r,'total_env_steps',0) for r in rows)),
    }

def calc_score(s):
    """Stage 2B score — penalizes random walk."""
    rw_rate = float(s.get('random_walk_rate', 0))
    ine_rate = float(s.get('inefficient_success_rate', 0))
    reg_rate = float(s.get('regression_rate', 0))
    cs_rate = float(s.get('collision_stuck_rate', 0))
    return (
        120 * float(s.get('success_pose_2mm_10deg_rate', 0)) / 100
        + 60 * float(s.get('success_pose_5mm_10deg_rate', 0)) / 100
        + 25 * float(s.get('reached_pose_10mm_10deg_once_rate', 0)) / 100
        - 25 * rw_rate / 100
        - 20 * ine_rate / 100
        - 15 * reg_rate / 100
        - 10 * cs_rate / 100
        - 10 * float(s.get('mean_final_pos_dist_m', 0))
        - 3 * float(s.get('mean_wasted_motion_ratio_capped',
                       s.get('mean_wasted_motion_ratio', 0)))
        + 10 * float(s.get('mean_progress_efficiency_ee', 0))
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
    by_speed = defaultdict(list)
    by_temp = defaultdict(list)
    by_speed_temp = defaultdict(list)
    by_family = defaultdict(list)
    by_config = defaultdict(list)

    for r in rows:
        sp = str(r.get('speed_mps','?'))
        tm = str(r.get('temperature','?'))
        fm = r.get('family','?')
        cfg = r.get('config', f"T{tm}_sp{sp}_{fm}")
        by_speed[sp].append(r)
        by_temp[tm].append(r)
        by_speed_temp[(sp, tm)].append(r)
        by_family[fm].append(r)
        by_config[cfg].append(r)

    # 1. summary_by_speed.csv
    ss = []
    for k in sorted(by_speed, key=lambda x: float(x) if x.replace('.','').isdigit() else 999):
        s = summarize(by_speed[k], f"speed={k}")
        if s: s['speed_mps'] = float(k) if k.replace('.','').isdigit() else 0; ss.append(s)
    fields_speed = ['speed_mps'] + base_fields
    write_csv(out_dir/"summary_by_speed.csv", ss, fields_speed)

    # 2. summary_by_temperature.csv
    st_ = []
    for k in sorted(by_temp, key=float):
        s = summarize(by_temp[k], f"T={k}")
        if s: s['temperature'] = float(k); st_.append(s)
    fields_temp = ['temperature'] + base_fields
    write_csv(out_dir/"summary_by_temperature.csv", st_, fields_temp)

    # 3. summary_by_speed_temperature.csv
    sst = []
    for (sp, tm) in sorted(by_speed_temp, key=lambda x: (float(x[0]) if x[0].replace('.','').isdigit() else 999, float(x[1]))):
        s = summarize(by_speed_temp[(sp,tm)], f"sp={sp}_T={tm}")
        if s: s['speed_mps'] = float(sp) if sp.replace('.','').isdigit() else 0; s['temperature'] = float(tm); sst.append(s)
    fields_st = ['speed_mps','temperature'] + base_fields
    write_csv(out_dir/"summary_by_speed_temperature.csv", sst, fields_st)

    # 4. summary_by_family.csv
    sf_ = []
    for k in sorted(by_family):
        s = summarize(by_family[k], k)
        if s: s['family'] = k; sf_.append(s)
    write_csv(out_dir/"summary_by_family.csv", sf_, ['family'] + base_fields)

    # 5. summary_by_config.csv
    sc = []
    for k in sorted(by_config):
        s = summarize(by_config[k], k)
        if s and by_config[k]:
            r0 = by_config[k][0]
            s['speed_mps'] = float(r0.get('speed_mps', 0))
            s['temperature'] = float(r0.get('temperature', 0))
            s['family'] = r0.get('family','')
            sc.append(s)
    fields_cfg = ['speed_mps','temperature','family'] + base_fields
    write_csv(out_dir/"summary_by_config.csv", sc, fields_cfg)

    # 6. top_mppi_stage2b_speed_configs.csv — score all speed-temp combos
    tops = sorted(sst, key=calc_score, reverse=True)
    for r in tops:
        r['score'] = round(calc_score(r), 2)
    fields_top = ['score','speed_mps','temperature'] + base_fields
    write_csv(out_dir/"top_mppi_stage2b_speed_configs.csv", tops, fields_top)

    # 7. random_walk_diagnostics.csv
    rw_rows = [r for r in rows if sb(r, 'random_walk_flag') or sb(r, 'inefficient_success_flag')]
    rw_fields = ['config','speed_mps','temperature','family','success_pose_2mm_10deg',
                 'ee_path_length_m','net_progress_m','wasted_motion_ratio',
                 'random_walk_flag','inefficient_success_flag','failure_type']
    rw_out = []
    for r in rw_rows:
        d = {k: r.get(k,'') for k in rw_fields}
        rw_out.append(d)
    write_csv(out_dir/"random_walk_diagnostics.csv", rw_out, rw_fields)

    # 8. training_data_summary.csv
    td = [
        ('total_episodes', len(rows)),
        ('total_transitions', int(sum(sf(r,'total_env_steps',0) for r in rows))),
        ('success_count', sum(1 for r in rows if sb(r,'success'))),
        ('failure_count', sum(1 for r in rows if not sb(r,'success'))),
        ('success_pose_2mm_10deg_count', cnt(rows, 'success_pose_2mm_10deg')),
        ('random_walk_count', cnt(rows, 'random_walk_flag')),
        ('inefficient_success_count', cnt(rows, 'inefficient_success_flag')),
        ('total_contact', int(sum(sf(r,'contact_count',0) for r in rows))),
        ('total_collision', int(sum(sf(r,'collision_count',0) for r in rows))),
        ('speed_count', len(set(r.get('speed_mps','') for r in rows))),
        ('temperature_count', len(set(r.get('temperature','') for r in rows))),
        ('family_count', len(set(r.get('family','') for r in rows))),
        ('template_count', len(set(r.get('template_id','') for r in rows))),
        ('mean_transitions_per_episode', round(sum(sf(r,'total_env_steps',0) for r in rows) / max(len(rows),1), 1)),
        ('state16_training_ready', 'false'),
    ]
    write_csv(out_dir/"training_data_summary.csv",
              [{'metric': str(m), 'value': str(v)} for m, v in td],
              ['metric','value'])

    # 9. stage2b_speed_summary.md
    overall = summarize(rows)
    md = []
    md.append("# MPPI Stage 2B Speed Sweep Summary\n")

    md.append("## 1. Purpose\n")
    md.append("本轮不是单纯追求最高成功率，而是判断 speed=0.75 是否存在无意义随机游走，")
    md.append("并寻找更干净的速度区间。扫描 3 temperatures × 5 speeds × 8 core8 模板 = 120 runs。\n")

    md.append("## 2. Best speed\n")
    # Best by success rate from by_speed summary
    if ss:
        best_succ = max(ss, key=lambda s: s['success_pose_2mm_10deg_rate'])
        best_eff = max(ss, key=lambda s: s['mean_progress_efficiency_ee'])
        best_score = max(ss, key=lambda s: calc_score(s))
        md.append(f"- **Highest success rate**: speed={best_succ['speed_mps']} ({best_succ['success_pose_2mm_10deg_rate']}%)")
        md.append(f"- **Best path efficiency**: speed={best_eff['speed_mps']} (progress_eff_ee={best_eff['mean_progress_efficiency_ee']:.4f})")
        md.append(f"- **Best composite score**: speed={best_score['speed_mps']} (score={calc_score(best_score):.2f})")
        md.append("")
        # Can speed 0.2/0.3/0.5 replace 0.75?
        sp075 = next((s for s in ss if abs(s['speed_mps'] - 0.75) < 0.01), None)
        md.append("- **Can lower speeds replace 0.75?**")
        if sp075:
            for s in ss:
                if s['speed_mps'] < 0.7:
                    better = s['success_pose_2mm_10deg_rate'] >= sp075['success_pose_2mm_10deg_rate']
                    wcap_s = s.get('mean_wasted_motion_ratio_capped', s.get('mean_wasted_motion_ratio', 0))
                    wcap_075 = sp075.get('mean_wasted_motion_ratio_capped', sp075.get('mean_wasted_motion_ratio', 0))
                    cleaner = wcap_s < wcap_075
                    md.append(f"  - speed={s['speed_mps']}: succ={s['success_pose_2mm_10deg_rate']}% (vs {sp075['success_pose_2mm_10deg_rate']}%), "
                             f"wasted_capped={wcap_s:.1f} (vs {wcap_075:.1f})"
                             f"{' ✅ higher or equal success, cleaner' if better and cleaner else ''}")
        md.append("")

    # Speed table
    md.append("### Speed comparison table\n")
    fields_tbl = ['speed_mps','count','success_pose_2mm_10deg_rate','success_pose_5mm_10deg_rate',
                  'mean_final_pos_dist_m','mean_ee_path_length_m','mean_wasted_motion_ratio','mean_wasted_motion_ratio_capped',
                  'random_walk_rate','inefficient_success_rate','mean_progress_efficiency_ee','mean_runtime_sec']
    md.append("| Speed | Runs | 2mm10° | 5mm10° | FinalDist | EE_Path | WasteRatio | WasteCap | RW% | IneffSucc% | ProgEff | Runtime |")
    md.append("|-------|------|--------|--------|-----------|---------|-------------|----------|-----|------------|---------|--------|")
    for s in ss:
        vals = [str(s.get(k,'?')) for k in fields_tbl]
        md.append(f"| {vals[0]} | {vals[1]} | {vals[2]}% | {vals[3]}% | {vals[4]} | {vals[5]} | {vals[6]} | {vals[7]}% | {vals[8]}% | {vals[9]} | {vals[10]}s |")
    md.append("")

    md.append("## 3. Speed × temperature interaction\n")
    if st_:
        md.append("### Temperature summary\n")
        md.append("| T | Runs | 2mm10° | 5mm10° | Reach10mm10° | Regress% | FinalDist | θErr | RW% | WasteRatio |")
        md.append("|---|------|--------|--------|-------------|----------|-----------|------|-----|------------|")
        for s in st_:
            md.append(f"| {s['temperature']} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['success_pose_5mm_10deg_rate']}% | {s['reached_pose_10mm_10deg_once_rate']}% | {s['regression_rate']}% | {s['mean_final_pos_dist_m']} | {s.get('mean_final_theta_error_deg','?')} | {s['random_walk_rate']}% | {s['mean_wasted_motion_ratio']} |")
        md.append("")

    if sst:
        md.append("### Speed × Temperature interaction table\n")
        # Pivot: rows=speed, cols=temperature
        temps = sorted(set(float(r.get('temperature',0)) for r in rows))
        speeds = sorted(set(float(r.get('speed_mps',0)) for r in rows))
        md.append("| Speed \\ T | " + " | ".join(f"T={t}" for t in temps) + " |")
        md.append("|" + "---|" * (len(temps)+1))
        for sp in speeds:
            vals = []
            for t in temps:
                key = (str(sp) if sp == int(sp) else str(sp), str(t) if t == int(t) else str(t))
                # try both representations
                s = None
                for (spk, tmk) in by_speed_temp:
                    try:
                        if abs(float(spk)-sp) < 0.001 and abs(float(tmk)-t) < 0.001:
                            s = summarize(by_speed_temp[(spk,tmk)], f"sp={sp}_T={t}")
                            break
                    except ValueError:
                        continue
                if s:
                    vals.append(f"{s['success_pose_2mm_10deg_rate']}%")
                else:
                    vals.append("?")
            md.append(f"| speed={sp} | " + " | ".join(vals) + " |")
        md.append("")
        md.append("**Sweet spots**: speed-temperature combinations where success ≥ average AND wasted_motion_ratio ≤ average.")
        avg_succ = overall.get('success_pose_2mm_10deg_rate', 0)
        avg_waste = overall.get('mean_wasted_motion_ratio_capped', overall.get('mean_wasted_motion_ratio', 999))
        sweet = []
        for (sp, tm) in by_speed_temp:
            s = summarize(by_speed_temp[(sp,tm)], f"sp={sp}_T={tm}")
            if s and s['success_pose_2mm_10deg_rate'] >= avg_succ and s.get('mean_wasted_motion_ratio_capped', s.get('mean_wasted_motion_ratio', 999)) <= avg_waste:
                sweet.append((sp, tm, s))
        if sweet:
            for sp, tm, s in sorted(sweet, key=lambda x: calc_score(x[2]), reverse=True):
                wcap = s.get('mean_wasted_motion_ratio_capped', s.get('mean_wasted_motion_ratio', 0))
                md.append(f"- speed={sp}, T={tm}: succ={s['success_pose_2mm_10deg_rate']}%, waste_capped={wcap:.1f}, score={calc_score(s):.1f}")
        else:
            md.append("- No sweet spots above both averages.")
        md.append("")

    md.append("## 4. Random-walk diagnostics\n")
    if sp075:
        md.append(f"### speed=0.75 analysis\n")
        md.append(f"- ee_path_length_m: {sp075.get('mean_ee_path_length_m','?')}")
        md.append(f"- wasted_motion_ratio: {sp075.get('mean_wasted_motion_ratio','?')}")
        md.append(f"- wasted_motion_ratio_capped: {sp075.get('mean_wasted_motion_ratio_capped','?')}")
        md.append(f"- random_walk_rate: {sp075.get('random_walk_rate','?')}%")
        md.append(f"- inefficient_success_rate: {sp075.get('inefficient_success_rate','?')}%")
    md.append("")
    for s in ss:
        md.append(f"- speed={s['speed_mps']}: EE_path={s['mean_ee_path_length_m']:.3f}m, waste={s['mean_wasted_motion_ratio']:.1f}, RW={s['random_walk_rate']}%, IneffSucc={s['inefficient_success_rate']}%")
    md.append("")
    # Compare 0.75 vs lower speeds
    if sp075:
        for s in ss:
            if s['speed_mps'] < 0.7:
                ratio = sp075['mean_ee_path_length_m'] / max(s['mean_ee_path_length_m'], 1e-9)
                waste_ratio = sp075['mean_wasted_motion_ratio'] / max(s['mean_wasted_motion_ratio'], 1e-9)
                md.append(f"- speed=0.75 vs speed={s['speed_mps']}: EE_path={ratio:.1f}× higher, waste={waste_ratio:.1f}× higher")
    md.append("")

    md.append("## 5. Family-wise behavior\n")
    md.append("| Family | Runs | 2mm10° | 5mm10° | Reach10mm10° | EE_Path | Waste | RW% | FinalDist | Top Failure |")
    md.append("|--------|------|--------|--------|-------------|---------|-------|-----|-----------|------------|")
    for s in sf_:
        fam = s['family']
        ftop = "?"
        if fam in by_family and by_family[fam]:
            ftypes = [r.get('failure_type','?') for r in by_family[fam] if r.get('failure_type','') not in ('success','')]
            if ftypes:
                ftop = max(set(ftypes), key=ftypes.count)
        md.append(f"| {fam} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['success_pose_5mm_10deg_rate']}% | {s['reached_pose_10mm_10deg_once_rate']}% | {s['mean_ee_path_length_m']} | {s['mean_wasted_motion_ratio']} | {s['random_walk_rate']}% | {s['mean_final_pos_dist_m']} | {ftop} |")
    md.append("")

    # Detailed sub-analysis per family × speed
    for fam_key in sorted(by_family):
        fam_rows = by_family[fam_key]
        fam_by_sp = defaultdict(list)
        for r in fam_rows:
            sp = r.get('speed_mps','?')
            fam_by_sp[sp].append(r)
        if len(fam_by_sp) <= 1:
            continue
        md.append(f"### {fam_key}: speed breakdown\n")
        md.append(f"| Speed | Runs | 2mm10° | EE_Path | Waste | ProgEff |")
        md.append("|-------|------|--------|---------|-------|---------|")
        for sp in sorted(fam_by_sp, key=lambda x: float(x) if x.replace('.','').isdigit() else 999):
            s = summarize(fam_by_sp[sp], f"{fam_key}_sp{sp}")
            if s:
                md.append(f"| {sp} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['mean_ee_path_length_m']} | {s['mean_wasted_motion_ratio']} | {s['mean_progress_efficiency_ee']} |")
        md.append("")

    md.append("## 6. Training data quality\n")
    # Best speed for training data
    best_train = max(ss, key=lambda s: s['mean_progress_efficiency_ee'] - s['random_walk_rate']*0.5, default=None)
    if best_train:
        md.append(f"- **Cleanest trajectories**: speed={best_train['speed_mps']} (prog_eff={best_train['mean_progress_efficiency_ee']:.4f}, RW={best_train['random_walk_rate']}%)")
    if sp075:
        md.append(f"- **speed=0.75 caution**: wasted_motion_ratio={sp075['mean_wasted_motion_ratio']:.1f}, RW={sp075['random_walk_rate']}% — many trajectories contain excessive wandering.")
        md.append(f"- **Recommendation**: avoid speed=0.75 as primary training data. Use it as aggressive oracle / failure-rich augmentation instead.")
    md.append(f"- **Preferred speeds for learned dynamics**: lower speeds with high progress_efficiency_ee and low random_walk_rate.")
    md.append("")

    md.append("## 7. Recommendation for Stage 2C\n")
    # Top 3 speed-temp combos
    top3 = sorted(sst, key=lambda s: calc_score(s), reverse=True)[:3]
    for i, s in enumerate(top3, 1):
        md.append(f"{i}. speed={s['speed_mps']}, T={s['temperature']}: succ={s['success_pose_2mm_10deg_rate']}%, waste={s['mean_wasted_motion_ratio']:.1f}, score={calc_score(s):.1f}")
    md.append("")
    md.append("Stage 2C will scan num_samples=[1024,2048] and init_std=[0.5,0.7,1.0] around these top speed-temperature configurations.")
    md.append("**Do NOT auto-run Stage 2C. Await user decision.**")
    md.append("")

    # Ranking score for all configs
    md.append("## Appendix: Top 10 speed-temperature combinations by score\n")
    md.append("| Rank | Speed | T | 2mm10° | 5mm10° | Reach10mm10° | RW% | IneffSucc% | Waste | ProgEff | Score |")
    md.append("|------|-------|---|--------|--------|-------------|-----|------------|-------|---------|-------|")
    for i, s in enumerate(sorted(sst, key=lambda s: calc_score(s), reverse=True)[:10], 1):
        sc = calc_score(s)
        md.append(f"| {i} | {s['speed_mps']} | {s['temperature']} | {s['success_pose_2mm_10deg_rate']}% | {s['success_pose_5mm_10deg_rate']}% | {s['reached_pose_10mm_10deg_once_rate']}% | {s['random_walk_rate']}% | {s['inefficient_success_rate']}% | {s['mean_wasted_motion_ratio']} | {s['mean_progress_efficiency_ee']} | {sc:.1f} |")
    md.append("")

    md.append(f"\n*Raw data: {manifest_path}*")
    md.append(f"\n*Generated: MPPI Stage 2B Speed Sweep*")

    report_path = out_dir / "stage2b_speed_summary.md"
    report_path.write_text('\n'.join(md))
    print(f"\n  Report: {report_path}")
    print(f"  Outputs: {out_dir}/")
    for f in sorted(out_dir.glob("summary_*.csv")):
        print(f"    {f.name}")
    for f in sorted(out_dir.glob("top_*.csv")):
        print(f"    {f.name}")
    for f in sorted(out_dir.glob("*diagnostics*.csv")):
        print(f"    {f.name}")
    for f in sorted(out_dir.glob("training_*.csv")):
        print(f"    {f.name}")

if __name__ == "__main__":
    main()
