#!/usr/bin/env python3
"""MPPI Stage 2A summarizer — 8 output files + full stats + scoring."""
import csv, sys, json, math
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
    try: return float(r.get(key, default))
    except: return default

def sb(r, key):
    return str(r.get(key,'false')).lower() in ('true','1','yes')

def rate(cnt, total):
    return round(cnt/total*100, 1) if total > 0 else 0.0

def avg(rows, key):
    vs = [sf(r,key) for r in rows if key in r]
    return round(sum(vs)/len(vs), 4) if vs else 0.0

def med(rows, key):
    vs = sorted(sf(r,key) for r in rows if key in r)
    return round(vs[len(vs)//2], 4) if vs else 0.0

def cnt(rows, key): return sum(1 for r in rows if sb(r, key))
def ft_cnt(rows, ft): return sum(1 for r in rows if r.get('failure_type','')==ft)

def summarize(rows, group_label='all'):
    t = len(rows)
    if t == 0: return {}
    s2 = cnt(rows, 'success_pose_2mm_10deg')
    s5 = cnt(rows, 'success_pose_5mm_10deg')
    s10 = cnt(rows, 'success_pose_10mm_10deg')
    return {
        'group': group_label, 'count': t,
        'success_pose_2mm_10deg_rate': rate(s2,t),
        'success_pos_1mm_rate': rate(cnt(rows,'success_pos_1mm'),t),
        'success_pos_2mm_rate': rate(cnt(rows,'success_pos_2mm'),t),
        'success_pos_5mm_rate': rate(cnt(rows,'success_pos_5mm'),t),
        'success_pos_10mm_rate': rate(cnt(rows,'success_pos_10mm'),t),
        'success_pos_50mm_rate': rate(cnt(rows,'success_pos_50mm'),t),
        'success_pose_5mm_10deg_rate': rate(s5,t),
        'success_pose_10mm_10deg_rate': rate(s10,t),
        'success_pose_5mm_5deg_rate': rate(cnt(rows,'success_pose_5mm_5deg'),t),
        'success_pose_10mm_5deg_rate': rate(cnt(rows,'success_pose_10mm_5deg'),t),
        'reached_pos_5mm_once_rate': rate(cnt(rows,'reached_pos_5mm_once'),t),
        'reached_pos_10mm_once_rate': rate(cnt(rows,'reached_pos_10mm_once'),t),
        'reached_pose_5mm_10deg_once_rate': rate(cnt(rows,'reached_pose_5mm_10deg_once'),t),
        'reached_pose_10mm_10deg_once_rate': rate(cnt(rows,'reached_pose_10mm_10deg_once'),t),
        'regression_rate': rate(cnt(rows,'regressed_after_near_success'),t),
        'no_contact_rate': rate(ft_cnt(rows,'no_contact'),t),
        'collision_stuck_rate': rate(ft_cnt(rows,'collision_stuck'),t),
        'low_progress_rate': rate(ft_cnt(rows,'low_progress'),t),
        'pos_theta_failed_rate': rate(ft_cnt(rows,'position_reached_theta_failed'),t),
        'near_success_5mm_rate': rate(ft_cnt(rows,'near_success_pose_5mm_10deg'),t),
        'near_success_10mm_rate': rate(ft_cnt(rows,'near_success_pose_10mm_10deg'),t),
        'regressed_rate': rate(ft_cnt(rows,'regressed_after_near_success'),t),
        'mean_final_pos_dist_m': avg(rows,'final_pos_dist_m'),
        'median_final_pos_dist_m': med(rows,'final_pos_dist_m'),
        'mean_best_pos_dist_m': avg(rows,'best_pos_dist_m'),
        'median_best_pos_dist_m': med(rows,'best_pos_dist_m'),
        'mean_final_theta_error_deg': avg(rows,'final_theta_error_deg'),
        'median_final_theta_error_deg': med(rows,'final_theta_error_deg'),
        'mean_best_pose_score': avg(rows,'best_pose_score'),
        'mean_collision_count': avg(rows,'collision_count'),
        'mean_contact_count': avg(rows,'contact_count'),
        'mean_mpc_steps': avg(rows,'mpc_steps'),
        'mean_total_env_steps': avg(rows,'total_env_steps'),
        'mean_runtime_sec': avg(rows,'runtime_sec'),
        'mean_total_progress_m': avg(rows,'total_progress_m'),
        'mean_initial_pos_dist_m': avg(rows,'initial_pos_dist_m'),
    }

def calc_score(s):
    return (
        120 * s['success_pose_2mm_10deg_rate'] / 100
        + 60 * s['success_pose_5mm_10deg_rate'] / 100
        + 30 * s['success_pose_10mm_10deg_rate'] / 100
        + 15 * s['reached_pose_10mm_10deg_once_rate'] / 100
        - 25 * s['regression_rate'] / 100
        - 15 * s['collision_stuck_rate'] / 100
        - 8 * s['no_contact_rate'] / 100
        - 10 * s['mean_final_pos_dist_m']
        - 0.5 * s['mean_collision_count']
    )

def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)

def main():
    if len(sys.argv) < 2:
        print("Usage: summarize_stage2a.py <manifest.csv>"); sys.exit(1)
    manifest = sys.argv[1]
    rows = load_manifest(manifest)
    print(f"Loaded {len(rows)} completed runs")

    out_dir = Path(manifest).parent
    fields = list(summarize(rows[:1]).keys()) if rows else []
    config_fields = ['temperature','execute_steps','max_mpc_steps','family'] + fields

    by_temp, by_exec, by_fam, by_te, by_cfg = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        t, es, f = str(r.get('temperature','?')), str(r.get('execute_steps','?')), r.get('family','?')
        by_temp[t].append(r); by_exec[es].append(r); by_fam[f].append(r)
        by_te[(t,es)].append(r)
        by_cfg[r.get('config',f'{t}_{es}_{f}')].append(r)

    # 1. by temperature
    st = []
    for k in sorted(by_temp, key=float):
        s = summarize(by_temp[k], f"T={k}")
        if s: s['temperature'] = float(k); st.append(s)
    write_csv(out_dir/"summary_by_temperature.csv", st, ['temperature']+fields)

    # 2. by execute_steps
    se = []
    for k in sorted(by_exec):
        s = summarize(by_exec[k], f"es={k}")
        if s: s['execute_steps'] = int(k); se.append(s)
    write_csv(out_dir/"summary_by_execute_steps.csv", se, ['execute_steps']+fields)

    # 3. by temp × execute
    ste = []
    for (t,es) in sorted(by_te, key=lambda x: (float(x[0]),int(x[1]))):
        s = summarize(by_te[(t,es)], f"T={t}_es={es}")
        if s: s['temperature'] = float(t); s['execute_steps'] = int(es); ste.append(s)
    write_csv(out_dir/"summary_by_temp_execute.csv", ste, ['temperature','execute_steps']+fields)

    # 4. by family
    sf_ = []
    for k in sorted(by_fam):
        sf_.append(summarize(by_fam[k], k))
    write_csv(out_dir/"summary_by_family.csv", sf_, fields)

    # 5. by config
    sc = []
    for k in sorted(by_cfg):
        s = summarize(by_cfg[k], k)
        if s and by_cfg[k]:
            r0 = by_cfg[k][0]
            s['temperature'] = float(r0.get('temperature',0))
            s['execute_steps'] = int(r0.get('execute_steps',10))
            s['max_mpc_steps'] = int(r0.get('max_mpc_steps',100))
            s['family'] = r0.get('family','')
            sc.append(s)
    write_csv(out_dir/"summary_by_config.csv", sc, config_fields)

    # 6. top by score
    tops = sorted(sc, key=calc_score, reverse=True)
    for r in tops: r['score'] = round(calc_score(r), 2)
    write_csv(out_dir/"top_mppi_stage2a_configs.csv", tops, ['score']+config_fields)

    # 7. training data summary
    td = [
        ('total_episodes', len(rows)),
        ('total_transitions', sum(sf(r,'total_env_steps',0) for r in rows)),
        ('success_count', sum(1 for r in rows if sb(r,'success'))),
        ('failure_count', sum(1 for r in rows if not sb(r,'success'))),
        ('success_ratio_pct', str(rate(sum(1 for r in rows if sb(r,'success')), len(rows)))),
        ('total_contact', sum(sf(r,'contact_count',0) for r in rows)),
        ('total_collision', sum(sf(r,'collision_count',0) for r in rows)),
        ('family_count', len(set(r.get('family','') for r in rows))),
        ('execute_settings', len(set(r.get('execute_steps','') for r in rows))),
        ('temperature_count', len(set(r.get('temperature','') for r in rows))),
        ('state16_training_ready', 'false'),
    ]
    write_csv(out_dir/"training_data_summary.csv",
              [{'metric':m,'value':str(v)} for m,v in td],
              ['metric','value'])

    # 8. Markdown
    overall = summarize(rows)
    md = []
    md.append("# MPPI Stage 2A Temperature × Execute-Steps Summary\n")

    md.append("## 1. Success definition\n")
    md.append("- **Primary early stop**: pose_success_2mm_10deg (position < 2mm AND angle < 10°)")
    md.append("- 1mm position-only retained as strict precision metric")
    md.append("- 5mm/10mm metrics indicate near-success / practical control ability")
    md.append("- best_pos_dist vs final_pos_dist identifies overshoot / regression")
    md.append("- execute_steps=20 with max_mpc_steps=50 vs es=10 with mpc=100 — equal total_budget=1000\n")

    md.append("## 2. Overall best config\n")
    if tops:
        best = tops[0]
        md.append(f"- **Best**: {best['group']} | score={best['score']} | 2mm10°={best['success_pose_2mm_10deg_rate']}%")
        md.append("- Top 3:")
        for t in tops[:3]:
            md.append(f"  - {t['group']}: score={t['score']} (2mm10°={t['success_pose_2mm_10deg_rate']}%)")
    md.append("")

    md.append("## 3. Temperature fine sweep\n")
    md.append("| T | Count | 2mm10° | Pos1mm | Pos5mm | Reach5mm | Reach10mm10° | Regress | FinalDist | θErr |")
    md.append("|---|-------|--------|--------|--------|----------|-------------|---------|----------|------|")
    for s in st:
        md.append(f"| {s['temperature']} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['success_pos_1mm_rate']}% | {s['success_pos_5mm_rate']}% | {s['reached_pos_5mm_once_rate']}% | {s['reached_pose_10mm_10deg_once_rate']}% | {s['regression_rate']}% | {s['mean_final_pos_dist_m']} | {s['mean_final_theta_error_deg']}° |")
    md.append("")

    md.append("## 4. Execute steps analysis\n")
    md.append("| ES | Count | 2mm10° | 5mm10° | Reach10mm10° | Regress | Collision | FinalDist | BestDist | Runtime |")
    md.append("|---|-------|--------|--------|-------------|---------|----------|----------|---------|--------|")
    for s in se:
        md.append(f"| {s['execute_steps']} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['success_pose_5mm_10deg_rate']}% | {s['reached_pose_10mm_10deg_once_rate']}% | {s['regression_rate']}% | {s['mean_collision_count']} | {s['mean_final_pos_dist_m']} | {s['mean_best_pos_dist_m']} | {s['mean_runtime_sec']:.0f}s |")
    e10 = next((s for s in se if str(s.get('execute_steps',''))=='10'), None)
    e20 = next((s for s in se if str(s.get('execute_steps',''))=='20'), None)
    if e10 and e20:
        w = "es=20" if e20['success_pose_2mm_10deg_rate'] > e10['success_pose_2mm_10deg_rate'] else "es=10"
        md.append(f"\n**Verdict**: {w} has higher primary success rate.")
        if e20.get('regression_rate',0) > e10.get('regression_rate',0):
            md.append("⚠️ es=20 shows higher regression — long open-loop execution causes overshoot/drift.")
    md.append("")

    md.append("## 5. Family-wise behavior\n")
    md.append("| Family | Count | 2mm10° | Pos5mm | Reach10mm10° | Regress | FinalDist | Top Failure |")
    md.append("|---|---|--------|--------|-------------|---------|----------|------------|")
    for s in sf_:
        fam = s['group']
        ftop = max(set(r.get('failure_type','?') for r in by_fam[fam]),
                   key=lambda ft: sum(1 for r in by_fam[fam] if r.get('failure_type','')==ft))
        md.append(f"| {fam} | {s['count']} | {s['success_pose_2mm_10deg_rate']}% | {s['success_pos_5mm_rate']}% | {s['reached_pose_10mm_10deg_once_rate']}% | {s['regression_rate']}% | {s['mean_final_pos_dist_m']} | {ftop} |")
    md.append("")

    md.append("## 6. Training data summary\n")
    for m,v in td:
        md.append(f"- {m}: {v}")
    md.append("")
    md.append("> **state16_training_ready: false**")
    md.append("> Stage 2A episode npz is intended for planner-rollout diagnostics.")
    md.append("> It is not yet the canonical state16 training dataset.")
    md.append("> For learned dynamics training, use data/sim/layout_ood_state16_v0")
    md.append("> generated by collect_layout_ood_state16.py, or add a dedicated state16 writer later.\n")

    md.append("## 7. Recommendation for Stage 2B\n")
    if tops:
        md.append("Top configs for next round:")
        for t in tops[:4]:
            md.append(f"- {t['group']}: score={t['score']}")
        md.append("\nDo NOT auto-run Stage 2B.")
    md.append("")

    md.append(f"\n*Raw data: {manifest}*")
    (out_dir/"stage2a_summary.md").write_text('\n'.join(md))
    print(f"\n  Report: {out_dir/'stage2a_summary.md'}")
    print(f"  Outputs: {out_dir}/")

if __name__ == "__main__":
    main()
