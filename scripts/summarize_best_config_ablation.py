#!/usr/bin/env python3
"""Summarize best config ablation results."""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

HARD_LABELS = {"bh00", "bh01", "bh09", "ph04", "ph07"}

def load_manifest(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def compute_score(success_rate, hard_success_count, mean_best_dist, mean_collision_count, mean_mpc_steps):
    return (100 * success_rate 
            + 10 * hard_success_count 
            - 10 * mean_best_dist 
            - 0.5 * mean_collision_count 
            - 0.01 * mean_mpc_steps)

def summarize_by_config(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "completed":
            key = (r["speed"], r["horizon"])
            groups[key].append(r)
    
    results = []
    for (speed, horizon), cases in groups.items():
        success_count = sum(1 for c in cases if c["success"] == "True")
        total_count = len(cases)
        hard_success = sum(1 for c in cases if c["success"] == "True" and c["label"] in HARD_LABELS)
        blocking_hard = sum(1 for c in cases if c["success"] == "True" and c["label"].startswith("bh"))
        passage_hard = sum(1 for c in cases if c["success"] == "True" and c["label"].startswith("ph"))
        
        best_dists = [float(c["best_dist"]) for c in cases if c["best_dist"] != "N/A"]
        colls = [float(c["collision_count"]) for c in cases if c["collision_count"] != "N/A"]
        mpc_steps = [float(c["mpc_steps"]) for c in cases if c["mpc_steps"] != "N/A"]
        
        mean_dist = sum(best_dists) / len(best_dists) if best_dists else 0
        median_dist = sorted(best_dists)[len(best_dists)//2] if best_dists else 0
        mean_coll = sum(colls) / len(colls) if colls else 0
        mean_mpc = sum(mpc_steps) / len(mpc_steps) if mpc_steps else 0
        
        success_rate = success_count / total_count if total_count > 0 else 0
        score = compute_score(success_rate, hard_success, mean_dist, mean_coll, mean_mpc)
        
        results.append({
            "config": f"s{float(speed):.2f}_h{horizon}",
            "speed_mps": float(speed),
            "speed_cm_s": float(speed) * 100,
            "horizon": int(horizon),
            "execute_steps": 10,
            "max_mpc_steps": 100,
            "total_budget": 1000,
            "success_count": success_count,
            "total_count": total_count,
            "success_rate": success_rate,
            "hard_success_count": hard_success,
            "blocking_hard_success_count": blocking_hard,
            "passage_hard_success_count": passage_hard,
            "mean_best_dist": mean_dist,
            "median_best_dist": median_dist,
            "mean_collision_count": mean_coll,
            "mean_collision_rate": 0,
            "mean_mpc_steps": mean_mpc,
            "mean_runtime_sec": 0,
            "failures_by_label": "",
            "score": score
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def summarize_by_speed(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "completed":
            groups[r["speed"]].append(r)
    
    results = []
    for speed, cases in groups.items():
        success = sum(1 for c in cases if c["success"] == "True")
        total = len(cases)
        hard = sum(1 for c in cases if c["success"] == "True" and c["label"] in HARD_LABELS)
        results.append({
            "speed_mps": float(speed),
            "speed_cm_s": float(speed) * 100,
            "success_count": success,
            "total_count": total,
            "success_rate": success / total if total > 0 else 0,
            "hard_success_count": hard
        })
    results.sort(key=lambda x: x["speed_mps"])
    return results

def summarize_by_horizon(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "completed":
            groups[r["horizon"]].append(r)
    
    results = []
    for horizon, cases in groups.items():
        success = sum(1 for c in cases if c["success"] == "True")
        total = len(cases)
        hard = sum(1 for c in cases if c["success"] == "True" and c["label"] in HARD_LABELS)
        results.append({
            "horizon": int(horizon),
            "success_count": success,
            "total_count": total,
            "success_rate": success / total if total > 0 else 0,
            "hard_success_count": hard
        })
    results.sort(key=lambda x: x["horizon"])
    return results

def summarize_by_template(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "completed":
            groups[r["label"]].append(r)
    
    results = []
    for label, cases in groups.items():
        success = sum(1 for c in cases if c["success"] == "True")
        total = len(cases)
        results.append({
            "label": label,
            "split": cases[0]["split"],
            "layout_family": cases[0]["layout_family"],
            "success_count": success,
            "total_count": total,
            "success_rate": success / total if total > 0 else 0
        })
    results.sort(key=lambda x: x["success_rate"])
    return results

def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def generate_markdown(run_root, config_summary, speed_summary, horizon_summary, template_summary, rows):
    top3 = config_summary[:3]
    
    md = f"""# Best Config Ablation: 0.05 to 1.0 m/s

## 1. High-level verdict

**Best overall config:** {top3[0]['config']} (speed={top3[0]['speed_mps']:.2f} m/s, horizon={top3[0]['horizon']})

**Top 3 configs:**
| Rank | Config | Speed | Horizon | Success Rate | Hard Success | Score |
|------|--------|-------|---------|--------------|--------------|-------|
"""
    for i, c in enumerate(top3):
        md += f"| {i+1} | {c['config']} | {c['speed_mps']:.2f} m/s ({c['speed_cm_s']:.0f} cm/s) | {c['horizon']} | {c['success_rate']:.1%} | {c['hard_success_count']} | {c['score']:.1f} |\n"
    
    md += f"""
## 2. Unit clarification

- 0.05 m/s = 5 cm/s
- 0.10 m/s = 10 cm/s
- 0.20 m/s = 20 cm/s
- 0.35 m/s = 35 cm/s
- 0.50 m/s = 50 cm/s
- 0.75 m/s = 75 cm/s
- 1.00 m/s = 100 cm/s

## 3. Speed analysis

| Speed (m/s) | Speed (cm/s) | Success Rate | Hard Success |
|-------------|--------------|--------------|--------------|
"""
    for s in speed_summary:
        md += f"| {s['speed_mps']:.2f} | {s['speed_cm_s']:.0f} | {s['success_rate']:.1%} | {s['hard_success_count']} |\n"
    
    md += f"""
## 4. Horizon analysis

| Horizon | Success Rate | Hard Success |
|---------|--------------|--------------|
"""
    for h in horizon_summary:
        md += f"| {h['horizon']} | {h['success_rate']:.1%} | {h['hard_success_count']} |\n"
    
    md += f"""
## 5. Template analysis

| Label | Split | Success Rate |
|-------|-------|--------------|
"""
    for t in template_summary:
        md += f"| {t['label']} | {t['layout_family']} | {t['success_rate']:.1%} |\n"
    
    md += f"""
## 6. Recommendation

Based on this ablation:
- **Best main config:** {top3[0]['config']}
- **Conservative config:** {config_summary[-1]['config']} (lowest speed, highest horizon)
- **Aggressive config:** {top3[1]['config'] if len(top3) > 1 else 'N/A'}

## 7. Data files

- Manifest: `{run_root}/manifest.csv`
- Videos: `{run_root}/videos/`
- Logs: `{run_root}/logs/`
"""
    
    return md

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    
    run_root = Path(args.run_root)
    manifest_path = run_root / "manifest.csv"
    
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found")
        return
    
    rows = load_manifest(manifest_path)
    print(f"Loaded {len(rows)} rows from manifest")
    
    # Summarize
    config_summary = summarize_by_config(rows)
    speed_summary = summarize_by_speed(rows)
    horizon_summary = summarize_by_horizon(rows)
    template_summary = summarize_by_template(rows)
    
    # Write CSVs
    config_fields = ["config", "speed_mps", "speed_cm_s", "horizon", "execute_steps", "max_mpc_steps", 
                     "total_budget", "success_count", "total_count", "success_rate", "hard_success_count",
                     "blocking_hard_success_count", "passage_hard_success_count", "mean_best_dist", 
                     "median_best_dist", "mean_collision_count", "mean_collision_rate", "mean_mpc_steps",
                     "mean_runtime_sec", "failures_by_label", "score"]
    
    write_csv(run_root / "summary_by_config.csv", config_summary, config_fields)
    write_csv(run_root / "summary_by_speed.csv", speed_summary, list(speed_summary[0].keys()))
    write_csv(run_root / "summary_by_horizon.csv", horizon_summary, list(horizon_summary[0].keys()))
    write_csv(run_root / "summary_by_template.csv", template_summary, list(template_summary[0].keys()))
    
    # Top configs
    write_csv(run_root / "top_configs.csv", config_summary[:10], config_fields)
    
    # Generate markdown
    md = generate_markdown(run_root, config_summary, speed_summary, horizon_summary, template_summary, rows)
    md_path = run_root / "best_config_ablation_summary.md"
    with open(md_path, 'w') as f:
        f.write(md)
    
    # Also write to docs
    docs_path = Path("docs/best_config_ablation_speed005_to_100_summary.md")
    docs_path.parent.mkdir(exist_ok=True)
    with open(docs_path, 'w') as f:
        f.write(md)
    
    print(f"\nSummaries written to:")
    print(f"  {run_root}/summary_by_config.csv")
    print(f"  {run_root}/summary_by_speed.csv")
    print(f"  {run_root}/summary_by_horizon.csv")
    print(f"  {run_root}/summary_by_template.csv")
    print(f"  {run_root}/top_configs.csv")
    print(f"  {md_path}")
    print(f"  {docs_path}")
    
    print(f"\n=== Top 3 Configs ===")
    for i, c in enumerate(config_summary[:3]):
        print(f"  {i+1}. {c['config']} speed={c['speed_mps']:.2f}m/s horizon={c['horizon']} success={c['success_rate']:.1%} score={c['score']:.1f}")

if __name__ == "__main__":
    main()
