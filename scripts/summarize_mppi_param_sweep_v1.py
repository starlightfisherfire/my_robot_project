#!/usr/bin/env python3
"""Phase M: Summarize MPPI parameter sweep results.

Usage:
  python scripts/summarize_mppi_param_sweep_v1.py --run-root RUN_ROOT

Outputs:
  - CSV summaries by temperature, config, family, type, template
  - Top config ranking
  - Training data summary
  - Markdown checkpoint summary doc
"""
from __future__ import annotations

import argparse, csv, json, os, sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--data-dir", default="data/sim/mppi_sweep_v1")
    return p.parse_args()


def load_manifest(run_root: Path) -> list[dict]:
    manifest_path = run_root / "manifest.csv"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)
    rows = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_episodes_metadata(data_dir: Path) -> list[dict]:
    meta_path = data_dir / "metadata" / "episodes.jsonl"
    if not meta_path.exists():
        return []
    episodes = []
    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def safe_int(v, default=0):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def safe_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)


def compute_ranking_score(row: dict) -> float:
    """Section N: config ranking score."""
    success = 1.0 if safe_bool(row.get("success")) else 0.0
    family = row.get("family", "")
    best_dist = safe_float(row.get("best_dist"), 1.0)
    collision = safe_float(row.get("collision_count"), 0)
    mpc_steps = safe_float(row.get("mpc_steps"), 100)
    collapse_rate = safe_float(row.get("collapse_rate"), 0)

    score = 100.0 * success
    if "bypass" in family:
        score += 12.0 * success
    if family == "blocking_hard":
        score += 8.0 * success
    if family == "passage_direct_narrow":
        score += 8.0 * success
    score -= 10.0 * best_dist
    score -= 0.5 * collision
    score -= 0.01 * mpc_steps
    score -= 5.0 * collapse_rate
    return score


def summarize_by_temperature(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        T = r.get("temperature", "?")
        groups[T].append(r)

    result = []
    for T in sorted(groups.keys(), key=lambda x: float(x) if x.replace(".","").replace("-","").isdigit() else 0):
        grp = groups[T]
        total = len(grp)
        successes = sum(1 for r in grp if safe_bool(r.get("success")))
        result.append({
            "temperature": T,
            "total_runs": total,
            "success_count": successes,
            "success_rate": round(successes / max(total, 1), 4),
            "mean_best_dist": round(sum(safe_float(r.get("best_dist"), 0) for r in grp) / max(total, 1), 4),
            "mean_collision": round(sum(safe_float(r.get("collision_count"), 0) for r in grp) / max(total, 1), 2),
            "mean_runtime": round(sum(safe_float(r.get("runtime_sec"), 0) for r in grp) / max(total, 1), 1),
            "mean_collapse_rate": round(sum(safe_float(r.get("collapse_rate"), 0) for r in grp) / max(total, 1), 4),
            "collapse_flags": sum(1 for r in grp if safe_bool(r.get("temperature_collapse_flag"))),
        })
    return result


def summarize_by_family(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        family = r.get("family", "?")
        groups[family].append(r)

    order = [
        "open", "blocking_easy", "blocking_medium", "blocking_hard",
        "passage_direct_wide", "passage_direct_medium", "passage_direct_narrow",
        "passage_bypass_wide", "passage_bypass_medium", "passage_bypass_narrow",
    ]
    result = []
    for family in order:
        if family not in groups:
            continue
        grp = groups[family]
        total = len(grp)
        successes = sum(1 for r in grp if safe_bool(r.get("success")))
        result.append({
            "family": family,
            "type": grp[0].get("type", ""),
            "obstacle_count": grp[0].get("obstacle_count", ""),
            "total_runs": total,
            "success_count": successes,
            "success_rate": round(successes / max(total, 1), 4),
            "mean_best_dist": round(sum(safe_float(r.get("best_dist"), 0) for r in grp) / max(total, 1), 4),
            "mean_collision": round(sum(safe_float(r.get("collision_count"), 0) for r in grp) / max(total, 1), 2),
        })
    return result


def summarize_by_type(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        t = r.get("type", "?")
        groups[t].append(r)

    order = ["open", "blocking", "passage_direct", "passage_bypass"]
    result = []
    for t in order:
        if t not in groups:
            continue
        grp = groups[t]
        total = len(grp)
        successes = sum(1 for r in grp if safe_bool(r.get("success")))
        result.append({
            "type": t,
            "total_runs": total,
            "success_count": successes,
            "success_rate": round(successes / max(total, 1), 4),
            "mean_best_dist": round(sum(safe_float(r.get("best_dist"), 0) for r in grp) / max(total, 1), 4),
            "mean_collision": round(sum(safe_float(r.get("collision_count"), 0) for r in grp) / max(total, 1), 2),
        })
    return result


def summarize_by_config(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        cfg = f"T={r.get('temperature','?')}"
        groups[cfg].append(r)

    configs = []
    for cfg, grp in groups.items():
        T = grp[0].get("temperature", "?")
        total = len(grp)
        successes = sum(1 for r in grp if safe_bool(r.get("success")))
        by_family = defaultdict(lambda: {"total": 0, "success": 0})
        for r in grp:
            f = r.get("family", "?")
            by_family[f]["total"] += 1
            if safe_bool(r.get("success")):
                by_family[f]["success"] += 1
        configs.append({
            "temperature": T,
            "total_runs": total,
            "success_count": successes,
            "success_rate": round(successes / max(total, 1), 4),
            "mean_best_dist": round(sum(safe_float(r.get("best_dist"), 0) for r in grp) / max(total, 1), 4),
            "mean_collapse_rate": round(sum(safe_float(r.get("collapse_rate"), 0) for r in grp) / max(total, 1), 4),
            **{f"bypass_success_{f}": by_family[f]["success"] for f in ["passage_bypass_wide", "passage_bypass_medium", "passage_bypass_narrow"] if f in by_family},
            **{f"hard_success_{f}": by_family[f]["success"] for f in ["blocking_hard", "passage_direct_narrow"] if f in by_family},
        })
    return sorted(configs, key=lambda x: safe_float(x.get("temperature", "0")))


def top_configs(rows: list[dict], top_n: int = 20) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        cfg = f"mppi_T{r.get('temperature','?')}_{r.get('family','?')}"
        groups[cfg].append(r)

    scored = []
    for cfg, grp in groups.items():
        avg_score = sum(compute_ranking_score(r) for r in grp) / len(grp)
        r0 = grp[0]
        scored.append({
            "config": cfg,
            "temperature": r0.get("temperature", "?"),
            "family": r0.get("family", "?"),
            "type": r0.get("type", "?"),
            "runs": len(grp),
            "success_rate": round(sum(1 for r in grp if safe_bool(r.get("success"))) / len(grp), 4),
            "mean_best_dist": round(sum(safe_float(r.get("best_dist"), 0) for r in grp) / len(grp), 4),
            "score": round(avg_score, 2),
        })
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_n]


def training_data_summary(data_dir: Path, rows: list[dict]) -> dict:
    episodes = load_episodes_metadata(data_dir)
    if not episodes:
        return {"saved_episodes": 0, "saved_transitions": 0, "note": "No episode metadata found"}

    success_count = sum(1 for e in episodes if e.get("success"))
    failure_count = len(episodes) - success_count
    total_transitions = sum(e.get("num_transitions", 0) for e in episodes)

    contact_transitions = sum(
        e.get("contact_count", 0) for e in episodes if e.get("contact_count")
    )
    collision_transitions = sum(
        e.get("collision_count", 0) for e in episodes if e.get("collision_count")
    )

    family_dist = defaultdict(int)
    for e in episodes:
        family_dist[e.get("family", "?")] += 1

    type_dist = defaultdict(int)
    for e in episodes:
        type_dist[e.get("type", "?")] += 1

    # Calculate recommended sampling weights
    total_eps = len(episodes)
    sampling_weights = {}
    for family, count in family_dist.items():
        if count > 0:
            # Inverse frequency weighting, capped
            w = total_eps / (count * len(family_dist))
            sampling_weights[family] = round(min(w, 5.0), 2)

    # Bypass families get extra weight
    for f in ["passage_bypass_wide", "passage_bypass_medium", "passage_bypass_narrow"]:
        if f in sampling_weights and sampling_weights[f] < 5.0:
            sampling_weights[f] = min(sampling_weights[f] * 2, 5.0)

    return {
        "saved_episodes": len(episodes),
        "saved_transitions": total_transitions,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_failure_ratio": round(success_count / max(failure_count, 1), 2),
        "contact_transition_ratio": round(contact_transitions / max(total_transitions, 1), 4),
        "collision_transition_ratio": round(collision_transitions / max(total_transitions, 1), 4),
        "family_distribution": dict(family_dist),
        "type_distribution": dict(type_dist),
        "recommended_sampling_weights": sampling_weights,
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("(no data)\n")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def generate_markdown(
    run_root: Path, data_dir: Path, rows: list[dict],
    by_temp: list[dict], by_family: list[dict], by_type: list[dict],
    by_config: list[dict], top: list[dict], train_summary: dict,
    output_path: Path,
):
    total = len(rows)
    completed = sum(1 for r in rows if r.get("status") == "completed")
    failed = sum(1 for r in rows if r.get("status") == "failed")
    success_count = sum(1 for r in rows if safe_bool(r.get("success")))

    # Priority breakdown
    priority_counts = defaultdict(lambda: {"completed": 0, "success": 0})
    for r in rows:
        p = r.get("priority", "?")
        priority_counts[p]["completed"] += 1
        if safe_bool(r.get("success")):
            priority_counts[p]["success"] += 1

    lines = []
    def w(s=""):
        lines.append(s)

    w("# MPPI Parameter Sweep Checkpoint 8h Summary")
    w()
    w("## 1. What was completed")
    w()
    w(f"- **RUN_ROOT**: `{run_root}`")
    w(f"- Completed runs: {completed}")
    w(f"- Failed runs: {failed}")
    w(f"- Total runs in manifest: {total}")
    w(f"- Success count: {success_count}")
    w(f"- Saved episodes: {train_summary.get('saved_episodes', 0)}")
    w(f"- Saved transitions: {train_summary.get('saved_transitions', 0)}")
    w()
    for p in ["A", "B", "C"]:
        if p in priority_counts:
            pc = priority_counts[p]
            w(f"- Priority {p}: {pc['completed']} runs, {pc['success']} successes")
    w()
    w("> ⚠️ **Partial checkpoint result, not final conclusion.**")
    w()

    w("## 2. Template inventory")
    w()
    w("| Family | Type | Obstacles | Runs | Success Rate |")
    w("|--------|------|-----------|------|-------------|")
    for row in by_family:
        w(f"| {row['family']} | {row['type']} | {row['obstacle_count']} | "
          f"{row['total_runs']} | {row['success_rate']:.1%} |")
    w()

    w("## 3. Best temperature so far")
    w()
    w("### By success rate")
    w("| Temperature | Runs | Success Rate | Collapse Flags |")
    w("|-------------|------|-------------|----------------|")
    for row in by_temp:
        w(f"| {row['temperature']} | {row['total_runs']} | "
          f"{row['success_rate']:.1%} | {row['collapse_flags']} |")
    w()

    w("### By hard/bypass success")
    if by_config:
        w("| Temperature | bypass_wide | bypass_medium | bypass_narrow | hard | narrow |")
        w("|-------------|-------------|---------------|---------------|------|--------|")
        for row in by_config:
            w(f"| {row['temperature']} | {row.get('bypass_success_passage_bypass_wide', 0)} | "
              f"{row.get('bypass_success_passage_bypass_medium', 0)} | "
              f"{row.get('bypass_success_passage_bypass_narrow', 0)} | "
              f"{row.get('hard_success_blocking_hard', 0)} | "
              f"{row.get('hard_success_passage_direct_narrow', 0)} |")
    w()

    w("## 4. Temperature diagnostics")
    w()
    if by_temp:
        for row in by_temp:
            T = row["temperature"]
            collapse = row["mean_collapse_rate"]
            flags = row["collapse_flags"]
            if collapse > 0.5:
                w(f"- **T={T}**: ⚠️ High collapse rate ({collapse:.1%}) — temperature too low")
            elif collapse < 0.05 and row["success_rate"] < 0.5:
                w(f"- **T={T}**: Low collapse but low success — may need higher temperature for exploration")
            else:
                w(f"- **T={T}**: Collapse rate {collapse:.1%}, success rate {row['success_rate']:.1%}")
    w()

    w("## 5. Bypass analysis")
    w()
    bypass_families = [r for r in by_family if "bypass" in r.get("family", "")]
    if bypass_families:
        w("| Family | Total | Success | Success Rate | Mean Dist | Mean Collision |")
        w("|--------|-------|---------|-------------|-----------|---------------|")
        for row in bypass_families:
            w(f"| {row['family']} | {row['total_runs']} | {row['success_count']} | "
              f"{row['success_rate']:.1%} | {row['mean_best_dist']:.4f} | {row['mean_collision']} |")
        w()
        w("> Bypass templates test true obstacle avoidance (go-around vs go-through).")
        w("> If bypass success is low, MPPI may not be exploring lateral paths effectively.")
    else:
        w("No bypass data collected yet.")
    w()

    w("## 6. Blocking hard analysis")
    w()
    hard_rows = [r for r in rows if r.get("family") == "blocking_hard" and r.get("status") == "completed"]
    if hard_rows:
        hard_success = sum(1 for r in hard_rows if safe_bool(r.get("success")))
        w(f"- Blocking hard: {len(hard_rows)} runs, {hard_success} successes ({hard_success/max(len(hard_rows),1):.1%})")
        w(f"- Mean best dist: {sum(safe_float(r.get('best_dist'),0) for r in hard_rows)/max(len(hard_rows),1):.4f}")
    else:
        w("No blocking hard data collected yet.")
    w()

    w("## 7. Training data summary")
    w()
    w(f"- Saved episodes: {train_summary.get('saved_episodes', 0)}")
    w(f"- Saved transitions: {train_summary.get('saved_transitions', 0)}")
    w(f"- Success/Failure ratio: {train_summary.get('success_failure_ratio', 'N/A')}")
    w(f"- Contact transition ratio: {train_summary.get('contact_transition_ratio', 'N/A')}")
    w(f"- Collision transition ratio: {train_summary.get('collision_transition_ratio', 'N/A')}")
    w()

    fd = train_summary.get("family_distribution", {})
    if fd:
        w("### Family distribution")
        w("| Family | Episodes |")
        w("|--------|----------|")
        for f, c in sorted(fd.items()):
            w(f"| {f} | {c} |")
        w()

    sw = train_summary.get("recommended_sampling_weights", {})
    if sw:
        w("### Recommended sampling weights for training")
        w("| Family | Weight | Note |")
        w("|--------|--------|------|")
        for f, w_val in sorted(sw.items()):
            note = "⚠️ Oversample (bypass, low count)" if "bypass" in f else ""
            w(f"| {f} | {w_val} | {note} |")
        w()

    w("## 8. Recommendation for next 8h node")
    w()

    # Make recommendation based on data
    total_planned = 100
    completed_pct = completed / max(total_planned, 1)

    bypass_total = sum(r["total_runs"] for r in by_family if "bypass" in r.get("family", ""))
    bypass_done = sum(r["success_count"] for r in by_family if "bypass" in r.get("family", ""))

    if completed < 30:
        w("**A. Continue Stage 1 temperature sweep** — insufficient data for conclusions.")
        w(f"   Only {completed}/{total_planned} runs completed. Restart with `--phase1` or `--resume`.")
    elif completed_pct < 0.67:
        w(f"**A. Continue Stage 1** — {completed}/{total_planned} runs done ({completed_pct:.0%}).")
        w("   Resume to complete remaining priority groups.")
    elif completed_pct >= 0.67 and bypass_done >= 10:
        w("**B. Fixed top 2 temperature, begin Stage 2** — sufficient temperature data.")
        w("   Analyze temperature results, then sweep num_samples and init_std.")
    elif bypass_done < 5:
        w("**C. Prioritize bypass variants** — bypass data is sparse.")
        w("   Consider running more bypass template variations before moving to Stage 2.")
    else:
        w("**B. Stage 2 sample/std refinement** — temperature sweep has good coverage.")
        w("   Select best 2 temperatures and sweep num_samples × init_std.")

    w()
    w("> ⚠️ Do not auto-enter next stage. User must review and decide.")
    w()

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ Markdown summary written to {output_path}")


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    data_dir = Path(args.data_dir)

    if not run_root.exists():
        print(f"ERROR: run-root not found: {run_root}")
        sys.exit(1)

    rows = load_manifest(run_root)
    if not rows:
        print("WARNING: manifest is empty — no data to summarize")
        return

    completed_rows = [r for r in rows if r.get("status") == "completed"]
    print(f"Loaded {len(rows)} manifest rows ({len(completed_rows)} completed)")

    # Generate summaries
    by_temp = summarize_by_temperature(completed_rows)
    by_family = summarize_by_family(completed_rows)
    by_type = summarize_by_type(completed_rows)
    by_config = summarize_by_config(completed_rows)
    top = top_configs(completed_rows)
    train_summary = training_data_summary(data_dir, completed_rows)

    # Write CSVs
    write_csv(run_root / "summary_by_temperature.csv", by_temp)
    write_csv(run_root / "summary_by_config.csv", by_config)
    write_csv(run_root / "summary_by_family.csv", by_family)
    write_csv(run_root / "summary_by_type.csv", by_type)
    by_template = summarize_by_family(completed_rows)  # reuse family as template granularity
    write_csv(run_root / "summary_by_template.csv", by_template)
    write_csv(run_root / "top_mppi_configs.csv", top)

    # Training data CSV
    td_rows = [train_summary] if train_summary else []
    write_csv(run_root / "training_data_summary.csv", td_rows)

    print(f"✅ Written CSV summaries to {run_root}/")

    # Generate markdown
    md_path = PROJECT_ROOT / "docs" / "mppi_param_sweep_checkpoint8h_summary.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    generate_markdown(run_root, data_dir, completed_rows,
                      by_temp, by_family, by_type, by_config, top,
                      train_summary, md_path)


if __name__ == "__main__":
    main()
